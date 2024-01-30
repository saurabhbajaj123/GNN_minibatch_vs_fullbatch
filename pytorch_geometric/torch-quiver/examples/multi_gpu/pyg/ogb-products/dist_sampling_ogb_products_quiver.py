# Reaches around 0.7870 ± 0.0036 test accuracy.
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv, GATConv
# from torch_geometric.nn.models import GAT
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
from parser import create_parser
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from datetime import timedelta

####################
# Import Quiver
####################
import quiver

import warnings
warnings.filterwarnings("ignore")

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        # self.skips = torch.nn.ModuleList()
        # self.skips.append(Lin(in_channels, hidden_channels * heads))
        # for _ in range(num_layers - 2):
        #     self.skips.append(
        #         Lin(hidden_channels * heads, hidden_channels * heads))
        # self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for skip in self.skips:
        #     skip.reset_parameters()
        


    def forward(self, x, adjs):
        # print(f"len(edge_index) = {len(edge_index)}")
        # # for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
        # for i, (conv) in enumerate(self.convs):
        #     x = conv(x, edge_index) # + skip(x)
        #     if i != self.num_layers - 1:
        #         x = F.elu(x)
        #         x = F.dropout(x, p=0.5, training=self.training)
        # return x
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index) # + self.skips[i](x)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                # pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        # total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                # total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


def run(rank, world_size, quiver_sampler, quiver_feature, y, edge_index, split_idx, num_features, num_classes, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(days=1))
    torch.cuda.set_device(rank)

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True)

    if rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=1000000,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    if "sage" in args.model.lower():
        model = SAGE(num_features, args.n_hidden, num_classes, num_layers=args.n_layers).to(rank)
    elif "gat" in args.model.lower():
        model = GAT(num_features, args.n_hidden, out_channels=num_classes,num_layers=args.n_layers, heads=args.heads).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = y.to(rank)
    train_time = 0
    count = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()

        epoch_start = time.time()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]
            
            optimizer.zero_grad()
            out = model(quiver_feature[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

        dist.barrier()
        train_time += time.time() - epoch_start
        count += 1
        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')

        if rank == 0 and epoch % args.log_every == 0:  # We evaluate on a single GPU for now
            model.cpu()
            model.eval()
            with torch.no_grad():
                # out = model.module.inference(quiver_feature, rank, subgraph_loader)
                out = model.module.inference(quiver_feature, 'cpu', subgraph_loader)
            res = out.argmax(dim=-1) == y.cpu()
            acc1 = int(res[train_idx].sum()) / train_idx.numel()
            acc2 = int(res[val_idx].sum()) / val_idx.numel()
            acc3 = int(res[test_idx].sum()) / test_idx.numel()
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
            model.cuda(rank)
        dist.barrier()
    if rank == 0:
        print(f"Time per epoch = {train_time/count}")
        print(f"epoch per second = {count/train_time}")
        print(f"total train time = {train_time}")
        print(f"num epochs = {count}")
    dist.destroy_process_group()

def main():
    args = create_parser()

    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    world_size = args.n_gpus # torch.cuda.device_count()
    
    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [args.fanout for _ in range(args.n_layers)], 0, mode="GPU")
    # quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="GPU")
    feature = torch.zeros(data.x.shape)
    feature[:] = data.x
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(feature)

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, quiver_sampler, quiver_feature, data.y.squeeze(), data.edge_index, split_idx, dataset.num_features, dataset.num_classes, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()