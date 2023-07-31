# Reaches around 0.7870 ± 0.0036 test accuracy.
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid

from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
from parser import create_parser
from torch_geometric.utils import add_remaining_self_loops, to_undirected

import numpy as np
####################
# Import Quiver
####################
import quiver

import wandb

import warnings
warnings.filterwarnings("ignore")


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

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
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
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
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, pin_memory=True)
    if rank == 0:
        wandb.init(
            project="Quiver-{}-{}-{}".format(args.dataset, args.model, args.sampling),
            name=f"n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, agg-{args.agg}, batch_size-{args.batch_size}, fanout-{args.fanout}",
        )

    if rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=args.eval_batch_size,
                                          shuffle=False, num_workers=6)

    if args.seed:
        torch.manual_seed(args.seed)
    model = SAGE(num_features, args.n_hidden, num_classes, num_layers=args.n_layers, dropout=args.dropout).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    y = y.to(rank)
    train_dur = []
    eval_dur = []
    best_val_acc = 0
    best_test_acc = 0
    train_time = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        total_loss = 0
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
            total_loss += loss
        t1 = time.time()
        train_time += t1 - t0
        dist.barrier()
        train_dur.append(t1-t0)

        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')
        if rank == 0 and (epoch+1) % args.log_every == 0:  # We evaluate on a single GPU for now
            
            t2 = time.time()
            model.eval()
            with torch.no_grad():
                out = model.module.inference(quiver_feature, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y.cpu()
            acc1 = int(res[train_idx].sum()) / train_idx.numel()
            acc2 = int(res[val_idx].sum()) / val_idx.numel()
            acc3 = int(res[test_idx].sum()) / test_idx.numel()
            t3 = time.time()
            
            eval_dur.append(t3-t2)

            print(f'Epoch: {epoch:03d}, Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
            if best_val_acc < acc2:
                best_train_acc = acc1
                best_val_acc = acc2
                best_test_acc = acc3
            
            wandb.log({'val_acc': acc2,
                'test_acc': acc3,
                'train_acc': acc1,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'best_train_acc': best_train_acc,
                'train_time': train_time,
            })

        dist.barrier()
    train_dur_sum_tensor = torch.tensor(np.sum(train_dur)).cuda()
    dist.reduce(train_dur_sum_tensor, 0)
    train_dur_sum = train_dur_sum_tensor.item() / args.n_gpus 

    train_dur_mean_tensor = torch.tensor(np.mean(train_dur)).cuda()
    dist.reduce(train_dur_mean_tensor, 0)
    train_dur_mean = train_dur_mean_tensor.item() / args.n_gpus

    if rank == 0:
        eval_dur_sum = np.sum(eval_dur)
        eval_dur_mean = np.mean(eval_dur)

        print(
            "Epoch {:05d} | Time per epoch {:.4f} | Time to train {:.4f} | Time to eval {:.4f} | Avg time to eval {:.4f}".format(epoch, train_dur_mean, train_dur_sum, eval_dur_sum, eval_dur_mean)
        )

        print(f"best val acc = {best_val_acc} | best test acc = {best_test_acc}")

        wandb.log({
            "epoch": epoch,
            "Time_per_epoch": train_dur_mean,
            "Time_to_train": train_dur_sum,
            "Time_to_eval": eval_dur_sum,
        })
    dist.destroy_process_group()

def main():
    args = create_parser()
    
    wandb.init(
        project="Quiver-{}-{}-{}".format(args.dataset, args.model, args.sampling),
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "fanout": args.fanout,
            "batch_size": args.batch_size,
            "agg": args.agg,
            }
    )
    config = wandb.config
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers
    args.dropout = config.dropout
    args.lr = config.lr
    args.fanout = config.fanout 
    args.batch_size = config.batch_size
    args.agg = config.agg

    root = args.dataset_dir
    dataset = PygNodePropPredDataset(args.dataset, root=root)
    data = dataset[0]
    print(data)
    # add_remaining_self_loops(data.edge_index)
    print(data)
    if args.dataset == "ogbn-arxiv":
        # to_undirected(data.edge_index)
        print(data)

    split_idx = dataset.get_idx_split()
    world_size = torch.cuda.device_count()
    
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
    wandb.login()
    # main()
    args = create_parser()
    sweep_configuration = {
        'name': "multiple runs for best params",
        'method': 'random',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
            # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
            # 'lr': {'distribution': 'uniform', 'min': 5e-4, 'max': 1e-2},
            # "agg": {'values': ["mean", "max", "lstm"]},
            # 'batch_size': {'values': [256, 512, 1024]},
            # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 15},
            'dummy': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project="Quiver-{}-{}-{}".format(args.dataset, args.model, args.sampling))

    wandb.agent(sweep_id, function=main, count=5)