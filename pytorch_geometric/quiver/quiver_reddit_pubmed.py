import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.loader import NeighborSampler

import time
import numpy as np
from parser import create_parser
######################
# Import From Quiver
######################
import quiver

import wandb

import warnings
warnings.filterwarnings("ignore")


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, aggr):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr))
        self.dropout = dropout

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


def run(rank, world_size, data_split, edge_index, x, quiver_sampler, y, num_features, num_classes, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.torch.cuda.set_device(rank)

    train_mask, val_mask, test_mask = data_split
    # print(train_mask, val_mask, test_mask)    
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    print(train_idx)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # print(len(train_loader))
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
    model = SAGE(num_features, args.n_hidden, num_classes, num_layers=args.n_layers, dropout=args.dropout, aggr=args.agg).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Simulate cases those data can not be fully stored by GPU memory
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
            out = model(x[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            total_loss += loss
        t1 = time.time()
        train_time += t1 - t0
        dist.barrier()
        train_dur.append(t1-t0)

        if rank == 0 and epoch > 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time : {time.time() - epoch_start :.4f}')

        if rank == 0 and (epoch+1) % args.log_every == 0:  # We evaluate on a single GPU for now
            t2 = time.time()

            model.eval()
            with torch.no_grad():
                out = model.module.inference(x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
            acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
            acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
            t3 = time.time()
            eval_dur.append(t3-t2)
            
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
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

    print(args)
    # root= '../dataset' 
    if args.dataset == 'reddit':
        dataset = Reddit(root)
    elif args.dataset == 'pubmed':
        dataset = Planetoid(root=root, name='Pubmed')
    print(dataset)

    world_size = torch.cuda.device_count()

    data = dataset[0]

    csr_topo = quiver.CSRTopo(data.edge_index)
    
    ##############################
    # Create Sampler And Feature
    ##############################
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [args.fanout for _ in range(args.n_layers)], 0, mode="GPU")

    # quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15, 10, 5], 0, mode='GPU')

    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)

    print('Let\'s use', world_size, 'GPUs!')
    data_split = (data.train_mask, data.val_mask, data.test_mask)
    mp.spawn(
        run, 
        args=(world_size, data_split, data.edge_index, quiver_feature, quiver_sampler, data.y, dataset.num_features, dataset.num_classes, args),
        nprocs=world_size,
        join=True
    )
if __name__ == '__main__':
    wandb.login()
    main()

    # args = create_parser()

    # sweep_configuration = {
    #     'name': "multiple runs for best params",
    #     'method': 'random',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 5},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #         # 'lr': {'distribution': 'uniform', 'min': 5e-4, 'max': 1e-2},
    #         # "agg": {'values': ["mean", "max", "lstm"]},
    #         # 'batch_size': {'values': [256, 512, 1024]},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         'dummy': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="Quiver-{}-{}-{}".format(args.dataset, args.model, args.sampling))

    # wandb.agent(sweep_id, function=main, count=5)

