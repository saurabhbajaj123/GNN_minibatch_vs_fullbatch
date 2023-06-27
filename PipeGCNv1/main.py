from helper.parser import *
import random
import torch.multiprocessing as mp
from helper.utils import *
import train
import warnings

import random
import wandb
wandb.login()

def main():
    args = create_parser()
    wandb.init(
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            }
    )
    config = wandb.config
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers

    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn('Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)

    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%d-%s-%s-induc' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        else:
            args.graph_name = '%s-%d-%s-%s-trans' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)

    if args.skip_partition:
        if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
            warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
            g, n_feat, n_class = load_data(args.dataset)
            args.n_feat = n_feat
            args.n_class = n_class
            args.n_train = g.ndata['train_mask'].int().sum().item()
    else:
        g, n_feat, n_class = load_data(args.dataset)
        if args.node_rank == 0:
            if args.inductive:
                graph_partition(g.subgraph(g.ndata['train_mask']), args)
            else:
                graph_partition(g, args)
        args.n_class = n_class
        args.n_feat = n_feat
        args.n_train = g.ndata['train_mask'].int().sum().item()

    print(f"args = {args}")

    if args.backend == 'gloo':
        processes = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = torch.cuda.device_count()
            devices = [f'{i}' for i in range(n)]
        mp.set_start_method('spawn', force=True)
        start_id = args.node_rank * args.parts_per_node
        for i in range(start_id, min(start_id + args.parts_per_node, args.n_partitions)):
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
            p = mp.Process(target=train.init_processes, args=(i, args.n_partitions, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        del os.environ['CUDA_VISIBLE_DEVICES']
    elif args.backend == 'nccl':
        raise NotImplementedError
    elif args.backend == 'mpi':
        raise NotImplementedError
    else:
        raise ValueError

if __name__ == '__main__':
    dataset = 'reddit'
    model = 'graphsage'
    # main()
    sweep_configuration = {
        'name': "n_layers, n_hidden",
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'n_hidden': {'distribution': 'int_uniform', 'min': 16, 'max': 256},
            'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 9},
            # 'dropout': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
            # "agg": {'values': ["mean", "gcn", "pool"]},
            # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
            # 'batch_size': {'values': [128, 256, 512]},
            # 'budget': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project="PipeGCN-{}-{}".format(dataset, model))

    wandb.agent(sweep_id, function=main, count=30)

