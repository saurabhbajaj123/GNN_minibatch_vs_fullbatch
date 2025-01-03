from helper.parser import *
import random
import torch.multiprocessing as mp
from helper.utils import *
import train
import warnings

import random
import wandb
wandb.login()
import warnings
warnings.filterwarnings("ignore")

def main():
    args = create_parser()
    # wandb.init(
    #     project="PipeGCN-{}-{}".format(args.dataset, args.model),
    #     config={
    #         "n_hidden": args.n_hidden,
    #         "n_layers": args.n_layers,
    #         "num_heads": args.num_heads,
    #         "dropout": args.dropout,
    #         "n_partitions": args.n_partitions,
    #         "lr": args.lr,
    #         "n_epochs": args.n_epochs,
    #         }
    # )
    # config = wandb.config
    # args.n_epochs = config.n_epochs
    
    # args.n_hidden = config.n_hidden
    # args.n_layers = config.n_layers
    # args.dropout = config.dropout
    # args.n_partitions = config.n_partitions
    # args.lr = config.lr
    # args.num_heads = config.num_heads

    args.node_rank = int(os.environ["SLURM_PROCID"])
    print(f"args.node_rank = {args.node_rank}")

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

    # if args.skip_partition:
    #     if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
    #         warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
    #         g, n_feat, n_class = load_data(args.dataset)
    #         args.n_feat = n_feat
    #         args.n_class = n_class
    #         args.n_train = g.ndata['train_mask'].int().sum().item()
    # else:
    #     g, n_feat, n_class = load_data(args.dataset)
    #     if args.node_rank == 0:
    #         if args.inductive:
    #             graph_partition(g.subgraph(g.ndata['train_mask']), args)
    #         else:
    #             graph_partition(g, args)
    #     args.n_class = n_class
    #     args.n_feat = n_feat
    #     args.n_train = g.ndata['train_mask'].int().sum().item()


    if args.skip_partition:
        if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
            warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
            # g, n_feat, n_class = load_data(args.dataset)
            if args.dataset_subgraph_path == '':
                g, n_feat, n_class = load_data(args.dataset)
            else:
                g, n_feat, n_class = load_subgraph(args.dataset_subgraph_path)
            args.n_feat = n_feat
            args.n_class = n_class
            args.n_train = g.ndata['train_mask'].int().sum().item()
    else:
        if args.dataset_subgraph_path == '':
            g, n_feat, n_class = load_data(args.dataset)
        else:
            g, n_feat, n_class = load_subgraph(args.dataset_subgraph_path)
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
        print(f"devices = {devices}, len devices = {len(devices)}")
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
    # dataset = 'pubmed'
    # model = 'graphsage'
    main()

    # args = create_parser() 
    # sweep_configuration = {
    #     'name': f"ablation enable_pipeline - {args.enable_pipeline}",
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         'n_hidden': {'values': [64, 128, 256]},
    #         'n_layers': {'values': [2, 3, 4]},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 256},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 2, 'max': 5},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.9},
    #         # 'dropout': {'values': [0.6, 0.7, 0.8, 0.9]},
    #         # 'lr': {'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
    #         # 'num_heads': {'distribution': 'int_uniform', 'min': 1, 'max': 10},
    #         # 'num_heads': {'values': [1, 2, 3]},
    #         # 'lr': {'distribution': 'uniform', 'min': args.lr*0.1, 'max': args.lr*10},
    #         # "lr": {'values': [0.003, 0.005, 0.01, 0.03, 0.05]},
    #         # 'n_partitions': {'distribution': 'int_uniform', 'min': 1, 'max': 4},
    #         # 'n_partitions': {'values': [2,3,4]},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512]},
    #         # 'budget': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #         # 'dummy': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="PipeGCN-{}-{}".format(args.dataset, args.model))

    # wandb.agent(sweep_id, function=main, count=1000)

