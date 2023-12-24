from helper.parser import *
import random
import torch.multiprocessing as mp
from helper.utils import *
import train
import warnings
import subprocess
import random
import sys

import wandb
# wandb.login()
import warnings
warnings.filterwarnings("ignore")
from socket import gethostname
import psutil 

def print_memory(s):
    torch.cuda.synchronize()
    print(f"cpu memory  = {psutil.virtual_memory()}")
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))

def main():
    args = create_parser()
    # wandb.init(
    #     project="PipeGCN-{}-{}".format(args.dataset, args.model),
    #     config={
    #         "n_hidden": args.n_hidden,
    #         "n_layers": args.n_layers,
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



    # args.node_rank = int(os.environ["SLURM_PROCID"]) # int( int(os.environ["SLURM_PROCID"]) // args.parts_per_node)
    # print(f"args.node_rank = {args.node_rank}")

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
    
    print_memory('before skip partition')

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
    print_memory('after skip partition')
    print(f"args = {args}")
    # return

    if args.backend == 'gloo':
        processes = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = torch.cuda.device_count()
            devices = [f'{i}' for i in range(n)]
        print(f"devices = {devices}, len devices = {len(devices)}")
        if len(devices) == 0:
            mp.set_start_method('spawn', force=True)
            start_id = args.node_rank * args.parts_per_node
            for i in range(start_id, min(start_id + args.parts_per_node, args.n_partitions)):
                p = mp.Process(target=train.init_processes, args=(i, args.n_partitions, args))
                p.start()
                processes.append(p)
        else:
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
        # world_size    = int(os.environ["WORLD_SIZE"])
        # rank          = int(os.environ["SLURM_PROCID"])
        # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(rank - gpus_per_node * (rank // gpus_per_node))
        # train.init_processes(rank, world_size, args)

        raise NotImplementedError
    elif args.backend == 'mpi':
        # gcn_arg = []
        # for k, v in vars(args).items():
        #     if v is True:
        #         gcn_arg.append(f'--{k}')
        #     elif v is not False:
        #         gcn_arg.extend([f'--{k}', f'{v}'])
        # mpi_arg = []
        # mpi_arg.extend(['-n', f'{args.n_partitions}'])
        # command = ['mpirun'] + mpi_arg + ['python', 'train.py'] + gcn_arg
        # print(' '.join(command))
        # subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)
        raise NotImplementedError
    else:
        raise ValueError

if __name__ == '__main__':
    # dataset = 'pubmed'
    # model = 'graphsage'
    main()
    # args = create_parser() 
    # sweep_configuration = {
    #     'name': f"ablation with num partitions",
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'values': [64, 128]},
    #         # 'n_layers': {'values': [4,5,6,7,8]},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 256},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 5},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #         # 'dropout': {'values': [0.2, 0.4, 0.6, 0.8]},
    #         # 'lr': {'distribution': 'log_uniform', 'min': 1e-4, 'max': 1e-1},
    #         # 'lr': {'distribution': 'uniform', 'min': args.lr*0.1, 'max': args.lr*10},
    #         # "lr": {'values': [0.001, 0.002, 0.003, 0.005]},
    #         # 'n_partitions': {'distribution': 'int_uniform', 'min': 1, 'max': 4},
    #         'n_partitions': {'values': [3, 4, 5, 6]},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'num_epochs': {'values': [2000, 4000, 6000, 8000]},
    #         # 'batch_size': {'values': [128, 256, 512]},
    #         # 'budget': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #         # 'dummy': {'distribution': 'int_uniform', 'min': 100, 'max': 10000},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="PipeGCN-{}-{}".format(args.dataset, args.model))

    # wandb.agent(sweep_id, function=main, count=5000)

