import argparse
import os
import time

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from model import GraphSAGE, SAGE
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")

from saint_train import *
from utils import load_data

import wandb
wandb.login()

def main():
    args = create_parser()
    wandb.init(
        project="MultiGPU-{}-{}-{}".format(args.dataset, args.model, args.sampling),
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "fanout": args.fanout,
            "batch_size": args.batch_size,
            "budget_node_edge": args.budget_node_edge,
            }
    )

    config = wandb.config
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers
    args.dropout = config.dropout
    args.lr = config.lr
    args.fanout = config.fanout 
    args.batch_size = config.batch_size
    args.budget_node_edge = config.budget_node_edge

    devices = list(range(args.n_gpus))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")


    # load and preprocess dataset
    print("Loading data")
    dataset = load_data(args.dataset)


    # dataset = AsNodePredDataset(
    #     DglNodePropPredDataset(args.dataset, root=args.dataset_dir)
    # )
    g = dataset[0]
    # print(g)
    
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    if args.dataset == "ogbn-arxiv":
        g.edata.clear()
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    else:
        g.edata.clear()
        # g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )
    # print(data)
    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args),
        nprocs=nprocs,
    )
    

if __name__ == "__main__":
    
    # dataset = 'ogbn-products'
    # model = 'graphsage'
    # sampling = 'NS'
    main()
    # args = create_parser()


    # sweep_configuration = {
    #     'name': "batch_size",
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_hidden': {'values': [64, 128, 256]},
    #         # 'n_layers': {'values': [2,3,4,5]},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 1024},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #         # 'lr': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-2},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         'batch_size': {'values': [1024, 2048, 4096]},
    #         # "batch_size": {'distribution': 'int_uniform', 'min': 100, 'max': 2000},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # "budget_node_edge": {'values': [2000, 4000, 8000, 10000]}
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="MultiGPU-{}-{}-{}".format(args.dataset, args.model, args.sampling))

    # wandb.agent(sweep_id, function=main, count=300)

