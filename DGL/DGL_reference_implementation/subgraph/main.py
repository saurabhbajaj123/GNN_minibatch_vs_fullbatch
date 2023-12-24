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

from train import *
from utils import load_data

import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["DGLDEFAULTDIR"] = "/home/ubuntu/gnn_mini_vs_full/.dgl"
os.environ["DGL_DOWNLOAD_DIR"] = "/home/ubuntu/gnn_mini_vs_full/.dgl"
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
            "agg": args.agg,
            "n_gpus": args.n_gpus,
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
    args.n_gpus = config.n_gpus

    # devices = list(map(int, args.gpu.split(",")))
    devices = list(range(args.n_gpus))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")


    # load and preprocess dataset
    print("Loading data")

    
    
    # loading data from saved subgraph
    
    g, _ = dgl.load_graphs(args.dataset_subgraph_path)
    print(g)
    g = g[0]


    # # loading data using API
    # dataset = load_data(args.dataset)
    # g = dataset[0]



    print(f"graph.ndata = {g.ndata}")
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    if args.dataset == "ogbn-arxiv":
        g.edata.clear()
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    else:
        g.edata.clear()
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)


    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    print(f"g.ndata = {g.ndata}")
    print("")
    node_features = g.ndata['feat']
    num_features = node_features.shape[1]

    g.ndata['label'] = g.ndata['label'].to(torch.int64)
    num_classes = (g.ndata['label'].max() + 1).item()

    train_nids = g.ndata['train_mask'].nonzero()
    valid_nids = g.ndata['val_mask'].nonzero()
    test_nids = g.ndata['test_mask'].nonzero()
    
    print(f"len of train_nid = {len(train_nids)}, val_nids = {len(valid_nids)}, test_nids = {len(test_nids)}")
    data = (
        num_classes,
        train_nids.flatten(),
        valid_nids.flatten(),
        test_nids.flatten(),
    )

    # print(torch.all(torch.eq(torch.sort(test_nids.flatten())[0], torch.sort(dataset.test_idx)[0])) == True) # Gives True
    # print(torch.all(torch.eq(torch.sort(valid_nids.flatten())[0], torch.sort(dataset.val_idx)[0])) == True) # Gives True

    # data = (
    #     dataset.num_classes,
    #     dataset.train_idx,
    #     dataset.val_idx,
    #     dataset.test_idx,
    # )
    # print(g)
    n_data = g.ndata
    # print("g.ndata")

    shared_graph = g.shared_memory("train_graph") 
    # print(shared_graph.ndata)
    # mp.spawn(
    #     run,
    #     args=(nprocs, devices, g, data, args),
    #     nprocs=nprocs,
    # )

    print("starting spawn")
    mp.spawn(
        run,
        args=(nprocs, devices, n_data, data, args),
        nprocs=nprocs,
    )
    

if __name__ == "__main__":
    
    # dataset = 'ogbn-products'
    # model = 'graphsage'
    # sampling = 'NS'
    main()

    # args = create_parser()


    # sweep_configuration = {
    #     # 'name': f"Multiple runs best parameters {args.n_gpus}",
    #     'name': f"Scalability {args.mode}",
    #     # 'name': "checking if 5 layers is the best",
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_acc'},
    #     'parameters': 
    #     {
    #         # 'n_layers': {'values': [2, 3, 4, 5, 6]},
    #         # 'n_hidden': {'values': [64, 128, 256, 512, 1024]},
    #         # 'n_hidden': {'distribution': 'int_uniform', 'min': 64, 'max': 256},
    #         # 'n_layers': {'distribution': 'int_uniform', 'min': 15, 'max': 20},
    #         # 'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.8},
    #         # 'dropout': {'values': [0.3, 0.5, 0.8]},
    #         # 'lr': {'distribution': 'uniform', 'min': 1e-3, 'max': 1e-2},
    #         # "agg": {'values': ["mean", "gcn", "pool"]},
    #         # 'batch_size': {'values': [256, 512, 1024, 2048, 4096]},
    #         'n_gpus': {'values': [1, 2, 3, 4]},
    #         # 'dummy': {'values': [1, 2, 3, 4, 5]},
    #         # 'fanout': {'values': [5, 10, 15, 20, 25]},
    #         # 'num_heads': {'values': [2, 4, 8]},
    #         # 'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
    #         # 'dummy': {'distribution': 'int_uniform', 'min': 3, 'max': 7},
    #     }
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration,
    #                        project="MultiGPU-{}-{}-{}".format(args.dataset, args.model, args.sampling))

    # wandb.agent(sweep_id, function=main, count=500)

