import argparse
import json
import logging
import os
import sys
import pickle

import dgl
from dgl.nn import GATConv

import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import random
import wandb
wandb.login()

import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics


import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import SAGEConv
# from dgl.data import RedditDataset
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset


def _get_data_loader(sampler, device, dataset, batch_size=1024):
    logger.info("Get train-val-test data loader")
    

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    graph, node_labels = dataset[0]
    # graph = dgl.add_self_loop(graph)
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()

    logger.info("Get train data loader")
    train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
    )
    logger.info("Get val data loader")
    valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Get test data loader")
    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    logger.info("Train-val-test data loader created")
    
    return (train_dataloader, valid_dataloader, test_dataloader, (in_feats, n_classes))




def load_data(dataset):
    if dataset == 'reddit':
        return load_reddit()
    elif 'ogbn' in dataset:
        return load_ogb_dataset(dataset)
    elif dataset == 'pubmed':
        return load_pubmed()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def load_reddit():
    root = "/home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(dgl.data.RedditDataset(raw_dir=root))

    return dataset


def load_pubmed():
    root = "/home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(dgl.data.PubmedGraphDataset(raw_dir=root))

    return dataset


def load_ogb_dataset(name):
    root = "/home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(DglNodePropPredDataset(name=name, root=root))

    return dataset
