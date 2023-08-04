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
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

import argparse
import time
import traceback
from functools import partial

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from sampler import ClusterIter, subgraph_collate_fn
from torch.utils.data import DataLoader


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata["feat"]
        model = model.cpu()
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata["feat"][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

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