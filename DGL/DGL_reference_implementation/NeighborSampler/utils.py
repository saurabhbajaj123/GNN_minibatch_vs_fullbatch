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
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(dgl.data.RedditDataset(raw_dir=root))

    return dataset


def load_pubmed():
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(dgl.data.PubmedGraphDataset(raw_dir=root))

    return dataset


def load_ogb_dataset(name):
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset"
    dataset = AsNodePredDataset(DglNodePropPredDataset(name=name, root=root))

    return dataset
