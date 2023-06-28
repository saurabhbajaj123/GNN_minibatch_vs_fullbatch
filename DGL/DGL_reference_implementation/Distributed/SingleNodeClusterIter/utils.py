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
from model import SAGE
# from dgl.data import RedditDataset
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
    root = "../../dataset/"
    dataset = dgl.data.RedditDataset(raw_dir=root)
    graph = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    train_nids = np.where(graph.ndata['train_mask'])[0]
    valid_nids = np.where(graph.ndata['val_mask'])[0]
    test_nids = np.where(graph.ndata['test_mask'])[0]
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = dataset.num_classes
    
    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return graph, n_classes, in_feats, train_nids, valid_nids, test_nids


def load_pubmed():
    root = "../../dataset/"
    dataset = dgl.data.PubmedGraphDataset(raw_dir=root)
    graph = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    train_nids = np.where(graph.ndata['train_mask'])[0]
    valid_nids = np.where(graph.ndata['val_mask'])[0]
    test_nids = np.where(graph.ndata['test_mask'])[0]
    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = dataset.num_classes
    
    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return graph, n_classes, in_feats, train_nids, valid_nids, test_nids


def load_ogb_dataset(name):
    root = "../../dataset/"
    dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset(name=name, root=root))
    g = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata

    # idx_split = dataset.get_idx_split()
    # train_nids = idx_split["train"]
    # valid_nids = idx_split["valid"]
    # test_nids = idx_split["test"]

    # node_data['label'] = label.view(-1).long()
    # node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    # node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    # node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    # node_data['train_mask'][idx_split["train"]] = True
    # node_data['val_mask'][idx_split["valid"]] = True
    # node_data['test_mask'][idx_split["test"]] = True


    in_feats = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_classes = g.ndata['label'].max().item() + 1
    else:
        n_classes = g.ndata['label'].shape[1]
    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    return g, n_classes, in_feats
    
# d = return ef load_data(dataset="ogbn-products"):
#     root = "../../dataset/"
#     dataset = DglNodePropPredDataset(dataset, root=root)

#     graph, node_labels = dataset[0]
#     # Add reverse edges since ogbn-arxiv is unidirectional.
#     graph = dgl.add_reverse_edges(graph)
#     graph.ndata["label"] = node_labels[:, 0]

#     node_features = graph.ndata["feat"]
#     in_feats = node_features.shape[1]
#     n_classes = (node_labels.max() + 1).item()

#     idx_split = dataset.get_idx_split()
#     train_nids = idx_split["train"]
#     valid_nids = idx_split["valid"]
#     test_nids = idx_split["test"]
#     return graph, in_feats, n_classes
