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
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import DGLDataset
import pandas as pd



def load_data(dataset):
    if dataset == 'reddit':
        return load_reddit()
    elif 'ogbn' in dataset:
        return load_ogb_dataset(dataset)
    elif dataset == 'pubmed':
        return load_pubmed()
    elif dataset == "orkut":
        return load_orkut()
    
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def load_subgraph(dataset_path):
    g, _ = dgl.load_graphs(dataset_path)
    # print(g)
    g = g[0]
    # g.ndata['label'] = g.ndata['label'].to(torch.int64)
    n_feat = g.ndata['feat'].shape[1]
    print("train_mask shape = {}".format(g.ndata['train_mask'].shape))
    print("label shape = {}".format(g.ndata['label'].shape))
    
    if g.ndata['label'].dim() == 1:
    
        n_class = int(torch.max(torch.unique(g.ndata['label'][torch.logical_not(torch.isnan(g.ndata['label']))])).item()) + 1 # g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]
    
    return g, n_feat, n_class

class OrkutDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="orkut")

    def process(self):
        root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
        edges_data = pd.read_csv(root + "/orkut/orkut_edges.csv")
        node_labels = pd.read_csv(root + "/orkut/orkut_labels.csv")


        node_features = torch.load(root + '/orkut/orkut_features.pt')
        # print(f"node_features = {node_features}")

        node_labels = torch.from_numpy(
            node_labels.astype("category").to_numpy()
        ).view(-1)
        # print(f"node_labels = {node_labels}")

        self.num_classes = (node_labels.max() + 1).item()
        # edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())
        # print(f"node_features.shape = {node_features.shape}")
        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=node_features.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        # self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

        self.train_idx = self.graph.ndata["train_mask"].nonzero().view(-1)
        self.val_idx = self.graph.ndata["val_mask"].nonzero().view(-1)
        self.test_idx = self.graph.ndata["test_mask"].nonzero().view(-1)


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def load_orkut():
    dataset = OrkutDataset()
    return dataset

def load_reddit():
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
    dataset = AsNodePredDataset(dgl.data.RedditDataset(raw_dir=root))
    # graph = dataset[0]
    # graph = dgl.add_reverse_edges(graph)
    # train_nids = np.where(graph.ndata['train_mask'])[0]
    # valid_nids = np.where(graph.ndata['val_mask'])[0]
    # test_nids = np.where(graph.ndata['test_mask'])[0]
    # node_features = graph.ndata['feat']
    # in_feats = node_features.shape[1]
    # n_classes = dataset.num_classes
    
    # g.edata.clear()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    return dataset


def load_pubmed():
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
    dataset = AsNodePredDataset(dgl.data.PubmedGraphDataset(raw_dir=root))
    # graph = dataset[0]
    # graph = dgl.add_reverse_edges(graph)
    # train_nids = np.where(graph.ndata['train_mask'])[0]
    # valid_nids = np.where(graph.ndata['val_mask'])[0]
    # test_nids = np.where(graph.ndata['test_mask'])[0]
    # node_features = graph.ndata['feat']
    # in_feats = node_features.shape[1]
    # n_classes = dataset.num_classes
    
    # g.edata.clear()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    return dataset


def load_ogb_dataset(name):
    root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
    dataset = AsNodePredDataset(DglNodePropPredDataset(name=name, root=root))
    # g, label = dataset[0]
    # n_node = g.num_nodes()
    # node_data = g.ndata

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


    # in_feats = g.ndata['feat'].shape[1]
    # if g.ndata['label'].dim() == 1:
    #     n_classes = g.ndata['label'].max().item() + 1
    # else:
    #     n_classes = g.ndata['label'].shape[1]
    # g.edata.clear()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)

    return dataset
    
# d = return ef load_data(dataset="ogbn-products"):
#     root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset"
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
