from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch

root = '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset'
x = torch.load(root + '/orkut/orkut_features.pt')

node_labels = pd.read_csv(root + "/orkut/orkut_labels.csv")

y = torch.from_numpy(
            node_labels.astype("category").to_numpy()
        ).view(-1)

edge_data = pd.read_csv(root + "/orkut/orkut_edges.csv").to_numpy()
edge_data = torch.from_numpy(edge_data)
edge_index = torch.transpose(edge_data, 0, 1)
print(f"edge_index.shape = {edge_index.shape}") 

n_nodes = x.shape[0]
n_train = int(n_nodes * 0.6)
n_val = int(n_nodes * 0.2)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train : n_train + n_val] = True
test_mask[n_train + n_val :] = True

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

torch.save(data, '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver/preprocess/orkut.bin')


import torch
from torch_geometric.data import InMemoryDataset, download_url


import torch
from torch_geometric.data import InMemoryDataset, download_url

import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url


class OrkutDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)

    def process(self):
        root = '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset'
        self.data = torch.load(root + '/orkut/orkut.bin')
        self.num_classes = (self.data.y + 1).item()
        self.num_features = self.data.x.shape[1]
    
    def get_idx_split(self):
        train_idx = self.data.train_mask.nonzero().view(-1)
        val_idx = self.data.val_mask.nonzero().view(-1)
        test_idx = self.data.test_mask.nonzero().view(-1)
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }

    def len(self):
        return

    def get(self, idx):
        return self.data