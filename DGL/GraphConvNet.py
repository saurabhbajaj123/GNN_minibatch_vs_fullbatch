import time
import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl.nn import GraphConv
from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset

class GraphConvNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=6, hidden_channels=64, num_layers=6):
        super(GraphConvNet, self).__init__()

        self.gcn = []
        self.gcn.append(GraphConv(in_channels, hidden_channels, allow_zero_in_degree=True))
        for _ in range(num_layers):
            self.gcn.append(GraphConv(hidden_channels, hidden_channels, allow_zero_in_degree=True))
        self.gcn.append(GraphConv(hidden_channels, out_channels, allow_zero_in_degree=True))
        self.gcn = nn.ModuleList(self.gcn)
    def forward(self, g, features):
        x = torch.relu(self.gcn[0](g, features))
        for i in range(1, len(self.gcn) - 1):
            x = torch.relu(self.gcn[i](g, x))
        x = torch.dropout(x, p=0.25, train=self.training)
        x = self.gcn[-1](g, x)
        x = torch.sigmoid(x)
        return x
    
if __name__ == "__main__":
    pass