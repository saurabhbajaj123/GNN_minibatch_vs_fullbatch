# Following the tutorial - https://colab.research.google.com/github/sachinsharma9780/interactive_tutorials/blob/master/notebooks/example_output/Comprehensive_GraphSage_Guide_with_PyTorchGeometric_Output.ipynb#scrollTo=r-JrMVQK9KJz

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
import os.path as osp
import pandas as pd
import numpy as np
import collections
from pandas.core.common import flatten
# importing obg datatset
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from pandas.core.common import flatten
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(16.7,8.27)})
sns.set_theme(style="ticks")
import collections
from scipy.special import softmax
import umap
from GraphSAGE_algorithm import SAGE
from utils import train
# download and loading the obg dataset
print(osp.join(osp.dirname(osp.realpath('./')), 'data', 'products'))
root = "./dataset/data/products"
dataset = PygNodePropPredDataset('ogbn-products', root)

# split_idx contains a dictionary of train, validation and test node indices
split_idx = dataset.get_idx_split()
# print("split_idx = {}".format(split_idx))
# # split_idx = {'train': tensor([     0,      1,      2,  ..., 196612, 196613, 196614]), 'valid': tensor([196615, 196616, 196617,  ..., 235935, 235936, 235937]), 'test': tensor([ 235938,  235939,  235940,  ..., 2449026, 2449027, 2449028])}

# predefined ogb evaluator method used for validation of predictions
evaluator = Evaluator(name='ogbn-products')

# lets check the node ids distribution of train, test and val
print('Number of training nodes:', split_idx['train'].size(0))
print('Number of validation nodes:', split_idx['valid'].size(0))
print('Number of test nodes:', split_idx['test'].size(0))

# loading the dataset
data = dataset[0]
# print(len(dataset)) # Here we have only a single dataset

# lets check some graph statistics of ogb-product graph
print("Number of nodes in the graph:", data.num_nodes)
print("Number of edges in the graph:", data.num_edges)
print("Node feature matrix with shape:", data.x.shape) # [num_nodes, num_node_features]
print("Graph connectivity in COO format with shape:", data.edge_index.shape) # [2, num_edges]
print("Target to train against :", data.y.shape) 
print("Node feature length", dataset.num_features)

# checking the number of unique labels
# there are 47 unique categories of product
print("Unique labels = {}".format(data.y.unique()))

# load integer to real product category from label mapping provided inside the dataset
df = pd.read_csv('./dataset/data/products/ogbn_products/mapping/labelidx2productcategory.csv.gz')

# lets see some of the product categories
print(df[:10])

# creating a dictionary of product category and corresponding integer label
label_idx, prod_cat = df.iloc[: ,0].values, df.iloc[: ,1].values
label_mapping = dict(zip(label_idx, prod_cat))

# counting the numbers of samples for each category
y = data.y.tolist()
y = list(flatten(y))
count_y = collections.Counter(y)
print(count_y)

# Neighborhood Sampling
train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device = {}".format(device))
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

print("model to device")
# loading node feature matrix and node labels
x = data.x.to(device)
y = data.y.squeeze().to(device)

print("data to device")





for epoch in range(20, 21):
    print("epoch = {}".format(epoch))
    loss, acc = train(epoch, model, train_loader, train_idx, device, x, y)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')