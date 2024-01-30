import torch 
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Reddit

import time

######################
# Import From Quiver
######################
import quiver

root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset"
root2 = '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric'
data1 = torch.load(root + '/orkut/orkut_pyg_subgraph.bin') 
data2 = torch.load(root + '/ogbn_products/products_pyg_graph.bin') 
data3 = PygNodePropPredDataset('ogbn-products', root)[0]
data4 = Reddit(root2 + '/torch-quiver/examples/data/Reddit')[0]
data5 = Planetoid(root=root, name='Pubmed')[0]


for data in [data1, data2, data3, data4, data5]:
    print(data)
    csr_topo = quiver.CSRTopo(data.edge_index)
    world_size = torch.cuda.device_count()

    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [4, 4], 0, mode='GPU')


    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="2G", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(data.x)
    print(quiver_feature)

# train_idx = subgraph_loaded.train_mask.nonzero(as_tuple=False).view(-1)


# print(len(train_idx))
# print(subgraph_loaded.keys())

# print(f"shape x  = {subgraph_loaded.x.shape}")
# print(f"shape y  = {subgraph_loaded.y.shape}")
# print(f"shape edge_index  = {subgraph_loaded.edge_index.shape}")
# print(f"shape of train_mask = {subgraph_loaded.train_mask.shape}")
# print(f"shape of val_mask = {subgraph_loaded.val_mask.shape}")
# print(f"shape of test_mask = {subgraph_loaded.test_mask.shape}")