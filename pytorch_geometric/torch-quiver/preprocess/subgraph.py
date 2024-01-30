from ogb.nodeproppred import PygNodePropPredDataset
import torch 
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Planetoid



# dataset = PygNodePropPredDataset(name = "ogbn-papers100M", root="/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset") 

# split_idx = dataset.get_idx_split()
# train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]



# graph = dataset[0] # pyg graph object
# n_node = graph.num_nodes

# train_mask = torch.zeros(n_node, dtype=torch.bool)
# val_mask = torch.zeros(n_node, dtype=torch.bool)
# test_mask = torch.zeros(n_node, dtype=torch.bool)

# train_mask[split_idx["train"]] = True
# val_mask[split_idx["valid"]] = True
# test_mask[split_idx["test"]] = True

# print(f"total train ids = {len(train_idx)}, train_mask.sum() = {train_mask.int().sum().item()}")

# ids = torch.cat([train_idx, valid_idx, test_idx])
# subset, edge_index, mapping, edge_mask = k_hop_subgraph(ids, 2, graph.edge_index, relabel_nodes=True)

# print(f"len(subset) = {len(subset)}")

# subset_x = graph.x[subset]
# subset_y = graph.y[subset]

# train_mask = train_mask[subset]
# val_mask = val_mask[subset]
# test_mask = test_mask[subset]


# data = Data(x=subset_x, edge_index=edge_index, y=subset_y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# torch.save(data, '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver/preprocess/papers100M_pyg_subgraph.bin')

################################################################################

# # dataset = Planetoid(root='/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset', name='Pubmed')
# root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset"
# dataset = PygNodePropPredDataset('ogbn-products', root)

root = "/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset"
data = torch.load(root + '/orkut/orkut.bin')
# print(dataset)
# data = dataset[0]
# data_split = (data.train_mask, data.val_mask, data.test_mask)
# split_idx = dataset.get_idx_split()
split_idx = {
        'train': data.train_mask.nonzero(as_tuple=False).view(-1),
        'valid': data.train_mask.nonzero(as_tuple=False).view(-1),
        'test': data.train_mask.nonzero(as_tuple=False).view(-1)
    }
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
print(f"total train ids = {len(train_idx)}, train_mask.sum() = {data.train_mask.int().sum().item()}")


n_node = data.x.shape[0] 
print(f"num nodes = {n_node}")

train_mask = torch.zeros(n_node, dtype=torch.bool)
val_mask = torch.zeros(n_node, dtype=torch.bool)
test_mask = torch.zeros(n_node, dtype=torch.bool)

train_mask[split_idx["train"]] = True
val_mask[split_idx["valid"]] = True
test_mask[split_idx["test"]] = True

ids = torch.cat([train_idx, valid_idx, test_idx])
percent = len(ids) // 10

subset, edge_index, mapping, edge_mask = k_hop_subgraph(ids[:percent], 2, data.edge_index, relabel_nodes=True)

print(f"len(subset) = {len(subset)}")

subset_x = data.x[subset]
subset_y = data.y[subset]

train_mask = train_mask[subset]
val_mask = val_mask[subset]
test_mask = test_mask[subset]

data = Data(x=subset_x, edge_index=edge_index, y=subset_y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# data = Data(x=data.x, edge_index=data.edge_index, y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
torch.save(data, '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver/preprocess/orkut_pyg_subgraph.bin')
