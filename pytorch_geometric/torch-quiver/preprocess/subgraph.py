from ogb.nodeproppred import PygNodePropPredDataset
import torch 
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
dataset = PygNodePropPredDataset(name = "ogbn-papers100M", root="/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset") 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]



graph = dataset[0] # pyg graph object
n_node = graph.num_nodes

train_mask = torch.zeros(n_node, dtype=torch.bool)
val_mask = torch.zeros(n_node, dtype=torch.bool)
test_mask = torch.zeros(n_node, dtype=torch.bool)

train_mask[split_idx["train"]] = True
val_mask[split_idx["valid"]] = True
test_mask[split_idx["test"]] = True

print(f"total train ids = {len(train_idx)}, train_mask.sum() = {train_mask.int().sum().item()}")

ids = torch.cat([train_idx, valid_idx, test_idx])
subset, edge_index, mapping, edge_mask = k_hop_subgraph(ids, 2, graph.edge_index)

print(f"len(subset) = {len(subset)}")

subset_x = graph.x[subset]
subset_y = graph.y[subset]

train_mask = train_mask[subset]
val_mask = val_mask[subset]
test_mask = test_mask[subset]
# subgraph_obj = {
#     'x': subset_x,
#     'y': subset_y,
#     'train_mask': train_mask,
#     'val_mask': val_mask,
#     'test_mask': test_mask
# }

data = Data(x=subset_x, edge_index=edge_index, y=subset_y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

torch.save(data, '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver/preprocess/papers100M_pyg_subgraph.bin')

