import torch 

subgraph_loaded = torch.load('/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/torch-quiver/preprocess/papers100M_pyg_subgraph.bin')





train_idx = subgraph_loaded.train_mask.nonzero(as_tuple=False).view(-1)


print(len(train_idx))
print(subgraph_loaded.keys())

print(f"shape x  = {subgraph_loaded.x.shape}")
print(f"shape y  = {subgraph_loaded.y.shape}")
print(f"shape edge_index  = {subgraph_loaded.edge_index.shape}")
print(f"shape of train_mask = {subgraph_loaded.train_mask.shape}")
print(f"shape of val_mask = {subgraph_loaded.val_mask.shape}")
print(f"shape of test_mask = {subgraph_loaded.test_mask.shape}")