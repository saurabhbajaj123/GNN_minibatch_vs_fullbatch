import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset


root="../dataset/"
dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)


dgl.distributed.initialize('ip_config.txt')
torch.distributed.init_process_group(backend='gloo')

graph, node_labels = dataset[0]
graph = dgl.add_reverse_edges(graph)


dgl.distributed.partition_graph(graph, 'test', 4, num_hops=1, part_method='metis',
                            out_path='output/part_config.json', reshuffle=True,
                            balance_ntypes=graph.ndata['train_mask'],
                            balance_edges=True)