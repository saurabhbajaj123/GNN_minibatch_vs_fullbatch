import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset


import os
import torch.multiprocessing as mp



def train_process(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '172.31.79.39'
    os.environ['MASTER_PORT'] = '8888'
    torch.distributed.init_process_group(backend='gloo')
    
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)

    idx_split = dataset.get_idx_split()
    train_mask = idx_split['train']
    val_mask = idx_split['valid']
    test_mask = idx_split['test']

    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    train_data = graph.subgraph(train_mask)
    print(train_data)
    # dgl.distributed.partition_graph(train_data, 'part_config', 4, num_hops=1, part_method='metis',
    #                         out_path='output',
    #                         balance_ntypes=train_data,
    #                         balance_edges=True)
    dgl.distributed.partition_graph(train_data, 'part_config', 4, num_hops=1, part_method='random',
                                 out_path='./../dataset/partition_data',)







if __name__ == '__main__':
    dgl.distributed.initialize('ip_config.txt')

    world_size = 4
    mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)