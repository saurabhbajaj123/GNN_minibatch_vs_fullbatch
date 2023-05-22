import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import argparse

import os
import torch.multiprocessing as mp


def partition_data(num_parts):
    # root="../dataset/"
    root = "/home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)


    graph, node_labels = dataset[0]
    # graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    val_nids = idx_split['valid']
    test_nids = idx_split['test']

    train_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    val_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    test_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)

    train_mask[train_nids] = True
    val_mask[val_nids] = True
    test_mask[test_nids] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    dgl.distributed.partition_graph(graph, 
                                    graph_name='part_config', 
                                    num_parts=num_parts, 
                                    num_hops=1, 
                                    part_method='metis',
                                    out_path='/home/ubuntu/GNN_mini_vs_full/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/dataset/partition_dataset',
                                    balance_ntypes=graph.ndata['train_mask'],
                                    balance_edges=True)




# def train_process(rank, world_size):
#     os.environ['RANK'] = str(rank)
#     os.environ['WORLD_SIZE'] = str(world_size)
#     os.environ['MASTER_ADDR'] = '172.31.79.39'
#     os.environ['MASTER_PORT'] = '8888'
#     torch.distributed.init_process_group(backend='gloo')
    
    

#     train_data = graph.subgraph(train_mask)
#     print(train_data)


def parse_args_fn():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parts", type=int, default=1)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # dgl.distributed.initialize('ip_config.txt')

    # world_size = 4
    # mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)
    args = parse_args_fn()

    partition_data(args.num_parts)