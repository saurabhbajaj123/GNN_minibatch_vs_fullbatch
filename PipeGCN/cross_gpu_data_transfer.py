from helper.utils import *
from collections import defaultdict, Counter
def count_neighbors_in_other_partitions(neighbors, partition_assignment, partition_assignment_dict):
    vertices_transfered = {}

    for partition, indices in partition_assignment_dict.items():
        print(partition, len(indices))
        partition_count = 0
        for node in indices:
            count = set()
            # print(f"num neighbors = {len(neighbors[node])}")
            for neighbor in neighbors[node]:
                # print(f"neighbor partition index ={partition_assignment[neighbor]}")
                if partition != partition_assignment[neighbor]:
                    count.add(partition_assignment[neighbor].tolist())
            count2 = set(partition_assignment[list(neighbors[node])].tolist()) - set([partition_assignment[node].tolist()])
            # set([partition_assignment[neighbor]  if partition != partition_assignment[neighbor] else 0 for neighbor in neighbors[node]])
            # if len(count) > 0: print(count)
            if len(count) != len(count2): print(f"count = {(count)}, count2 = {(count2)}")
            partition_count += len(count2)
        # print(partition_count)
        vertices_transfered[partition] = partition_count
    print(f"vertices_transfered = {vertices_transfered}")
    return vertices_transfered




node_map = {
    # 'ogbn-arxiv': [[0,41100],[41100,84636],[84636,125736],[125736,169343]],
    'pubmed': [[0,5009],[5009,9796],[9796,14673],[14673,19717]],
    # 'ogbn-products': [[0, 594402],[594402,1188797],[1188797,1819054],[1819054,2449029]],
    # 'reddit': [[0,56543],[56543,113430],[113430,172974],[172974,232965]],
    # 'orkut': [[1, 2]],
    # 'ogbn-papers100m': '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/subgraph/ogbn-papers100M_frac_100.0_hops_2_subgraph.bin'
    } 


def count_data_transfer(dataset_subgraph_path, name):
    if 'papers' in name:
        g, n_feat, n_class = load_subgraph(dataset_subgraph_path)
    else: 
        g, n_feat, n_class = load_data(name)
    
    print('train_mask', g.ndata['train_mask'])
    g = dgl.remove_self_loop(g)
    adj = g.adj().indices()
    # print(adj.shape)
    # print(adj[0][:10])
    # print(adj[1][:10])
    # print(sum(abs(partition_assignment[adj[0][:10]] - partition_assignment[adj[1][:10]])))

    # print(sum(abs(partition_assignment[adj[0]] - partition_assignment[adj[1]])))
    # print(partition_assignment[adj[0][:10]])
    # print(partition_assignment[adj[1][:10]])

    neighbors = defaultdict(set)
    for u, v in zip(adj[0].tolist(), adj[1].tolist()):
        neighbors[u].add(v)
        neighbors[v].add(u)

    for n_gpus in [8]:

        partition_assignment = dgl.metis_partition_assignment(g, n_gpus)
        partition_assignment_dict = defaultdict(list)

        for i, partition in enumerate(partition_assignment.tolist()):
            partition_assignment_dict[partition].append(i)
        


        
        vertices_transfered = count_neighbors_in_other_partitions(neighbors, partition_assignment, partition_assignment_dict)

        print(f"vertices_transfered = {vertices_transfered}")
        with open(f"/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/PipeGCN/data_transfer/{name}_{n_gpus}.json", 'w') as f:
            json.dump(vertices_transfered, f)

        data_transfered = sum(vertices_transfered.values())
        print(g.ndata['feat'].shape)
        tensor = g.ndata['feat'][1]
        tensor_size_bytes = tensor.element_size() * tensor.numel()
        print(f"Tensor size: {tensor_size_bytes} bytes")


        print(f"Total data transfered for {name} = {round(data_transfered*tensor_size_bytes / (1024 * 1024), 2)} MB")

"""

def layerwise_data_transfer(adj, target, partition_assignment, visited):
    next_neighbors = []
    vertices_transfered = 0

    # for each layer we have the target vertices, and for each vertex we check
    # whether any of its neighbor is present in a different partition
    # if it is present in a different partition, then we increment the vertices_transfered by 1
    # for each neighbor we also check if that neighbor has already been visited, if it is not visited yet
    # then we add that to the list of next_neighbors for the next_layer
    for ind in target:
        visited.add(ind)

        for neighbor in adj[ind]:
            if neighbor not in visited:
                next_neighbors.append(neighbor)
            if partition_assignment[neighbor] != partition_assignment[ind]:
                vertices_transfered += 1
    return vertices_transfered, next_neighbors

def count_data_transfer(dataset_subgraph_path, name):
    from collections import defaultdict
    # load data
    if 'papers' in name:
        g, n_feat, n_class = load_subgraph(dataset_subgraph_path)
    else: 
        g, n_feat, n_class = load_data(name)

    # get partition assignment (which node belongs to which partition)
    partition_assignment = dgl.metis_partition_assignment(g, 4)
    
    # remove self loops (not necessary in this)
    g = dgl.remove_self_loop(g)

    # get the adjacency matrix, zeroth row corresponds to the source, and the first row corresponds to the destination
    adj_list = g.adj().indices()

    # converting to a dict, where keys are the source, and the value is a list of destinations
    adj = defaultdict(list)
    for u, v in zip(adj_list[0], adj_list[1]):
        adj[u.item()].append(v.item())
    # print("adj", adj)

    # getting list of train indices, this will be used to calculate data transfer for the first layer
    train_indices = [i for i, x in enumerate(g.ndata['train_mask']) if x]

    vertices_transfered = 0

    num_layers = 5
    visited = set()
    # now for each layer, we get the number of vertices transfered, and the target_vertices for the next layer
    for _ in range(num_layers):
        transfer, train_indices = layerwise_data_transfer(adj, train_indices, partition_assignment, visited)
        print(transfer)
        vertices_transfered += transfer
    print("total transfer", vertices_transfered)

"""

for key, val in node_map.items():
    count_data_transfer(val, key)



# neighbors = {
#     0: [1, 2, 4],
#     1: [2, 0, 3],
#     2: [0, 1],
#     3: [1, 5],
#     4: [0, 5],
#     5: [3, 4]
# }
# partition_assignment = [0, 1, 1, 2, 2, 2]
# partition_assignment_dict = {
#     0: [0],
#     1: [1, 2],
#     2: [3, 4, 5]
# }

# v_t = count_neighbors_in_other_partitions(neighbors, partition_assignment, partition_assignment_dict)


# print(v_t)