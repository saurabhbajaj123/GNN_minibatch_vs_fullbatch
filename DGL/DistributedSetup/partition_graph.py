import argparse
import time

import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset


def load_reddit(self_loop=True):
    """Load reddit dataset."""
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    # g.ndata["features"] = g.ndata.pop("feat")
    # g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    """Load ogbn dataset."""
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    # graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["label"] = labels
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph, num_labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DistributedSetup/partitions/partitioned_graph",
        help="Output path of partitioned graph.",
    )

    argparser.add_argument(
        "--subgraph-dataset", "--subgraph_dataset",
        type=str,
        default="/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset/papers_subgraphs/ogbn-papers100M_frac_1_hops_3_subgraph.bin",
        help="Input path of subgraph.",
    )
    args = argparser.parse_args()
    print(f"args = {args}")
    start = time.time()
    # if args.dataset == "reddit":
    #     g, _ = load_reddit(root='/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset')
    # elif args.dataset in ["ogbn-products", "ogbn-papers100M"]:
    #     g, _ = load_ogb(args.dataset, root='/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/dataset')
    # elif 'subgraph' in args.dataset:
    #     g, _ = dgl.load_graphs(args.subgraph_dataset)
    #     g = g[0]
    # else:
    #     raise RuntimeError(f"Unknown dataset: {args.dataset}")
    def load_subgraph(dataset_path):
        g, _ = dgl.load_graphs(dataset_path)
        g = g[0]
        g.ndata['label'] = g.ndata['label'].to(torch.int64)
        n_feat = g.ndata['feat'].shape[1]
        print("train_mask shape = {}".format(g.ndata['train_mask'].shape))
        print("label shape = {}".format(g.ndata['label'].shape))
        g.ndata['in_degree'] = g.in_degrees()
        if g.ndata['label'].dim() == 1:
            # n_class = g.ndata['label'].max().item() + 1
            n_class = int(torch.max(torch.unique(g.ndata['label'][torch.logical_not(torch.isnan(g.ndata['label']))])).item()) + 1
        else:
            n_class = g.ndata['label'].shape[1]
        return g, n_feat, n_class

    g, n_feat, n_class = load_subgraph(args.subgraph_dataset)
    # print(
    #     "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    # )
    # print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    # print(
    #     "train: {}, valid: {}, test: {}".format(
    #         torch.sum(g.ndata["train_mask"]),
    #         torch.sum(g.ndata["val_mask"]),
    #         torch.sum(g.ndata["test_mask"]),
    #     )
    # )
    # if args.balance_train:
    #     balance_ntypes = g.ndata["train_mask"]
    # else:
    #     balance_ntypes = None

    # if args.undirected:
    #     sym_g = dgl.to_bidirected(g, readonly=True)
    #     for key in g.ndata:
    #         sym_g.ndata[key] = g.ndata[key]
    #     g = sym_g

    # dgl.distributed.partition_graph(
    #     g,
    #     args.dataset,
    #     args.num_parts,
    #     args.output,
    #     part_method=args.part_method,
    #     balance_ntypes=balance_ntypes,
    #     balance_edges=args.balance_edges,
    #     num_trainers_per_machine=args.num_trainers_per_machine,
    # )
    graph_name = f'{args.dataset}-{args.num_parts}-{args.part_method}-vol-trans'
    n_partitions = args.num_parts
    graph_dir = 'partitions/' + graph_name + '/'
    partition_method = args.part_method
    partition_obj = 'vol'
    dgl.distributed.partition_graph(g, graph_name, n_partitions, graph_dir, part_method=partition_method, balance_edges=False, objtype=partition_obj)