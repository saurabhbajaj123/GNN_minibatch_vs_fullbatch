import argparse


def create_parser():

    parser = argparse.ArgumentParser(description='DistDGL')

    parser.add_argument("--dataset", type=str, default='ogbn-products',
                        help="the input dataset")
    parser.add_argument("--model", type=str, default='graphsage',
                        help="model for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=200,
                        help="the number of training epochs")
    parser.add_argument("--n-gpus", "--n_gpus", type=int, default=2,
                        help="the number of partitions")
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=16,
                        help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2,
                        help="the number of GCN layers")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1024,
                        help="batch size for each GPU")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-4,
                        help="weight for L2 loss")
    parser.add_argument("--num-partitions", "--num_partitions", type=int, default=4)
    parser.add_argument("--agg", "--agg", type=str, default='mean',
                        help="aggregator for GraphSAGE")
    parser.add_argument("--log-every", "--log_every", type=int, default=10)
    parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--port", type=int, default=12345,
                        help="the network port for communication")
    parser.add_argument("--master-addr", "--master_addr", type=str, default="127.0.0.1")

    return parser.parse_args()