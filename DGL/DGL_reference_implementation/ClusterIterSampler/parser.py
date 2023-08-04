import argparse


def create_parser():

    parser = argparse.ArgumentParser(description='GCN-fullbatch')

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Manual seed.",
    )

    parser.add_argument("--dataset", type=str, default='ogbn-arxiv',
                        help="the input dataset")
    
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=10,
                        help="the number of training epochs")
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=256,
                        help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=3,
                        help="the number of GCN layers")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-4,
                        help="weight for L2 loss")
    parser.add_argument("--log-every", "--log_every", type=int, default=5)
    parser.add_argument("--agg", type=str, default='mean')
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1024,
                        help="batch size for each GPU")
    parser.add_argument("--num-partitions", "--num_partitions", type=int, default=5000)
    parser.add_argument("--device-id", "--device_id", type=int, default=0,
                        help="the number of training epochs")
    return parser.parse_args()
