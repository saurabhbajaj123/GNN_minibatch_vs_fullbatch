import torch.multiprocessing as mp
from train import run
from parser import *
import signal
import wandb
wandb.login()
def main():
    args = create_parser()
    wandb.init(
        project="SingleNode-MultiGpu-{}-{}".format(args.dataset, args.model),
        config={
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "agg": args.agg,
            "batch_size": args.batch_size,
            "fanout": args.fanout,
            "lr": args.lr, 
            }
    )

    config = wandb.config
    
    args.n_hidden = config.n_hidden
    args.n_layers = config.n_layers
    args.dropout = config.dropout
    args.agg = config.agg
    args.batch_size = config.batch_size
    args.fanout = config.fanout
    args.lr = config.lr

    print(f"args = {args}")

    num_gpus = args.n_gpus
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    mp.spawn(run, args=(list(range(num_gpus)), args), nprocs=num_gpus)


# Say you have four GPUs.
if __name__ == '__main__':
    dataset = 'ogbn-products'
    model = 'graphsage'
    # main()
    sweep_configuration = {
        # 'name': "n_hidden",
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'n_hidden': {'distribution': 'int_uniform', 'min': 16, 'max': 512},
            'n_layers': {'distribution': 'int_uniform', 'min': 2, 'max': 5},
            # 'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.8},
            # "agg": {'values': ["mean", "gcn", "pool"]},
            # 'lr': {'distribution': 'uniform', 'min':0.00001, 'max':0.0001},
            # 'batch_size': {'values': [128, 256, 512, 1024, 2048]},
            'fanout': {'distribution': 'int_uniform', 'min': 3, 'max': 4},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project="SingleNode-MultiGpu-{}-{}".format(dataset, model))

    wandb.agent(sweep_id, function=main, count=2)