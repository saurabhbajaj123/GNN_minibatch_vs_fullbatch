import torch.multiprocessing as mp
from train import run
from parser import *
import signal

def main():
    args = create_parser()
    print(f"args = {args}")
    num_gpus = args.n_gpus
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    mp.spawn(run, args=(list(range(num_gpus)), args), nprocs=num_gpus)


# Say you have four GPUs.
if __name__ == '__main__':
    main()