import torch.multiprocessing as mp
from train import run
def main()
    num_gpus = 7
    mp.spawn(run, args=(list(range(num_gpus)),), nprocs=num_gpus)


# Say you have four GPUs.
if __name__ == '__main__':