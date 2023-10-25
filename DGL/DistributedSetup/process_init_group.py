import torch
import argparse
import os


def main(args):
    os.environ['MASTER_ADDR'] = '10.100.30.12'
    os.environ['MASTER_PORT'] = '29591'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    print("begin")
    torch.distributed.init_process_group(backend='gloo', world_size=2, rank=args.local_rank)
    print("test1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    main(args)