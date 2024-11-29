import argparse
import os
import time

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from model import GraphSAGE, SAGE
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")

from train import *
from utils import load_data, load_subgraph

import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["DGLDEFAULTDIR"] = "/work/sbajaj_umass_edu/.dgl"
os.environ["DGL_DOWNLOAD_DIR"] = "/work/sbajaj_umass_edu/.dgl"


import argparse
import os
import time

import dgl
import dgl.nn as dglnn

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torchprofile
import psutil


import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from model import *
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

import wandb


os.environ["DGLDEFAULTDIR"] = "/work/sbajaj_umass_edu/.dgl"
os.environ["DGL_DOWNLOAD_DIR"] = "/work/sbajaj_umass_edu/.dgl"

fanout = 20
n_layers = 3
n_hidden = 256
batch_size = 1024
device = 2
nprocs = 4
devices = list(range(nprocs))

dataset = load_data("ogbn-arxiv")
g = dataset[0]
data = (
    dataset.num_classes,
    dataset.train_idx,
    dataset.val_idx,
    dataset.test_idx,
    )
mp.spawn(
    run,
    args=(nprocs, devices, g, data),
    nprocs=nprocs,
)

def run(proc_id, nprocs, devices, g_or_n_data, data):

    dist.init_process_group(
        backend="gloo", # "gloo", #"nccl"
        init_method=f"tcp://gypsum-gpu085:1234",
        world_size=nprocs,
        rank=proc_id,
    )

    n_classes, train_idx, val_idx, test_idx = data
    g = g_or_n_data
    sampler = NeighborSampler(
            [fanout for _ in range(n_layers)], prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=16,
        use_ddp=True,
        use_uva=True,
        )


    for it, (_, _,blocks) in enumerate(train_dataloader):
        print(it)