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
def evaluate(model, g, n_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=n_classes,
    )


def layerwise_infer(
    proc_id, device, g, n_classes, nid, model, use_uva, batch_size=2**10
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=n_classes
        )
        print("Test accuracy {:.4f}".format(acc.item()))
        return acc.item()

def whole_infer(proc_id, device, model, g, nid, n_classes):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g.to(device), g.ndata['feat'].to(device))[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", n_classes=n_classes
        )
        print("Test accuracy {:.4f}".format(acc.item()))
        return acc.item()


def train(
    proc_id,
    nprocs,
    device,
    g,
    n_classes,
    train_idx,
    val_idx,
    test_idx,
    model,
    use_uva,
    n_epochs,
    args,
):
    sampler = NeighborSampler(
        [args.fanout for _ in range(args.n_layers)], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    test_dataloader = DataLoader(
        g,
        test_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (_, _, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        t1 = time.time()

        if (epoch + 1) % args.log_every == 0:
            acc = (
                evaluate(model, g, n_classes, val_dataloader).to(device) / nprocs
            )
            test_acc = 0
            # test_acc = layerwise_infer(proc_id, device, g, n_classes, test_idx, model, use_uva)
            # test_acc = whole_infer(proc_id, device, model, g, test_idx, n_classes)
            
            # test_acc_1 = (
            #     evaluate(model, g, n_classes, test_dataloader).to(device) / nprocs
            # )
            t2 = time.time()
            dist.reduce(acc, 0)
            # dist.reduce(test_acc_1, 0)

            if proc_id == 0:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Val Acc {:.4f} | "
                    "Train Time {:.4f} | Eval Time {:.4f}".format(
                        epoch, total_loss / (it + 1), acc.item(), t1 - t0, t2 - t1
                    )
                )


def run(proc_id, nprocs, devices, g, data, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    # print(torch.initial_seed())
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.port}",
        world_size=nprocs,
        rank=proc_id,
    )
    n_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    # test_idx = test_idx.to(device)
    g = g.to(device if args.mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_feats = g.ndata["feat"].shape[1]
    # model = SAGE(in_size, 256, n_classes, 3, 0.5, F.relu).to(device)
    activation = F.relu
    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = args.mode == "mixed"
    train(
        proc_id,
        nprocs,
        device,
        g,
        n_classes,
        train_idx,
        val_idx,
        test_idx,
        model,
        use_uva,
        args.n_epochs,
        args,
    )
    layerwise_infer(proc_id, device, g, n_classes, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()

