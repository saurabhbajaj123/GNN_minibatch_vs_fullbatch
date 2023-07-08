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
import wandb

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
        # print(pred.device)
        nid = nid.to(pred.device)
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
        # nid = nid.to(pred.device)
        g = g.to(pred.device)
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=n_classes
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

    if proc_id == 0:
        wandb.init(
            project="MultiGPU-{}-{}-{}".format(args.dataset, args.model, args.sampling),
            name=f"n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, agg-{args.agg}, batch_size-{args.batch_size}, fanout-{args.fanout}",
            # notes="HPO by varying only the n_hidden and n_layers"
        # project="PipeGCN-{}-{}".format(args.dataset, args.model),
        )
 
        wandb.log({
            "torch seed": torch.initial_seed()  & ((1<<63)-1)
        })
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
    train_dur = []
    eval_dur = []
    best_val_acc = 0
    best_test_acc = 0
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

        train_dur.append(t1-t0)

        if (epoch + 1) % args.log_every == 0:
            train_acc = (
                evaluate(model, g, n_classes, train_dataloader).to(device) / nprocs
            )
            val_acc = (
                evaluate(model, g, n_classes, val_dataloader).to(device) / nprocs
            )
            test_acc = 0
            # test_acc = layerwise_infer(proc_id, device, g, n_classes, test_idx, model, use_uva)
            test_acc = whole_infer(proc_id, device, model, g, test_idx, n_classes)
            
            # test_acc_1 = (
            #     evaluate(model, g, n_classes, test_dataloader).to(device) / nprocs
            # )
            t2 = time.time()

            eval_dur.append(t2 - t1)
            dist.reduce(train_acc, 0)
            dist.reduce(val_acc, 0)
            # dist.reduce(test_acc_1, 0)
            train_acc = train_acc.item()
            val_acc = val_acc.item()
            if proc_id == 0:
                print(
                    "Epoch {:05d} | Train Acc {:.4f} | Val Acc {:.4f} | Loss {:.4f} | "
                    "Train Time {:.4f} | Eval Time {:.4f}".format(
                        epoch, train_acc, val_acc, total_loss / (it + 1), t1 - t0, t2 - t1
                    )
                )
                if best_val_acc < val_acc:
                    best_train_acc = train_acc
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
                wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                    })

    dist.barrier()
    train_dur_sum_tensor = torch.tensor(np.sum(train_dur)).cuda()
    dist.reduce(train_dur_sum_tensor, 0)
    train_dur_sum = train_dur_sum_tensor.item() / args.n_gpus 

    train_dur_mean_tensor = torch.tensor(np.mean(train_dur)).cuda()
    dist.reduce(train_dur_mean_tensor, 0)
    train_dur_mean = train_dur_mean_tensor.item() / args.n_gpus
    
    eval_dur_tensor = torch.tensor(np.sum(eval_dur)).cuda()
    dist.reduce(eval_dur_tensor, 0)
    eval_dur_sum = eval_dur_tensor.item() / args.n_gpus

    # print(train_dur_sum)
    if proc_id == 0:
        print(
            "Epoch {:05d} | Time per epoch {:.4f} | Time to train {:.4f} | Time to eval {:.4f}".format(epoch, train_dur_mean, train_dur_sum, eval_dur_sum)
        )

        print(f"best val acc = {best_val_acc} | best test acc = {best_test_acc}")

        wandb.log({
            "epoch": epoch,
            "Time_per_epoch": train_dur_mean,
            "Time_to_train": train_dur_sum,
            "Time_to_eval": eval_dur_sum,
        })

def run(proc_id, nprocs, devices, g, data, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    if args.seed:
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
    test_idx = test_idx.to(device)
    g = g.to(device if args.mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_feats = g.ndata["feat"].shape[1]
    activation = F.relu
    # model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
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
    # layerwise_infer(proc_id, device, g, n_classes, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()

