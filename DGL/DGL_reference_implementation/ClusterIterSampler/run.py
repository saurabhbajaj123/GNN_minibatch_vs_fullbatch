import argparse
import time
import traceback
from functools import partial

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from sampler import ClusterIter, subgraph_collate_fn
from torch.utils.data import DataLoader



#### Entry point
def run(args, device, data):
    # Unpack data
    (
        train_nid,
        val_nid,
        test_nid,
        in_feats,
        labels,
        n_classes,
        g,
        cluster_iterator,
    ) = data

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        iter_load = 0
        iter_far = 0
        iter_back = 0
        iter_tl = 0
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_start = time.time()
        for step, cluster in enumerate(cluster_iterator):
            cluster = cluster.int().to(device)
            mask = cluster.ndata["train_mask"].to(device)
            if mask.sum() == 0:
                continue
            feat = cluster.ndata["feat"].to(device)
            batch_labels = cluster.ndata["labels"].to(device)
            tic_step = time.time()

            batch_pred = model(cluster, feat)
            batch_pred = batch_pred[mask]
            batch_labels = batch_labels[mask]
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            tic_far = time.time()
            loss.backward()
            optimizer.step()
            tic_back = time.time()
            iter_load += tic_step - tic_start
            iter_far += tic_far - tic_step
            iter_back += tic_back - tic_far

            tic_start = time.time()
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                # print(
                #     "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB".format(
                #         epoch, step, loss.item(), acc.item(), gpu_mem_alloc
                #     )
                # )

        toc = time.time()
        # print(
        #     "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
        #         toc - tic, iter_load, iter_far, iter_back
        #     )
        # )
        if epoch >= 5:
            avg += toc - tic

        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(
                model, g, labels, val_nid, test_nid, args.val_batch_size, device
            )
            model = model.to(device)
            # if args.save_pred:
            #     np.savetxt(
            #         args.save_pred + "%02d" % epoch,
            #         pred.argmax(1).cpu().numpy(),
            #         "%d",
            #     )
            # print("Eval Acc {:.4f}".format(eval_acc))
            if eval_acc > best_eval_acc:
                best_train_acc = train_acc
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            # print(
            #     "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
            #         best_eval_acc, best_test_acc
            #     )
            # )
    print("Avg epoch time: {}".format(avg / (epoch - 4)))
    return best_test_acc
