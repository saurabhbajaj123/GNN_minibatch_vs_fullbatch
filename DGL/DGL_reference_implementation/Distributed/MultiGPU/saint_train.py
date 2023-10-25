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
    SAINTSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch.nn.parallel import DistributedDataParallel
from model import GraphSAGE, SAGE, SaintSAGE
from parser import create_parser
import warnings
warnings.filterwarnings("ignore")
import wandb




def evaluate(g, features, labels, masks, model):
    # model = model.to(g.device)
    model = model.cpu()
    model.eval()
    # print(f"graph = {g.device}")
    # print(f"features = {features.device}")
    # print(f"labels = {labels.device}")
    # print(f"model = {model.device}")
    with torch.no_grad():
        train_mask = masks[0]
        val_mask = masks[1]
        test_mask = masks[2]
        logits = model.module.inference(g, features)

        val_logits = logits[val_mask]
        val_labels = labels[val_mask]
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        test_logits = logits[test_mask]
        test_labels = labels[test_mask]


        _, val_indices = torch.max(val_logits, dim=1)
        val_correct = torch.sum(val_indices == val_labels)
        val_acc = val_correct.item() * 1.0 / len(val_labels)

        _, train_indices = torch.max(train_logits, dim=1)
        train_correct = torch.sum(train_indices == train_labels)
        train_acc = train_correct.item() * 1.0 / len(train_labels)

        _, test_indices = torch.max(test_logits, dim=1)
        test_correct = torch.sum(test_indices == test_labels)
        test_acc = test_correct.item() * 1.0 / len(test_labels)

        return train_acc, val_acc, test_acc

# def evaluate(model, g, n_classes, dataloader):
#     model.eval()
#     ys = []
#     y_hats = []
#     for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
#         with torch.no_grad():
#             x = blocks[0].srcdata["feat"]
#             ys.append(blocks[-1].dstdata["label"])
#             y_hats.append(model(blocks, x))
#     return MF.accuracy(
#         torch.cat(y_hats),
#         torch.cat(ys),
#         task="multiclass",
#         num_classes=n_classes,
#     )



# @torch.no_grad()
# def evaluate(evaluator, predictions, labels):
#     # print(labels.size(), predictions.size())
#     acc = evaluator.eval({
#         'y_true': torch.reshape(labels, (-1, 1)),
#         'y_pred': torch.reshape(predictions, (-1, 1)),
#     })['acc']
#     # eacc = sklearn.metrics.accuracy_score(labels, predictions)
#     return acc


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
            name=f"n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, batch_size-{args.batch_size},budget-{args.budget_node_edge}",
            # notes="HPO by varying only the n_hidden and n_layers"
        # project="PipeGCN-{}-{}".format(args.dataset, args.model),
        )
 
        wandb.log({
            "torch seed": torch.initial_seed()  & ((1<<63)-1)
        })
    if args.mode_saint == 'walk':
        budget = (args.budget_rw_0,args.budget_rw_1) 
    else:
        budget = args.budget_node_edge 
    # print(budget)
    sampler = SAINTSampler(
        mode=args.mode_saint,
        budget=budget,
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )
    # sampler = NeighborSampler(
    #     [args.fanout for _ in range(args.n_layers)], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    # )
    print(f"device = {device}")
    dataloader = DataLoader(
        g,
        torch.arange(args.num_iters),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    # print(val_idx)
    # eval_dataloader = DataLoader(
    #     g,
    #     torch.arange((len(train_idx))/args.batch_size).to(device),
    #     sampler,
    #     device=device,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     use_ddp=True,
    #     use_uva=use_uva,
    # )
    # for sg in eval_dataloader:
    #     print(sg.ndata['val_mask'])
    # print(len(eval_dataloader))
    evaluator = Evaluator(name='ogbn-arxiv')
    train_dur = []
    eval_dur = []
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    masks = (train_idx, val_idx, test_idx)
    no_improvement_count = 0
    train_time = 0
    for epoch in range(n_epochs):
        model = model.to(device)
        t0 = time.time()
        model.train()
        total_loss = 0
        # print(epoch)
        for it, sg in enumerate(dataloader):
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool()
            
            y_hat = model(sg, x)
            loss = F.cross_entropy(y_hat[m], y[m])

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        t1 = time.time()
        train_time += t1 - t0
        no_improvement_count = 0
        train_dur.append(t1-t0)

        if (epoch + 1) % args.log_every == 0 and proc_id == 0:
            # model.eval()
            train_acc, val_acc, test_acc = evaluate(g, g.ndata["feat"].to('cpu'), g.ndata["label"], masks, model)
            # model = model.to('cpu')
            # print(f"features device = {g.ndata['feat'].device}, graph device = {g.device}, model = {model.to('cpu').device}")
            # pred = model(g, g.ndata['feat'])
            # print(f"pred device = {pred.device}")
            # train_acc = evaluate(evaluator, pred[train_idx].argmax(1), g.ndata['label'][train_idx])
            # val_acc = evaluate(evaluator, pred[val_idx].argmax(1), g.ndata['label'][val_idx])
            # test_acc = evaluate(evaluator, pred[test_idx].argmax(1), g.ndata['label'][test_idx])
            # print(train_acc, val_acc, test_acc)
            # train_acc, val_acc, test_acc = train_acc/nprocs, val_acc/nprocs, test_acc/nprocs

            # model.eval()
            # with torch.no_grad():
            #     train_preds, val_preds, test_preds = [], [], []
            #     train_labels, val_labels, test_labels = [], [], []
            #     for it, sg in enumerate(dataloader):
            #         x = sg.ndata["feat"]
            #         y = sg.ndata["label"]
            #         m_train = sg.ndata["train_mask"].bool()
            #         m_val = sg.ndata["val_mask"].bool()
            #         m_test = sg.ndata["test_mask"].bool()
            #         y_hat = model(sg, x)
            #         train_preds.append(y_hat[m_train])
            #         train_labels.append(y[m_train])
            #         val_preds.append(y_hat[m_val])
            #         val_labels.append(y[m_val])
            #         test_preds.append(y_hat[m_test])
            #         test_labels.append(y[m_test])
            #     train_preds = torch.cat(train_preds, 0)
            #     train_labels = torch.cat(train_labels, 0)
            #     val_preds = torch.cat(val_preds, 0)
            #     val_labels = torch.cat(val_labels, 0)
            #     test_preds = torch.cat(test_preds, 0)
            #     test_labels = torch.cat(test_labels, 0)

            #     train_acc = MF.accuracy(
            #         train_preds,
            #         train_labels,
            #         task="multiclass",
            #         num_classes=n_classes,
            #     ).to(device) / nprocs
                
            #     val_acc = MF.accuracy(
            #         val_preds,
            #         val_labels,
            #         task="multiclass",
            #         num_classes=n_classes,
            #     ).to(device) / nprocs
            #     test_acc = MF.accuracy(
            #         test_preds,
            #         test_labels,
            #         task="multiclass",
            #         num_classes=n_classes,
            #     ).to(device) / nprocs

            t2 = time.time()
            # print(f"Proc_id: {proc_id}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")
            # print(type(train_acc), type(val_acc), type(test_acc))
            # eval_dur.append(t2 - t1)
            # dist.reduce(train_acc, 0)
            # dist.reduce(val_acc, 0)
            # dist.reduce(test_acc, 0)
            # # dist.reduce(test_acc_1, 0)
            # train_acc = train_acc.item()
            # val_acc = val_acc.item()
            # test_acc = test_acc.item()
            if proc_id == 0:
                print("Test Acc {:.4f}".format(test_acc))
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
                    no_improvement_count = 0
                else:
                    no_improvement_count += args.log_every

                wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'train_time': train_time,
                    })

        # dist.barrier()
        # break_condition = False
        # if epoch > 50 and no_improvement_count >= args.patience:
        #     break_condition = True
        # dist.barrier()
        # break_condition_tensor = torch.tensor(int(break_condition)).cuda()
        # dist.all_reduce(break_condition_tensor, op=dist.ReduceOp.BOR)
        # break_condition = bool(break_condition_tensor.item())
        # # print(break_condition)
        
        # if break_condition:
        #     print(f'Early stopping after {epoch + 1} epochs.')
        #     break


    # dist.barrier()
    # train_dur_sum_tensor = torch.tensor(np.sum(train_dur)).cuda()
    # dist.reduce(train_dur_sum_tensor, 0)
    # train_dur_sum = train_dur_sum_tensor.item() / args.n_gpus 

    # train_dur_mean_tensor = torch.tensor(np.mean(train_dur)).cuda()
    # dist.reduce(train_dur_mean_tensor, 0)
    # train_dur_mean = train_dur_mean_tensor.item() / args.n_gpus
    
    # eval_dur_tensor = torch.tensor(np.sum(eval_dur)).cuda()
    # dist.reduce(eval_dur_tensor, 0)
    # eval_dur_sum = eval_dur_tensor.item() / args.n_gpus

    # print(train_dur_sum)
    if proc_id == 0:
        print(
            "Epoch {:05d} | Time per epoch {:.4f} | Time to train {:.4f} | Time to eval {:.4f}".format(epoch, train_dur_mean, train_dur_sum, eval_dur_sum)
        )

        print(f"best val acc = {best_val_acc} | best test acc = {best_test_acc}")

        # wandb.log({
        #     "epoch": epoch,
        #     "Time_per_epoch": train_dur_mean,
        #     "Time_to_train": train_dur_sum,
        #     "Time_to_eval": eval_dur_sum,
        # })

def run(proc_id, nprocs, devices, g, data, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    if args.seed:
        torch.manual_seed(args.seed)
    # print(torch.initial_seed())
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.port}",
        world_size=nprocs,
        rank=proc_id,
    )
    n_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device if args.mode == "puregpu" else "cpu")
    val_idx = val_idx.to(device if args.mode == "puregpu" else "cpu")
    test_idx = test_idx.to(device if args.mode == "puregpu" else "cpu")
    # print(f"print device = {device}")
    g = g.to(device if args.mode == "puregpu" else "cpu")
    # print(f"graph device = {g.device}")
    # create GraphSAGE model (distributed)
    in_feats = g.ndata["feat"].shape[1]
    activation = F.relu
    # model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    # model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    model = SaintSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation).to(device)
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

