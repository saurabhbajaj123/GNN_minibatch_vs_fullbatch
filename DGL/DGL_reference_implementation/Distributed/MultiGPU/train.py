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

def evaluate(model, g, n_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_pred, _ = model(blocks, x)
            y_hats.append(y_pred)
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

def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))
    return torch.cuda.max_memory_allocated() / 1024 / 1024

# def print_memory(s):
#     torch.cuda.synchronize()
#     print(f"cpu memory  = {psutil.virtual_memory()}")
#     print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
#         torch.cuda.memory_allocated() / 1024 / 1024,
#         torch.cuda.max_memory_allocated() / 1024 / 1024,
#         torch.cuda.memory_reserved() / 1024 / 1024
#     ))

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
            name=f"n_gpus-{args.n_gpus}, n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, num_heads-{args.num_heads}, batch_size-{args.batch_size}, fanout-{args.fanout}",
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
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1, eta_min=1e-4)

    scheduler2 = ReduceLROnPlateau(opt, mode='max', cooldown=10, factor=0.99, patience=20, min_lr=1e-5)
    train_time = 0
    no_improvement_count = 0
    train_acc = 0
    val_acc = 0
    test_acc = 0

    sampling_microbatch_accumulator = []
    train_microbatch_accumulator = []
    backward_pass_accumulator = []

    # print("start training")

    for epoch in range(n_epochs):
        print('epoch ', epoch)
        # total_edges = torch.tensor(0)
        # total_dst_nodes = torch.tensor(0)
        total_flops = torch.tensor(0.0)

        # print(f"total_flops = {total_flops}, type = {type(total_flops)}")
        model.train()
        total_loss = 0
        number_of_batches = 0
        timers  = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        # d = iter(train_dataloader)
        # print(f"d = {d}")
        t0 = time.time()
        # try:
        # while d:
        timers[0].record()
        for it, (_, _,blocks) in enumerate(train_dataloader):
            # print(d)
            # timers[0].record()
            # _,_,blocks = next(d)
            # timers[1].record()

            # start_micro_batch_time = time.time()

            number_of_batches += 1
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            try:
                y = y.type(torch.cuda.LongTensor)
            except:
                y = y.to(torch.int64)

            # timers[2].record()
            y_hat, flops = model(blocks, x)
            # print(f"edges = {edges}")
            # flops = torchprofile.profile_macs(model, args=[blocks, x])


            # timers[3].record()
            # timers[4].record()
            loss = F.cross_entropy(y_hat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()  # Gradients are synchronized in DDP
            # timers[5].record()
            total_loss += loss

            # timers[3].synchronize()
            # timers[5].synchronize()
            # if epoch > 5:
            #     sampling_microbatch = timers[0].elapsed_time(timers[1])/1000
            #     train_microbatch = timers[2].elapsed_time(timers[3])/1000
            #     backward_pass = timers[4].elapsed_time(timers[5])/1000
            #     sampling_microbatch_accumulator.append(sampling_microbatch)
            #     train_microbatch_accumulator.append(train_microbatch)
            #     backward_pass_accumulator.append(backward_pass)
            #     total = sampling_microbatch + train_microbatch + backward_pass
                # print(f"sampling_microbatch = {sampling_microbatch}, train_microbatch = {train_microbatch}, backward_pass = {backward_pass}")
                # print(f"sampling_microbatch fraction = {sampling_microbatch/total}, train_microbatch = {train_microbatch/total}, backward_pass = {backward_pass/total}")
            # # total_edges += edges
            # # total_dst_nodes += dst_nodes
            # # print(f"flops = {flops}, type = {type(flops)}")
            total_flops += flops
            # # print(f"rank = {proc_id}, total_edges = {total_edges}")
            # timers[0].record()

        # except Exception as e:
        #     print("error: ", e)
        #     continue
        dist.barrier()
        dist.all_reduce(total_flops, op=dist.ReduceOp.SUM)
        dist.barrier()
        print(f"total_flops = {total_flops}")
        if proc_id == 0 and epoch <= 1: 
            data = {
                'n_layers': [args.n_layers],
                'n_hidden': [args.n_hidden],
                'fanout': [args.fanout],
                'batch_size': [args.batch_size],
                'n_gpus': [args.n_gpus],
                'total_flops': [total_flops.item()]
                # 'total_dst_nodes': [total_dst_nodes.item()], 
                # 'total_edges': [total_edges.item()]

            }
            df = pd.DataFrame(data)
            print(df)
            file_path = f'/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/{args.dataset}_{args.model}_flops.csv'
            try:
                df.to_csv(file_path, mode='a', index=False, header=False)
            except Exception as e:
                print(e)
        #     print(f"layers = {args.n_layers}, n_hidden = {args.n_hidden}, fanout = {args.fanout}, batch_size = {args.batch_size}")
        #     # print(f"total_dst_nodes = {total_dst_nodes}, total_edges = {total_edges}")
        #     print(f"total_flops = {total_flops}")
        dist.barrier()




        # return 
        # wandb.log(
        #     {
        #         'total_edges': total_edges
        #     }
        # )
        # if proc_id == 0: 
        #     print(f"micro batches per gpu = {number_of_batches}")
        #     print(f"sampling timers avg = {np.mean(sampling_microbatch_accumulator[int(0.2*len(sampling_microbatch_accumulator)):])}")
        #     print(f"train timers avg = {np.mean(train_microbatch_accumulator[int(0.2*len(train_microbatch_accumulator)):])}")
        # peak_mem = print_memory("after_forward and backward pass")
        # peak_mem = torch.tensor(peak_mem)



        # dist.barrier()
        # dist.all_reduce(peak_mem, op=dist.ReduceOp.SUM)
        # dist.barrier()
        # if proc_id == 0: 
        #     data = {
        #     'n_layers': [args.n_layers],
        #     'n_hidden': [args.n_hidden],
        #     'fanout': [args.fanout],
        #     'batch_size': [args.batch_size],
        #     'n_gpus': [args.n_gpus],
        #     'peak_mem': [peak_mem],
        #     }
        #     df = pd.DataFrame(data)
        #     print(df)
        #     file_path = f'/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/{args.dataset}_mem.csv'
        #     try:
        #         df.to_csv(file_path, mode='a', index=False, header=False)
        #     except Exception as e:
        #         print(e)

        
        t1 = time.time()
        train_time += t1 - t0
        train_dur.append(t1-t0)
        
        # if proc_id == 0: print(f"Train time = {t1 - t0}")
        # # continue
        # # scheduler.step()
        # # scheduler2.step(best_val_acc)
        # # print(f"train time = {t1 - t0}")
        if (epoch + 1) % args.log_every == 0:
            train_acc = (
                evaluate(model, g, n_classes, train_dataloader).to(device) / nprocs
            )
            val_acc = (
                evaluate(model, g, n_classes, val_dataloader).to(device) / nprocs
            )
            # val_acc = layerwise_infer(proc_id, device, g, n_classes, val_idx, model, use_uva)
            test_acc = 0
            # print("layerwise infer")
            test_acc = layerwise_infer(proc_id, device, g, n_classes, test_idx, model, use_uva)
            
            # print("whole infer")
            # test_acc = whole_infer(proc_id, device, model, g, test_idx, n_classes)
            
            # print("evaluate")
            # test_acc = (
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
                # if best_test_acc < test_acc:
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
                        'lr': opt.param_groups[0]['lr'],

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
    # peak_mem = print_memory("after training")

    # print(f"train time = {np.mean(train_dur[int(0.2*len(train_dur)):])}")
    # sample_avg = torch.tensor(np.mean(sampling_microbatch_accumulator))
    # forward_avg = torch.tensor(np.mean(train_microbatch_accumulator))
    # backward_avg = torch.tensor(np.mean(backward_pass_accumulator))
    # total_avg = sample_avg + forward_avg + backward_avg
    # print(f"avg_sampling_microbatch = {sample_avg}, avg_train_microbatch = {forward_avg}, backward_pass_accumulator = {backward_avg}")
    # print(f"avg_sampling_microbatch frac = {sample_avg/(total_avg)}, avg_train_microbatch frac = {forward_avg/total_avg}, backward_pass_accumulator frac = {backward_avg/total_avg}")
    # dist.barrier()
    # dist.all_reduce(sample_avg, op=dist.ReduceOp.SUM)
    # dist.all_reduce(forward_avg, op=dist.ReduceOp.SUM)
    # dist.all_reduce(backward_avg, op=dist.ReduceOp.SUM)
    # dist.all_reduce(total_avg, op=dist.ReduceOp.SUM)
    # dist.barrier()
    # if proc_id == 0: 
    #     data = {
    #         'n_layers': [args.n_layers],
    #         'n_hidden': [args.n_hidden],
    #         'fanout': [args.fanout],
    #         'batch_size': [args.batch_size],
    #         'n_gpus': [args.n_gpus],
    #         'sampling': [sample_avg.item()/args.n_gpus],
    #         'forwad': [forward_avg.item()/args.n_gpus],
    #         'backward': [backward_avg.item()/args.n_gpus],
    #         'total': [total_avg.item()/args.n_gpus],
    #         'sampling_frac': [sample_avg.item()/(total_avg.item())],
    #         'forwad_frac': [forward_avg.item()/(total_avg.item())],
    #         'backward_frac': [backward_avg.item()/(total_avg.item())],
    #     }
    #     df = pd.DataFrame(data)
    #     print(df)
    #     file_path = f'/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/DGL/DGL_reference_implementation/Distributed/MultiGPU/{args.dataset}_time.csv'
    #     try:
    #         df.to_csv(file_path, mode='a', index=False, header=False)
    #     except Exception as e:
    #         print(e)

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
        # print(
        #     "Epoch {:05d} | Time per epoch {:.4f} | Time to train {:.4f} | Time to eval {:.4f}".format(epoch, train_dur_mean, train_dur_sum, eval_dur_sum)
        # )

        print(f"best val acc = {best_val_acc} | best test acc = {best_test_acc}")

        wandb.log({
            "epoch": epoch,
            # "Time_per_epoch": train_dur_mean,
            # "Time_to_train": train_dur_sum,
            # "Time_to_eval": eval_dur_sum,
            "torch seed": torch.initial_seed()  & ((1<<63)-1),

        })
    # wandb.finish()
    return

# def run(proc_id, nprocs, devices, g, data, args):
def run(proc_id, nprocs, devices, g_or_n_data, data, args):
    # find corresponding device for my rank
    device = devices[proc_id] % 4
    torch.cuda.set_device(device)
    if args.seed:
        torch.manual_seed(args.seed)
    # print(torch.initial_seed())
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="gloo", # "gloo", #"nccl"
        init_method=f"tcp://{args.master_addr}:{args.port}",
        world_size=nprocs,
        rank=proc_id,
    )
    if args.dataset.lower() == 'ogbn-papers100m':
        g = dgl.hetero_from_shared_memory("train_graph")
        g.ndata['label'] = g_or_n_data['label']
        g.ndata['feat'] = g_or_n_data['feat']
        g = g.to(device if args.mode == "puregpu" else "cpu")  

    else:
        g = g_or_n_data
        del g.ndata['val_mask']
        del g.ndata['test_mask']
        del g.ndata['train_mask']
        # print(g)
        # print(g.ndata)
        import sys

        # print(sys.getsizeof(g))
        g = g.to(device if args.mode == "puregpu" else "cpu")  


    # g.ndata = n_data
    # print(f"g.ndata = {g.ndata}")
    n_classes, train_idx, val_idx, test_idx = data
    # removing transfer of unecessary things to gpu
    train_idx = train_idx.to(device if args.mode == "puregpu" else "cpu")
    val_idx = val_idx.to(device if args.mode == "puregpu" else "cpu")
    test_idx = test_idx.to(device if args.mode == "puregpu" else "cpu")

    # create GraphSAGE model (distributed)
    in_feats = g.ndata["feat"].shape[1]
    # print("g.ndata[feat].device = {}".format(g.ndata["feat"].device))
    activation = F.relu
    # model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    if "sage" in args.model.lower():
        model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)
    elif args.model == "gcn":
        model = NSGCN(in_feats, args.n_hidden, n_classes, args.n_layers, activation=F.elu, dropout=args.dropout).to(device)
    elif args.model == "gat":
        model = NSGAT(in_feats, args.num_heads, args.n_hidden, n_classes, args.n_layers, dropout=args.dropout).to(device)

    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing


    # print_memory("before train function")
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
    # writer.flush()
    dist.destroy_process_group()

