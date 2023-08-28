import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.nn import SAGEConv
from model import SAGE, GAT, SAGE_CLUSTER
# from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils import *
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


graph, n_classes, in_feats = load_data('ogbn-arxiv')
def run(proc_id, devices, args):
    # print(proc_id, devices, args)
    # Initialize distributed training context.
    if proc_id == 0:
        wandb.init(
            name=f"n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, agg-{args.agg}, batch_size-{args.batch_size}, num_partitions-{args.num_partitions}",
        )
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=args.master_addr, master_port= '%d' % args.port
    )
    
    if torch.cuda.device_count() < 1:
        device = torch.device("cpu")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device("cuda:" + str(dev_id))
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )

    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    # sampler = dgl.dataloading.NeighborSampler([args.fanout for _ in range(args.n_layers)])
    sampler = dgl.dataloading.ClusterGCNSampler(
        graph,
        args.num_partitions,
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )

    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(args.num_partitions).to(device),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        use_ddp=True,  # Make it work with distributed data parallel
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    # train_dataloader = dgl.dataloading.DataLoader(
    #     # The following arguments are specific to DataLoader.
    #     graph,  # The graph
    #     train_nids,  # The node IDs to iterate over in minibatches
    #     sampler,  # The neighbor sampler
    #     device=device,  # Put the sampled MFGs on CPU or GPU
    #     use_ddp=True,  # Make it work with distributed data parallel
    #     # The following arguments are inherited from PyTorch DataLoader.
    #     batch_size=args.batch_size,  # Per-device batch size.
    #     # The effective batch size is this number times the number of GPUs.
    #     shuffle=True,  # Whether to shuffle the nodes for every epoch
    #     drop_last=False,  # Whether to drop the last incomplete batch
    #     num_workers=0,  # Number of sampler processes
    # )


    # valid_dataloader = dgl.dataloading.DataLoader(
    #     graph,
    #     valid_nids,
    #     sampler,
    #     device=device,
    #     use_ddp=False,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    # )
    # test_dataloader = dgl.dataloading.DataLoader(
    #     graph,
    #     test_nids,
    #     sampler,
    #     device=device,
    #     use_ddp=False,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    # )
    # model = SAGE(in_feats, 128, n_classes).to(device)
    activation=F.relu
    # in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'
    # print(args.dropout)
    if args.model == 'graphsage':
        Model = SAGE
    elif args.model == 'gat':
        Model = GAT
    elif args.model == 'clustersage':
        Model = SAGE_CLUSTER
    else:
        ValueError('Unknown model: {}'.format(args.model))

    model = Model(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)

    # Wrap the model nh distributed data parallel module.
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(opt, mode='max', cooldown=10, factor=0.95, patience=20, min_lr=1e-5)

    best_val_acc, best_test_acc, best_train_acc = 0, 0, 0
    best_model_path = "./model.pt"

    # Copied from previous tutorial with changes highlighted.
    for epoch in range(args.n_epochs):
        model.train()

        # with tqdm.tqdm(train_dataloader) as tq:
        for step, sg in enumerate(dataloader):
            # feature copy from CPU to GPU takes place here
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool()
            y_hat = model(sg, x)
            loss = F.cross_entropy(y_hat[m], y[m])

            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step(best_val_acc)
        if epoch % args.log_every == 0:
            print("Epoch: {} | Step: {} | Loss: {}".format(epoch, step, "%.03f" % loss.item()))


        # Evaluate on only the first GPU.
        if proc_id == 0 and epoch % args.log_every == 0:
            model.eval()
            train_predictions = []
            train_labels = []
            val_predictions = []
            val_labels = []
            test_predictions = []
            test_labels = []
            with torch.no_grad():
                for it, sg in enumerate(dataloader):
                    x = sg.ndata["feat"]
                    y = sg.ndata["label"]
                    m_train = sg.ndata["train_mask"].bool()
                    m_val = sg.ndata["val_mask"].bool()
                    m_test = sg.ndata["test_mask"].bool()
                    y_hat = model(sg, x)
                    train_predictions.append(y_hat[m_train])
                    train_labels.append(y[m_train])
                    val_predictions.append(y_hat[m_val])
                    val_labels.append(y[m_val])
                    test_predictions.append(y_hat[m_test])
                    test_labels.append(y[m_test])
                train_predictions = torch.cat(train_predictions, 0)
                train_labels = torch.cat(train_labels, 0)
                val_predictions = torch.cat(val_predictions, 0)
                val_labels = torch.cat(val_labels, 0)
                test_predictions = torch.cat(test_predictions, 0)
                test_labels = torch.cat(test_labels, 0)

                train_acc = MF.accuracy(
                    train_predictions,
                    train_labels,
                    task="multiclass",
                    num_classes=n_classes,
                )
                val_acc = MF.accuracy(
                    val_predictions,
                    val_labels,
                    task="multiclass",
                    num_classes=n_classes,
                )
                test_acc = MF.accuracy(
                    test_predictions,
                    test_labels,
                    task="multiclass",
                    num_classes=n_classes,
                )


                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'lr': opt.param_groups[0]['lr'],
                })
                print("Epoch: {} | Test Acc {} | Val Acc {}".format(epoch,"%.04f" % test_acc,"%.04f" % val_acc))



