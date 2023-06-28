import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import SAGEConv
from model import SAGE
# from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils import *


def run(proc_id, devices, args):
    # print(proc_id, devices, args)
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=args.master_addr, master_port= '%d' % args.port
    )
    graph, n_classes, in_feats, train_nids, valid_nids, test_nids = load_data(args.dataset)
    # train_nids, valid_nids, test_nids = graph.ndata['train_nids'], graph.ndata['val_nids'], graph.ndata['test_nids']
    
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
    sampler = dgl.dataloading.NeighborSampler([args.fanout for _ in range(args.n_layers)])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph,
        valid_nids,
        sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_dataloader = dgl.dataloading.DataLoader(
        graph,
        test_nids,
        sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    # model = SAGE(in_feats, 128, n_classes).to(device)
    activation=F.relu
    # in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'
    # print(args.dropout)
    model = SAGE(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, activation, aggregator_type=args.agg).to(device)

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

    best_val_acc, best_test_acc, best_train_acc = 0, 0, 0
    best_model_path = "./model.pt"

    # Copied from previous tutorial with changes highlighted.
    for epoch in range(args.n_epochs):
        model.train()

        # with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata["feat"]
            labels = mfgs[-1].dstdata["label"]

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # if epoch % args.log_every == 0:
            # print("Epoch: {} | Step: {} | Loss: {}".format(epoch, step, "%.03f" % loss.item()))


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

                # pred = model.inference(graph.to(device), graph.ndata['feat'].to(device))

                # train_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][train_nids].cpu().numpy(), pred[train_nids].argmax(1).cpu().numpy())
                # val_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][valid_nids].cpu().numpy(), pred[valid_nids].argmax(1).cpu().numpy())
                # test_acc_fullgraph_no_sample = sklearn.metrics.accuracy_score(graph.ndata['label'][test_nids].cpu().numpy(), pred[test_nids].argmax(1).cpu().numpy())

                for input_nodes, output_nodes, mfgs in train_dataloader:
                    inputs = mfgs[0].srcdata["feat"]
                    train_labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    train_predictions.append(
                        model(mfgs, inputs).argmax(1).cpu().numpy()
                    )
                train_predictions = np.concatenate(train_predictions)
                train_labels = np.concatenate(train_labels)
                train_acc = sklearn.metrics.accuracy_score(train_labels, train_predictions)


                for input_nodes, output_nodes, mfgs in valid_dataloader:
                    inputs = mfgs[0].srcdata["feat"]
                    val_labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    val_predictions.append(
                        model(mfgs, inputs).argmax(1).cpu().numpy()
                    )
                val_predictions = np.concatenate(val_predictions)
                val_labels = np.concatenate(val_labels)
                val_acc = sklearn.metrics.accuracy_score(val_labels, val_predictions)

                
                for input_nodes, output_nodes, mfgs in test_dataloader:
                    inputs = mfgs[0].srcdata["feat"]
                    test_labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    test_predictions.append(
                        model(mfgs, inputs).argmax(1).cpu().numpy()
                    )
                test_predictions = np.concatenate(test_predictions)
                test_labels = np.concatenate(test_labels)
                test_acc = sklearn.metrics.accuracy_score(test_labels, test_predictions)
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc

                print("Epoch: {} | Test Acc {} | Val Acc {} | Train Acc {}".format(epoch,"%.04f" % test_acc,"%.04f" % val_acc, "%.04f" % train_acc))