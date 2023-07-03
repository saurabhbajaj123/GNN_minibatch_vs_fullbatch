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
from model import SAGE, GAT
import time
# from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils import *
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
def run(proc_id, devices, args, dataset_args):
    graph, n_classes, in_feats, train_nids, valid_nids, test_nids = dataset_args
    # print(proc_id, devices, args)
    # Initialize distributed training context.
    if proc_id == 0:
        wandb.init(
            name=f"n_hidden-{args.n_hidden}, n_layers-{args.n_layers}, agg-{args.agg}, batch_size-{args.batch_size}, fanout-{args.fanout}",
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
            backend="gloo",
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
        # timeout=60,
        use_uva=True,
        # use_prefetch_thread=True,
    )

    if proc_id == 0 :
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
    if args.model == 'graphsage':
        Model = SAGE
    elif args.model == 'gat':
        Model = GAT
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

    # Synchronize the model parameters across processes
    torch.cuda.synchronize()
    running_time = 0

    start_time = time.time()
    training_time = 0
    for epoch in range(args.n_epochs):
        
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
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

            running_loss += loss.item()

        scheduler.step(best_val_acc)
        training_time += time.time() - epoch_start_time

        # if epoch % args.log_every == 0:
            # print("Epoch: {} | Step: {} | Loss: {}".format(epoch, step, "%.03f" % loss.item()))


        # avg_loss = running_loss / len(train_dataloader)
        # # Synchronize the loss across processes
        # avg_loss_tensor = torch.tensor(avg_loss).to(proc_id)
        # dist.all_reduce(avg_loss_tensor)
        # avg_loss = avg_loss_tensor.item() / len(devices)

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # if epoch > 0:
        #     if abs(avg_loss - prev_loss) < convergence_threshold:
        #         break
        # prev_loss = avg_loss



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
                wandb.log({'val_acc': val_acc,
                        'test_acc': test_acc,
                        'train_acc': train_acc,
                        'best_val_acc': best_val_acc,
                        'best_test_acc': best_test_acc,
                        'best_train_acc': best_train_acc,
                        'lr': opt.param_groups[0]['lr'],
                })
                print("Epoch: {} | Test Acc {} | Val Acc {}".format(epoch,"%.04f" % test_acc,"%.04f" % val_acc))
 
    end_time = time.time()
    total_time = end_time - start_time

    # Synchronize the total time across processes
    total_time_tensor = torch.tensor(total_time).to(proc_id)
    dist.all_reduce(total_time_tensor)
    total_time = total_time_tensor.item()

    print(f"Rank {proc_id}: Time taken for training: {training_time:.2f} seconds")

    total_training_time_tensor = torch.tensor(training_time).to(proc_id)
    dist.all_reduce(total_training_time_tensor)
    total_training_time = total_training_time_tensor.item()

    # # Synchronize the number of epochs across processes
    # num_epochs_tensor = torch.tensor(epoch + 1).to(proc_id)
    # dist.all_reduce(num_epochs_tensor)
    # num_epochs = num_epochs_tensor.item()
    if proc_id == 0:    
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Time taken for training: {total_training_time:.2f} seconds")
    # print(f"Rank {rank}: Number of epochs: {num_epochs}")
    dist.destroy_process_group()



