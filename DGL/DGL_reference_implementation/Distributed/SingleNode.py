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
from ogb.nodeproppred import DglNodePropPredDataset

root = "../dataset/"
dataset = DglNodePropPredDataset("ogbn-products", root=root)

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata["label"] = node_labels[:, 0]

node_features = graph.ndata["feat"]
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

idx_split = dataset.get_idx_split()
train_nids = idx_split["train"]
valid_nids = idx_split["valid"]
test_nids = idx_split["test"]

def run(proc_id, devices):
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
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
    sampler = dgl.dataloading.NeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,  # Per-device batch size.
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
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = SAGE(num_features, 128, num_classes).to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    # Define optimizer
    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0
    best_model_path = "./model.pt"

    # Copied from previous tutorial with changes highlighted.
    for epoch in range(10):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )

                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )

        model.eval()

        # Evaluate on only the first GPU.
        if proc_id == 0:
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata["feat"]
                    labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    predictions.append(
                        model(mfgs, inputs).argmax(1).cpu().numpy()
                    )
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    # torch.save(model.state_dict(), best_model_path)

# Say you have four GPUs.
if __name__ == '__main__':
    num_gpus = 4
    import torch.multiprocessing as mp
    mp.spawn(run, args=(list(range(num_gpus)),), nprocs=num_gpus)