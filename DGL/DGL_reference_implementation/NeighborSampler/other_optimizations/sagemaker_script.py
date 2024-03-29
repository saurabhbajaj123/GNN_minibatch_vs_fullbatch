import argparse
import json
import logging
import os
import sys
import pickle

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time 
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Model(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout, activation, aggregator_type='mean'
    ):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type=aggregator_type))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type=aggregator_type))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type=aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation



    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.layers[0](mfgs[0], (x, h_dst))
        for i in range(1, self.n_layers - 1):
            h_dst = h[:mfgs[i].num_dst_nodes()]  # <---
            h = self.layers[i](mfgs[i], (h, h_dst))
            # h = F.relu(h)
            h = self.activation(h)
            h = self.dropout(h)
        h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        h = self.layers[-1](mfgs[-1], (h, h_dst))
        return h

def parse_args_fn():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--n_hidden", type=int, default=2**6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    # parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    # parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    return args

def load_dataset(path):
    """
    Load entire dataset
    """
    # find all files with pkl extenstion and load the first one
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith("pkl")]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))

    dataset = pickle.load(open(files[0], 'rb'))
def _get_data_loader(sampler, device, dataset, batch_size=1024):
    logger.info("Get train-val-test data loader")
    

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    in_feats = node_features.shape[1]
    n_classes = (node_labels.max() + 1).item()
    
    train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
    )

    valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    test_dataloader = dgl.dataloading.DataLoader(
    graph, test_nids, sampler,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
    )

    return (train_dataloader, valid_dataloader, test_dataloader, (in_feats, n_classes))

def train(args, data, device):
    n_layers = args.n_layers
    n_hidden = args.n_hidden
    num_epochs = args.num_epochs
    dropout = args.dropout

    print(args)
    train_dataloader, valid_dataloader, test_dataloader, (in_feats, n_classes) = data

    input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))

    activation = F.relu

    model = Model(in_feats, n_hidden, n_classes, n_layers, dropout, activation).to(device)
    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0
    
    best_eval_acc = 0
    best_test_acc = 0

    best_model_path = 'model.pt'
    best_model = None
    total_time = 0

    time_load = 0
    time_forward = 0
    time_backward = 0
    total_time = 0
    for epoch in range(num_epochs):
        model.train()
        tic = time.time()
        
        

        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            tic_start = time.time()
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            tic_step = time.time()
            predictions = model(mfgs, inputs)
            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            tic_forward = time.time()
            loss.backward()
            opt.step()
            tic_backward = time.time()

            time_load += tic_step - tic_start
            time_forward += tic_forward - tic_step
            time_backward += tic_backward - tic_forward

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
            if step % args.log_every == 0:
                logger.debug(
                        "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}".format(
                            epoch, step, loss.item(), accuracy.item()
                        )
                    )
                # print(
                #         "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}".format(
                #             epoch, step, loss.item(), accuracy.item()
                #         )
                #     )
        toc = time.time()
        total_time += toc - tic
        logger.debug(
            "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
                toc - tic, time_load, time_forward, time_backward
            )
        )        
        # print(
        #     "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
        #         toc - tic, time_load, time_forward, time_backward
        #     )
        # )

        if epoch % args.eval_every == 0:
            model.eval()

            predictions = []
            labels = []
            # with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
            with torch.no_grad():
                # for input_nodes, output_nodes, mfgs in tq:
                for input_nodes, output_nodes, mfgs in valid_dataloader:
                
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                eval_acc = sklearn.metrics.accuracy_score(labels, predictions)
                if best_eval_acc < eval_acc:
                    best_eval_acc = eval_acc
                    best_model = model
                    # torch.save(model.state_dict(), best_model_path)
                logger.debug('Epoch {}, Val Acc {}, Best Val Acc {}'.format(epoch, eval_acc, best_eval_acc))

    logger.debug("total time for {} epochs = {}".format(num_epochs, total_time))
    logger.debug("avg time per epoch = {}".format(total_time/num_epochs))
    return best_eval_acc, model

if __name__ == "__main__":
    
    args = parse_args_fn()
    print(args)
    best_eval_acc = 0
    root="../dataset/"
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    open('eval_acc.txt', 'w').close()
    for n_layers in range(3, 10):
        for fanout in range(3, 6):
                for dropout in [0.0, 0.25, 0.5, 0.75]:
                    for n_hidden in range(6, 11):
                        pass
                        args.num_epochs= 50
                        args.n_layers = n_layers
                        args.fanout= fanout
                        args.dropout= 0
                        args.eval_every= 5
                        args.log_every= 100
                        args.n_hidden = 2**n_hidden
                        args.batch_size= 512 # *1024*9//(args.n_layers*args.n_hidden)

                        batch_size = args.batch_size
                        n_layers = args.n_layers
                        fanout = args.fanout

                        
                        sampler = dgl.dataloading.NeighborSampler([fanout for _ in range(n_layers)])
                        
                        
                        # dataset = load_dataset(args.train)
                        data = _get_data_loader(sampler, device, dataset, batch_size)


                        
                        curr_seed = torch.seed()
                        eval_acc, model = train(args, data, device)
                        
                        if best_eval_acc < eval_acc:
                            best_eval_acc = eval_acc
                            # best_model_path = str(best_eval_acc).split(".")[1] + ".pt"
                            best_model_path = "model.pt"
                            torch.save(model.state_dict(), best_model_path)
                            with open("eval_acc.txt", "a+") as f:
                                f.write("{}, {}, {}, {} ".format(n_layers, fanout, dropout, n_hidden) + str(best_eval_acc) + '\n')

                            

