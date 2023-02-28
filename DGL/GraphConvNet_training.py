import time
import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl.nn import GraphConv
from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
from GraphConvNet import GraphConvNet

def main():
    # hyperparameter
    num_epochs = 10000
    lr = 1e-3
    seed = 42
    hidden_channels=64
    num_layers=6

    dataset = PPIDataset()
    print(dir(dataset))
    test_dataset, train_dataset = split_dataset(dataset, [0.1, 0.9], shuffle=True)
    train_loader = GraphDataLoader(train_dataset, batch_size=8)
    test_loader = GraphDataLoader(test_dataset, batch_size=1)
    
    for batch in train_loader:
        # this for loop is only to obtain a single batch to get the feature size
        break
    print(batch)
    
    in_channels = batch.ndata['feat'].shape[1]
    out_channels = dataset.num_labels
    print(in_channels, out_channels)
    
    model = GraphConvNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, num_layers=num_layers)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("my_device = {}".format(my_device))
    model = model.to(my_device)

    # losses = []
    # time_elapsed = []
    # epochs = []
    # t0 = time.time()

    # for epoch in range(num_epochs):
    #     total_loss = 0.0
    #     batch_count = 0
    #     for batch in train_loader:
    #         optimizer.zero_grad()
    #         batch = batch.to(my_device)
    #         pred = model(batch, batch.ndata["feat"].to(my_device))
    #         loss = loss_fn(pred, batch.ndata["label"].to(my_device))
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.detach()
    #         batch_count += 1
        
    #     mean_loss = total_loss / batch_count
    #     losses.append(mean_loss)
    #     epochs.append(epoch)
    #     time_elapsed.append(time.time() - t0)

    #     if epoch % 100 == 0:
    #         print("loss at epoch {} = {}".format(epoch, mean_loss))

    # num_correct = 0
    # num_total = 0
    # model.eval()

    # for batch in test_loader:
    #     batch = batch.to(my_device)
    #     pred = model(batch, batch.ndata['feat'])
    #     num_correct += (pred.round() == batch.ndata['label'].to(my_device)).sum()
    #     num_total += pred.shape[0]*pred.shape[1]

    #     np.save("dgl.npy", \
    #     {
    #         "epochs": epochs, \
    #         "losses": losses, \
    #         "time_elapsed": time_elapsed

    #     })

    # print("test accuracy = {}".format(num_correct / num_total))

    
if __name__ == "__main__":
    main()