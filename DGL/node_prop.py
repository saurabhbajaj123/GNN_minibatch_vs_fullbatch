import dgl
import torch
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import GraphDataLoader
from GraphConvNet import GraphConvNet
from dgl.data.utils import split_dataset
import torch.nn.functional as F
import time

def _collate_fn(batch):
    graphs = [e[0] for e in batch]
    labels = [e[1] for e in batch]
    g = dgl.batch(graphs)
    labels = torch.stack(labels, 0)
    return g, labels

def main():
        
    root='../dataset/data'

    # dataset = DglGraphPropPredDataset(name='ogbg-molhiv', root=root)
    # dataset = DglGraphPropPredDataset(name='ogbg-molpcba', root=root)
    # data = dataset[0]
    # print(data)
    
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=root)
    # data = dataset[0]
    split_idx = dataset.get_idx_split()
    print(dataset)
    # print(data[0], data[1].size())
    # print(dir(dataset))
    # print(split_idx)
    test_dataset, train_dataset = split_dataset(dataset, [0.1, 0.9], shuffle=True)
    train_loader = GraphDataLoader(train_dataset, batch_size=8)
    test_loader = GraphDataLoader(test_dataset, batch_size=1)

    # train_loader = GraphDataLoader(dataset[split_idx['train']], batch_size=1, shuffle=True, collate_fn=collate_dgl)
    # valid_loader = GraphDataLoader(dataset[split_idx['valid']], batch_size=1, shuffle=True, collate_fn=collate_dgl)
    # test_loader = GraphDataLoader(dataset[split_idx['test']], batch_size=1, shuffle=True, collate_fn=collate_dgl)
    # print(len(train_loader))

    for batch in train_loader:
        # this for loop is only to obtain a single batch to get the feature size
        print(batch[0].ndata)
        break

    in_channels = batch[0].ndata['feat'].shape[1] # this needs to be a parameter dependent on the dataset
    out_channels = 1 # int(dataset.num_classes) # this needs to be a parameter dependent on the dataset
    print(in_channels, out_channels)

    num_epochs = 10
    lr = 1e-3
    seed = 42
    hidden_channels=64
    num_layers=6

    model = GraphConvNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("my_device = {}".format(my_device))
    model = model.to(my_device)

    losses = []
    time_elapsed = []
    epochs = []
    t0 = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            batch = batch.to(my_device)
            pred = model(batch, batch.ndata["feat"].to(my_device))
            print(pred.size(), labels.size())
            loss = loss_fn(pred, labels.to(my_device))
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            batch_count += 1
        
        mean_loss = total_loss / batch_count
        losses.append(mean_loss)
        epochs.append(epoch)
        time_elapsed.append(time.time() - t0)

        if epoch % 100 == 0:
            print("loss at epoch {} = {}".format(epoch, mean_loss))

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