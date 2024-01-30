import torch
from torch_geometric.data import Dataset
import os.path as osp

class OrkutDataset(Dataset):
    def __init__(self):
        super().__init__()

    def process(self):
        root = '/work/sbajaj_umass_edu/GNN_minibatch_vs_fullbatch/pytorch_geometric/dataset'
        self.data = torch.load(root + '/orkut/orkut.bin')
        self.num_classes = (self.data.y + 1).item()
        self.num_features = self.data.x.shape[1]
    
    def get_idx_split(self):
        train_idx = self.data.train_mask.nonzero().view(-1)
        val_idx = self.data.val_mask.nonzero().view(-1)
        test_idx = self.data.test_mask.nonzero().view(-1)
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }

    def len(self):
        return 1

    def get(self, idx):
        return self.data

dataset = OrkutDataset()
graph = dataset[0]