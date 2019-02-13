
from torch.utils.data import DataLoader, Dataset
import torch
from options import opt
import numpy as np

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx]


def my_collate(batch):
    x, y = zip(*batch)

    x, lengths, y = pad(x, y)

    if opt.gpu >= 0 and torch.cuda.is_available():
        x = x.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return x, lengths, y

def pad(x, y):
    tokens = x

    lengths = [len(row) for row in tokens]
    max_len = max(lengths)

    tokens = pad_sequence(tokens, max_len)
    lengths = torch.LongTensor(lengths)

    y = torch.LongTensor(y).view(-1)


    return tokens, lengths, y


def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x