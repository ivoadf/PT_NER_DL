import torch

def FloatTensor(gpu):
    if gpu:
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor

def LongTensor(gpu):
    if gpu:
        return torch.cuda.LongTensor
    else:
        return torch.LongTensor
