import torch

def FloatTensor():
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor

def LongTensor():
    if torch.cuda.is_available():
        return torch.cuda.LongTensor
    else:
        return torch.LongTensor
