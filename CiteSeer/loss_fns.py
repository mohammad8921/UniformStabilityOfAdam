import torch

def CE(y_hat, y):
    return torch.sum(-1*torch.log(y_hat[range(y.size()[0]), y.long()])) / y.size()[0]

def RJM(y_hat, y):
    return torch.sum(1 - torch.sqrt(y_hat[range(y.size()[0]), y.long()])) / y.size()[0]