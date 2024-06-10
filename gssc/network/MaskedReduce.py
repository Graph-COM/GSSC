import torch
from torch import Tensor, BoolTensor
'''
x (B, N, d)
mask (B, N, 1)
'''
def maskedSum(x: Tensor, mask: BoolTensor, dim: int):
    '''
    mask true elements
    '''
    return torch.sum(torch.where(mask, 0, x), dim=dim)

def maskedMean(x: Tensor, mask: BoolTensor, dim: int, gsize: Tensor = None):
    '''
    mask true elements
    '''
    if gsize is None:
        gsize = x.shape[dim] - torch.sum(mask, dim=dim)
    return torch.sum(torch.where(mask, 0, x), dim=dim)/gsize

def maskedMax(x: Tensor, mask: BoolTensor, dim: int):
    return torch.max(torch.where(mask, -torch.inf, x), dim=dim)[0]

def maskedMin(x: Tensor, mask: BoolTensor, dim: int):
    return torch.min(torch.where(mask, torch.inf, x), dim=dim)[0]

def maskednone(x: Tensor, mask: BoolTensor, dim: int):
    return x

reduce_dict =  {
            "sum": maskedSum,
            "mean": maskedMean,
            "max": maskedMax,
            "none": maskednone
        }