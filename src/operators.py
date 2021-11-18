# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian(n, requires_grad=False):
    assert n in [3, 5, 7], "n must be one of 3, 5 or 7"
    if n == 3:
        hood = [[1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.],]

        zeros = [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],]

        weights = torch.tensor([
            [hood, zeros, zeros],
            [zeros, hood, zeros],
            [zeros, zeros, hood],
        ]) / 16.
        res = nn.Conv2d(3, 3, 3, stride=1, padding=1, 
            bias=False, padding_mode='replicate')
    elif n == 5:        
        hood = [[1,  4,  7,  4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1,  4,  7,  4, 1]]

        zero = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]

        weights = torch.tensor([
            [hood, zero, zero],
            [zero, hood, zero],
            [zero, zero, hood],
        ]).float() / 273.
        res = nn.Conv2d(3, 3, 5, stride=1, padding=2, 
            bias=False, padding_mode='replicate') 
    elif n == 7:          
        hood = [[0,  0,  1,   2,  1,  0, 0],
                [0,  3, 13,  22, 13,  3, 0],
                [1, 13, 59,  97, 59, 13, 1],
                [2, 22, 97, 159, 97, 22, 2],
                [1, 13, 59,  97, 59, 13, 1],
                [0,  3, 13,  22, 13,  3, 0],
                [0,  0,  1,   2,  1,  0, 0],]

        zero = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]

        weights = torch.tensor([
            [hood, zero, zero],
            [zero, hood, zero],
            [zero, zero, hood],
        ]).float() / 1003.
        res = nn.Conv2d(3, 3, 7, stride=1, padding=3, 
            bias=False, padding_mode='replicate')            
        
    res.requires_grad_(requires_grad)
    res.weight.data = weights
    return res         