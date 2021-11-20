# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn

class DiscreteLaplacian(nn.Module):
    def __init__(self):
        super(DiscreteLaplacian, self).__init__()        
        self.seq = nn.Sequential(OrderedDict([
          #('padd', nn.ReflectionPad2d(1)),
          ('conv', nn.Conv2d(3, 3, 3, stride=1, padding=0,
                             bias=False, groups=3)),
        ]))
        self.seq.requires_grad_(False)
        self.weights_init()

    def forward(self, x):
        return self.seq(x).abs().mean()

    def weights_init(self):
        w = torch.tensor([[ 1.,   4., 1.],
                          [ 4., -20., 4.],
                          [ 1.,   4., 1.],])
        for _, f in self.named_parameters():           
            f.data.copy_(w)