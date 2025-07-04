# pyright: reportMissingImports=false
import numpy  as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

class DiscreteGaussian(nn.Module):
    def __init__(self, kernel_size, sigma=1, padding=True):
        super(DiscreteGaussian, self).__init__()
        print(kernel_size)
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.side = (kernel_size-1) // 2
        self.sigma = sigma
        self.padding = self.side if padding else 0
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.padding), 
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=None, groups=3)
        )
        self.seq.requires_grad_(False)
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((self.kernel_size, self.kernel_size))
        n[self.side, self.side] = 1
        k = gaussian_filter(n,sigma=self.sigma)
        for _, f in self.named_parameters():            
            f.data.copy_(torch.from_numpy(k))
