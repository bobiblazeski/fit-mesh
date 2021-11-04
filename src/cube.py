# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d
import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from src.util import make_faces

class Cube(nn.Module):
    def __init__(self, n, start=-0.5, end=0.5):
        super(Cube, self).__init__()
        self.n = n

        self.params = self.make_params(n, start, end)
        self.source = self.make_source(n, start, end)            

    def make_params(self, n, start, end):        
        return nn.ParameterDict({
            'front': nn.Parameter(torch.zeros((n, n, 3))),
            'back' : nn.Parameter(torch.zeros((n, n, 3))),            
            'left' : nn.Parameter(torch.zeros((n, n, 3))),
            'right': nn.Parameter(torch.zeros((n, n, 3))),            
            'top'  : nn.Parameter(torch.zeros((n, n, 3))),
            'down' : nn.Parameter(torch.zeros((n, n, 3))),
        })

    def make_vert(self, params):
        return torch.stack(list(params.values())).reshape(-1, 3)

    def make_source(self, n, start, end):
        d1, d2 = torch.meshgrid(
            torch.linspace(start, end, steps=n), 
            torch.linspace(start, end, steps=n))
        d3 = torch.full_like(d1, end)
        sides =  {
            'front': torch.stack((+d3, d1, d2), dim=-1),
            'back' : torch.stack((-d3, d1, d2), dim=-1),            
            'left' : torch.stack((d1, +d3, d2), dim=-1),
            'right': torch.stack((d1, -d3, d2), dim=-1),
            'top'  : torch.stack((d1, d2, +d3), dim=-1),
            'down' : torch.stack((d1, d2, -d3), dim=-1),
        }
        vert = self.make_vert(sides)
        offset, faces = n ** 2, make_faces(n, n)
        faces = torch.cat([
            i * offset + torch.tensor(faces)
            for i in range(6)])
        textures = TexturesVertex(verts_features=[torch.ones_like(vert)])
        return Meshes(verts=[vert], faces=[faces], textures=textures)
    
    def forward(self):
        deform_vert = self.make_vert(self.params)
        new_mesh = self.source.offset_verts(deform_vert)        
        return new_mesh

    def to(self, device):
        module = super(Cube, self).to(device)        
        module.source = self.source.to(device)        
        return module