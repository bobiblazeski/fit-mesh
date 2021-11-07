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

        self.params = self.make_params(n)
        self.source = self.make_source(n, start, end)            

    def make_params(self, n):
        return nn.ParameterDict({
            'front': nn.Parameter(torch.zeros((n - 2, n - 2, 3))),
            'back' : nn.Parameter(torch.zeros((n - 2, n - 2, 3))),            
            'left' : nn.Parameter(torch.zeros((n - 2, n - 2, 3))),
            'right': nn.Parameter(torch.zeros((n - 2, n - 2, 3))),            
            'top'  : nn.Parameter(torch.zeros((n - 2, n - 2, 3))),
            'down' : nn.Parameter(torch.zeros((n - 2, n - 2, 3))),
            # Edges
            'lf' : nn.Parameter(torch.zeros((n - 2, 1, 3))),
            'fr' : nn.Parameter(torch.zeros((n - 2, 1, 3))),
            'rb' : nn.Parameter(torch.zeros((n - 2, 1, 3))),
            'bl' : nn.Parameter(torch.zeros((n - 2, 1, 3))),
            
            'lt' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'ft' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'rt' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'bt' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            
            'ld' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'fd' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'rd' : nn.Parameter(torch.zeros((1, n - 2, 3))),
            'bd' : nn.Parameter(torch.zeros((1, n - 2, 3))),         
            # Corners
            'ltf' : nn.Parameter(torch.zeros((1, 1, 3))),
            'ftr' : nn.Parameter(torch.zeros((1, 1, 3))),
            'rtb' : nn.Parameter(torch.zeros((1, 1, 3))),
            'btl' : nn.Parameter(torch.zeros((1, 1, 3))),
            
            'ldf' : nn.Parameter(torch.zeros((1, 1, 3))),
            'fdr' : nn.Parameter(torch.zeros((1, 1, 3))),
            'rdb' : nn.Parameter(torch.zeros((1, 1, 3))),
            'bdl' : nn.Parameter(torch.zeros((1, 1, 3))),
        })

    def make_vert(self, p):        
        tr = torch.cat((p['btl'], p['lt'], p['ltf']), dim=1)
        dr = torch.cat((p['bdl'], p['ld'], p['ldf']), dim=1)
        mr = torch.cat((p['bl'], p['left'], p['lf']), dim=1)
        left = torch.cat((tr, mr, dr), dim=0)        

        tr = torch.cat((p['ltf'], p['ft'], p['ftr']), dim=1)
        dr = torch.cat((p['ldf'], p['fd'], p['fdr']), dim=1)
        mr = torch.cat((p['lf'], p['front'], p['fr']), dim=1)
        front = torch.cat((tr, mr, dr), dim=0)        

        tr = torch.cat((p['ftr'], p['rt'], p['rtb']), dim=1)
        dr = torch.cat((p['fdr'], p['rd'], p['rdb']), dim=1)
        mr = torch.cat((p['fr'], p['right'], p['rb']), dim=1)
        right = torch.cat((tr, mr, dr), dim=0)        

        tr = torch.cat((p['rtb'], p['bt'], p['btl']), dim=1)
        dr = torch.cat((p['rdb'], p['bd'], p['bdl']), dim=1)
        mr = torch.cat((p['rb'], p['back'], p['bl']), dim=1)
        back = torch.cat((tr, mr, dr), dim=0)        

        tr = torch.cat((p['btl'], p['bt'], p['rtb']), dim=1)
        dr = torch.cat((p['ltf'], p['ft'], p['ftr']), dim=1)
        mr = torch.cat((p['lt'].permute(1, 0, 2), p['top'],
            p['rt'].permute(1, 0, 2)), dim=1).contiguous()
        top = torch.cat((tr, mr, dr), dim=0)        

        tr = torch.cat((p['ldf'], p['fd'], p['fdr']), dim=1)
        dr = torch.cat((p['bdl'], p['bd'], p['rdb']), dim=1)
        mr = torch.cat((p['ld'].permute(1, 0, 2), p['down'],
            p['rd'].permute(1, 0, 2)), dim=1).contiguous()
        down = torch.cat((tr, mr, dr), dim=0)

        # tr = torch.cat((p['fdr'], p['rd'], p['rdb']), dim=1)
        # dr = torch.cat((p['bdl'], p['ld'], p['ldf']), dim=1)
        # mr = torch.cat((p['bd'].permute(1, 0, 2), p['down'],
        #     p['ld'].permute(1, 0, 2)), dim=1).contiguous()
        # down = torch.cat((tr, mr, dr), dim=0)

        #top, down = torch.zeros_like(top), torch.zeros_like(down)
        # right, back = torch.zeros_like(right), torch.zeros_like(back)
        # front = torch.zeros_like(front)
        sides = [left, front, right, back, top, down]
        return torch.stack(sides).reshape(-1, 3)

    def make_source(self, n, start, end):
        d1, d2 = torch.meshgrid(
            torch.linspace(start, end, steps=n), 
            torch.linspace(start, end, steps=n))
        d3 = torch.full_like(d1, end)
        sides =  {
            'front': torch.stack((+d3,  d1,  d2), dim=-1),
            'back' : torch.stack((-d3,  d1,  d2), dim=-1),            
            'left' : torch.stack(( d1, +d3,  d2), dim=-1),
            'right': torch.stack(( d1, -d3,  d2), dim=-1),
            #'top'  : torch.stack(( d1,  d2, +d3), dim=-1),
            #'down' : torch.stack(( d1,  d2, -d3), dim=-1),
        }
        vert = torch.stack(list(sides.values())).reshape(-1, 3)
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