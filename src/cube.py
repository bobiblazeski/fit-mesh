# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import save_obj


from src.util import make_faces
from src.operators import get_gaussian

from src.discrete_gaussian import DiscreteGaussian
from src.discrete_laplacian import DiscreteLaplacian

edges = torch.tensor([
    [ 0, 22,  2],
    [ 2, 22, 23],#
    [ 3,  7,  2],
    [ 7,  2,  6],#   
    [ 1, 18, 19],
    [ 1, 19,  3],#    
    [ 1, 14, 15],
    [ 1, 14,  0],#    
    [13, 15, 16],
    [15, 16, 18],#    
    [12, 14, 20],
    [14, 22, 20],#    
    [23, 21,  6],
    [ 6,  4, 21],#    
    [ 7, 19, 17],
    [ 5,  7, 17],#    
    [11, 16, 17],
    [11,  9, 16],#
    [ 8,  9, 13],
    [13,  8, 12],#
    [10,  4,  5],
    [11,  5, 10],#    
    [10,  8, 21],
    [20, 21,  8],
])

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def get_edge_vertices(tri, pair, n):
    square = torch.arange(n**2).reshape(n, n)
    if pair == [0, 1]:
        r = square[0, :]    
    elif pair == [0, 2]:
        r = square[:, 0]
    elif pair == [1, 3]:
        r = square[:, -1]
    elif pair == [2, 3]:
        r = square[-1, :]
    else:
        raise Exception(f'Unknown pair {pair}')
    return ((n ** 2) * tri  + r).tolist()

def single_edge_faces(l1, l2):
    t1 = [[a, b, c] for a, b, c 
          in zip(l1, l1[1:], l2)]
    t2 = [[a, b, c] for a, b, c 
          in zip(l2, l2[1:], l1[1:])]
    return t1 + t2

def make_edges(n):
    res  = []
    for x, y in pairwise(edges.tolist()):
        m = list(set(x + y))
        m.sort()
        a1, a2, b1, b2 = m              
        pair_a = [a1 % 4, a2 % 4]
        pair_b = [b1 % 4, b2 % 4]         
        tri_a, tri_b = a1 // 4, b1 // 4                
        l1 = get_edge_vertices(tri_a, pair_a, n)
        l2 = get_edge_vertices(tri_b, pair_b, n)        
        res = res + single_edge_faces(l1, l2)     
    return torch.tensor(res)

def make_corners(n):
    s0 = 0
    s1 = n-1
    s2 = n**2-n
    s3 = n**2-1
    tris = torch.tensor([
        [0, 5, 3],
        [0, 4, 1],
        [4, 2, 3],
        [4, 2, 1],
        [0, 4, 3],
        [0, 5, 1],
        [5, 2, 3],
        [5, 2, 1]])  
    rmn = torch.tensor([
        [s0, s2, s2],
        [s3, s3, s3],
        [s0, s1, s1],
        [s1, s3, s1],
        [s1, s2, s3],
        [s2, s3, s2],
        [s0, s0, s0],
        [s1, s2, s0]])
    return tris * n**2+ rmn

def make_sides(n):
    offset, faces = n ** 2, make_faces(n, n)
    sides = torch.cat([
        i * offset + torch.tensor(faces)
        for i in range(6)])
    return sides

def make_cube_faces(n):
    sides = make_sides(n)
    corners = make_corners(n)
    edges = make_edges(n)
    return torch.cat((sides, corners, edges))

def make_cube_mesh(n, start=-0.5, end=0.5):
    d1, d2 = torch.meshgrid(
        torch.linspace(start, end, steps=n),
        torch.linspace(start, end, steps=n))
    d3 = torch.full_like(d1, end) + 1 / n
    sides =  OrderedDict({
        'front': torch.stack((+d3,  d1,  d2), dim=-1),
        'right': torch.stack(( d1, +d3,  d2), dim=-1),    
        'back' : torch.stack((-d3,  d1,  d2), dim=-1),         
        'left' : torch.stack(( d1, -d3,  d2), dim=-1),
        'top'  : torch.stack(( d1,  d2, +d3), dim=-1),
        'down' : torch.stack(( d1,  d2, -d3), dim=-1),
    })
    vert = torch.stack(list(sides.values())).reshape(-1, 3)
    faces = make_cube_faces(n)
    textures = TexturesVertex(verts_features=[torch.ones_like(vert)])
    mesh = Meshes(verts=[vert], faces=[faces], textures=textures)
    return mesh

class Cube(nn.Module):
    def __init__(self, n, kernel=21, sigma=7, start=-0.5, end=0.5):
        super(Cube, self).__init__()        
        self.n = n
        self.params = nn.ParameterDict({
            'front': nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=True)),
            'back' : nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=False)),
            'left' : nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=False)),
            'right': nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=False)),
            'top'  : nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=False)),
            'down' : nn.Parameter(torch.zeros((1, 3, n, n), requires_grad=False)),
        })
        self.source = make_cube_mesh(n, start, end)
        self.gaussian = get_gaussian(kernel)
        #self.gaussian = DiscreteGaussian(kernel, sigma=sigma)
        self.laplacian = DiscreteLaplacian()
        clip_value = 1. / n
        for p in self.params.values():
            #p.register_hook(lambda grad: torch.nan_to_num(grad))
            p.register_hook(lambda grad: self.gaussian(torch.nan_to_num(grad)))
            #p.register_hook(lambda grad: torch.clamp(self.gaussian(grad), -clip_value, clip_value))
            #p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))        

    def make_vert(self):
        return torch.cat([p[0].reshape(3, -1).t()
                          for p in self.params.values()]) 

    def forward(self):
        ps = torch.cat([p for p in self.params.values()])
        deform_verts = ps.permute(0, 2, 3, 1).reshape(-1, 3)        
        new_src_mesh = self.source.offset_verts(deform_verts)
        # deform_verts = self.make_vert()
        # new_src_mesh = self.source.offset_verts(deform_verts)
        return new_src_mesh, 0 #self.laplacian(ps)
    
    def to(self, device):
        module = super(Cube, self).to(device)        
        module.source = self.source.to(device)        
        return module
    
    def export(self, f):        
        mesh, _ = self.forward()
        mesh = mesh.detach()
        save_obj(f, mesh.verts_packed(), mesh.faces_packed())   

