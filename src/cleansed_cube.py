# pyright: reportMissingImports=false
from collections import (
    namedtuple,
    OrderedDict,
)
import math
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

#from src.util import make_faces
from src.operators import get_gaussian

from src.discrete_gaussian import DiscreteGaussian
from src.discrete_laplacian import DiscreteLaplacian
from src.padding import pad_side


from trimesh.util import triangle_strips_to_faces

def create_strips(n, m):
    res = []
    for i in range(n-1):
        strip = []
        for j in range(m):            
            strip.append(j+(i+1)*m)
            strip.append(j+i*m)
            #strip.append(j+(i+1)*m)
        res.append(strip)
    return res

def make_faces(n, m):
    strips = create_strips(n, m)    
    return triangle_strips_to_faces(strips)

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

def sides_dict(n):
    return nn.ParameterDict({
        'front': nn.Parameter(torch.zeros((1, 3, n, n))),
        'back' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'left' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'right': nn.Parameter(torch.zeros((1, 3, n, n))),
        'top'  : nn.Parameter(torch.zeros((1, 3, n, n))),
        'down' : nn.Parameter(torch.zeros((1, 3, n, n))),
    })

Mesh = namedtuple('Mesh', ['vertices', 'faces', 'colors'])

class SourceCube(nn.Module):
    def __init__(self, n, start=-0.5, end=0.5):
        super(SourceCube, self).__init__()
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
        vertices = torch.stack(list(sides.values())).reshape(-1, 3)
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', make_cube_faces(n).int())
        self.register_buffer('colors', torch.ones_like(vertices) * 0.5)

    def to_homogeneous(_, t):
        return torch.cat((t,
            torch.ones(t.size(0), 1, device=t.device)), dim=-1)

    def forward(self, deform_verts):
        vert = self.vertices + deform_verts
        pos = self.to_homogeneous(vert)[None]
        return Mesh(pos, self.faces, self.colors)

class SimpleCube(nn.Module):
    def __init__(self, n, kernel=5, sigma=1, clip_value = 0.1, start=-0.5, end=0.5):
        super(SimpleCube, self).__init__()        
        self.n = n
        self.kernel = kernel
        self.params = sides_dict(n)
        self.source = SourceCube(n, start, end)
        #self.gaussian = get_gaussian(kernel)
        self.gaussian = DiscreteGaussian(kernel, sigma=sigma, padding=False)
        self.laplacian = DiscreteLaplacian()          
        for p in self.params.values():            
            p.register_hook(lambda grad: torch.clamp(
                torch.nan_to_num(grad), -clip_value, clip_value))

    def make_vert(self):
        return torch.cat([p[0].reshape(3, -1).t()
                          for p in self.params.values()]) 

    def forward(self):
        ps = torch.cat([p for p in self.params.values()])
        deform_verts = ps.permute(0, 2, 3, 1).reshape(-1, 3)        
        new_src_mesh = self.source(deform_verts)        
        return new_src_mesh, self.laplacian(ps)
    
    def smooth(self):
        sides = {}
        for side_name in self.params:
            grad = self.params[side_name].grad[0]
            sides[side_name] = grad.permute(1, 2, 0)
            
        for side_name in self.params:
            padded = pad_side(sides, side_name, self.kernel)
            padded = padded.permute(2, 0, 1)[None]
            padded = self.gaussian(padded)
            self.params[side_name].grad.copy_(padded)    
      
    def export(self, f):        
        mesh, _ = self.forward()
        vertices = mesh.vertices[0].cpu().detach()
        faces = mesh.faces.cpu().detach()        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f)

class ProgressiveCube(nn.Module):
    def __init__(self, n, kernel=3, sigma=1, clip=None, start=-0.5, end=0.5):
        super(ProgressiveCube, self).__init__()        
        self.n = n
        self.kernel= kernel
        self.side_names = list(sides_dict(1).keys())
        self.params = nn.ModuleList([sides_dict(2**i)
            for i in range(1, int(math.log2(n))+1)])
        
        self.source = SourceCube(n, start, end)
        #self.gaussian = get_gaussian(kernel)
        self.gaussian = DiscreteGaussian(kernel, sigma=sigma,  padding=False)
        self.laplacian = DiscreteLaplacian()
        clip = clip or 1. / n
        for d in self.params:
            for p in d.values():                
                p.register_hook(lambda grad:
                    torch.clamp(torch.nan_to_num(grad), -clip, clip))

    def make_vert(self):
        return torch.cat([p[0].reshape(3, -1).t()
                          for p in self.params.values()])

    def scale(self, t):
        return  F.interpolate(t, self.n, mode='bilinear', align_corners=True)

    def forward(self):
        summed = {}
        for d in self.params:            
            for key in self.side_names:
                if key in summed:
                    summed[key] = summed[key] + self.scale(d[key])
                else:
                    summed[key] = self.scale(d[key])        
        ps = torch.cat([p for p in summed.values()])        
        deform_verts = ps.permute(0, 2, 3, 1).reshape(-1, 3)         
        new_src_mesh = self.source(deform_verts)        
        return new_src_mesh, 0#self.laplacian(ps)    
    
    def smooth(self):
        for i in range(len(self.params)):
            params, sides = self.params[i], {}
            for side_name in params:
                grad = params[side_name].grad[0]        
                sides[side_name] = grad.permute(1, 2, 0)

            for side_name in params:
                padded = pad_side(sides, side_name, self.kernel)
                padded = padded.permute(2, 0, 1)[None]
                padded = self.gaussian(padded)
                self.params[i][side_name].grad.copy_(padded)

    def export(self, f):        
        mesh, _ = self.forward()
        vertices = mesh.vertices[0].cpu().detach()
        faces = mesh.faces.cpu().detach()        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f)