import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr


from src.cleansed_cube import (
  SimpleCube,
  ProgressiveCube,
)

device = torch.device('cuda')
n = 8
cube = SimpleCube(n).to(device)
cube = ProgressiveCube(n).to(device)
(vert, faces, colors), _ = cube()
print([f.shape for f in [vert, faces, colors]])

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

glctx = dr.RasterizeGLContext()

vert = vert + torch.randn_like(vert, device=device) * 0.1
rast, _ = dr.rasterize(glctx, vert, faces, resolution=[256, 256])
out, _ = dr.interpolate(colors, rast, faces)

img = out.cpu().detach().numpy()[0, ::-1, :, :] # Flip vertically.
img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

print("Saving to './data/cube.png'.")
imageio.imsave('./data/cube.png', img)