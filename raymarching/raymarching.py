import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend



#########################################
### inference functions
#########################################

### march_rays
# inputs:
#   n_alive: n
#   n_step: int
#   rays_alive: int [n], only the alive IDs in N (n may > n_alive, but we only work on first n_alive)
#   rays_t: float [n], input & output
#   rays_o/d: float [N, 3], all rays
#   bound: float
#   density_grid: float [H, H, H]
#   mean_density: float
#   near/far: float [N]
# outputs:
#   xyzs, dirs, dt: float [n_alive * n_step, 3/3/2], output
class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_grid, mean_density, near, far, align=-1):
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        H = density_grid.shape[0] # grid resolution
        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # 2 vals, one for rgb, one for depth

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, H, density_grid, mean_density, near, far, xyzs, dirs, deltas)

        return xyzs, dirs, deltas

march_rays = _march_rays.apply


### composite_rays 
# modify rays_alive to -1 if it is dead.(actual_step < step, indicated by dt <= 0)
# inputs:
#   n_alive: int
#   n_step: int
#   rays_alive: int [n]
#   sigmas, rgbs, deltas: float [n_alive * n_step, 1/3/2]
#   depth, image, weight: float [N, 1/3/1]
class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights, depth, image):
        _backend.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights, depth, image)


composite_rays = _composite_rays.apply

### compact_rays
# inputs:
#   rays_alive_old
#   rays_t_old
# outputs:
#   rays_alive
#   rays_t
class _compact_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter):
        _backend.compact_rays(n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter)

compact_rays = _compact_rays.apply