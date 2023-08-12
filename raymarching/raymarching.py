import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

#########################################
### training functions
#########################################

class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, rays_o, rays_d, bound, density_grid, mean_density, iter_density, step_counter=None, mean_count=-1, perturb=False, align=-1, force_all_rays=False):
        
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays
        H = density_grid.shape[0] # grid resolution

        M = N * 2048 # init max points number in total, hardcoded

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count
        
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, dtype=rays_o.dtype, device=rays_o.device)
        xyzstorays = torch.zeros(M, dtype=torch.int32, device=rays_o.device)
        rays = torch.zeros(N, 3, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps

        if step_counter is None:
            step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device) # point counter, ray counter


        _backend.march_rays_train(rays_o, rays_d, density_grid, mean_density, iter_density, bound, N, H, M, xyzs, dirs, deltas,xyzstorays, rays, step_counter, perturb) # m is the actually used points number

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item() # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]
            xyzstorays = xyzstorays[:m]


        return xyzs, dirs, deltas, rays, xyzstorays

march_rays_train = _march_rays_train.apply


### accumulate rays (need backward)
# inputs: sigmas: [M], rgbs: [M, 3], rays: [N, 3], points [M, 7]
# outputs: depth: [N], image: [N, 3]
class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, sigmas, rgbs, deltas, rays, bound, bg_color):
        
        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()
        deltas = deltas.contiguous()
        rays = rays.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        depth = torch.zeros(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.zeros(N, 3, dtype=sigmas.dtype, device=sigmas.device)
        A = torch.zeros(N,1, dtype=sigmas.dtype, device=sigmas.device)

        _backend.composite_rays_train_forward(sigmas, rgbs, deltas, rays, bound, bg_color, M, N, depth, image,A)
        
        ctx.save_for_backward(sigmas, rgbs, deltas, rays, image,A, bg_color)
        ctx.dims = [M, N, bound]

        return depth, image,A
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_depth, grad_image,grad_A):

        grad_image = grad_image.contiguous()
        grad_A = grad_A.contiguous()

        sigmas, rgbs, deltas, rays, image,A, bg_color= ctx.saved_tensors
        M, N, bound = ctx.dims
        
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.composite_rays_train_backward(grad_image,grad_A, sigmas, rgbs, deltas, rays, image,A, bound, M, N, grad_sigmas, grad_rgbs)
        return grad_sigmas, grad_rgbs, None, None, None, None


composite_rays_train = _composite_rays_train.apply

#########################################
### inference functions
#########################################
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

class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights, depth, image):
        _backend.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights, depth, image)
        return tuple()


composite_rays = _composite_rays.apply

class _compact_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter):
        _backend.compact_rays(n_alive, rays_alive, rays_alive_old, rays_t, rays_t_old, alive_counter)

compact_rays = _compact_rays.apply