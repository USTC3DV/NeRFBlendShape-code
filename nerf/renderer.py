import imp
import time
import mcubes
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching 

from datetime import datetime



def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=0.05)
        # print(near)

    return near, far





class NeRFRenderer(nn.Module):
    def __init__(self,
                 cuda_ray=False,no_pru=False,
                 ):
        super().__init__()

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        self.no_pru=no_pru
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([128 + 1] * 3)+1e-3 
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(64, 2, dtype=torch.int32) 
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d,exps, bound):
        raise NotImplementedError()

    def density(self, x,exps, bound):
        raise NotImplementedError()



    def run_cuda(self, rays_o, rays_d,exps,exp_ori, bound, num_steps, bg_color,index=None):

        B, N = rays_o.shape[:2]
        device = rays_o.device
        A=None

        if bg_color is None:
            bg_color = torch.ones(3, dtype=rays_o.dtype, device=device)

        if self.training:
            pass
        else:
            dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # print(dtype)
            
            weights_sum = torch.zeros(B * N, dtype=dtype, device=device)
            depth = torch.zeros(B * N, dtype=dtype, device=device)
            image = torch.zeros(B * N, 3, dtype=dtype, device=device)
            
            n_alive = B * N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
            # print(near,far)
            near = near.view(B * N)
            far = far.view(B * N)

            step = 0
            i = 0
            while step < 1024: # max step
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = near
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    # print("break")
                    break

                # decide compact_steps
                n_step = max(min(B * N // n_alive, 8), 1)


                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, bound, self.density_grid, self.mean_density, near, far, 128)
                sigmas, rgbs,exps_code = self(xyzs, exps,exp_ori,d=dirs, bound=bound,index=torch.tensor([0],device="cuda",dtype=torch.long))
                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], sigmas, rgbs, deltas, weights_sum, depth, image)

                step += n_step
                i += 1
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = (depth - near) / (far - near)

        depth = depth.reshape(B, N)
        image = image.reshape(B, N, 3)

        
        if A==None:
            return depth, image,exps_code
        else:
            A = A.reshape(B,N,1)
            return depth,image,A,exps_code


    def render(self, rays_o, rays_d,exps,exp_ori,index=None, bound=1, num_steps=128,  staged=False, bg_color=None, **kwargs):
        _run = self.run_cuda

        B, N = rays_o.shape[:2]
        device = rays_o.device
        A=None
        # print(B,N)

        if self.training :
            depth, image,A,exps_code = _run(rays_o, rays_d,exps,exp_ori, bound, num_steps,  bg_color,index=index)
        else:
            depth, image,exps_code = _run(rays_o, rays_d,exps,exp_ori, bound, num_steps, bg_color)

        results = {}
        results['depth'] = depth
        results['rgb'] = image[:,:,0:3]
        results["exps_code"] =exps_code
        if A!=None:
            results['A'] = A
        
            
        return results
