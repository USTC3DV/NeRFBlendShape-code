import imp
import time
# import mcubes
# import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching 


from datetime import datetime



def near_far_from_bound(rays_o, rays_d, bound, type='cube'):

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound 
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        near = torch.clamp(near, min=0.05)

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



    def run_cuda(self, rays_o, rays_d,exps, bound, num_steps, bg_color,index=None):
        B, N = rays_o.shape[:2]
        device = rays_o.device
        A=None

        if bg_color is None:
            bg_color = torch.ones(3, dtype=rays_o.dtype, device=device)

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 64]
            counter.zero_() # set to 0
            self.local_step += 1

            # print(rays_o.shape)
            # print(rays_d.shape)

            xyzs, dirs, deltas, rays,xyzstorays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, self.training, 128, True)

            sigmas, rgbs,exps_code = self(xyzs, exps, d=dirs, bound=bound,xyzstorays=xyzstorays,index=index)#(GX) seems iter_denisty is not used by this function?

            depth, image,A = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound, bg_color)

        else:
            dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            
            weights_sum = torch.zeros(B * N, dtype=dtype, device=device)
            depth = torch.zeros(B * N, dtype=dtype, device=device)
            image = torch.zeros(B * N, 3, dtype=dtype, device=device)
            
            n_alive = B * N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
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
                n_step = max(min(B * N // n_alive, 8), 1)


                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, bound, self.density_grid, self.mean_density, near, far, 128)
                sigmas, rgbs,exps_code = self(xyzs, exps, d=dirs, bound=bound,index=torch.tensor([0],device="cuda",dtype=torch.long))
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


    def render(self, rays_o, rays_d,exps,index=None, bound=1, num_steps=128,  staged=False, bg_color=None, **kwargs):

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        A=None

        if self.training :
            depth, image,A,exps_code = _run(rays_o, rays_d,exps, bound, num_steps,  bg_color,index=index)
        else:
            depth, image,exps_code = _run(rays_o, rays_d,exps, bound, num_steps, bg_color)

        results = {}
        results['depth'] = depth
        results['rgb'] = image[:,:,0:3]
        results["exps_code"] =exps_code
        if A!=None:
            results['A'] = A
        
            
        return results

    def update_extra_state(self, bound, decay=0.9):
        # call before each epoch to update extra states.

        if self.no_pru:
            return
        if not self.cuda_ray:
            return 
        
        ### update density grid
        resolution = self.density_grid.shape[0]
        X = torch.linspace(-1, 1, resolution).split(128)
        Y = torch.linspace(-1, 1, resolution).split(128)
        Z = torch.linspace(-1, 1, resolution).split(128)
        half_grid_size=bound / 128

        tmp_grid=torch.zeros([len(self.max_per),resolution,resolution,resolution]).cuda()

        with torch.no_grad():
            for exp_count in range(len(self.max_per)):
                exps=torch.zeros([1,self.basis_num]).cuda()
                exps[0][0]=1
                exps[0][exp_count]=self.max_per[exp_count]
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            lx, ly, lz = len(xs), len(ys), len(zs)
                            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                            pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)*(bound - half_grid_size) # [N, 3]
                            pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size
                            n = pts.shape[0]
                            pad_n = 128 - (n % 128)
                            if pad_n != 0:
                                pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                            density = self.forward(pts.to(tmp_grid.device),exps, bound,only_density=True)[:n].reshape(lx, ly, lz).detach()
                            tmp_grid[exp_count,xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density
        tmp_grid,_=tmp_grid.max(0)
        
        # smooth by maxpooling
        tmp_grid = F.pad(tmp_grid, (0, 1, 0, 1, 0, 1))
        tmp_grid = F.max_pool3d(tmp_grid.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=1).squeeze(0).squeeze(0)

        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
        self.mean_density = 1*torch.mean(self.density_grid).item()
        self.iter_density += 1

        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f} | [step counter] mean={self.mean_count}')
