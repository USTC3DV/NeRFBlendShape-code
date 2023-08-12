import os
import glob
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# import trimesh
# import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import lpips
import ipdb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def normalize_for_percep(input, mod_n = 64):
    h, w = input.shape[2:4]
    return (input*2.-1.).permute(0,3,1,2)

def lift(x, y, z, intrinsics):
    device = x.device
    
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    sk = intrinsics[..., 0, 1].unsqueeze(-1)

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)

def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.tensor([10.],device="cuda"))


def get_rays(c2w, intrinsics, H, W, N_rays=-1,rect=None,sample_rate=0.9,use_lpips=False,win_size=32):

    device = c2w.device
    rays_o = c2w[..., :3, 3] 
    prefix = c2w.shape[:-2]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij') # for torch < 1.10, should remove indexing='ij'
    i = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    if use_lpips==True:
        if rect==None:
            select_h=torch.randint(0,H-win_size,size=[*prefix],device=device)
            select_w=torch.randint(0,W-win_size,size=[*prefix],device=device)
            select_hs=select_h.unsqueeze(1)+(torch.range(0,win_size*win_size-1,device=device)//win_size).reshape([1,win_size*win_size])
            select_ws=select_w.unsqueeze(1)+(torch.range(0,win_size-1,device=device).expand([1,win_size,win_size]).reshape(-1))
            select_inds = select_hs * W + select_ws
            select_inds = select_inds.expand([*prefix, win_size*win_size]).to(dtype=torch.long)
            i = torch.gather(i, -1, select_inds)
            j = torch.gather(j, -1, select_inds)
        else:
            select_h=rect[:,1]+torch.rand(*prefix,device=device)*(rect[:,3]-rect[:,1]-win_size-1)
            select_w=rect[:,0]+torch.rand(*prefix,device=device)*(rect[:,2]-rect[:,0]-win_size-1)
            if torch.sum(select_h>H-win_size-2)==0 and torch.sum(select_w>W-win_size-2)==0 and torch.sum(select_h<win_size+2)==0 and torch.sum(select_w<win_size+2)==0:

                select_hs=select_h.unsqueeze(1)+(torch.range(0,win_size*win_size-1,device=device)//win_size).reshape([1,win_size*win_size])
                select_ws=select_w.unsqueeze(1)+(torch.range(0,win_size-1,device=device).expand([1,win_size,win_size]).reshape(-1))
                select_inds = select_hs * W + select_ws
                select_inds = select_inds.expand([*prefix, win_size*win_size]).to(dtype=torch.long)
                i = torch.gather(i, -1, select_inds)
                j = torch.gather(j, -1, select_inds)

            else:
                select_h=torch.randint(0,H-win_size,size=[*prefix],device=device)
                select_w=torch.randint(0,W-win_size,size=[*prefix],device=device)
                select_hs=select_h.unsqueeze(1)+(torch.range(0,win_size*win_size-1,device=device)//win_size).reshape([1,win_size*win_size])
                select_ws=select_w.unsqueeze(1)+(torch.range(0,win_size-1,device=device).expand([1,win_size,win_size]).reshape(-1))
                select_inds = select_hs * W + select_ws
                select_inds = select_inds.expand([*prefix, win_size*win_size]).to(dtype=torch.long)
                i = torch.gather(i, -1, select_inds)
                j = torch.gather(j, -1, select_inds)
    else:
        if N_rays > 0 and rect==None:
            N_rays = min(N_rays, H*W)
            select_hs = torch.randint(0, H, size=[N_rays], device=device)
            select_ws = torch.randint(0, W, size=[N_rays], device=device)
            select_inds = select_hs * W + select_ws
            select_inds = select_inds.expand([*prefix, N_rays])
            i = torch.gather(i, -1, select_inds)
            j = torch.gather(j, -1, select_inds)
        elif N_rays > 0 and rect!=None:

            N_rays1=int(min(N_rays*sample_rate,H*W))
            N_rays2=N_rays-N_rays1
            select_hs1 =( rect[:,1].unsqueeze(1)+(rect[:,3]-rect[:,1]).unsqueeze(1)*torch.rand(size=[rect.shape[0],N_rays1], device=device)).to(dtype=torch.int64)
            select_ws1 =( rect[:,0].unsqueeze(1)+(rect[:,2]-rect[:,0]).unsqueeze(1)*torch.rand(size=[rect.shape[0],N_rays1], device=device)).to(dtype=torch.int64)

            select_inds1 = select_hs1 * W + select_ws1
            select_inds1 = select_inds1.expand([*prefix, N_rays1])

            i1 = torch.gather(i, -1, select_inds1)
            j1 = torch.gather(j, -1, select_inds1)
 

            select_hs2 = torch.randint(0, H, size=[N_rays2], device=device)
            select_ws2 = torch.randint(0, W, size=[N_rays2], device=device)
            select_inds2 = select_hs2 * W + select_ws2
            select_inds2 = select_inds2.expand([*prefix, N_rays2])
            i2 = torch.gather(i, -1, select_inds2)
            j2 = torch.gather(j, -1, select_inds2)

            i=torch.cat((i1,i2),dim=1)
            j=torch.cat((j1,j2),dim=1)

            select_inds=torch.cat((select_inds1,select_inds2),dim=1)


        else:
            select_inds = torch.arange(H*W, device=device).expand([*prefix, H*W])

    pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsics)
    pixel_points_cam = pixel_points_cam.transpose(-1, -2)

    world_coords = torch.bmm(c2w, pixel_points_cam).transpose(-1, -2)[..., :3]
    
    rays_d = world_coords - rays_o[..., None, :]
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = rays_o[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds



class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 conf, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 lr_scheduler=None, # scheduler
                 local_rank=0, # which GPU am I
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 workspace='workspace', # workspace to save logs & ckpts
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 bc_img=None,
                 ):
        
        self.name = name
        self.conf = conf
        self.mute = mute
        self.local_rank = local_rank
        self.workspace = workspace
        self.fp16 = fp16
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.bc_img=torch.from_numpy(bc_img).to(self.device)

        self.percep_module=lpips.LPIPS(net="vgg")
        self.percep_module.to(device="cuda").eval()
        
        model.to(self.device)
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "valid_loss": [],
            "results": [], 
            "checkpoints": [], 
            "best_result": None,
            }


        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)

    ### ------------------------------	

    def train_step(self, data,lpips_flag=True,mask_loss_flag=False,if_mouth=True):
        images = data["image"] 
        poses = data["pose"] 
        intrinsics = data["intrinsic"] 
        exps =data["exp"]
        mask=data["mask"]
        if if_mouth==True and lpips_flag==True:
            rect=data["rects_mouth"]
        else:
            rect=data["rects"]
        index=data["index"]

        # sample rays 
        B, H, W, C = images.shape

        if lpips_flag==True:
            rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, self.conf['num_rays'],rect,use_lpips=self.conf['use_lpips'])
        else:
            rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, self.conf['num_rays'])

        images = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]
        mask = torch.gather(mask.reshape(B, -1, 1), 1, torch.stack(1*[inds], -1))#[B,N,1]

        bg_color=self.bc_img.expand([B,H,W,C])
        bg_color = torch.gather(bg_color.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d,exps=exps,index=index, staged=False, bg_color=bg_color, **self.conf)
    
        pred_rgb = outputs['rgb']
        pred_A = outputs['A']

        loss_exp=torch.tensor([0],device="cuda")

        if (self.conf['use_lpips']==True)and(lpips_flag==True):
            loss_vgg=1e-1*torch.mean(self.percep_module.forward(normalize_for_percep(pred_rgb.reshape(B,32,32,3)),normalize_for_percep(gt_rgb.reshape(B,32,32,3))  )  )
        else:
            loss_vgg=torch.tensor([0],device="cuda")

        if mask_loss_flag==True:
            mask_loss=1*self.criterion(pred_A, mask.to(dtype=torch.float32))
        else:
            mask_loss=torch.tensor([0],device="cuda")
        
        loss_reg=1*torch.mean(torch.abs(self.model.encoder.embeddings))
        if (self.conf['use_lpips']==True)and(lpips_flag==True):
            loss_rgb=0.1*self.criterion(pred_rgb, gt_rgb)
        else:
            loss_rgb=self.criterion(pred_rgb, gt_rgb)
        loss = loss_rgb+mask_loss+1*loss_exp+loss_vgg +loss_reg

        psnr=mse2psnr(torch.mean((pred_rgb-gt_rgb)**2))

        return pred_rgb, gt_rgb, loss,loss_exp,psnr,loss_vgg , loss_reg

    def eval_step(self, data):
        images = data["image"] 
        poses = data["pose"] 
        intrinsics = data["intrinsic"] 
        exps =data["exp"]
        mask=data["mask"]

        B, H, W, C = images.shape
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, -1)

        bg_color=torch.ones_like(self.bc_img.expand([B,H,W,C]))
        bg_color = torch.gather(bg_color.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) 
        
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
            
        outputs = self.model.render(rays_o, rays_d,exps, index=[0], staged=True, bg_color=bg_color, **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)


        return pred_rgb, pred_depth, gt_rgb

    def test_step(self, data):  
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...
        exps =data["exp"]
        C=3

        B = poses.shape[0]
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, -1)
        bg_color=self.bc_img.expand([B,H,W,C])
        bg_color=torch.ones_like(self.bc_img.expand([B,H,W,C]))
        bg_color = torch.gather(bg_color.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]
        outputs = self.model.render(rays_o, rays_d,exps, index=[0], staged=True,bg_color=bg_color, **self.conf)
        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth

    def train(self, train_loader, valid_loader, max_epochs,max_path,min_path):

        self.model.load_max(max_path,min_path)
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        self.train_time=0
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch


            self.train_one_epoch(train_loader)
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def test(self, loader, save_path=None,max_path=None,min_path=None):
        self.model.load_max(max_path,min_path)

        if save_path is None:
            if self.conf["use_checkpoint"]!="latest":
                save_path = os.path.join(self.workspace, os.path.basename(self.conf["use_checkpoint"]))
            else:
                save_path = os.path.join(self.workspace, 'res')

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                
                data = self.prepare_data(data)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)     
                path = os.path.join(save_path, f'{i:04d}.png')
                preds=preds.reshape([-1,int(data["H"][0]),int(data["W"][0]),3])
                print(path)
                cv2.imwrite(path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))  

                if self.conf["mode"]=="normal_test":
                    data["pose"]=data["pose_l1"]  
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth = self.test_step(data)     
                    path = os.path.join(save_path, f'{i:04d}_nvs.png')
                    preds=preds.reshape([-1,int(data["H"][0]),int(data["W"][0]),3])
                    cv2.imwrite(path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))  

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")
    

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        self.model.train()
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            if self.global_step in [1000,2000]:#pruning first in the beginning.
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state(self.conf['bound'])
                    print("pruning early")
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            if self.global_step <=4000:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss,loss_exp,psnr,loss_vgg,loss_reg = self.train_step(data,lpips_flag=False,mask_loss_flag=True)
            elif self.global_step >4000 and self.global_step<=20000:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss,loss_exp,psnr,loss_vgg,loss_reg = self.train_step(data,lpips_flag=False,mask_loss_flag=False)
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds1, truths1, loss1,loss_exp1,psnr1,loss_vgg1,loss_reg1  = self.train_step(data,lpips_flag=True,mask_loss_flag=False,if_mouth=(self.global_step%2==0))
                    preds2, truths2, loss2,loss_exp2,psnr2,loss_vgg2,loss_reg2  = self.train_step(data,lpips_flag=False,mask_loss_flag=False,if_mouth=(self.global_step%2==0))

                    preds,truths,loss,loss_exp,psnr,loss_vgg,loss_reg=(preds1+preds2)/2,(truths1+truths2)/2,(loss1+loss2)/2,(loss_exp1+loss_exp2)/2,(psnr1+psnr2)/2,(loss_vgg1+loss_vgg2)/2,(loss_reg1+loss_reg2)/2

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                pbar.set_description(f"loss={loss.item():.4f} loss_exp={loss_exp.item():.4f} loss_vgg={loss_vgg.item():.4f} loss_reg={loss_reg.item():.4f}  PSNR:{psnr.item():.4f}")
                pbar.update(loader.batch_size)

            if self.global_step %10000==0 :
                if self.workspace is not None:
                    self.save_checkpoint(full=True)

        # update grid
        if self.model.cuda_ray and self.epoch>=4:#only do the pruning operation per epoch
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state(self.conf['bound'])


        if self.local_rank == 0:
            pbar.close()

        self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
        self.log(f"loss={loss.item():.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                data = self.prepare_data(data)


                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths = self.eval_step(data)

                if self.local_rank == 0:

                    save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}.png')
                    save_path_gt = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_gt.png')

                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_gt, cv2.cvtColor((truths[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.update(loader.batch_size)
                    self.log(f"==> Saving validation image to {save_path}")

        if self.local_rank == 0:
            pbar.close()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False):

        state = {
            'epoch': self.epoch,
            'global_step':self.global_step,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
        

        state['model'] = self.model.state_dict()

        file_path = f"{self.ckpt_path}/{self.name}_{int(self.global_step):04d}.pth.tar"

        torch.save(state, file_path)

            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return
        # print(self.ckpt_path)
        # print(checkpoint)
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        self.epoch = checkpoint_dict['epoch']
        if "global_step" in checkpoint_dict.keys():
            self.global_step=checkpoint_dict['global_step']

        if self.model.cuda_ray:
            self.model.mean_count = checkpoint_dict['mean_count']
            self.model.mean_density = checkpoint_dict['mean_density']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")
        
        if 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler, use default.")
