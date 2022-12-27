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

import trimesh
import mcubes
from rich.console import Console
# from torch_ema import ExponentialMovingAverage


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


def get_rays(c2w, intrinsics, H, W, N_rays=-1):
    device = c2w.device
    rays_o = c2w[..., :3, 3] # [B, 3]
    prefix = c2w.shape[:-2]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    if N_rays > 0 :
        N_rays = min(N_rays, H*W)
        select_hs = torch.randint(0, H, size=[N_rays], device=device)
        select_ws = torch.randint(0, W, size=[N_rays], device=device)
        select_inds = select_hs * W + select_ws
        select_inds = select_inds.expand([*prefix, N_rays])
        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
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
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 bc_img=None,
                 ):
        
        self.name = name
        self.conf = conf
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.bc_img=torch.from_numpy(bc_img).to(self.device)
        
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
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

        # if ema_decay is not None:
        #     self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        # else:
        self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
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

    def test_step(self, data):  
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...
        exps =data["exp"]
        exp_ori= data["exp_ori"]
        C=3

        B = poses.shape[0]
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, -1)
        bg_color=self.bc_img.expand([B,H,W,C])
        # bg_color=torch.ones_like(self.bc_img.expand([B,H,W,C]))
        bg_color = torch.gather(bg_color.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]
        outputs = self.model.render(rays_o, rays_d,exps,exp_ori=exp_ori,index=[0], staged=True,bg_color=bg_color, **self.conf)
        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth

    def test(self, loader, save_path=None,max_path=None,min_path=None):
        self.model.load_max(max_path,min_path)

        if save_path is None:
            if self.conf["use_checkpoint"]!="latest":
                save_path = os.path.join(self.workspace, os.path.basename(self.conf["use_checkpoint"]))
            else:
                save_path = os.path.join(self.workspace, 'res2')

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


    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return
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

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
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
        
        # strange bug: keyerror 'lr_lambdas'
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
