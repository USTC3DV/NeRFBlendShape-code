from operator import index
import os
import time
import glob
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json
import math


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class NeRFDataset(Dataset):
    def __init__(self, img_idpath,exp_idpath,pose_idpath,intr_idpath,basis_num, type='train', downscale=1, n_test=10,test_basis_inter=1,add_mean=False,to_mem=False,rot_cycle=100):
        super().__init__()
        # path: the json file path.
        self.img_idpath=img_idpath
        self.exp_idpath=exp_idpath
        self.pose_idpath=pose_idpath
        self.intr_idpath=intr_idpath
        self.basis_num=basis_num
        self.add_mean=add_mean
        self.root_img = os.path.dirname(img_idpath)
        self.root_exp = os.path.dirname(exp_idpath)
        self.type = type
        self.downscale = downscale
        self.test_basis_inter=test_basis_inter
        max_path=os.path.join(os.path.dirname(self.img_idpath),f"max_{self.basis_num}.txt")
        min_path=os.path.join(os.path.dirname(self.img_idpath),f"min_{self.basis_num}.txt")
        self.load_max(max_path,min_path)

        
        
        # load nerf-compatible format data.
        with open(img_idpath, 'r') as f:
            transform_img = json.load(f)

        with open(exp_idpath, 'r') as f:
            transform_exp = json.load(f)

        with open(pose_idpath, 'r') as f:
            transform_pose = json.load(f)

        with open(intr_idpath, 'r') as f:
            transform_intr = json.load(f)

        self.H = int(transform_img['h']) // downscale
        self.W = int(transform_img['w']) // downscale


        self.bc_img=np.ones([self.H,self.W,3])

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float16)
        self.intrinsic[0, 0] = transform_intr['fx'] / downscale
        self.intrinsic[1, 1] = transform_intr['fy'] / downscale
        self.intrinsic[0, 2] = transform_intr['cx'] / downscale
        self.intrinsic[1, 2] = transform_intr['cy'] / downscale

        frames_img = transform_img["frames"]
        frames_img = sorted(frames_img, key=lambda d: d['img_id'])
        frames_exp = transform_exp["frames"]
        frames_exp = sorted(frames_exp, key=lambda d: d['img_id'])
        frames_pose = transform_pose["frames"]
        frames_pose = sorted(frames_pose, key=lambda d: d['img_id'])

        if type=="train":
            
            def get_train_frames(frames):
                out_frames=frames[0:-500]
                return out_frames

            frames_img = get_train_frames(frames_img)
            frames_exp = get_train_frames(frames_exp)
            frames_pose = get_train_frames(frames_pose)
            assert len(frames_img)==len(frames_exp) and len(frames_img)==len(frames_pose)
        elif type=="valid":
            frames_img = frames_img[-500::50]
            frames_exp = frames_exp[-500::50]
            frames_pose = frames_pose[-500::50]
            assert len(frames_img)==len(frames_exp) and len(frames_img)==len(frames_pose)
        elif type=="normal_test":
            frames_img = frames_img[-500:]
            frames_exp = frames_exp[-500:]
            frames_pose = frames_pose[-500:]
        else:
            frames_pose = frames_pose[:]




        self.poses = []
        self.images_list = []
        self.exps = []
        self.parsings = []
        self.lms = []
        self.exps_ori = []
        self.num_frames=len(frames_exp)

        for f_id in range(self.num_frames):
            if  (self.type=="normal_test"):
                pass

            # print(f_id)
            
            pose = np.array(frames_pose[f_id]['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose)

            if add_mean == True:
                frames_exp[f_id]['exp_ori'].insert(0,1.0)
            exp = np.array(frames_exp[f_id]['exp_ori'], dtype=np.float32)
            exp_ori=np.array(frames_exp[f_id]['exp_ori'], dtype=np.float32)

            self.poses.append(pose)
            self.exps.append(exp)
            self.exps_ori.append(exp_ori)

        # print(self.poses)
        self.poses = np.stack(self.poses, axis=0).astype(np.float16)
        self.exps = np.stack(self.exps, axis=0).astype(np.float16)
        self.exps_ori = np.stack(self.exps_ori,axis=0).astype(np.float16)

        
        if self.type=="normal_test":
            self.poses_l1=[]
            self.poses_l2=[]
            self.poses_r1=[]
            self.poses_r2=[]

            vec=np.array([0,0,0.3493212163448334])
            for i in range(self.num_frames):
                tmp_pose=np.identity(4,dtype=np.float32)
                r1 = Rotation.from_euler('y', 15+(-30)*((i%rot_cycle)/rot_cycle), degrees=True)
                tmp_pose[:3,:3]=r1.as_matrix()
                trans=tmp_pose[:3,:3]@vec
                tmp_pose[0:3,3]=trans
                self.poses_l1.append(nerf_matrix_to_ngp(tmp_pose) )
            return



        

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
            'exp':self.exps[index],
            'exp_ori':self.exps_ori[index],
        }
        results['H'] = str(self.H)
        results['W'] = str(self.W)

        if self.type=="normal_test":
            results['pose_l1']=self.poses_l1[index]
            return results
        else:

            return results

    def load_max(self,max_path,min_path):
        self.max_per=torch.from_numpy(np.loadtxt(max_path)).to(dtype=torch.float16).cuda()
        if self.add_mean:
            self.max_per=torch.cat([torch.ones([1]).cuda(),self.max_per],dim=0)

        print("load max_per successfully:max_per is :")
        print(self.max_per)

        self.min_per=torch.from_numpy(np.loadtxt(min_path)).to(dtype=torch.float16).cuda()
        if self.add_mean:
            self.min_per=torch.cat([torch.ones([1]).cuda(),self.min_per],dim=0)
        print("load min_per successfully:min_per is :")
        print(self.min_per)