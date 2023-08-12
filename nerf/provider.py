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

import json
import math

def nerf_matrix_to_ngp(pose, scale=0.33):
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class NeRFDataset(Dataset):
    def __init__(self, img_idpath,exp_idpath,pose_idpath,intr_idpath,basis_num, type='train', downscale=1, test_basis_inter=1,add_mean=False,to_mem=False,rot_cycle=100, test_start=None):
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
        self.to_mem = to_mem
        self.test_basis_inter=test_basis_inter
        max_path=os.path.join(os.path.dirname(self.img_idpath),f"max_{self.basis_num}.txt")
        min_path=os.path.join(os.path.dirname(self.img_idpath),f"min_{self.basis_num}.txt")
        self.load_max(max_path,min_path)
        self.test_start=test_start

        
        
        # load nerf-compatible format data.
        with open(img_idpath, 'r') as f:
            transform_img = json.load(f)

        with open(exp_idpath, 'r') as f:
            transform_exp = json.load(f)

        with open(pose_idpath, 'r') as f:
            transform_pose = json.load(f)

        with open(intr_idpath, 'r') as f:
            transform_intr = json.load(f)


        self.bc_img_path=os.path.join(self.root_img,"bc.jpg")
        self.bc_img=cv2.cvtColor(cv2.imread(self.bc_img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.bc_img = self.bc_img.astype(np.float32) / 255

        # load image size
        self.H = int(transform_img['h']) // downscale
        self.W = int(transform_img['w']) // downscale

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
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
                if test_start!=None:
                    out_frames=frames[0:self.test_start]
                else:
                    out_frames=frames[0:-500]
                return out_frames

            frames_img = get_train_frames(frames_img)
            frames_exp = get_train_frames(frames_exp)
            frames_pose = get_train_frames(frames_pose)
            assert len(frames_img)==len(frames_exp) and len(frames_img)==len(frames_pose)
        elif type=="valid":

            if test_start!=None:
                frames_img=frames_img[self.test_start::70]
                frames_exp=frames_exp[self.test_start::70]
                frames_pose=frames_pose[self.test_start::70]
                print(len(frames_img))
            else:
                frames_img = frames_img[-500::70]
                frames_exp = frames_exp[-500::70]
                frames_pose = frames_pose[-500::70]
            assert len(frames_img)==len(frames_exp) and len(frames_img)==len(frames_pose)
        elif type=="normal_test":
            if test_start!=None:
                frames_img=frames_img[self.test_start:]
                frames_exp=frames_exp[self.test_start:]
                frames_pose=frames_pose[self.test_start:]
            else:
                frames_img = frames_img[-500:]
                frames_exp = frames_exp[-500:]
                frames_pose = frames_pose[-500:]
        else:
            frames_pose = frames_pose[:]

        self.num_frames=len(frames_exp)


        self.poses = []
        self.images_list = []
        self.exps = []
        self.parsings = []
        self.lms = []


        for f_id in range(self.num_frames):
            if (self.type=="train") or (self.type=="valid") :
                f_path = os.path.join(self.root_img,"head_imgs", str(frames_img[f_id]['img_id'])+".jpg")
                parsing_path = os.path.join(self.root_img,"parsing", str(frames_img[f_id]['img_id'])+".png")
                lms_path= os.path.join(self.root_img,"ori_imgs", str(frames_img[f_id]['img_id'])+".lms")
            if  (self.type=="normal_test"):
                f_path = os.path.join(self.root_exp,"ori_imgs", str(frames_exp[f_id]['img_id'])+".jpg")
                parsing_path = os.path.join(self.root_exp,"parsing", str(frames_exp[f_id]['img_id'])+".png")
                lms_path= os.path.join(self.root_exp,"ori_imgs", str(frames_exp[f_id]['img_id'])+".lms")
            if not os.path.exists(f_path):
                continue
            
            pose = np.array(frames_pose[f_id]['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose)

            if add_mean == True:
                frames_exp[f_id]['exp_ori'].insert(0,1.0)
            exp = np.array(frames_exp[f_id]['exp_ori'], dtype=np.float32)
            lms=np.loadtxt(lms_path)

            self.poses.append(pose)
            self.images_list.append(f_path)
            self.exps.append(exp)
            self.parsings.append(parsing_path)
            self.lms.append(lms)

        # print(self.poses)
        self.poses = np.stack(self.poses, axis=0).astype(np.float32)
        self.exps = np.stack(self.exps, axis=0).astype(np.float32)
        self.lms = np.stack(self.lms, axis=0).astype(np.int32)#(N,478,2)
        
        if self.type=="normal_test":
            self.rects,self.rects_mouth,self.rects_eyes=self.get_rect_test(self.lms,self.W,self.H)
        else:
            self.rects,self.rects_mouth,self.rects_eyes=self.get_rect(self.lms,self.W,self.H)


        if self.to_mem==True and (self.type=="train" or self.type=="valid"):
            self.mem_images=[]
            self.mem_masks=[]
            for index in range(self.num_frames):
                f_path=self.images_list[index]
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.uint8) # [H, W, 3/4]
                self.mem_images.append(image)

                parsing_path=self.parsings[index]
                seg = cv2.imread(parsing_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if seg.shape[-1] == 3: 
                    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
                else:
                    seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
                seg = cv2.resize(seg, (self.W, self.H), interpolation=cv2.INTER_AREA)
                mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
                self.mem_masks.append(mask)
            return
        
        if self.type=="normal_test":
            self.poses_l1=[]

            vec=np.array([0,0,0.3493212163448334])
            for i in range(self.num_frames):
                tmp_pose=np.identity(4,dtype=np.float32)
                r1 = Rotation.from_euler('y', 15+(-30)*((i%rot_cycle)/rot_cycle), degrees=True)
                tmp_pose[:3,:3]=r1.as_matrix()
                trans=tmp_pose[:3,:3]@vec
                # print(trans)
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
        }
        results['H'] = str(self.H)
        results['W'] = str(self.W)

        if self.to_mem==False and (self.type=="train" or self.type=="valid"):
            results["rects"]=self.rects[index]
            results["rects_mouth"]=self.rects_mouth[index]
            f_path=self.images_list[index]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) 
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255 
            results['image'] = image

            parsing_path=self.parsings[index]
            seg = cv2.imread(parsing_path, cv2.IMREAD_UNCHANGED) 
            if seg.shape[-1] == 3: 
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            else:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
            seg = cv2.resize(seg, (self.W, self.H), interpolation=cv2.INTER_AREA)
            mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
            results['mask'] = mask
            return results
        elif self.to_mem==True and (self.type=="train" or self.type=="valid"):
            results["rects"]=self.rects[index]
            results["rects_mouth"]=self.rects_mouth[index]
            image=self.mem_images[index]
            results['image'] = image.astype(np.float32) / 255

            mask=self.mem_masks[index]
            results['mask'] = mask
            return results
        elif self.type=="normal_test":
            results['pose_l1']=self.poses_l1[index]

            results["rects"]=self.rects[index]
            results["rects_mouth"]=self.rects_mouth[index]
            results["rects_eyes"]=self.rects_eyes[index]
            f_path=self.images_list[index]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) 
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255 
            results['image'] = image
            return results
        else:
            return results

    def get_rect(self,lms,W,H):
        max_w=np.max(lms[:,:,0],axis=1)
        min_w=np.min(lms[:,:,0],axis=1)
        max_h=np.max(lms[:,:,1],axis=1)
        min_h=np.min(lms[:,:,1],axis=1)
        w0=(max_w+min_w)/2
        h0=(max_h+min_h)/2
        radius=1.2*np.sqrt((max_w-min_w)**2+(max_h-min_h)**2)/2   #1.1
        w1=w0-radius
        h1=h0-radius*1.2
        w2=w0+radius
        h2=h0+radius*.8
        rect=np.stack([w1,h1,w2,h2],axis=1).astype(np.int32)
        rect[:,[0,2]]=np.clip(rect[:,[0,2]],1,self.W-1)
        rect[:,[1,3]]=np.clip(rect[:,[1,3]],1,self.H-1)

        mouth_l, mouth_r, mouth_t, mouth_b = lms[:,57, 0], lms[:,287, 0], lms[:,0, 1], lms[:,17, 1]
        eye_l, eye_r, eye_t, eye_b = lms[:,130, 0], lms[:,359, 0], np.minimum(lms[:,27, 1], lms[:,257, 1]), np.maximum(lms[:,23, 1], lms[:,253, 1])

        rect_mouth=np.stack([mouth_l,mouth_t,mouth_r,mouth_b],axis=1).astype(np.int32)
        rect_eye=np.stack([eye_l,eye_t,eye_r,eye_b],axis=1).astype(np.int32)
        

        rect_mouth[:,[0,2]]=np.clip(rect_mouth[:,[0,2]],1,self.W-1)
        rect_mouth[:,[1,3]]=np.clip(rect_mouth[:,[1,3]],1,self.H-1)


        rect_eye[:,[0,2]]=np.clip(rect_eye[:,[0,2]],1,self.W-1)
        rect_eye[:,[1,3]]=np.clip(rect_eye[:,[1,3]],1,self.H-1)


        rect_sp=rect_mouth
        rect_sp2=rect_eye

        return rect,rect_sp,rect_sp2

    def get_rect_test(self,lms,W,H,scale=1.2):
        max_w=np.max(lms[:,:,0],axis=1)
        min_w=np.min(lms[:,:,0],axis=1)
        max_h=np.max(lms[:,:,1],axis=1)
        min_h=np.min(lms[:,:,1],axis=1)
        w0=(max_w+min_w)/2
        h0=(max_h+min_h)/2
        radius=scale*np.sqrt((max_w-min_w)**2+(max_h-min_h)**2)/2  
        w1=w0-radius
        h1=h0-radius*1.2
        w2=w0+radius
        h2=h0+radius*.8
        rect=np.stack([w1,h1,w2,h2],axis=1).astype(np.int32)
        rect[:,[0,2]]=np.clip(rect[:,[0,2]],1,self.W-1)
        rect[:,[1,3]]=np.clip(rect[:,[1,3]],1,self.H-1)

        mouth_l, mouth_r, mouth_t, mouth_b = lms[:,57, 0], lms[:,287, 0], lms[:,0, 1], lms[:,17, 1]
        eye_l, eye_r, eye_t, eye_b = lms[:,130, 0], lms[:,359, 0], np.minimum(lms[:,27, 1], lms[:,257, 1]), np.maximum(lms[:,23, 1], lms[:,253, 1])

        radius_mouth=0.5*np.sqrt((mouth_r-mouth_l)**2+(mouth_b-mouth_t)**2)/2
        radius_eye=0.3*np.sqrt((eye_r-eye_l)**2+(eye_b-eye_t)**2)/2
        mouth_l=mouth_l-radius_mouth
        mouth_t=mouth_t-radius_mouth
        mouth_r=mouth_r+radius_mouth
        mouth_b=mouth_b+radius_mouth
        eye_l=eye_l-radius_eye
        eye_t=eye_t-radius_eye
        eye_r=eye_r+radius_eye
        eye_b=eye_b+radius_eye

        rect_mouth=np.stack([mouth_l,mouth_t,mouth_r,mouth_b],axis=1).astype(np.int32)
        rect_eye=np.stack([eye_l,eye_t,eye_r,eye_b],axis=1).astype(np.int32)
        rect_mouth[:,[0,2]]=np.clip(rect_mouth[:,[0,2]],1,self.W-1)
        rect_mouth[:,[1,3]]=np.clip(rect_mouth[:,[1,3]],1,self.H-1)
        rect_eye[:,[0,2]]=np.clip(rect_eye[:,[0,2]],1,self.W-1)
        rect_eye[:,[1,3]]=np.clip(rect_eye[:,[1,3]],1,self.H-1)
        rect_sp=rect_mouth
        rect_sp2=rect_eye

        return rect,rect_sp,rect_sp2

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