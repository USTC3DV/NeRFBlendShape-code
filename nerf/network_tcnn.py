import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from .renderer import NeRFRenderer
from .gridencoder import ExpHashEncoder
from torch.cuda.amp import autocast as autocast

import numpy as np

torch.set_printoptions(profile="full")

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 cuda_ray=False,
                 basis_num=None,
                 no_pru=False,
                 level_dim=4,
                 num_levels=16,
                 add_mean=False,
                 mode="train",
                 ):
        super().__init__(cuda_ray,no_pru)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.basis_num = basis_num

        self.num_levels=num_levels
        self.add_mean=add_mean
        self.mode=mode

        
        self.encoder = ExpHashEncoder(input_dim=3, num_levels=self.num_levels, level_dim=level_dim, base_resolution=16, log2_hashmap_size=14,basis_num=basis_num,desired_resolution=1024)

        sigma_net_m=[]
        sigma_net_begin=self.num_levels*level_dim

        self.sigma_net = tcnn.Network(
            n_input_dims=sigma_net_begin,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )


        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color ,
            },
        )



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
    
    def forward(self, x, exps, bound,d=None,index=None,xyzstorays=None,only_density=False):
        if xyzstorays == None:
            xyzstorays=torch.zeros([x.shape[0]],device="cuda")

        exps=torch.minimum(exps,self.max_per.expand([exps.shape[0],self.max_per.shape[0]]))
        exps=torch.maximum(exps,self.min_per.expand([exps.shape[0],self.min_per.shape[0]]))
        exps_code=exps[:,:self.basis_num]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder.embed(x,exps_code,xyzstorays)

        h = self.sigma_net(x)
        sigma=F.softplus(h[..., 0])

        if only_density==True:
            return sigma



        geo_feat = h[..., 1:]
        d = d.view(-1, 3)
        
        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        color = torch.sigmoid(h)
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)
        
        return sigma, color,exps_code

    def density(self, x,exps, bound):
        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = (x + bound) / (2 * bound) # to [0, 1]
        x=x.to(dtype=torch.float32)
        x = self.encoder.embed(x,exps)
        
        h = self.sigma_net(x)
        sigma = F.relu(h[..., 0])

        sigma = sigma.view(*prefix)

        return sigma