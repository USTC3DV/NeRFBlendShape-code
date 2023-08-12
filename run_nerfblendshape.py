import torch
from nerf.provider import NeRFDataset
from nerf.utils import *
import argparse
import shutil

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_idpath', type=str)
    parser.add_argument('--exp_idpath', type=str)
    parser.add_argument('--pose_idpath', type=str)
    parser.add_argument('--intr_idpath', type=str)
    parser.add_argument('--use_checkpoint', type=str,default="latest")
    parser.add_argument('--mode', type=str,default="train")
    parser.add_argument('--to_mem', action='store_true')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=1024)
    parser.add_argument('--use_lpips', action="store_true")
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--bound', type=float, default=0.06, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--basis_num',type=int, default=46, help="basis")
    parser.add_argument('--add_mean',action='store_true', help="add one for exp coef ")
    parser.add_argument('--no_pru',action='store_true', help="do not update density grid")    
    parser.add_argument('--num_layers',type=int,default=4 )
    parser.add_argument('--hidden_dim',type=int,default=64 )
    parser.add_argument('--geo_feat_dim',type=int,default=64 )
    parser.add_argument('--num_layers_col',type=int,default=1 )
    parser.add_argument('--hidden_dim_col',type=int,default=64 )
    parser.add_argument('--test_start',type=int,default=-500)

    opt = parser.parse_args()

    print(opt)
    from nerf.network_tcnn import NeRFNetwork

    seed_everything(opt.seed)

    if opt.add_mean==True:
        model_basis_num=opt.basis_num+1
    else:
        model_basis_num=opt.basis_num

    dataset = NeRFDataset(opt.img_idpath,opt.exp_idpath,opt.pose_idpath,opt.intr_idpath, type=opt.mode, add_mean=opt.add_mean,basis_num=opt.basis_num,to_mem=opt.to_mem,test_start=opt.test_start)

    model = NeRFNetwork(
        encoding="hashgrid",
        num_layers=opt.num_layers, hidden_dim=opt.hidden_dim, geo_feat_dim=opt.geo_feat_dim, num_layers_color=opt.num_layers_col, hidden_dim_color=opt.hidden_dim_col, 
        cuda_ray=opt.cuda_ray,basis_num=model_basis_num,no_pru=opt.no_pru,add_mean=opt.add_mean,mode=opt.mode,
    )
    print(model)

    if opt.mode=="train" :
    
        valid_dataset = NeRFDataset(opt.img_idpath,opt.exp_idpath,opt.pose_idpath,opt.intr_idpath, type='valid', downscale=1, add_mean=opt.add_mean,basis_num=opt.basis_num,to_mem=opt.to_mem,test_start=opt.test_start)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
        
        criterion = torch.nn.L1Loss()

        params_list=[]
        params_list.append({'name': 'encoding1', 'params': model.encoder.embeddings})
        params_list.append({'name': 'encoding2', 'params': model.encoder.embeddings_mean})
        params_list.append({'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()) , 'weight_decay': 1e-6})


        optimizer = lambda model: torch.optim.Adam(params_list, lr=1e-3, betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,100], gamma=0.33)


        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.use_checkpoint, eval_interval=1,bc_img=dataset.bc_img)
        shutil.copy("run_train.sh",os.path.join(opt.workspace,"run_train.sh"))
        shutil.copy("run_nerfblendshape.py",os.path.join(opt.workspace,"run_nerfblendshape.py"))
        shutil.copy(os.path.join("nerf","network_tcnn.py"),os.path.join(opt.workspace,"network_tcnn.py"))
        shutil.copy(os.path.join("nerf","provider.py"),os.path.join(opt.workspace,"provider.py"))
        shutil.copy(os.path.join("nerf","utils.py"),os.path.join(opt.workspace,"utils.py"))
        shutil.copy(os.path.join("nerf","renderer.py"),os.path.join(opt.workspace,"renderer.py"))
       
        trainer.train(train_loader, valid_loader, 200,max_path=os.path.join(os.path.dirname(opt.img_idpath),f"max_{opt.basis_num}.txt"),min_path=os.path.join(os.path.dirname(opt.img_idpath),f"min_{opt.basis_num}.txt"))

    elif opt.mode=="normal_test":
        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.use_checkpoint,bc_img=dataset.bc_img)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
        trainer.test(test_loader,max_path=os.path.join(os.path.dirname(opt.img_idpath),f"max_{opt.basis_num}.txt"),min_path=os.path.join(os.path.dirname(opt.img_idpath),f"min_{opt.basis_num}.txt"))
        exit()
    else:
        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.use_checkpoint,bc_img=dataset.bc_img)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
        trainer.test(test_loader,max_path=os.path.join(os.path.dirname(opt.img_idpath),f"max_{opt.basis_num}.txt"),min_path=os.path.join(os.path.dirname(opt.img_idpath),f"min_{opt.basis_num}.txt"))
        exit()