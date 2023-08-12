# Reconstructing Personalized Semantic Facial NeRF Models From Monocular Video

Official PyTorch implementation of the paper "Reconstructing Personalized Semantic Facial NeRF Models From Monocular Video". This repository contains code, data and released pretrained model.

**|[Project Page](https://ustc3dv.github.io/NeRFBlendShape/)|[Paper](https://arxiv.org/abs/2210.06108)|**

![teaser](fig/teaser.jpg)
We present a semantic model for human head defined with neural radiance field. In this model, multi-level voxel field is adopted as basis with corresponding expression coefficients, which enables strong representation ability on the aspect of rendering and fast training.

## Pipeline

We track the RGB sequence and get expression coefficients, poses and intrinsics. Then we use the tracked expression coefficients to combine multiple multi-level hash tables to get a hash table corresponding to a specific expression. Then the sampled point is queried in hash table to get voxel features, we use an MLP to interpret the voxel features as RGB and density. We fix the expression coefficients and optimize the hash tables and MLP to get our head model.

![pipeline](fig/pipeline.jpg)



## Setup

This code has been tested on RTX 3090. 

Install requirements.txt:

```
pip install -r requirements.txt
```

Install [PyTorch](https://pytorch.org/get-started/locally/) according to your OS and Compute Platform.

Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Train
Download our [preprocessed dataset](https://drive.google.com/drive/folders/1OiUvo7vHekVpy67Nuxnh3EuJQo7hlSq1?usp=sharing) and unzip it to `./dataset` or organize your own data in the same folder structure:

```
dataset
├── id1
│   ├── head_imgs #segmented head images 
│   ├── ori_imgs #extracted video frames and landmarks
│   ├── parsing #semantic segmentation
│   ├── transforms.json #intrinsics, poses and tracked expression coefficients
│   └── bc.jpg #background image
├── id2
│   ├── ...
...
```

Then you could run `run_train.sh` to train a NeRFBlendShape model

```
bash run_train.sh id1 -500 0
```
This means you want to run training with the dataset of `id1`, the last 500 frames are used for inference and GPU_ID is `0`

`run.sh` will first compute the range of expression coefficients with `get_max.py`, this will save `max_46.txt` and `min_46.txt` to `dataset/$NAME`. Then it will run `run_nerfblendshape.py` to start training. You could also adjust the following options of `run_nerfblendshape.py`:

`--workspace`: the workspace folder of your experiment. Checkpoints, inference results and script backup will all be placed here.

`--basis_num` the dimention of your expression coefficient. In our data, this value is 46.

`--to_mem` Choose this if you want `nerf/provider` to preload the whole dataset to memory before training and change needed batch of data from `.cpu()` to `.cuda()` in every step, otherwise, in every step `nerf/provider` will load needed batch of data from disk to memory first, then change the data from  `cpu()` to `cuda()`. 

`--use_lpips` whether you want to use perceptual loss to improve details.

`--add_mean` add '1.0' to your expression coefficient to represent neutral expression.

## Inference
When training is finished, the checkpoint will be saved in `the_name_of_workspace/checkpoints`.

to run inference with a given checkpoint: 

```
bash run_infer.sh id1 -500 0 the_path_of_your_checkpoint
```
This means you want to run inference of `the_path_of_your_checkpoint` with the dataset of `id1`, the last 500 frames are used for inference and GPU_ID is `0`. `run_infer.sh` also runs `run_nerfblendshape.py` but with `--mode normal_test`

You could use `generate_video.py` to convert rendered images into a video sequence.

You could also use our [pretrained model](https://drive.google.com/drive/folders/1OiUvo7vHekVpy67Nuxnh3EuJQo7hlSq1?usp=sharing) to run inference imediately.

## Citation

If you find our paper useful for your work please cite:

```
@article{Gao2022nerfblendshape,
         author = {Xuan Gao and Chenglai Zhong and Jun Xiang and Yang Hong and Yudong Guo and Juyong Zhang}, 
         title = {Reconstructing Personalized Semantic Facial NeRF Models From Monocular Video}, 
         journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)}, 
         volume = {41}, 
         number = {6}, 
         year = {2022}, 
         doi = {10.1145/3550454.3555501} }
```

## Acknowledgement

This code is developed on [torch-ngp](https://github.com/ashawkey/torch-ngp) code base. 

```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
```