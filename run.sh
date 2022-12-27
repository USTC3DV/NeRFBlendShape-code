
NAME=id4

CUDA_VISIBLE_DEVICES=7 python run_nerfblendshape.py\
    --img_idpath ./dataset/$NAME/transforms_train.json \
    --exp_idpath ./dataset/$NAME/transforms_train.json \
    --pose_idpath ./dataset/$NAME/transforms_train.json \
    --intr_idpath ./dataset/$NAME/transforms_train.json \
    --use_checkpoint ./dataset/$NAME/pretrained.pth.tar \
    --workspace trial_$NAME \
    --fp16 --tcnn  --cuda_ray --basis_num 46   --add_mean  --use_lpips  --mode normal_test
