DATA_PATH=./dataset

NAME=$1
TEST_START=$2
GPUID=$3


python get_max.py --path $DATA_PATH/$NAME/transforms.json --test_start $TEST_START --num 46

CUDA_VISIBLE_DEVICES=$GPUID python run_nerfblendshape.py\
    --img_idpath $DATA_PATH/$NAME/transforms.json \
    --exp_idpath $DATA_PATH/$NAME/transforms.json \
    --pose_idpath $DATA_PATH/$NAME/transforms.json \
    --intr_idpath $DATA_PATH/$NAME/transforms.json \
    --workspace trial_nerfblendshape_$NAME\
    --test_start $TEST_START\
    --fp16 --tcnn  --cuda_ray --basis_num 46   --add_mean  --use_lpips   --mode train --to_mem