#!/bin/bash
python train_base.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 200 \
    --stage 2 \
    --weight_decay 5e-4 \
    --num_part 15 \
    --batch_size 64 \
    --gpu 0

python finetune_cub.py \
    --opt adam \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 40 \
    --stage 1 \
    --weight_decay 0 \
    --num_part 15 \
    --batch_size 16 \
    --load_path model_Conv4-base.pth \
    --gpu 0