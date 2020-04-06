#!/bin/bash
python train_base.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 100 \
    --stage 2 \
    --weight_decay 1e-3 \
    --num_part 15 \
    --batch_size 64 \
    --resnet \
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
    --resnet \
    --load_path model_ResNet18-base.pth \
    --gpu 0