#!/bin/bash
python train_stage_1.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 25 \
    --stage 3 \
    --weight_decay 1e-3 \
    --resnet \
    --num_part 15 \
    --batch_size 64 \
    --gpu 0

python train_stage_2.py \
    --opt adam \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 200 \
    --stage 1 \
    --weight_decay 0 \
    --resnet \
    --num_part 15 \
    --load_path model_ResNet18-stage_1.pth \
    --gpu 0