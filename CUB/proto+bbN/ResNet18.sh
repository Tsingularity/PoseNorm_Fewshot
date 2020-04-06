#!/bin/bash
python train.py \
    --opt adam \
    --lr 1e-1 \
    --gamma 5e-1 \
    --epoch 160 \
    --stage 5 \
    --weight_decay 0 \
    --num_part 2 \
    --alpha 10 \
    --resnet \
    --gpu 0