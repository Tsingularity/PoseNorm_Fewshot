#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 200 \
    --stage 2 \
    --weight_decay 5e-3 \
    --num_part 15 \
    --resnet \
    --gpu 0