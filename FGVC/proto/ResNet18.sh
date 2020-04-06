#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 300 \
    --stage 2 \
    --weight_decay 1e-3 \
    --resnet \
    --val_epoch 40 \
    --gpu 0