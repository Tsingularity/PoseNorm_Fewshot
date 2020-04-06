#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 500 \
    --stage 2 \
    --weight_decay 1e-3 \
    --num_part 5 \
    --alpha 50 \
    --val_epoch 40 \
    --batch_size 400 \
    --gpu 0