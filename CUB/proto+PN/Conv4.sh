#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 600 \
    --stage 2 \
    --weight_decay 1e-3 \
    --num_part 15 \
    --alpha 100 \
    --gpu 0