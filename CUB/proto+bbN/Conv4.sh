#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-2 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 2 \
    --weight_decay 5e-4 \
    --num_part 2 \
    --alpha 10 \
    --gpu 0