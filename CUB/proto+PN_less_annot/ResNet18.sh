#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 300 \
    --stage 2 \
    --weight_decay 5e-3 \
    --num_part 15 \
    --alpha 200 \
    --percent 20 \
    --batch_size 7 \
    --resnet \
    --gpu 0