#!/bin/bash

RETRACTIONS=("steepest_manifold")
ITERS=(1 4 8 12)
BATCH_SIZES=(1024)
LRS=(0.01) # Add your desired learning rates here

for retraction in "${RETRACTIONS[@]}"; do
    for iters in "${ITERS[@]}"; do
        for lr in "${LRS[@]}"; do
            for batch in "${BATCH_SIZES[@]}"; do
                python train/train.py \
                    --dataset cifar10 \
                    --retraction $retraction \
                    --sharp_iters $iters \
                    --batch_size $batch \
                    --epochs 200 \
                    --variant learnable_ortho \
                    --lr $lr
            done
        done
    done
done

wait