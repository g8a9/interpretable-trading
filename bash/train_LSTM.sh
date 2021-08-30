#!/bin/bash

model=LSTM

for year in {2007..2017}; do

    for seed in {0..9}; do

        python src/models/train_model.py \
            output/year/out_${year}_${model}_${seed} \
            ${model} \
            ${year} \
            --training_type year \
            --normalize \
            --seed $seed \
            --log_comet \
            --seq_length 10 \
            --batch_size 1024 \
            --max_epochs 100 \
            --lr 2e-5 \
            --early_stop 10 \
            --gpus 1

    done


done
