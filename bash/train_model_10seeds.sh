#!/bin/bash

model=$1

for year in {2007..2017}; do

    for seed in {0..9}; do

        python src/models/train_model.py \
            output/year/out_${year}_${model}_${seed} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --normalize \
            --seed $seed \
            --log_comet

    done


done
