#!/bin/bash

model=MLP

for year in {2007..2017}; do

    for seed in {0..9}; do

        python src/models/train_model.py \
            output/sectors/out_${year}_${model}_${seed} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --use_sectors \
            --normalize \
            --seed $seed \
            --log_comet

    done


done
