#!/bin/bash

for year in {2007..2017}; do

    for model in MLP; do

        for seed in {0..9}; do

            python src/models/train_model.py \
                out_$year_$model \
                --training_type year \
                --do_grid_search \
                --normalize \
                --seed $seed

        done

    done

done