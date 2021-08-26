#!/bin/bash

year=$1

for model in GNB SVC KNN LG; do

    python src/models/train_model.py \
        output/year/out_cumulative-${year}_${model} \
        ${model} \
        ${year} \
        --training_type cumulative \
        --do_grid_search \
        --normalize \
        --parallel \
        --log_comet

done

for model in RFC; do

    python src/models/train_model.py \
        output/year/out_cumulative-${year}_${model} \
        ${model} \
        ${year} \
        --training_type cumulative \
        --do_grid_search \
        --log_comet

done