#!/bin/bash

for year in {2007..2017}; do

    for model in GNB SVC KNN LG; do

        python src/models/train_model.py \
            output/year/out_${year}_${model} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --normalize \
            --parallel \
            --log_comet

    done

    for model in RFC; do

        python src/models/train_model.py \
            output/year/out_${year}_${model} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --log_comet

    done

done