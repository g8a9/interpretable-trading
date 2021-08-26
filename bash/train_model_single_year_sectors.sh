#!/bin/bash

for year in {2007..2017}; do

    for model in GNB SVC KNN LG; do

        python src/models/train_model.py \
            output/sectors/out_${year}_${model} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --normalize \
            --log_comet \
            --use_sectors

    done

    for model in RFC; do

        python src/models/train_model.py \
            output/sectors/out_${year}_${model} \
            ${model} \
            ${year} \
            --training_type year \
            --do_grid_search \
            --log_comet \
            --use_sectors

    done

    python src/models/train_model.py \
        output/sectors/out_${year}_L3 \
        L3 \
        ${year} \
        --training_type year \
        --do_grid_search \
        --log_comet \
        --use_sectors

    python src/models/train_model.py \
        output/sectors/out_${year}_L3-LVL1 \
        L3 \
        ${year} \
        --training_type year \
        --do_grid_search \
        --log_comet \
        --rule_sets_modifier level1 \
        --use_sectors

done