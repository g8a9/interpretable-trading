#!/bin/bash

for year in {2007..2017}; do

    python src/models/train_model.py \
        output/year/out_${year}_L3 \
        L3 \
        ${year} \
        --training_type year \
        --do_grid_search \
        --log_comet

    python src/models/train_model.py \
        output/year/out_${year}_L3-LVL1 \
        L3 \
        ${year} \
        --training_type year \
        --do_grid_search \
        --log_comet \
        --rule_sets_modifier level1

done