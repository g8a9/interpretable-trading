#!/bin/bash

for year in {2007..2017}; do

    for model in SVC KNN LG; do

        python src/models/train_model.py \
            out_$year_$model \
            $model \
            $year \
            --training_type year \
            --do_grid_search \
            --normalize

    done

    for model in RFC; do

        python src/models/train_model.py \
            out_$year_$model \
            $model \
            $year \
            --training_type year \
            --do_grid_search

    done

done