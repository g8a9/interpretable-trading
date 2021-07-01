#!/bin/bash

training_type=$1

for year in {2007..2017}; do

    for model in RFC SVC KNN; do

        python src/models/train_model.py \
            out_$year_$model \
            --training_type $training_type

done

#