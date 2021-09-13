#!/bin/bash

for model in A2C DDPG; do
    for year in {2007..2017}; do
        python src/models/train_FinRL.py ${year} ${model}
    done
done
