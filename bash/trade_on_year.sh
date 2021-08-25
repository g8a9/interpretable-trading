#!/bin/bash

year=$1

mkdir output/trading/out_${year}

python src/trading/run_trading.py \
    --input_dir output/year \
    --year ${year} \
    --output_dir output/trading/out_${year} \
    --log_comet