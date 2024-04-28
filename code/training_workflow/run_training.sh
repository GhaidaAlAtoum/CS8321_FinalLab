#!/bin/bash

python3 ./train_small.py -m "./sample.yaml" -f "/notebooks/data/fairface" -e 4 -b 64 -s 3 -c 1 --overwrite_sample_number true --number_samples 100
