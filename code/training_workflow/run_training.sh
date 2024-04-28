#!/bin/bash

python3 ./train_small.py \
    --model_config_path /notebooks/code/training_workflow/model_8_layers_3_kernel_size_no_flat.yaml \
    --logging_output_dir /notebooks/code/training_workflow/outputs
