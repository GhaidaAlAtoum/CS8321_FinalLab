#!/bin/bash

if [ "$#" -lt 3 ]
then
  echo "No arguments supplied - Require 3 arguments:"
  echo "1. Run Directory"
  echo "2. Directory under layer_configs to get config from"
  echo "3. Kernel size for \"kernel_size_[2].yaml\" file"
  exit 1
fi

CONFIG_DIR="./layer_configs/${1}/${2}"

if [ ! -d $CONFIG_DIR ]; then
  echo "$CONFIG_DIR does not exist."
  exit 1
fi

CONFIG_FILE="$CONFIG_DIR/kernel_size_${3}.yaml"
if [ ! -f $CONFIG_FILE ]; then
    echo "$CONFIG_FILE File not found!"
    exit 1
fi

echo "CONFIG PATH $CONFIG_FILE"

set -e 

echo "-------------------------------------- 1 - Setup ENV"
source setup.sh
pip install /inputs/repo/code/BiasStudy/

echo "-------------------------------------- 3 - Run Training"
python3 ./train_small.py \
    --model_config_path $CONFIG_FILE \
    --logging_output_dir ./outputs/training-output-dataset
cp -R ./outputs/training-output-dataset /outputs
ls /outputs
ls -R /outputs/training-output-dataset