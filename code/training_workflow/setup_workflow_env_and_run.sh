#!/bin/bash

if [ "$#" -lt 3 ]
then
  echo "No arguments supplied - Require 3 arguments:"
  echo "1. Directory under layer_configs to get config from"
  echo "2. Kernel size for \"kernel_size_[2].yaml\" file"
  echo "3. with_flat or no_flat"
  exit 1
fi

# if [[ "${$3}" != @(no_flat|with_flat) ]]; then
#     echo "3rd Argument needs to be no_flat or with_flat"
# fi

if [ "$3" != "no_flat" -a "$1" != "with_flat" ];then
    echo "3rd Argument needs to be \"no_flat\" or \"with_flat\""
    exit 1
fi

CONFIG_DIR="./layer_configs/${1}/${3}"

if [ ! -d $CONFIG_DIR ]; then
  echo "$CONFIG_DIR does not exist."
  exit 1
fi

CONFIG_FILE="$CONFIG_DIR/kernel_size_${2}.yaml"
if [ ! -f $CONFIG_FILE ]; then
    echo "$CONFIG_FILE File not found!"
    exit 1
fi

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