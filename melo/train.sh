#!/bin/bash

#source data/melotts/bin/activate

# python -m unidic download
# Set the relative path to the config file
RELATIVE_CONFIG_PATH="data/example/config.json"

# Get the directory of the script
SCRIPT_DIR="$(dirname "$0")"

# Construct the full path to the config file
CONFIG="$SCRIPT_DIR/$RELATIVE_CONFIG_PATH"

# Extract the model name from the config path
MODEL_NAME="$(basename "$CONFIG" .json)"

# Loop for training
# while true; do
    # Run the training script
python train.py -c "$CONFIG" -m "$MODEL_NAME"
# python train_wandb.py -c "$CONFIG" -m "$MODEL_NAME"
    # Kill any running Python processes
    # pkill -f python

    # Wait for 30 seconds before restarting
#     sleep 30
# done
