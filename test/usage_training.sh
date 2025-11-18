#!/bin/bash

# Commands to run HuBERT-ECG model training
# This script demonstrates the parameter structure for the training utility
# More useful information available with: python train.py --help

# Configuration - Required Parameters
TRAIN_ITERATION=1
ECG_DIR_PATH="output/PTB_npy"
TRAIN_CSV="data/label/ptb/ptb_train_0.csv"
VAL_CSV="data/label/ptb/ptb_test_0.csv"
BATCH_SIZE=32
PATIENCE=10

# Pretraining-specific required parameters
MASK_TIME_PROB=0.065
ALPHA=1.0

# KMEANS_PATH="output/PTB_mfcc_only/kmeans_models.txt"
# TRAIN_FEATURES_PATH="output/PTB_mfcc_only/train_features"
# VAL_FEATURES_PATH="output/PTB_mfcc_only/val_features"
# VOCAB_SIZES=100

TRAIN_FEATURES_PATH="output/PTB_mfcc_only"
VAL_FEATURES_PATH="output/PTB_mfcc_only"
# KMEANS_PATH="output/PTB/sklearn-model/models-03.txt"
# VOCAB_SIZES=3

# Training schedule
TRAINING_STEPS=1000
VAL_INTERVAL=1
PATIENCE=500

# Optional parameters - Model configuration
LARGENESS="small"
ACCUMULATION_STEPS=1
DOWNSAMPLING_FACTOR=5

# Optional parameters - Optimization
LEARNING_RATE=5e-5
WEIGHT_DECAY_MULT=1.0
MODEL_DROPOUT_MULT=0.0

# Optional parameters - Regularization
DYNAMIC_REG="--dynamic_reg"
INTERVALS_FOR_PENALTY=4

# Optional parameters - Logging
WANDB_RUN_NAME="hubert_ecg_iteration1_experiment"

# Print header
echo "=== HuBERT-ECG Training Pipeline ==="
echo "Starting execution at $(date)"

LOAD_PATH="output/checkpoints/self-supervised/hubert_1_iteration_61.pt"

# Train iteration 1 (using morphological features)
echo ""
echo "[EXECUTING] Training iteration 1 with morphological features..."
python HuBert_ECG/pretrain.py "$TRAIN_ITERATION" "$ECG_DIR_PATH" "$TRAIN_CSV" "$VAL_CSV" "$BATCH_SIZE" "$PATIENCE" \
    "$MASK_TIME_PROB" "$ALPHA" "$KMEANS_PATH" \
    "$TRAIN_FEATURES_PATH" "$VAL_FEATURES_PATH" \
    "$VOCAB_SIZES" \
    --training_steps="$TRAINING_STEPS" \
    --val_interval="$VAL_INTERVAL" \
    --largeness="$LARGENESS" \
    --accumulation_steps="$ACCUMULATION_STEPS" \
    --lr="$LEARNING_RATE" \
    --weight_decay_mult="$WEIGHT_DECAY_MULT" \
    --model_dropout_mult="$MODEL_DROPOUT_MULT" \
    $DYNAMIC_REG \
    --intervals_for_penalty="$INTERVALS_FOR_PENALTY" \
    --wandb_run_name="$WANDB_RUN_NAME"