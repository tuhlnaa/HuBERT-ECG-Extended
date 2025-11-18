@echo off
setlocal

rem Commands to run HuBERT-ECG model training
rem This script demonstrates the parameter structure for the training utility
rem More useful information available with: python train.py --help

rem Configuration - Required Parameters
set "TRAIN_ITERATION=1"
set "ECG_DIR_PATH=output/PTB_npy"
set "TRAIN_CSV=data/label/ptb/ptb_train_0.csv"
set "VAL_CSV=data/label/ptb/ptb_test_0.csv"
set "BATCH_SIZE=32"
set "PATIENCE=10"

rem Pretraining-specific required parameters
set "MASK_TIME_PROB=0.065"
set "ALPHA=1.0"

@REM set "KMEANS_PATH=output/PTB_mfcc_only/kmeans_models.txt"
@REM set "TRAIN_FEATURES_PATH=output/PTB_mfcc_only/train_features"
@REM set "VAL_FEATURES_PATH=output/PTB_mfcc_only/val_features"
@REM set "VOCAB_SIZES=100"

set "TRAIN_FEATURES_PATH=output/PTB_mfcc_only"
set "VAL_FEATURES_PATH=output/PTB_mfcc_only"
@REM set "KMEANS_PATH=output/PTB/sklearn-model/models-03.txt"
@REM set "VOCAB_SIZES=3"

rem Training schedule
set "TRAINING_STEPS=1000"
set "VAL_INTERVAL=1"
set "PATIENCE=500"

rem Optional parameters - Model configuration
set "LARGENESS=small"
set "ACCUMULATION_STEPS=1"
set "DOWNSAMPLING_FACTOR=5"

rem Optional parameters - Optimization
set "LEARNING_RATE=5e-5"
set "WEIGHT_DECAY_MULT=1.0"
set "MODEL_DROPOUT_MULT=0.0"

rem Optional parameters - Regularization
set "DYNAMIC_REG=--dynamic_reg"
set "INTERVALS_FOR_PENALTY=4"

rem Optional parameters - Logging
set "WANDB_RUN_NAME=hubert_ecg_iteration1_experiment"

rem Print header
echo === HuBERT-ECG Training Pipeline ===
echo Starting execution at %date% %time%

set "LOAD_PATH=output/checkpoints/self-supervised/hubert_1_iteration_61.pt"

echo %KMEANS_PATH%
echo %VOCAB_SIZES%
rem Train iteration 1 (using morphological features)
echo.
echo [EXECUTING] Training iteration 1 with morphological features...
python HuBert_ECG/pretrain.py %TRAIN_ITERATION% %ECG_DIR_PATH% %TRAIN_CSV% %VAL_CSV% %BATCH_SIZE% %PATIENCE% ^
    %MASK_TIME_PROB% %ALPHA% %KMEANS_PATH% ^
    %TRAIN_FEATURES_PATH% %VAL_FEATURES_PATH% ^
    %VOCAB_SIZES% ^
    --training_steps=%TRAINING_STEPS% ^
    --val_interval=%VAL_INTERVAL% ^
    --largeness=%LARGENESS% ^
    --accumulation_steps=%ACCUMULATION_STEPS% ^
    --lr=%LEARNING_RATE% ^
    --weight_decay_mult=%WEIGHT_DECAY_MULT% ^
    --model_dropout_mult=%MODEL_DROPOUT_MULT% ^
    %DYNAMIC_REG% ^
    --intervals_for_penalty=%INTERVALS_FOR_PENALTY% ^
    --wandb_run_name=%WANDB_RUN_NAME%

    @REM --resume_pretraining ^
    @REM --load_path=%LOAD_PATH% ^

@REM rem Train iteration 2 (resume pretraining with latent features)
@REM echo.
@REM echo [EXECUTING] Training iteration 2 - resuming pretraining...
@REM set "TRAIN_ITERATION=2"
@REM set "KMEANS_PATH=output/PTB_latent_iter1/kmeans_models.txt"
@REM set "TRAIN_FEATURES_PATH=output/PTB_latent_iter1/train_features"
@REM set "VAL_FEATURES_PATH=output/PTB_latent_iter1/val_features"
@REM set "LOAD_PATH=output/model_weights/hubert_ecg_small_iter1.pt"
@REM set "WANDB_RUN_NAME=hubert_ecg_iteration2_experiment"

@REM python train.py %TRAIN_ITERATION% %TRAIN_CSV% %VAL_CSV% %BATCH_SIZE% %PATIENCE% ^
@REM     %MASK_TIME_PROB% %ALPHA% %KMEANS_PATH% ^
@REM     %TRAIN_FEATURES_PATH% %VAL_FEATURES_PATH% ^
@REM     %VOCAB_SIZES% ^
@REM     --training_steps=%TRAINING_STEPS% ^
@REM     --val_interval=%VAL_INTERVAL% ^
@REM     --resume_pretraining ^
@REM     --load_path=%LOAD_PATH% ^
@REM     --accumulation_steps=%ACCUMULATION_STEPS% ^
@REM     --lr=%LEARNING_RATE% ^
@REM     --weight_decay_mult=%WEIGHT_DECAY_MULT% ^
@REM     --wandb_run_name=%WANDB_RUN_NAME%

@REM rem Alternative: Train using epochs instead of training steps
@REM echo.
@REM echo [EXECUTING] Alternative training with epochs...
@REM set "TRAIN_ITERATION=1"
@REM set "EPOCHS=100"

@REM python train.py %TRAIN_ITERATION% %TRAIN_CSV% %VAL_CSV% %BATCH_SIZE% %PATIENCE% ^
@REM     %MASK_TIME_PROB% %ALPHA% %KMEANS_PATH% ^
@REM     %TRAIN_FEATURES_PATH% %VAL_FEATURES_PATH% ^
@REM     %VOCAB_SIZES% ^
@REM     --epochs=%EPOCHS% ^
@REM     --largeness=%LARGENESS% ^
@REM     --lr=%LEARNING_RATE% ^
@REM     --wandb_run_name=hubert_ecg_epochs_experiment

@REM echo.
@REM echo === Training completed at %date% %time% ===
@REM endlocal