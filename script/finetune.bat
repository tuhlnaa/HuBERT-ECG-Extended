@echo off
setlocal

rem Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
rem Standard train/val/test splits are used
rem Although previous commands actually reproduce the results we obtain on PTB-XL All, we consider this dataset only as an example
rem More useful information available with:   python finetune.py --help

rem Configuration
set "EXPERIMENT_NAME=finetune"
set "PATH_TO_DATASET_CSV_TRAIN=./reproducibility/ptbxl/ptbxl_all_train.csv"
set "PATH_TO_DATASET_CSV_VAL=./reproducibility/ptbxl/ptbxl_all_val.csv"
set "LOAD_PATH=./output/model_weights/hubert_ecg_small.pt"
set "ECG_DIR=./output/PTB"

set "TRAIN_ITERATION=3"
set "VOCAB_SIZE=13"
set "PATIENCE=8"
set "BATCH_SIZE=64"
set "TARGET_METRIC=auroc"

rem Print header
echo === HuBERT-ECG Fine-tuning Pipeline ===
echo Starting execution at %date% %time%

rem Fine-tune SMALL model
echo.
echo [EXECUTING] Fine-tuning SMALL model...
python HuBert_ECG/finetune.py %TRAIN_ITERATION% %PATH_TO_DATASET_CSV_TRAIN% ^
    %PATH_TO_DATASET_CSV_VAL% %ECG_DIR% %VOCAB_SIZE% %PATIENCE% %BATCH_SIZE% %TARGET_METRIC% ^
    --load_path=%LOAD_PATH% ^
    --experiment_name %EXPERIMENT_NAME%^
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights ^
    --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 ^
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptbxl_all


echo.
echo === Fine-tuning completed at %date% %time% ===
endlocal