@echo off
setlocal

rem Commands to run feature dumping for HuBERT-ECG model training
rem This script demonstrates the parameter structure for the dumping.py utility
rem More useful information available with: python dumping.py --help

rem Configuration
set "DF_PATH=./output/ptb_testV1.csv"
set "INPUT_DIR=./output/PTB_testV1"
set "OUTPUT_DIR=./output/PTB_testV1_outputV2"
@REM set "DF_PATH=data/label/ptbxl/ptbxl_all.csv"
@REM set "INPUT_DIR=D:/Kai/Dataset_Preprocessing/PTB-XL/PTB-XL"
@REM set "OUTPUT_DIR=output/PTB_XL_mfcc_only"
set "SUBSET_START=0.0"
set "SUBSET_END=1.0"
set "ITERATION=1"
set "FEATURE_MODE=mfcc_only"

rem Optional parameters
set "HUBERT_PATH=output/model_weights/hubert_ecg_small.pt"
set "SAMPLE_RATE=100"
set "BATCH_SIZE=3"
set "OUTPUT_LAYER=1"

rem Print header
echo === HuBERT-ECG Feature Dumping Pipeline ===
echo Starting execution at %date% %time%

@REM rem Dump morphological features (iteration 1)
@REM echo.
@REM echo [EXECUTING] Dumping morphological features (iteration 1)...
@REM python script/extract_features.py %ITERATION% %DF_PATH% %INPUT_DIR% %OUTPUT_DIR% ^
@REM     %SUBSET_START% %SUBSET_END% ^
@REM     --sample_rate=%SAMPLE_RATE% ^
@REM     --feature_mode=%FEATURE_MODE% ^
@REM     --overwrite ^
@REM     --save_csv

rem Dump latent features (iteration 2+)
echo.
echo [EXECUTING] Dumping latent features (iteration 2+)...
set "ITERATION=2"
python script/extract_features.py %ITERATION% %DF_PATH% %INPUT_DIR% %OUTPUT_DIR% ^
    %SUBSET_START% %SUBSET_END% ^
    --hubert_path=%HUBERT_PATH% ^
    --batch_size=%BATCH_SIZE% ^
    --output_layer=%OUTPUT_LAYER% ^
    --save_csv

@REM rem Dump time and frequency features only
@REM echo.
@REM echo [EXECUTING] Dumping time and frequency features...
@REM set "ITERATION=1"
@REM python dumping.py %ITERATION% %DF_PATH% %INPUT_DIR% %OUTPUT_DIR% ^
@REM     %SUBSET_START% %SUBSET_END% ^
@REM     --time_freq ^
@REM     --save_csv

@REM echo.
@REM echo === Feature dumping completed at %date% %time% ===
@REM endlocal