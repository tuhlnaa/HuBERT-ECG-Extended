@echo off
setlocal

rem Commands to run feature dumping for HuBERT-ECG model training
rem This script demonstrates the parameter structure for the dumping.py utility
rem More useful information available with: python dumping.py --help

rem Configuration
set "TRAIN_ITERATION=1"
@REM set "DATAFRAME_PATH=ptb_testV1.csv"
@REM set "IN_DIR=./output/PTB_testV1"
@REM set "DEST_DIR=./output/PTB_testV1_outputV2"
set "DATAFRAME_PATH=reproducibility/ptb/ptb_test_0.csv"
set "IN_DIR=output/PTB"
set "DEST_DIR=output/PTB_testV1_outputV2"
set "START_PERC=0.0"
set "END_PERC=1.0"

rem Optional parameters
set "HUBERT_PATH=./output/model_weights/hubert_ecg_small.pt"
set "SAMP_RATE=100"
set "BATCH_SIZE=3"
set "OUTPUT_LAYER=1"

rem Print header
echo === HuBERT-ECG Feature Dumping Pipeline ===
echo Starting execution at %date% %time%

rem Dump morphological features (iteration 1)
echo.
echo [EXECUTING] Dumping morphological features (iteration 1)...
python script/extract_features.py %TRAIN_ITERATION% %DATAFRAME_PATH% %IN_DIR% %DEST_DIR% ^
    %START_PERC% %END_PERC% ^
    --samp_rate=%SAMP_RATE% ^
    --mfcc_only ^
    --overwrite ^
    --save_csv_for_dumped_features

@REM rem Dump latent features (iteration 2+)
@REM echo.
@REM echo [EXECUTING] Dumping latent features (iteration 2+)...
@REM set "TRAIN_ITERATION=2"
@REM python script/extract_features.py %TRAIN_ITERATION% %DATAFRAME_PATH% %IN_DIR% %DEST_DIR% ^
@REM     %START_PERC% %END_PERC% ^
@REM     --hubert_path=%HUBERT_PATH% ^
@REM     --batch_size=%BATCH_SIZE% ^
@REM     --output_layer=%OUTPUT_LAYER% ^
@REM     --save_csv_for_dumped_features

@REM rem Dump time and frequency features only
@REM echo.
@REM echo [EXECUTING] Dumping time and frequency features...
@REM set "TRAIN_ITERATION=1"
@REM python dumping.py %TRAIN_ITERATION% %DATAFRAME_PATH% %IN_DIR% %DEST_DIR% ^
@REM     %START_PERC% %END_PERC% ^
@REM     --time_freq ^
@REM     --save_csv_for_dumped_features

@REM echo.
@REM echo === Feature dumping completed at %date% %time% ===
@REM endlocal