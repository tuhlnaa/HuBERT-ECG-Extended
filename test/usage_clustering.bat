@echo off
setlocal

rem Commands to run clustering for HuBERT-ECG model training
rem This script demonstrates the parameter structure for the clustering utility
rem More useful information available with: python clustering.py --help

rem Configuration
set "PATH_TO_DATASET_CSV_TRAIN=./data/label/ptb/ptb_train_0.csv"
set "PATH_TO_DATASET_CSV_VAL=./data/label/ptb/ptb_test_0.csv"
set "IN_DIR=D:/Kai/Dataset_Preprocessing/PTB/PTB_mfcc_only"

@REM set "PATH_TO_DATASET_CSV_TRAIN=data/label/ptbxl/ptbxl_all_train.csv"
@REM set "PATH_TO_DATASET_CSV_VAL=data/label/ptbxl/ptbxl_all_val.csv"
@REM set "IN_DIR=D:/Kai/Dataset_Preprocessing/PTB-XL/PTB-XL_mfcc_only"

set "TRAIN_ITERATION=1"
set "BATCH_SIZE=100"

rem Clustering parameters
set "N_CLUSTERS_START=3"
set "N_CLUSTERS_END=78"
set "STEP=5"

rem Optional parameters
set "MODEL_PATH=./kmeans_100_morphology_sse3.110000e+02.pkl"
set "LAYER=1"

rem Print header
echo === HuBERT-ECG Clustering Pipeline ===
echo Starting execution at %date% %time%

rem Cluster morphological features (iteration 1)
echo.
echo [EXECUTING] Clustering morphological features (iteration 1)...
python HuBert_ECG/kmeans_clustering.py %PATH_TO_DATASET_CSV_TRAIN% %PATH_TO_DATASET_CSV_VAL% ^
    %IN_DIR% %TRAIN_ITERATION% %BATCH_SIZE% ^
    --cluster ^
    --n_clusters_start=%N_CLUSTERS_START% ^
    --n_clusters_end=%N_CLUSTERS_END% ^
    --step=%STEP%

@REM rem Evaluate a clustering model (iteration 1)
@REM echo.
@REM echo [EXECUTING] Evaluating clustering model...
@REM set "TRAIN_ITERATION=1"
@REM python HuBert_ECG/kmeans_clustering.py %PATH_TO_DATASET_CSV_TRAIN% %IN_DIR% %TRAIN_ITERATION% %BATCH_SIZE% ^
@REM     --model_path=%MODEL_PATH%

@REM rem Resume clustering from a saved model (iteration 1)
@REM echo.
@REM echo [EXECUTING] Resuming clustering from saved model...
@REM python clustering.py %PATH_TO_DATASET_CSV_TRAIN% %IN_DIR% %TRAIN_ITERATION% %BATCH_SIZE% ^
@REM     --cluster ^
@REM     --n_clusters_start=100 ^
@REM     --n_clusters_end=%N_CLUSTERS_END% ^
@REM     --step=%STEP% ^
@REM     --model_path=%MODEL_PATH%

@REM rem Cluster latent features (iteration 2+)
@REM echo.
@REM echo [EXECUTING] Clustering latent features (iteration 2+)...
@REM set "TRAIN_ITERATION=2"
@REM python clustering.py %PATH_TO_DATASET_CSV_TRAIN% %IN_DIR% %TRAIN_ITERATION% %BATCH_SIZE% ^
@REM     --cluster ^
@REM     --n_clusters_start=%N_CLUSTERS_START% ^
@REM     --n_clusters_end=%N_CLUSTERS_END% ^
@REM     --step=%STEP% ^
@REM     --layer=%LAYER%

@REM rem Evaluate a clustering model (iteration 2+)
@REM echo.
@REM echo [EXECUTING] Evaluating clustering model (iteration 2+)...
@REM set "TRAIN_ITERATION=2"
@REM set "MODEL_PATH=./k_means_100_encoder_1_2_5e+05.pkl"
@REM python clustering.py %PATH_TO_DATASET_CSV_TRAIN% %IN_DIR% %TRAIN_ITERATION% %BATCH_SIZE% ^
@REM     --model_path=%MODEL_PATH% ^
@REM     --layer=%LAYER%

@REM echo.
@REM echo === Clustering pipeline completed at %date% %time% ===
@REM endlocal