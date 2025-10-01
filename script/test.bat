@echo off
setlocal

rem Configuration
set "DATASET_CSV=./reproducibility/ptb/ptb_test_0.csv"
set "ECG_DIR=E:/Kai_2/CODE_Repository/HuBERT-ECG-Extended/output/PTB"
set "BATCH_SIZE=64"
set "MODEL_PATH=D:/Kai/huggingface-model/HuBERT-ECG-SSL-Pretrained/hubert_ecg_small.pt"
set "DOWNSAMPLING_FACTOR=5"
set "LABEL_START_INDEX=4"
set "SAVE_ID=ptb_all_small"
set "TTA_AGGREGATION=max"

rem Print header
echo === ECG Model Testing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the testing script
echo [EXECUTING] Starting ECG model testing...
python src/test.py "%DATASET_CSV%" "%ECG_DIR%" %BATCH_SIZE% ^
    "%MODEL_PATH%" ^
    --downsampling_factor=%DOWNSAMPLING_FACTOR% ^
    --label_start_index=%LABEL_START_INDEX% ^
    --tta ^
    --save_id=%SAVE_ID% ^
    --tta_aggregation=%TTA_AGGREGATION%

echo === Testing completed at %date% %time% ===
endlocal