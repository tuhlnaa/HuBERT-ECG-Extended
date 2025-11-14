@echo off
setlocal

set "SEARCH_PATH=output/PTB/sklearn-model"
set "VOCAB_SIZES=8"

for %%f in ("%SEARCH_PATH%/*.txt") do (
    echo Processing: %%f
    set KMEANS_PATH=%%f
    set /a VOCAB_SIZES+=5
    call test/usage_training.bat
)

echo All files processed!