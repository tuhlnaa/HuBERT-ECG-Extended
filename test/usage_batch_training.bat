@echo off
setlocal

set "SEARCH_PATH=output/PTB/sklearn-model"
set "VOCAB_SIZES=3"

for %%f in ("%SEARCH_PATH%/*.txt") do (
    echo Processing: %%f
    set KMEANS_PATH="%SEARCH_PATH%/%%f"
    call test/usage_training.bat
    set /a VOCAB_SIZES+=5
)

echo All files processed!