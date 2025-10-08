@REM [Windows] - PowerShell
Get-Content requirements.txt | ForEach-Object {
    if ($_ -match "\S" -and -not $_.StartsWith("#")) {
        Write-Host "Installing $_..." -ForegroundColor Cyan
        pip install $_
    }
}