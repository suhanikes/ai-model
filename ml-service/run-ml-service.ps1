# Start the Garment Recolor ML service (FastAPI) on port 8000.
# Run this in a separate terminal before using the lasso tool.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# If .env doesn't exist but SAM checkpoint is in this folder, set it
if (-not (Test-Path ".env")) {
    $samPath = Join-Path $PSScriptRoot "sam_vit_h_4b8939.pth"
    if (Test-Path $samPath) {
        $env:SAM_CHECKPOINT_PATH = $samPath
        Write-Host "Using SAM checkpoint: $samPath" -ForegroundColor Green
    }
}

Write-Host "Starting ML service at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Keep this window open. Use Ctrl+C to stop." -ForegroundColor Gray
python main.py
