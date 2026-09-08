# PowerShell Runner for Vision Decision Agent
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting Deep Learning Vision Decision Agent" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

if (-not (Test-Path "venv")) {
    Write-Host "[*] Creating virtual environment 'venv'..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[-] Python was not found in PATH. Please install Python 3.10+ or activate your Conda environment." -ForegroundColor Red
        Exit 1
    }
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "[+] Launching Deployment Server on http://127.0.0.1:8000 ..." -ForegroundColor Green
python app.py
