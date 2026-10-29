# Starts Live Translator (no console window).
# Run:  powershell -ExecutionPolicy Bypass -File "D:\jason\jasen\AI interviewer assistance\live translator\start.ps1"

$ErrorActionPreference = "Stop"
$appDir = $PSScriptRoot
$venvPython = Join-Path $appDir ".venv\Scripts\python.exe"
$venvPythonw = Join-Path $appDir ".venv\Scripts\pythonw.exe"

# First run on a new computer: create the environment and install packages.
if (-not (Test-Path $venvPythonw)) {
    Write-Host "First run: installing packages. This takes a few minutes..."
    python -m venv (Join-Path $appDir ".venv")
    & $venvPython -m pip install -r (Join-Path $appDir "requirements.txt")
}

# Don't open a second copy if it is already running.
$running = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*app.py*" }
if ($running) {
    Write-Host "Live Translator is already running."
    exit 0
}

Start-Process -FilePath $venvPythonw -ArgumentList "app.py" -WorkingDirectory $appDir
Write-Host "Live Translator started. It is ready in about 20 seconds."
