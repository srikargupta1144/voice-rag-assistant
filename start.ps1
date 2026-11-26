# -----------------------------------------------------------------------------
# File: start.ps1
#
# Description:
#   Starts the Voice RAG Assistant on this Windows laptop with the local
#   Python 3.12 virtual environment (.venv). Checks that Ollama is running and
#   has the llama3.2:3b model, then serves the API and web page on
#   http://127.0.0.1:8000. Run it from the project folder:
#       powershell -ExecutionPolicy Bypass -File .\start.ps1
#   Optional: .\start.ps1 -Ask "What is the late fee on the Kova card?"
#   answers one question in the terminal instead of starting the server.
#
# Dependencies:
#   .venv (created with: python -m venv .venv, then pip install -e .)
#   Ollama for Windows with the llama3.2:3b model pulled
#   Models under models\ (python -m scripts.download_models) and the index
#   under data\index (vra ingest)
#
# Created by: Srikar Gupta
# Created date: 2026-11-26
# Updated by: Srikar Gupta
# Updated date: 2026-11-26
#
# Change log:
#   2026-11-26  Srikar Gupta  First version for local runs on Python 3.12.
#   2026-11-26  Srikar Gupta  Clear message when port 8000 is already in use.
#   2026-11-26  Srikar Gupta  Opens a new Edge window at a fresh address when the
#                             server is ready, so a saved old page is never shown.
# -----------------------------------------------------------------------------

param(
    [string]$Ask = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Settings needed on Windows: torch and CTranslate2 each ship an OpenMP runtime,
# and all models are already downloaded, so Hugging Face stays offline.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$vra = Join-Path $PSScriptRoot ".venv\Scripts\vra.exe"
if (-not (Test-Path $python)) {
    Write-Host "The .venv folder is missing. Create it first (see SETUP_WINDOWS.md)."
    exit 1
}

# Ollama: start it if it is not answering, then make sure the model is present.
$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$ollamaUp = $false
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
    $ollamaUp = $true
} catch {
    $ollamaUp = $false
}
if (-not $ollamaUp) {
    Write-Host "Starting Ollama..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
$models = & $ollama list
if (-not ($models -match "llama3.2:3b")) {
    Write-Host "Pulling llama3.2:3b (about 2 GB, one time only)..."
    & $ollama pull llama3.2:3b
}

if ($Ask -ne "") {
    & $vra ask $Ask
    exit $LASTEXITCODE
}

$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port 8000 is already in use (process $($busy.OwningProcess | Select-Object -First 1))."
    Write-Host "The assistant may already be running: open http://127.0.0.1:8000."
    Write-Host "To restart it, close that window or run: taskkill /PID <number above> /T /F"
    exit 1
}

# Open a new Edge window once the server answers. The address ends with a fresh number,
# so the browser cannot show a saved copy of an older page; it must fetch the current one.
$stamp = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$pageUrl = "http://127.0.0.1:8000/?v=$stamp"
Start-Job -Name "vra-open-browser" -ArgumentList $pageUrl -ScriptBlock {
    param($url)
    for ($i = 0; $i -lt 180; $i++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -UseBasicParsing -TimeoutSec 3 | Out-Null
            Start-Process "msedge.exe" -ArgumentList "--new-window", $url
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
} | Out-Null

Write-Host "Starting the API. Models warm up first (about 30 seconds)."
Write-Host "A new Edge window opens by itself when the server is ready:"
Write-Host "  $pageUrl"
Write-Host "Use that window, not the browser panel inside VS Code. Press Ctrl+C to stop."
& $vra serve
