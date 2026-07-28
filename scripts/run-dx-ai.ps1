<#
.SYNOPSIS
    Generic DirectX AI application runner with debug and capture support.

.DESCRIPTION
    This script provides a generic wrapper for running any DirectX AI application
    with proper debug layer configuration, GPU capture hooks, and DRED support.
    Works with GGUF models (llama.cpp), ONNX models (DirectML), or custom
    DirectX 12 compute applications.

.PARAMETER AppPath
    Path to the AI application executable

.PARAMETER ModelPath
    Path to the model file (GGUF, ONNX, or other)

.PARAMETER Prompt
    Prompt to send to the AI model

.PARAMETER EnableDebug
    Enable D3D12 and DirectML debug layers

.PARAMETER EnableDRED
    Enable Device Removed Extended Data (DRED)

.PARAMETER CapturePath
    Directory for GPU capture files

.PARAMETER Capture
    Capture a GPU frame using PIX or VS Graphics Debugger

.PARAMETER NPixels
    Number of tokens to predict

.PARAMETER ExtraArgs
    Additional arguments to pass to the application

.EXAMPLE
    .\run-dx-ai.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf" -Prompt "Hi"
    Run llama-cli with DirectX backend and debug layers.

.EXAMPLE
    .\run-dx-ai.ps1 -AppPath ".\my_onnx_app.exe" -ModelPath "model.onnx" -Prompt "Hello" -EnableDRED
    Run an ONNX-based DirectML app with DRED enabled.

.NOTES
    This script sets up the proper environment for DirectX debugging:
    - DX12_ENABLE_DEBUG_LAYER=1
    - DX12_FORCE_DEBUG_LAYER=1
    - DX12_ENABLE_DRED=1 (if -EnableDRED)
    - DX12_DRED_OUTPUT=<capture_path>/dred_log.txt
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$AppPath,

    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [string]$Prompt = "Hi",

    [switch]$EnableDebug,

    [switch]$EnableDRED,

    [string]$CapturePath = ".\captures\",

    [switch]$Capture,

    [int]$NPixels = 15,

    [string]$ExtraArgs = ""
)

# Set up environment for DirectX debugging
if ($EnableDebug) {
    $env:DX12_ENABLE_DEBUG_LAYER = "1"
    $env:DX12_FORCE_DEBUG_LAYER = "1"
    Write-Host "D3D12 Debug Layer: ENABLED" -ForegroundColor Green
}

if ($EnableDRED) {
    $env:DX12_ENABLE_DRED = "1"
    $dredLog = Join-Path $CapturePath "dred_log.txt"
    $env:DX12_DRED_OUTPUT = $dredLog
    Write-Host "DRED: ENABLED (output: $dredLog)" -ForegroundColor Green
}

# Create capture directory if needed
if (-not (Test-Path $CapturePath)) {
    New-Item -ItemType Directory -Path $CapturePath -Force | Out-Null
}

# Build command line arguments
$args = "--model `"$ModelPath`" --prompt `"$Prompt`" --no-display-prompt --temp 0 --n-predict $NPixels"
if ($ExtraArgs) {
    $args += " " + $ExtraArgs
}

# Check if PIX is available for capture
if ($Capture) {
    $pixPaths = @(
        "C:\Program Files\Microsoft PIX",
        "C:\Program Files (x86)\Microsoft PIX"
    )
    $pixExe = $null
    foreach ($pixPath in $pixPaths) {
        if (Test-Path $pixPath) {
            $pixExe = Get-ChildItem $pixPath -Filter "pix.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($pixExe) {
                $pixExe = $pixExe.FullName
                break
            }
        }
    }

    if ($pixExe) {
        $captureFile = Join-Path $CapturePath "dx12_capture_$(Get-Date -Format 'yyyyMMdd_HHmmss').piX"
        Write-Host "Starting PIX capture..." -ForegroundColor Cyan
        Write-Host "  App: $AppPath" -ForegroundColor Gray
        Write-Host "  Capture: $captureFile" -ForegroundColor Gray

        $pixArgs = "capture -outfile `"$captureFile`" -process `"$AppPath`" -cmdline `"$args`""
        & "$pixExe" $pixArgs
        exit $LASTEXITCODE
    } else {
        Write-Warning "PIX not found. Falling back to direct execution."
        Write-Warning "Use Visual Studio Graphics Debugger (ALT+F12) to capture frames."
    }
}

# Run the application directly
Write-Host "Running DirectX AI application..." -ForegroundColor Cyan
Write-Host "  App: $AppPath" -ForegroundColor Gray
Write-Host "  Model: $ModelPath" -ForegroundColor Gray
Write-Host "  Prompt: $Prompt" -ForegroundColor Gray

& "$AppPath" $args
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "Application exited with code: $exitCode" -ForegroundColor Red
    if ($EnableDRED -and (Test-Path $env:DX12_DRED_OUTPUT)) {
        Write-Host "DRED log available at: $env:DX12_DRED_OUTPUT" -ForegroundColor Yellow
    }
}

exit $exitCode
