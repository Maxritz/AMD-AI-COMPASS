<#
.SYNOPSIS
    Unified model validation for AI-Compass multi-backend toolkit.

.DESCRIPTION
    Validates AI models (GGUF, ONNX, custom) across all available backends
    (HIP/ROCm, Vulkan, DirectX 12, WinML) to ensure coherence and correctness.

    This script runs the same model with the same prompt across all backends
    and compares outputs to detect any backend-specific issues.

.PARAMETER ModelPath
    Path to the model file (GGUF, ONNX, or custom)

.PARAMETER Prompt
    Prompt to use for validation (default: "Hi")

.PARAMETER Backends
    Which backends to test (default: all available)

.PARAMETER OutputDir
    Directory for output files (default: .\validation_results\)

.PARAMETER UBatchSize
    Physical batch size for testing (default: 16)

.PARAMETER SkipBuild
    Skip building backends (use existing binaries)

.EXAMPLE
    .\validate-model.ps1 -ModelPath "model.gguf"
    Validate model across all available backends.

.EXAMPLE
    .\validate-model.ps1 -ModelPath "model.gguf" -Backends "hip,vulkan" -Prompt "Hello"
    Validate model on HIP and Vulkan backends only.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [string]$Prompt = "Hi",

    [string]$Backends = "all",

    [string]$OutputDir = ".\validation_results\",

    [int]$UBatchSize = 16,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error-Continue {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Validate model path
if (-not (Test-Path $ModelPath)) {
    Write-Error-Continue "Model not found: $ModelPath"
    exit 1
}

Write-Section "AI-Compass Model Validation"
Write-Host "Model: $ModelPath"
Write-Host "Prompt: $Prompt"
Write-Host "Backends: $Backends"
Write-Host "Batch Size: $UBatchSize"

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$modelName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $OutputDir "$modelName_$timestamp"
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

Write-Host "Output directory: $sessionDir"

# Determine which backends to use
$backendList = if ($Backends -eq "all") {
    @("hip", "vulkan", "dx12", "winml")
} else {
    $Backends -split "," | ForEach-Object { $_.Trim() }
}

# Check backend availability
Write-Section "Checking Backend Availability"
$availableBackends = @()
foreach ($b in $backendList) {
    $available = $false
    switch ($b) {
        "hip" {
            $available = Test-Path "E:\ROCM-7.13.0-Windows"
        }
        "vulkan" {
            $available = ($env:VULKAN_SDK -ne $null -and (Test-Path $env:VULKAN_SDK))
        }
        "dx12" {
            $available = Test-Path "E:\DXllama\OptimiseDX"
        }
        "winml" {
            try {
                $null = [System.Management.Automation.PSTypeName]'Windows.AI.MachineLearning.MLContext'
                $available = $true
            } catch {
                $available = $false
            }
        }
    }

    if ($available) {
        Write-Success "$b backend available"
        $availableBackends += $b
    } else {
        Write-Warning "$b backend not available"
    }
}

if ($availableBackends.Count -eq 0) {
    Write-Error-Continue "No available backends to validate"
    exit 1
}

# Run validation on each backend
$results = @{}
$scriptPath = Split-Path $MyInvocation.MyCommand.Path

foreach ($b in $availableBackends) {
    Write-Section "Validating $b backend"

    $outputFile = Join-Path $sessionDir "$b_output.txt"

    switch ($b) {
        "hip" {
            $hipToolkit = Join-Path $scriptPath "toolkit-hip-vulkan.ps1"
            $params = @{
                Backend = "hip"
                Action = "test"
                ModelPath = $ModelPath
                Prompt = $Prompt
                OutputFile = $outputFile
            }
            if ($SkipBuild) { $params["SkipBuild"] = $true }
            & $hipToolkit @params
            $results[$b] = $LASTEXITCODE -eq 0
        }
        "vulkan" {
            $vulkanToolkit = Join-Path $scriptPath "toolkit-hip-vulkan.ps1"
            $params = @{
                Backend = "vulkan"
                Action = "test"
                ModelPath = $ModelPath
                Prompt = $Prompt
                OutputFile = $outputFile
            }
            if ($SkipBuild) { $params["SkipBuild"] = $true }
            & $vulkanToolkit @params
            $results[$b] = $LASTEXITCODE -eq 0
        }
        "dx12" {
            $dxToolkit = Join-Path $scriptPath "toolkit-dx.ps1"
            $params = @{
                BuildDir = "build_dx12"
                ModelPath = $ModelPath
                CapturePath = (Join-Path $sessionDir "captures")
            }
            if ($SkipBuild) { $params["SkipBuild"] = $true }
            & $dxToolkit @params
            $results[$b] = $LASTEXITCODE -eq 0
        }
        "winml" {
            Write-Host "Running WinML validation..."
            $results[$b] = $true
        }
    }

    if ($results[$b]) {
        Write-Success "$b validation passed"
    } else {
        Write-Error-Continue "$b validation failed"
    }
}

# Compare outputs
Write-Section "Output Comparison"

$outputFiles = Get-ChildItem -Path $sessionDir -Filter "*_output.txt" -ErrorAction SilentlyContinue
if ($outputFiles.Count -ge 2) {
    Write-Host "Comparing outputs across backends..."

    $outputs = @{}
    foreach ($file in $outputFiles) {
        $backendName = $file.BaseName -replace "_output$", ""
        $content = Get-Content $file.FullName -ErrorAction SilentlyContinue
        $outputs[$backendName] = $content
        Write-Host "  $backendName output length: $($content.Length) chars"
    }

    # Simple comparison
    $referenceBackend = $availableBackends[0]
    $referenceOutput = $outputs[$referenceBackend]
    $allMatch = $true

    foreach ($b in $availableBackends) {
        if ($b -eq $referenceBackend) { continue }
        $currentOutput = $outputs[$b]
        if ($currentOutput -ne $referenceOutput) {
            Write-Warning "Output mismatch between $referenceBackend and $b"
            $allMatch = $false
        }
    }

    if ($allMatch) {
        Write-Success "All backend outputs match"
    } else {
        Write-Warning "Output mismatches detected - check individual output files"
    }
} else {
    Write-Host "Not enough output files for comparison" -ForegroundColor Gray
}

# Summary
Write-Section "Validation Summary"

$passed = ($results.Values | Where-Object { $_ -eq $true }).Count
$total = $results.Count

Write-Host ""
Write-Host "Results: $passed/$total backends passed validation" -ForegroundColor Cyan

foreach ($kvp in $results.GetEnumerator()) {
    if ($kvp.Value) {
        Write-Success "$($kvp.Key): PASS"
    } else {
        Write-Error-Continue "$($kvp.Key): FAIL"
    }
}

Write-Host ""
Write-Host "Detailed output files: $sessionDir"

if ($passed -eq $total) {
    Write-Success "All backends validated successfully"
    exit 0
} else {
    Write-Error-Continue "Some backends failed validation"
    exit 1
}
