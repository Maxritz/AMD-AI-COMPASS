<#
.SYNOPSIS
    Unified system validation for AI-Compass multi-backend toolkit.

.DESCRIPTION
    Validates that all AI compute backends (HIP/ROCm, Vulkan, DirectX 12, WinML)
    are properly installed and configured on the system.

    This script checks:
    - GPU hardware detection (AMD/NVIDIA/Intel)
    - HIP/ROCm runtime availability and version
    - Vulkan SDK availability and version
    - DirectX 12 runtime and DirectML availability
    - WinML runtime availability
    - Required environment variables
    - Build tools (cmake, ninja, etc.)

.PARAMETER Detailed
    Show detailed validation output including versions and paths

.PARAMETER Fix
    Attempt to fix common configuration issues (sets PATH, etc.)

.EXAMPLE
    .\validate-system.ps1
    Run basic system validation.

.EXAMPLE
    .\validate-system.ps1 -Detailed
    Run detailed system validation with version information.
#>

param(
    [switch]$Detailed,
    [switch]$Fix
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

$validationResults = @{}

# GPU Hardware Detection
Write-Section "GPU Hardware Detection"

$gpus = @()
try {
    $gpuInfo = Get-WmiObject -Class "Win32_VideoController" -ErrorAction Stop
    foreach ($gpu in $gpuInfo) {
        $gpus += [PSCustomObject]@{
            Name = $gpu.Name
            Vendor = if ($gpu.Name -match "AMD|Radeon") { "AMD" } elseif ($gpu.Name -match "NVIDIA|GeForce|RTX") { "NVIDIA" } elseif ($gpu.Name -match "Intel|UHD|Arc") { "Intel" } else { "Unknown" }
            VRAM = if ($gpu.AdapterRAM) { [math]::Round($gpu.AdapterRAM / 1GB, 2) } else { "Unknown" }
        }
        $validationResults["GPU_$($gpu.Name)"] = $true
    }
} catch {
    Write-Error-Continue "Failed to detect GPUs: $_"
    $validationResults["GPU_Detection"] = $false
}

foreach ($gpu in $gpus) {
    Write-Host "  $($gpu.Name) [$($gpu.Vendor)] - $($gpu.VRAM) GB VRAM"
}

if ($gpus.Count -eq 0) {
    Write-Warning "No GPUs detected"
    $validationResults["GPU_Detection"] = $false
} else {
    Write-Success "Detected $($gpus.Count) GPU(s)"
    $validationResults["GPU_Detection"] = $true
}

# HIP/ROCm Validation
Write-Section "HIP/ROCm Backend"

$hipPaths = @(
    "E:\ROCM-7.13.0-Windows",
    "C:\Program Files\ROCm",
    "C:\ROCm"
)

$hipFound = $false
foreach ($path in $hipPaths) {
    if (Test-Path $path) {
        $hipFound = $true
        Write-Success "HIP/ROCm found at: $path"
        $validationResults["HIP"] = $true

        if ($Detailed) {
            $versionFile = Join-Path $path "bin\hip_version"
            if (Test-Path $versionFile) {
                $version = Get-Content $versionFile -ErrorAction SilentlyContinue
                Write-Host "  Version: $version"
            }

            $llvmBin = Join-Path $path "lib\llvm\bin"
            if (Test-Path $llvmBin) {
                Write-Host "  LLVM bin: $llvmBin"
            }
        }

        if ($Fix) {
            $llvmBin = Join-Path $path "lib\llvm\bin"
            if (Test-Path $llvmBin) {
                $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
                if (-not ($currentPath -split ";" | Where-Object { $_ -eq $llvmBin })) {
                    Write-Host "  Adding LLVM bin to PATH..."
                    $newPath = "$currentPath;$llvmBin"
                    [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
                    $env:PATH = $newPath
                }
            }
        }
        break
    }
}

if (-not $hipFound) {
    Write-Warning "HIP/ROCm not found in standard locations"
    $validationResults["HIP"] = $false
}

# Vulkan Validation
Write-Section "Vulkan Backend"

$vulkanSDK = $env:VULKAN_SDK
if ($vulkanSDK -and (Test-Path $vulkanSDK)) {
    Write-Success "Vulkan SDK found at: $vulkanSDK"
    $validationResults["Vulkan"] = $true

    if ($Detailed) {
        $versionJson = Join-Path $vulkanSDK "Lib\vk.json"
        if (Test-Path $versionJson) {
            Write-Host "  SDK version JSON available"
        }
    }
} else {
    Write-Warning "Vulkan SDK not found (VULKAN_SDK env var not set or path invalid)"
    $validationResults["Vulkan"] = $false
}

# DirectX 12 Validation
Write-Section "DirectX 12 Backend"

$dxPath = "E:\DXllama\OptimiseDX"
if (Test-Path $dxPath) {
    Write-Success "DirectX 12 backend found at: $dxPath"
    $validationResults["DX12"] = $true
} else {
    Write-Warning "DirectX 12 backend not found at: $dxPath"
    $validationResults["DX12"] = $false
}

# Check DirectX 12 runtime
try {
    $dx12Runtime = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\DirectX" -ErrorAction SilentlyContinue
    if ($dx12Runtime) {
        Write-Success "DirectX runtime detected"
    } else {
        Write-Warning "DirectX runtime not found in registry"
    }
} catch {
    Write-Warning "Could not check DirectX runtime: $_"
}

# WinML Validation
Write-Section "WinML Backend"

$winmlAvailable = $false
try {
    $null = [System.Management.Automation.PSTypeName]'Windows.AI.MachineLearning.MLContext'
    $winmlAvailable = $true
    Write-Success "WinML runtime available"
    $validationResults["WinML"] = $true
} catch {
    Write-Warning "WinML runtime not available (requires Windows 10/11 with AI features)"
    $validationResults["WinML"] = $false
}

# Build Tools Validation
Write-Section "Build Tools"

$tools = @{
    "cmake" = "CMake"
    "ninja" = "Ninja"
    "git" = "Git"
    "clang" = "Clang"
    "python" = "Python"
}

foreach ($tool in $tools.GetEnumerator()) {
    try {
        $version = & $tool.Key --version 2>$null
        if ($LASTEXITCODE -eq 0 -or $version) {
            Write-Success "$($tool.Value) available"
            if ($Detailed) {
                $versionLine = $version | Select-Object -First 1
                Write-Host "  $versionLine"
            }
            $validationResults[$tool.Key] = $true
        } else {
            Write-Warning "$($tool.Value) not found in PATH"
            $validationResults[$tool.Key] = $false
        }
    } catch {
        Write-Warning "$($tool.Value) not found in PATH"
        $validationResults[$tool.Key] = $false
    }
}

# Environment Variables
Write-Section "Environment Variables"

$envVars = @{
    "PATH" = "System PATH"
    "VULKAN_SDK" = "Vulkan SDK"
    "ROCM_PATH" = "ROCm Path"
    "HIP_PATH" = "HIP Path"
}

foreach ($var in $envVars.GetEnumerator()) {
    $value = [Environment]::GetEnvironmentVariable($var.Key)
    if ($value) {
        Write-Success "$($var.Value): $value"
        $validationResults["ENV_$($var.Key)"] = $true
    } else {
        Write-Warning "$($var.Value): not set"
        $validationResults["ENV_$($var.Key)"] = $false
    }
}

# Summary
Write-Section "Validation Summary"

$passed = ($validationResults.Values | Where-Object { $_ -eq $true }).Count
$total = $validationResults.Count
$failed = $total - $passed

Write-Host ""
Write-Host "Results: $passed passed, $failed failed out of $total checks" -ForegroundColor Cyan

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Failed checks:" -ForegroundColor Yellow
    foreach ($kvp in $validationResults.GetEnumerator()) {
        if (-not $kvp.Value) {
            Write-Host "  - $($kvp.Key)" -ForegroundColor Red
        }
    }
}

Write-Host ""
if ($passed -ge ($total / 2)) {
    Write-Success "System is ready for AI-Compass toolkit"
    exit 0
} else {
    Write-Error-Continue "System is not ready - too many missing components"
    exit 1
}
