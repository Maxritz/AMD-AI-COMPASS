<#
.SYNOPSIS
    System validation for DirectX AI development environment.

.DESCRIPTION
    Validates that the system has all required components for DirectX AI development:
    - Windows version (10 1903+ or 11)
    - DirectX 12 compatible GPU
    - DirectX 12 Agility SDK
    - DXC (DirectX Shader Compiler)
    - Visual Studio 2022 with required workloads
    - Graphics Tools FOD (for debug layers)
    - Microsoft PIX (optional)
    - Vendor tools (AMD RGP, NVIDIA Nsight, Intel GPA)

.PARAMETER SkipVendorTools
    Skip checking for vendor-specific tools

.PARAMETER OutputFile
    Output file for validation report

.EXAMPLE
    .\validate-system.ps1
    Run full system validation and display results.

.EXAMPLE
    .\validate-system.ps1 -OutputFile "system_report.txt"
    Run validation and save report to file.
#>

param(
    [switch]$SkipVendorTools,
    [string]$OutputFile = ""
)

function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Fail { Write-Host $args -ForegroundColor Red }
function Write-Warn { Write-Host $args -ForegroundColor Yellow }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

$validationResults = @()

function Add-Result {
    param([string]$Component, [bool]$Passed, [string]$Details)
    $validationResults += [PSCustomObject]@{
        Component = $Component
        Status = if ($Passed) { "PASS" } else { "FAIL" }
        Details = $Details
    }
}

Write-Info "=== DirectX AI System Validation ==="
Write-Info ""

# 1. Windows Version
Write-Info "Checking Windows version..."
$osInfo = Get-CimInstance Win32_OperatingSystem
$buildNumber = [int]$osInfo.BuildNumber
$version = [System.Environment]::OSVersion.Version

if ($version.Major -ge 10 -and $buildNumber -ge 18362) {
    $winVer = if ($buildNumber -ge 22000) { "Windows 11" } else { "Windows 10" }
    Write-Success "Windows: $winVer (Build $buildNumber)"
    Add-Result "Windows Version" $true "$winVer (Build $buildNumber)"
} else {
    Write-Fail "Windows: Version $version (Build $buildNumber) - requires 10.0.18362+"
    Add-Result "Windows Version" $false "Build $buildNumber (requires 18362+)"
}

# 2. DirectX 12 Support
Write-Info "Checking DirectX 12 support..."
try {
    $dxVersion = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\DirectX").Version
    Write-Success "DirectX: Version $dxVersion"
    Add-Result "DirectX" $true "Version $dxVersion"
} catch {
    Write-Warn "DirectX: Version info not found in registry"
    Add-Result "DirectX" $true "Registry check inconclusive"
}

# 3. GPU Detection
Write-Info "Checking GPU..."
$gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterCompatibility -notmatch "Microsoft" }
if ($gpus.Count -gt 0) {
    foreach ($gpu in $gpus) {
        $gpuName = $gpu.Name
        $vramGB = [math]::Round($gpu.AdapterRAM / 1GB, 1)
        Write-Success "GPU: $gpuName ($vramGB GB VRAM)"
        Add-Result "GPU" $true "$gpuName ($vramGB GB)"
    }
} else {
    Write-Fail "No DirectX 12 compatible GPU detected"
    Add-Result "GPU" $false "No compatible GPU found"
}

# 4. DirectX 12 Agility SDK
Write-Info "Checking DirectX 12 Agility SDK..."
$agilityPaths = @(
    "C:\Program Files (x86)\Windows Kits\10\Include",
    "C:\Program Files\Windows Kits\10\Include"
)
$agilityFound = $false
foreach ($path in $agilityPaths) {
    if (Test-Path $path) {
        $latestInclude = Get-ChildItem $path -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latestInclude) {
            $d3d12Header = Join-Path $latestInclude.FullName "um\d3d12.h"
            if (Test-Path $d3d12Header) {
                Write-Success "DirectX 12 SDK: Found in $path\$($latestInclude.Name)"
                Add-Result "DX12 Agility SDK" $true "Found in $path"
                $agilityFound = $true
                break
            }
        }
    }
}
if (-not $agilityFound) {
    Write-Warn "DirectX 12 Agility SDK: Not found"
    Write-Warn "Install from: https://devblogs.microsoft.com/directx/directx12agility/"
    Add-Result "DX12 Agility SDK" $false "Not found"
}

# 5. DXC (DirectX Shader Compiler)
Write-Info "Checking DXC..."
$dxcPaths = @(
    "C:\Program Files (x86)\Windows Kits\10\bin",
    "C:\Program Files\Windows Kits\10\bin"
)
$dxcFound = $false
foreach ($basePath in $dxcPaths) {
    if (Test-Path $basePath) {
        $dxcExe = Get-ChildItem $basePath -Recurse -Filter "dxc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($dxcExe) {
            Write-Success "DXC: Found at $($dxcExe.FullName)"
            Add-Result "DXC" $true "Found at $($dxcExe.FullName)"
            $dxcFound = $true
            break
        }
    }
}
if (-not $dxcFound) {
    Write-Warn "DXC: Not found"
    Write-Warn "Install from: https://github.com/microsoft/DirectXShaderCompiler"
    Add-Result "DXC" $false "Not found"
}

# 6. Visual Studio 2022
Write-Info "Checking Visual Studio 2022..."
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022"
)
$vsFound = $false
foreach ($vsPath in $vsPaths) {
    if (Test-Path $vsPath) {
        $vsEditions = Get-ChildItem $vsPath -Directory | Select-Object -ExpandProperty FullName
        foreach ($edition in $vsEditions) {
            $vsDevShell = Join-Path $edition "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $vsDevShell) {
                Write-Success "Visual Studio: Found at $edition"
                Add-Result "Visual Studio 2022" $true "Found at $edition"
                $vsFound = $true
                break
            }
        }
    }
}
if (-not $vsFound) {
    Write-Warn "Visual Studio 2022: Not found"
    Write-Warn "Install from: https://visualstudio.microsoft.com/"
    Add-Result "Visual Studio 2022" $false "Not found"
}

# 7. Graphics Tools FOD
Write-Info "Checking Graphics Tools FOD..."
try {
    $graphicsTools = Get-WindowsCapability -Online -Filter "Name=Tools.Graphics.DirectX*" 2>$null
    if ($graphicsTools -and $graphicsTools.State -eq "Installed") {
        Write-Success "Graphics Tools FOD: Installed"
        Add-Result "Graphics Tools FOD" $true "Installed"
    } else {
        Write-Warn "Graphics Tools FOD: Not installed"
        Write-Warn "Install with: Add-WindowsCapability -Online -Name 'Tools.Graphics.DirectX~~~~0.0.1.0'"
        Add-Result "Graphics Tools FOD" $false "Not installed"
    }
} catch {
    Write-Warn "Graphics Tools FOD: Cannot check (requires admin)"
    Add-Result "Graphics Tools FOD" $false "Cannot check (admin required)"
}

# 8. Microsoft PIX
Write-Info "Checking Microsoft PIX..."
$pixPaths = @(
    "C:\Program Files\Microsoft PIX",
    "C:\Program Files (x86)\Microsoft PIX"
)
$pixFound = $false
foreach ($pixPath in $pixPaths) {
    if (Test-Path $pixPath) {
        Write-Success "Microsoft PIX: Found at $pixPath"
        Add-Result "Microsoft PIX" $true "Found at $pixPath"
        $pixFound = $true
        break
    }
}
if (-not $pixFound) {
    Write-Warn "Microsoft PIX: Not found (optional)"
    Write-Warn "Download from: https://devblogs.microsoft.com/directx/announcing-microsoft-pix-2024-1/"
    Add-Result "Microsoft PIX" $false "Not found (optional)"
}

# 9. Visual Studio Graphics Debugger
Write-Info "Checking Visual Studio Graphics Debugger..."
$vsGraphicsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\Extensions\Microsoft\Graphics\Debugger"
if (-not (Test-Path $vsGraphicsPath)) {
    $vsGraphicsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\IDE\Extensions\Microsoft\Graphics\Debugger"
}
if (Test-Path $vsGraphicsPath) {
    Write-Success "VS Graphics Debugger: Found"
    Add-Result "VS Graphics Debugger" $true "Found"
} else {
    Write-Warn "VS Graphics Debugger: Not found"
    Write-Warn "Install 'Graphics debugging' workload in VS Installer"
    Add-Result "VS Graphics Debugger" $false "Not found"
}

# 10. Vendor Tools
if (-not $SkipVendorTools) {
    Write-Info "Checking vendor tools..."

    # AMD Radeon GPU Profiler
    $rgpPath = "C:\Program Files\AMD\Radeon GPU Profiler"
    if (Test-Path $rgpPath) {
        Write-Success "AMD RGP: Found"
        Add-Result "AMD RGP" $true "Found"
    } else {
        Write-Warn "AMD RGP: Not found (optional)"
        Add-Result "AMD RGP" $false "Not found (optional)"
    }

    # NVIDIA Nsight
    $nsightPath = "C:\Program Files\NVIDIA Corporation\Nsight Graphics"
    if (Test-Path $nsightPath) {
        Write-Success "NVIDIA Nsight: Found"
        Add-Result "NVIDIA Nsight" $true "Found"
    } else {
        Write-Warn "NVIDIA Nsight: Not found (optional)"
        Add-Result "NVIDIA Nsight" $false "Not found (optional)"
    }

    # Intel GPA
    $gpaPath = "C:\Program Files\IntelSWTools\GPA"
    if (Test-Path $gpaPath) {
        Write-Success "Intel GPA: Found"
        Add-Result "Intel GPA" $true "Found"
    } else {
        Write-Warn "Intel GPA: Not found (optional)"
        Add-Result "Intel GPA" $false "Not found (optional)"
    }
}

# Summary
Write-Host ""
Write-Info "=== Validation Summary ==="
$passed = ($validationResults | Where-Object { $_.Status -eq "PASS" }).Count
$failed = ($validationResults | Where-Object { $_.Status -eq "FAIL" }).Count
$total = $validationResults.Count

Write-Host "Total: $total | Passed: $passed | Failed: $failed"
Write-Host ""

$validationResults | Format-Table -AutoSize

# Save to file if requested
if ($OutputFile) {
    $validationResults | Export-Csv -Path $OutputFile -NoTypeInformation
    Write-Info "Report saved to: $OutputFile"
}

# Exit code
if ($failed -gt 0) {
    Write-Warn "Some components are missing. See above for details."
    exit 1
} else {
    Write-Success "All components validated successfully!"
    exit 0
}
