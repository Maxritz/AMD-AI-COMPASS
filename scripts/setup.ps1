# AI-COMPASS Setup Script
# Sets up development environment and symlinks to third-party tools

Write-Host "🔧 AI-COMPASS Setup" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# Check prerequisites
$tools = @{
    "CMake" = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    "Python" = (Get-Command python -ErrorAction SilentlyContinue).Source
    "Git" = (Get-Command git -ErrorAction SilentlyContinue).Source
    "VS2022" = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
}

Write-Host "`nChecking prerequisites:" -ForegroundColor Yellow
foreach ($k in $tools.Keys) {
    $path = $tools[$k]
    if ($path -and (Test-Path $path -ErrorAction SilentlyContinue)) {
        Write-Host "  ✅ $k found at $path"
    } else {
        Write-Host "  ❌ $k not found"
    }
}

# Set up third-party tool symlinks
$links = @{
    "third_party\hip_tracer" = "F:\AMD-Ai\hip_tracer"
    "third_party\gpu_perf_api" = "F:\AMD-Ai\gpu_performance_api"
    "third_party\adlx_sdk" = "F:\AMD-Ai\ADLX-SDK"
    "third_party\rcv" = "F:\AMD-Ai\rocprof-compute-viewer"
    "third_party\rocm-systems" = "F:\AMD-Ai\rocm-systems"
    "third_party\geak" = "F:\AMD-Ai\GEAK-RDNA"
    "third_party\hyperloom" = "F:\AMD-Ai\Hyperloom-RDNA"
    "third_party\magpie" = "F:\AMD-Ai\Magpie"
    "third_party\intellikit" = "F:\AMD-Ai\intellikit"
    "third_party\apex" = "F:\AMD-Ai\Apex"
    "third_party\tracelens" = "F:\AMD-Ai\TraceLens"
}

Write-Host "`nCreating third-party symlinks:" -ForegroundColor Yellow
$ai_compass_root = Split-Path -Parent $PSScriptRoot
foreach ($link in $links.Keys) {
    $link_path = Join-Path $ai_compass_root $link
    $target = $links[$link]
    if (Test-Path $target) {
        if (-not (Test-Path $link_path)) {
            New-Item -ItemType Junction -Path $link_path -Target $target | Out-Null
            Write-Host "  🔗 $link → $target"
        } else {
            Write-Host "  ✓ $link already exists"
        }
    } else {
        Write-Host "  ⚠ Target $target not found, skipping $link"
    }
}

# Build the project
Write-Host "`nBuilding AI-COMPASS:" -ForegroundColor Yellow
$build_dir = Join-Path $ai_compass_root "build"
if (-not (Test-Path $build_dir)) {
    New-Item -ItemType Directory -Path $build_dir | Out-Null
}

Push-Location $build_dir
try {
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release -- /m:8
    Write-Host "  ✅ Build complete!"
} catch {
    Write-Host "  ❌ Build failed: $_"
} finally {
    Pop-Location
}

Write-Host "`nAI-COMPASS ready!" -ForegroundColor Green
Write-Host "  CLI: $ai_compass_root\build\Release\aicompass.exe"
Write-Host "  Run: aicompass help"