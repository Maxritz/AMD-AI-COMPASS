# Build llama.cpp for RDNA2 (RX 6700 XT) with ROCm 7.3 + CPU optimizations
param(
    [string]$SrcDir = "C:\Users\rr\Desktop\llama\llama.cpp-ROCM-Test",
    [string]$BuildDir = "C:\Users\rr\Desktop\llama\llama.cpp-ROCM-Test\build-hip-rdna2",
    [string]$VcVars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    [string]$GpuArch = "gfx1031",
    [switch]$NoBuild
)

Write-Host "=== Building llama.cpp for RDNA2 (RX 6700 XT) ===" -ForegroundColor Cyan
Write-Host "GPU:    RDNA2 / gfx1031 (RX 6700 XT)" -ForegroundColor Yellow
Write-Host "ROCm:   7.3 on Windows 11" -ForegroundColor Yellow
Write-Host "CPU:    AVX2 + FMA (Ryzen 5600X)" -ForegroundColor Yellow
Write-Host "Build:  $BuildDir"
Write-Host ""

$cmakeArgs = @(
    "-S", $SrcDir,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_HIP=ON",
    "-DGGML_HIP_UMA=ON",
    "-DAMDGPU_TARGETS=$GpuArch",
    "-DCMAKE_HIP_ARCHITECTURES=$GpuArch",
    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
    "-DGGML_OPENMP=OFF",
    "-DGGML_AVX=ON",
    "-DGGML_AVX2=ON",
    "-DGGML_FMA=ON",
    "-DGGML_AVX_VNNI=ON"
)

# Configure
Write-Host "=== Configuring (gfx1031) ===" -ForegroundColor Cyan
cmd /c "`"$VcVars`" && cmake @cmakeArgs 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed!" -ForegroundColor Red
    exit 1
}

if ($NoBuild) {
    Write-Host "Configure complete. Build with:" -ForegroundColor Green
    Write-Host "  cmake --build $BuildDir --config Release"
    exit 0
}

# Build
Write-Host "=== Building ===" -ForegroundColor Cyan
cmd /c "`"$VcVars`" && cmake --build $BuildDir --config Release 2>&1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild complete!" -ForegroundColor Green
    Write-Host "Benchmarks in: $BuildDir\bin\llama-bench.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
}
