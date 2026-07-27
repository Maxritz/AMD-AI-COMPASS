# Build llama.cpp with both HIP (GPU) and AVX2/FMA (CPU) optimizations
$src = "C:\Users\rr\Desktop\llama\llama.cpp-ROCM-Test"
$build = Join-Path $src "build-hip"

$cmakeArgs = @(
    "-S", $src,
    "-B", $build,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_HIP=ON",
    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
    "-DGGML_OPENMP=OFF",
    "-DGGML_AVX=ON",
    "-DGGML_AVX2=ON",
    "-DGGML_FMA=ON",
    "-DGGML_AVX_VNNI=ON"
)

Write-Host "=== Building llama.cpp with GPU + CPU optimizations ===" -ForegroundColor Cyan
Write-Host "GPU: HIP/ROCm for RDNA4 (RX 9070 XT)"
Write-Host "CPU: AVX2 + FMA + AVX_VNNI for Ryzen 5900XT"
Write-Host ""

# Configure
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && cmake @cmakeArgs 2>&1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed!" -ForegroundColor Red
    exit 1
}

# Build
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && cmake --build $build --config Release 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild complete!" -ForegroundColor Green
}
