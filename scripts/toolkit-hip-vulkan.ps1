<#
.SYNOPSIS
    Unified AI-Compass-HIP-Vulkan toolkit for coherence testing and debugging.

.DESCRIPTION
    Provides a unified interface for testing and debugging AI applications across
    HIP (ROCm) and Vulkan compute backends on Windows.

.PARAMETER Backend
    Which backend to use: 'hip', 'vulkan', or 'both'

.PARAMETER Action
    What action to perform: 'build', 'test', 'debug', 'validate', 'capture', 'profile'

.PARAMETER ModelPath
    Path to the model file (GGUF, ONNX, or custom)

.PARAMETER Prompt
    Prompt for coherence testing (default: "Hi")

.PARAMETER OutputFile
    Output file for test results

.EXAMPLE
    .\toolkit-hip-vulkan.ps1 -Backend hip -Action test -ModelPath "model.gguf"
    Run coherence test on HIP backend.

.EXAMPLE
    .\toolkit-hip-vulkan.ps1 -Backend vulkan -Action debug -ModelPath "model.gguf"
    Run debug test on Vulkan backend with validation layers.

.EXAMPLE
    .\toolkit-hip-vulkan.ps1 -Backend both -Action validate
    Validate system for both HIP and Vulkan.
#>

param(
    [ValidateSet("hip", "vulkan", "both")]
    [string]$Backend = "both",

    [ValidateSet("build", "test", "debug", "validate", "capture", "profile")]
    [string]$Action = "test",

    [string]$ModelPath = "",

    [string]$Prompt = "Hi",

    [string]$OutputFile = ""
)

function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Fail { Write-Host $args -ForegroundColor Red }
function Write-Warn { Write-Host $args -ForegroundColor Yellow }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

function Test-HIPBackend {
    Write-Info "=== HIP (ROCm) Backend ==="

    # Check ROCm
    $rocmPath = $env:ROCM_PATH
    if (-not $rocmPath) {
        $rocmPath = "C:\Program Files\ROCm"
    }
    if (Test-Path $rocmPath) {
        Write-Success "ROCm found at: $rocmPath"
    } else {
        Write-Fail "ROCm not found. Install ROCm 7.x for Windows."
        return $false
    }

    # Check hipcc
    $hipcc = Get-Command hipcc -ErrorAction SilentlyContinue
    if ($hipcc) {
        Write-Success "hipcc found: $($hipcc.Source)"
    } else {
        Write-Warn "hipcc not in PATH. Add $rocmPath\bin to PATH."
    }

    # Check hipblas
    $hipblasPath = Join-Path $rocmPath "lib\hipblas.lib"
    if (Test-Path $hipblasPath) {
        Write-Success "hipBLAS library found"
    } else {
        Write-Warn "hipBLAS library not found at $hipblasPath"
    }

    # Check GPU
    $hipDevices = & hipcc --offload-arch=gfx1201 -E -x c++ - <<< "#include <hip/hip_runtime.h>" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "HIP can target gfx1201 (RDNA4)"
    } else {
        Write-Warn "Cannot target gfx1201"
    }

    return $true
}

function Test-VulkanBackend {
    Write-Info "=== Vulkan Backend ==="

    # Check VULKAN_SDK
    $vulkanSdk = $env:VULKAN_SDK
    if ($vulkanSdk -and (Test-Path $vulkanSdk)) {
        Write-Success "Vulkan SDK found at: $vulkanSdk"
    } else {
        Write-Fail "Vulkan SDK not found. Install LunarG Vulkan SDK 1.3+."
        return $false
    }

    # Check glslc
    $glslc = Get-Command glslc -ErrorAction SilentlyContinue
    if ($glslc) {
        Write-Success "glslc found: $($glslc.Source)"
    } else {
        $glslcPath = Join-Path $vulkanSdk "Bin\glslc.exe"
        if (Test-Path $glslcPath) {
            Write-Success "glslc found at: $glslcPath"
        } else {
            Write-Warn "glslc not found"
        }
    }

    # Check Vulkan runtime
    $vulkanRT = Get-ChildItem "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -ErrorAction SilentlyContinue
    if ($vulkanRT) {
        Write-Success "Vulkan runtime drivers found"
    } else {
        Write-Warn "No Vulkan runtime drivers found"
    }

    # Check validation layers
    $layersPath = Join-Path $vulkanSdk "etc\vk_layer_settings.d"
    if (Test-Path $layersPath) {
        Write-Success "Vulkan validation layers found"
    } else {
        Write-Warn "Vulkan validation layers not found"
    }

    # Check RenderDoc
    $renderdocPath = "C:\Program Files\RenderDoc"
    if (Test-Path $renderdocPath) {
        Write-Success "RenderDoc found at: $renderdocPath"
    } else {
        Write-Warn "RenderDoc not found (optional for capture)"
    }

    return $true
}

function Invoke-HIPBuild {
    Write-Info "Building HIP backend..."
    $buildDir = "build-hip"
    if (-not (Test-Path $buildDir)) {
        cmake -S . -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_OPENMP=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX_VNNI=ON
    }
    cmake --build $buildDir --config Release --target llama-cli
    if ($LASTEXITCODE -eq 0) {
        Write-Success "HIP build completed"
    } else {
        Write-Fail "HIP build failed"
    }
}

function Invoke-VulkanBuild {
    Write-Info "Building Vulkan backend..."
    $buildDir = "build-vulkan"
    if (-not (Test-Path $buildDir)) {
        cmake -S . -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
    }
    cmake --build $buildDir --config Release --target llama-cli
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Vulkan build completed"
    } else {
        Write-Fail "Vulkan build failed"
    }
}

function Invoke-HIPCoherenceTest {
    param([string]$ModelPath, [string]$Prompt, [string]$OutputFile)

    Write-Info "Running HIP coherence test..."

    $exe = "build-hip\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) {
        $exe = "build-hip\llama-cli.exe"
    }
    if (-not (Test-Path $exe)) {
        Write-Fail "HIP binary not found. Build first."
        return $false
    }

    $outputFile = $OutputFile ?: "hip_coherence_output.txt"
    $textFile = "hip_coherence_text.txt"

    $args = @("-m", $ModelPath, "-p", $Prompt, "--n-gpu-layers", "99", "--no-display-prompt", "--temp", "0", "--no-penalties")

    & $exe $args > $outputFile 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Fail "HIP inference failed with exit code $exitCode"
        Get-Content $outputFile | Select-Object -Last 20
        return $false
    }

    # Extract response
    $content = Get-Content $outputFile
    $responseStart = $false
    $responseLines = @()
    foreach ($line in $content) {
        if ($line -match "llama_perf_context_.*eval") { $responseStart = $false }
        if ($responseStart) { $responseLines += $line }
        if ($line -match "^> User$") { $responseStart = $true }
    }
    $responseLines | Set-Content $textFile

    # Validate coherence
    $text = $responseLines -join "`n"
    $hasGarbled = $text -match "[\x00-\x08\x0E-\x1F\x7F]"
    $hasStructure = $text -match "[.!?]\s*$"
    $repeatCount = ($text -split "\s+").Count - ($text -split "\s+" | Select-Object -Unique).Count

    Write-Info "Coherence validation:"
    Write-Host "  Garbled chars: $(if ($hasGarbled) { 'FAIL' } else { 'PASS' })"
    Write-Host "  Sentence structure: $(if ($hasStructure) { 'PASS' } else { 'WARN' })"
    Write-Host "  Word repetition: $(if ($repeatCount -gt 10) { "FAIL ($repeatCount)" } else { "PASS ($repeatCount)" })"

    if ($hasGarbled -or $repeatCount -gt 10) {
        Write-Fail "HIP coherence test FAILED"
        return $false
    }

    Write-Success "HIP coherence test PASSED"
    return $true
}

function Invoke-VulkanCoherenceTest {
    param([string]$ModelPath, [string]$Prompt, [string]$OutputFile)

    Write-Info "Running Vulkan coherence test..."

    $exe = "build-vulkan\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) {
        $exe = "build-vulkan\llama-cli.exe"
    }
    if (-not (Test-Path $exe)) {
        Write-Fail "Vulkan binary not found. Build first."
        return $false
    }

    $outputFile = $OutputFile ?: "vulkan_coherence_output.txt"
    $textFile = "vulkan_coherence_text.txt"

    $args = @("-m", $ModelPath, "-p", $Prompt, "--n-gpu-layers", "99", "--no-display-prompt", "--temp", "0", "--no-penalties")

    & $exe $args > $outputFile 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Fail "Vulkan inference failed with exit code $exitCode"
        Get-Content $outputFile | Select-Object -Last 20
        return $false
    }

    # Extract response
    $content = Get-Content $outputFile
    $responseStart = $false
    $responseLines = @()
    foreach ($line in $content) {
        if ($line -match "llama_perf_context_.*eval") { $responseStart = $false }
        if ($responseStart) { $responseLines += $line }
        if ($line -match "^> User$") { $responseStart = $true }
    }
    $responseLines | Set-Content $textFile

    # Validate coherence
    $text = $responseLines -join "`n"
    $hasGarbled = $text -match "[\x00-\x08\x0E-\x1F\x7F]"
    $hasStructure = $text -match "[.!?]\s*$"
    $repeatCount = ($text -split "\s+").Count - ($text -split "\s+" | Select-Object -Unique).Count

    Write-Info "Coherence validation:"
    Write-Host "  Garbled chars: $(if ($hasGarbled) { 'FAIL' } else { 'PASS' })"
    Write-Host "  Sentence structure: $(if ($hasStructure) { 'PASS' } else { 'WARN' })"
    Write-Host "  Word repetition: $(if ($repeatCount -gt 10) { "FAIL ($repeatCount)" } else { "PASS ($repeatCount)" })"

    if ($hasGarbled -or $repeatCount -gt 10) {
        Write-Fail "Vulkan coherence test FAILED"
        return $false
    }

    Write-Success "Vulkan coherence test PASSED"
    return $true
}

function Invoke-HIPDebugTest {
    param([string]$ModelPath, [string]$Prompt)

    Write-Info "Running HIP debug test with validation..."

    $env:HIP_TRACE_API = "1"
    $env:HIP_VISIBLE_DEVICES = "0"

    $exe = "build-hip\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) { $exe = "build-hip\llama-cli.exe" }
    if (-not (Test-Path $exe)) {
        Write-Fail "HIP binary not found. Build first."
        return $false
    }

    $args = @("-m", $ModelPath, "-p", $Prompt, "--n-gpu-layers", "99", "--temp", "0")

    & $exe $args 2>&1 | Tee-Object -Variable output

    $outputStr = $output -join "`n"
    if ($outputStr -match "error|Error|ERROR|failed|Failed|FAILED") {
        Write-Fail "HIP debug test found errors"
        return $false
    }

    Write-Success "HIP debug test passed"
    return $true
}

function Invoke-VulkanDebugTest {
    param([string]$ModelPath, [string]$Prompt)

    Write-Info "Running Vulkan debug test with validation layers..."

    $env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
    $env:GGML_VK_DEBUG_MARKERS = "1"

    $exe = "build-vulkan\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) { $exe = "build-vulkan\llama-cli.exe" }
    if (-not (Test-Path $exe)) {
        Write-Fail "Vulkan binary not found. Build first."
        return $false
    }

    $args = @("-m", $ModelPath, "-p", $Prompt, "--n-gpu-layers", "99", "--temp", "0")

    & $exe $args 2>&1 | Tee-Object -Variable output

    $outputStr = $output -join "`n"
    if ($outputStr -match "Validation|VALIDATION|error|Error|ERROR") {
        Write-Warn "Vulkan validation layer messages detected"
        $output | Where-Object { $_ -match "Validation|VALIDATION|error|Error|ERROR" } | Select-Object -First 10
    }

    if ($outputStr -match "failed|Failed|FAILED") {
        Write-Fail "Vulkan debug test found errors"
        return $false
    }

    Write-Success "Vulkan debug test passed"
    return $true
}

function Invoke-HIPCapture {
    param([string]$ModelPath)

    Write-Info "Setting up HIP GPU capture..."

    # Check for RGP
    $rgpPath = "C:\Program Files\AMD\Radeon GPU Profiler"
    if (Test-Path $rgpPath) {
        Write-Success "RGP found. Launch RGP and attach to process."
        Write-Host "  1. Start RGP"
        Write-Host "  2. Run: build-hip\bin\Release\llama-cli.exe -m $ModelPath --n-gpu-layers 99"
        Write-Host "  3. Capture frame in RGP"
    } else {
        Write-Warn "RGP not found. Install from https://gpuopen.com/rgp/"
    }

    # Check for rocprof
    $rocprof = Get-Command rocprof -ErrorAction SilentlyContinue
    if ($rocprof) {
        Write-Success "rocprof found. Profile with:"
        Write-Host "  rocprof --stats --hip-profiling --build-hip\bin\Release\llama-cli.exe -m $ModelPath"
    } else {
        Write-Warn "rocprof not in PATH"
    }
}

function Invoke-VulkanCapture {
    param([string]$ModelPath)

    Write-Info "Setting up Vulkan GPU capture..."

    # Check for RenderDoc
    $renderdocPath = "C:\Program Files\RenderDoc"
    if (Test-Path $renderdocPath) {
        Write-Success "RenderDoc found."
        Write-Host "  1. Launch RenderDoc"
        Write-Host "  2. Set executable: build-vulkan\bin\Release\llama-cli.exe"
        Write-Host "  3. Set command line: -m $ModelPath --n-gpu-layers 99"
        Write-Host "  4. Capture frame"
    } else {
        Write-Warn "RenderDoc not found. Install from https://renderdoc.org/"
    }

    # Check for NSight Graphics
    $nsightPath = "C:\Program Files\NVIDIA Corporation\Nsight Graphics"
    if (Test-Path $nsightPath) {
        Write-Success "NSight Graphics found."
        Write-Host "  1. Launch NSight Graphics"
        Write-Host "  2. Set executable: build-vulkan\bin\Release\llama-cli.exe"
        Write-Host "  3. Set command line: -m $ModelPath --n-gpu-layers 99"
        Write-Host "  4. Capture frame"
    } else {
        Write-Warn "NSight Graphics not found (optional)"
    }
}

function Invoke-HIPProfile {
    param([string]$ModelPath)

    Write-Info "Profiling HIP backend..."

    $exe = "build-hip\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) { $exe = "build-hip\llama-cli.exe" }
    if (-not (Test-Path $exe)) {
        Write-Fail "HIP binary not found. Build first."
        return $false
    }

    $rocprof = Get-Command rocprof -ErrorAction SilentlyContinue
    if ($rocprof) {
        $args = @("--stats", "--hip-profiling", $exe, "-m", $ModelPath, "--n-gpu-layers", "99", "--temp", "0")
        & rocprof $args
        Write-Success "HIP profiling complete. Results in rocprof_stats.csv"
    } else {
        Write-Warn "rocprof not found. Install ROCm 7.x for profiling."
    }
}

function Invoke-VulkanProfile {
    param([string]$ModelPath)

    Write-Info "Profiling Vulkan backend..."

    $exe = "build-vulkan\bin\Release\llama-cli.exe"
    if (-not (Test-Path $exe)) { $exe = "build-vulkan\llama-cli.exe" }
    if (-not (Test-Path $exe)) {
        Write-Fail "Vulkan binary not found. Build first."
        return $false
    }

    # Use Vulkan's built-in timing
    $env:GGML_VULKAN_DEBUG = "1"
    $args = @("-m", $ModelPath, "--n-gpu-layers", "99", "--temp", "0")
    & $exe $args 2>&1 | Tee-Object -Variable output

    $outputStr = $output -join "`n"
    if ($outputStr -match "llama_perf_context_.*eval.*t/s") {
        Write-Success "Vulkan profiling complete. Check output for timing."
    }
}

# Main execution
Write-Info "=== AI-Compass-HIP-Vulkan Toolkit ==="
Write-Info "Backend: $Backend | Action: $Action"
Write-Host ""

$results = @{}

if ($Backend -eq "hip" -or $Backend -eq "both") {
    $hipOk = Test-HIPBackend
    $results["HIP"] = $hipOk

    if ($hipOk) {
        switch ($Action) {
            "build" { Invoke-HIPBuild }
            "test" {
                if ($ModelPath) {
                    $results["HIP_Test"] = Invoke-HIPCoherenceTest -ModelPath $ModelPath -Prompt $Prompt -OutputFile $OutputFile
                } else {
                    Write-Warn "No model path specified for coherence test"
                }
            }
            "debug" {
                if ($ModelPath) {
                    $results["HIP_Debug"] = Invoke-HIPDebugTest -ModelPath $ModelPath -Prompt $Prompt
                } else {
                    Write-Warn "No model path specified for debug test"
                }
            }
            "validate" { }
            "capture" {
                if ($ModelPath) { Invoke-HIPCapture -ModelPath $ModelPath }
            }
            "profile" {
                if ($ModelPath) { Invoke-HIPProfile -ModelPath $ModelPath }
            }
        }
    }
    Write-Host ""
}

if ($Backend -eq "vulkan" -or $Backend -eq "both") {
    $vulkanOk = Test-VulkanBackend
    $results["Vulkan"] = $vulkanOk

    if ($vulkanOk) {
        switch ($Action) {
            "build" { Invoke-VulkanBuild }
            "test" {
                if ($ModelPath) {
                    $results["Vulkan_Test"] = Invoke-VulkanCoherenceTest -ModelPath $ModelPath -Prompt $Prompt -OutputFile $OutputFile
                } else {
                    Write-Warn "No model path specified for coherence test"
                }
            }
            "debug" {
                if ($ModelPath) {
                    $results["Vulkan_Debug"] = Invoke-VulkanDebugTest -ModelPath $ModelPath -Prompt $Prompt
                } else {
                    Write-Warn "No model path specified for debug test"
                }
            }
            "validate" { }
            "capture" {
                if ($ModelPath) { Invoke-VulkanCapture -ModelPath $ModelPath }
            }
            "profile" {
                if ($ModelPath) { Invoke-VulkanProfile -ModelPath $ModelPath }
            }
        }
    }
    Write-Host ""
}

# Summary
Write-Info "=== Summary ==="
$results.GetEnumerator() | ForEach-Object {
    $status = if ($_.Value) { "PASS" } else { "FAIL" }
    Write-Host "$($_.Key): $status"
}

# Exit code
$allPassed = ($results.Values | Where-Object { $_ -eq $false }).Count -eq 0
if ($allPassed) {
    Write-Success "All checks passed!"
    exit 0
} else {
    Write-Fail "Some checks failed."
    exit 1
}
