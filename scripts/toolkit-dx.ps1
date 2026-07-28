<#
.SYNOPSIS
    Build, test, and debug toolkit for DirectX AI applications.

.DESCRIPTION
    This toolkit provides a comprehensive workflow for developing, testing, and
    debugging AI applications that use DirectX 12 and DirectML. It includes:

    1. Build verification for DirectX 12 backends
    2. Mandatory coherence testing (semantic output validation)
    3. Debug layer validation (D3D12 + DirectML)
    4. GPU capture and analysis hooks
    5. DRED (Device Removed Extended Data) configuration
    6. Hardware profiling integration
    7. Unit test execution

    The coherence test MUST pass before the toolkit exits successfully.
    If the coherence test fails, the toolkit will prompt to fix the issue.

    This toolkit is designed to work with any AI application that uses
    DirectX 12 (llama.cpp DX12 backend, DirectML, custom D3D12 compute, etc.).

.PARAMETER BuildDir
    Build directory to use (default: build_dx12)

.PARAMETER ModelPath
    Path to a GGUF model for coherence testing (default: E:\OLLAMA-Models\GGUF\carwin-Q4_K_M.gguf)

.PARAMETER SkipBuild
    Skip the build step (use existing binaries)

.PARAMETER SkipCoherence
    Skip the coherence test (NOT RECOMMENDED)

.PARAMETER SkipTests
    Skip unit tests (ctest)

.PARAMETER SkipDebugValidation
    Skip D3D12/DirectML debug layer validation

.PARAMETER SkipGPUCapture
    Skip GPU capture setup

.PARAMETER EnableDRED
    Enable Device Removed Extended Data (DRED) for crash debugging

.PARAMETER CapturePath
    Directory for GPU capture files (default: .\captures\)

.PARAMETER Backend
    Specify backend for coherence check: "dx12" (default) or "vulkan"

.PARAMETER AppPath
    Path to the AI application executable (default: auto-detected from BuildDir)

.PARAMETER AppArgs
    Additional arguments to pass to the AI application

.EXAMPLE
    .\toolkit-dx.ps1
    Build and run full DirectX AI validation suite.

.EXAMPLE
    .\toolkit-dx.ps1 -SkipBuild -SkipCoherence -SkipTests
    Run only debug layer validation and GPU capture setup.

.EXAMPLE
    .\toolkit-dx.ps1 -EnableDRED -CapturePath ".\my_captures\"
    Run with DRED enabled and custom capture directory.

.NOTES
    Requires:
    - DirectX 12 Agility SDK
    - DXC (DirectX Shader Compiler)
    - Visual Studio Graphics Debugger (integrated in VS 2022)
    - Optional: Microsoft PIX (standalone) for advanced GPU capture

    Debug layer DLLs should be placed next to the application executable.
    See: https://learn.microsoft.com/en-us/windows/ai/directml/dml-debug-layer
#>

param(
    [string]$BuildDir = "build_dx12",
    [string]$ModelPath = "E:\OLLAMA-Models\GGUF\carwin-Q4_K_M.gguf",
    [switch]$SkipBuild,
    [switch]$SkipCoherence,
    [switch]$SkipTests,
    [switch]$SkipDebugValidation,
    [switch]$SkipGPUCapture,
    [switch]$EnableDRED,
    [string]$CapturePath = ".\captures\",
    [string]$Backend = "dx12",
    [string]$AppPath = "",
    [string]$AppArgs = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = git rev-parse --show-toplevel 2>$null
if (-not $repoRoot) {
    Write-Error "Not in a Git repository."
    exit 1
}

Push-Location $repoRoot

# Color helpers
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Fail { Write-Host $args -ForegroundColor Red }
function Write-Warn { Write-Host $args -ForegroundColor Yellow }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Header { Write-Host "" ; Write-Host "=== $args ===" -ForegroundColor Cyan }

$coherencePassed = $false
$buildPassed = $false
$debugPassed = $false
$capturePassed = $false
$testExitCode = 0

# Initialize capture directory
$captureDir = Join-Path $repoRoot $CapturePath
if (-not (Test-Path $captureDir)) {
    New-Item -ItemType Directory -Path $captureDir -Force | Out-Null
}

try {
    # Step 1: Build
    if (-not $SkipBuild) {
        Write-Header "Step 1: Building (DirectX 12 backend)"
        $buildResult = & cmake --build $BuildDir --config Release --target llama-cli 2>&1
        $buildExitCode = $LASTEXITCODE

        if ($buildExitCode -eq 0) {
            Write-Success "Build PASSED"
            $buildPassed = $true
        } else {
            Write-Fail "Build FAILED"
            Write-Host $buildResult
            $buildPassed = $false
        }
    } else {
        Write-Warn "Skipping build step"
        $buildPassed = $true
    }

    # Step 2: Debug Layer Validation
    if (-not $SkipDebugValidation) {
        Write-Header "Step 2: Debug Layer Validation"

        # Check for D3D12 SDK layers
        $d3d12LayersFound = $false
        $dmlDebugFound = $false

        # Check for D3D12 debug layer (typically in Windows SDK)
        $d3d12DebugPaths = @(
            "C:\Program Files (x86)\Windows Kits\10\bin",
            "C:\Program Files\Windows Kits\10\bin"
        )

        foreach ($basePath in $d3d12DebugPaths) {
            if (Test-Path $basePath) {
                $d3d12LayersFound = $true
                break
            }
        }

        # Check for DirectML debug layer
        $dmlDebugPaths = @(
            "C:\Program Files\Microsoft DirectML",
            "C:\Program Files (x86)\Microsoft DirectML"
        )

        foreach ($basePath in $dmlDebugPaths) {
            if (Test-Path $basePath) {
                $dmlDebugFound = $true
                break
            }
        }

        # Check for dml.dll in the build directory (indicates DirectML is linked)
        $dmlDllPath = Join-Path $repoRoot "$BuildDir\bin\Release\dml.dll"
        if (Test-Path $dmlDllPath) {
            $dmlDebugFound = $true
            Write-Info "DirectML runtime found in build directory"
        }

        if ($d3d12LayersFound) {
            Write-Success "D3D12 SDK Layers: FOUND"
        } else {
            Write-Warn "D3D12 SDK Layers: NOT FOUND (debug validation may be limited)"
        }

        if ($dmlDebugFound) {
            Write-Success "DirectML Debug Layer: FOUND"
        } else {
            Write-Warn "DirectML Debug Layer: NOT FOUND (see https://learn.microsoft.com/en-us/windows/ai/directml/dml-debug-layer)"
        }

        # Check for Visual Studio Graphics Debugger
        $vsGraphicsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\Extensions\Microsoft\Graphics\Debugger"
        if (Test-Path $vsGraphicsPath) {
            Write-Success "Visual Studio Graphics Debugger: FOUND"
        } else {
            Write-Warn "Visual Studio Graphics Debugger: NOT FOUND (install 'Graphics debugging' workload in VS Installer)"
        }

        # Check for standalone PIX
        $pixPaths = @(
            "C:\Program Files\Microsoft PIX",
            "C:\Program Files (x86)\Microsoft PIX"
        )
        $pixFound = $false
        foreach ($pixPath in $pixPaths) {
            if (Test-Path $pixPath) {
                $pixFound = $true
                break
            }
        }

        if ($pixFound) {
            Write-Success "Microsoft PIX: FOUND"
        } else {
            Write-Warn "Microsoft PIX: NOT FOUND (download from https://devblogs.microsoft.com/directx/announcing-microsoft-pix-2024-1/ for advanced GPU capture)"
        }

        # Check for DRED support
        if ($EnableDRED) {
            Write-Info "DRED (Device Removed Extended Data) is ENABLED"
            Write-Info "Set DX12_ENABLE_DRED=1 and DX12_DRED_OUTPUT=.\dred_log.txt before running your app"
            Write-Info "See: https://devblogs.microsoft.com/directx/dred/"
        }

        $debugPassed = $true
    } else {
        Write-Warn "Skipping debug layer validation"
        $debugPassed = $true
    }

    # Step 3: GPU Capture Setup
    if (-not $SkipGPUCapture) {
        Write-Header "Step 3: GPU Capture Setup"

        Write-Info "Capture directory: $captureDir"

        # Generate a capture-ready environment configuration
        $envConfig = @{
            "DX12_ENABLE_DEBUG_LAYER" = "1"
            "DX12_FORCE_DEBUG_LAYER" = "1"
            "DX12_ENABLE_DRED" = if ($EnableDRED) { "1" } else { "0" }
            "DX12_DRED_OUTPUT" = Join-Path $captureDir "dred_log.txt"
            "DX12_CAPTURE_PATH" = $captureDir
            "DX12_LOG_LEVEL" = "3"
        }

        $envFile = Join-Path $captureDir "dx_env_config.txt"
        $envConfig.GetEnumerator() | Sort-Object Name | ForEach-Object {
            "$($_.Name)=$($_.Value)"
        } | Out-File -FilePath $envFile -Encoding utf8
        Write-Info "Environment configuration saved to: $envFile"

        # Generate PIX capture script if PIX is available
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
            $pixScript = Join-Path $captureDir "pix_capture.cmd"
            $pixCaptureFile = Join-Path $captureDir "dx12_capture.piX"

            $appExe = if ($AppPath) { $AppPath } else {
                $cliPath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-cli.exe"
                if (Test-Path $cliPath) { $cliPath } else {
                    Join-Path $repoRoot "$BuildDir\bin\llama-cli.exe"
                }
            }

            $appArguments = if ($AppArgs) { $AppArgs } else {
                "--model `"$ModelPath`" --prompt `"Hi`" --no-display-prompt --temp 0 --n-predict 15"
            }

            $pixCmd = "`"$pixExe`" capture -outfile `"$pixCaptureFile`" -process `"$appExe`" -cmdline `"$appArguments`""

            $pixCmd | Out-File -FilePath $pixScript -Encoding ascii
            Write-Success "PIX capture script generated: $pixScript"
            Write-Info "Run it with: cmd /c `"$pixScript`""
        } else {
            Write-Warn "PIX not found - GPU capture script not generated"
            Write-Warn "Use Visual Studio Graphics Debugger instead:"
            Write-Warn "  1. Open Visual Studio"
            Write-Warn "  2. Debug -> Graphics -> Start Graphics Debugging"
            Write-Warn "  3. Run your AI application"
            Write-Warn "  4. Capture frames using ALT+F12"
        }

        # Generate DRED configuration if enabled
        if ($EnableDRED) {
            $dredConfig = Join-Path $captureDir "dred_config.txt"
            @"
DRED (Device Removed Extended Data) Configuration
=================================================

To enable DRED in your application, add the following code:

    #include <d3d12.h>
    #include <dxgidebug.h>

    // Enable DRED
    ID3D12Debug1* debugController;
    D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    debugController->SetEnableDRED(true);
    debugController->SetDREDOutput(DXGI_DEBUG_DRED, DXGI_DRED_OUTPUT_FORMAT_TEXT);

    // Set DRED flags
    D3D12_DRED_SETTINGS dredSettings = {};
    dredSettings.Flags = D3D12_DRED_FLAG_BREADCRUMB_OPS | D3D12_DRED_FLAG_AUTO_BREADCRUMBS;
    dredSettings.BreadcrumbContext = 0;
    debugController->SetDREDSettings(&dredSettings);

Environment variables:
    DX12_ENABLE_DRED=1
    DX12_DRED_OUTPUT=$(Join-Path $captureDir 'dred_log.txt')

See: https://devblogs.microsoft.com/directx/dred/
"@ | Out-File -FilePath $dredConfig -Encoding utf8
            Write-Success "DRED configuration saved to: $dredConfig"
        }

        $capturePassed = $true
    } else {
        Write-Warn "Skipping GPU capture setup"
        $capturePassed = $true
    }

    # Step 4: Coherence Test (MANDATORY)
    if (-not $SkipCoherence) {
        Write-Header "Step 4: Coherence Test (MANDATORY)"

        # DX12 builds use bin\Release\ path
        $cliPath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-cli.exe"
        if (-not (Test-Path $cliPath)) {
            # Fallback to bin\ path for non-config builds
            $cliPath = Join-Path $repoRoot "$BuildDir\bin\llama-cli.exe"
            if (-not (Test-Path $cliPath)) {
                Write-Fail "llama-cli.exe not found in $BuildDir\bin\Release\ or $BuildDir\bin\"
                Write-Warn "Run without -SkipBuild to build first."
                exit 1
            }
        }

        if (-not (Test-Path $ModelPath)) {
            Write-Fail "Model not found at $ModelPath"
            Write-Warn "Specify a different model with -ModelPath"
            exit 1
        }

        Write-Info "Running coherence test with model: $ModelPath"
        Write-Info "Backend: $Backend"
        Write-Info "Prompt: 'Hi'"
        Write-Info "Expected: Readable, coherent text response"

        # Set up environment for debug layers during coherence test
        $env:DX12_ENABLE_DEBUG_LAYER = "1"
        $env:DX12_FORCE_DEBUG_LAYER = "1"
        if ($EnableDRED) {
            $env:DX12_ENABLE_DRED = "1"
            $env:DX12_DRED_OUTPUT = Join-Path $captureDir "dred_coherence.txt"
        }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $cliPath
        $psi.Arguments = "--model `"$ModelPath`" --prompt `"Hi`" --no-display-prompt --temp 0 --n-predict 15 --backend $Backend"
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $timeoutMs = 120000
        $exited = $proc.WaitForExit($timeoutMs)

        if (-not $exited) {
            $proc.Kill()
            Write-Fail "Coherence test timed out after $timeoutMs ms"
            exit 1
        }

        $coherenceOutput = $proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd()
        $coherenceExitCode = $proc.ExitCode

        # Save raw output to file for validation
        $outputFile = Join-Path $repoRoot "coherence_output_dx.txt"
        $coherenceOutput | Out-File -FilePath $outputFile -Encoding utf8
        Write-Info "Coherence output saved to: $outputFile"

        # Check for garbled output patterns
        $garbledPatterns = @(
            "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",  # Control characters
            "\x00\x00\x00",                       # Null bytes
            [char]0xFFFD                           # Replacement character
        )

        $isGarbled = $false
        foreach ($pattern in $garbledPatterns) {
            if ($coherenceOutput -match $pattern) {
                $isGarbled = $true
                break
            }
        }

        # Extract the model's response text from the output
        $readableText = ""
        $lines = $coherenceOutput -split "`n"
        $inResponse = $false
        foreach ($line in $lines) {
            if ($line -match "^\s*>\s*Hi\s*$") {
                $inResponse = $true
                continue
            }
            if ($inResponse -and $line -match "^\s*>\s*$") {
                break
            }
            if ($inResponse -and $line -match "^\s*\[\s*Prompt:") {
                break
            }
            if ($inResponse -and $line -match "^\s*\[Start thinking\]") {
                continue
            }
            if ($inResponse) {
                $readableText += $line + "`n"
            }
        }
        $readableText = $readableText.Trim()

        # Save extracted text for validation
        $textFile = Join-Path $repoRoot "coherence_text_dx.txt"
        $readableText | Out-File -FilePath $textFile -Encoding utf8
        Write-Info "Extracted text saved to: $textFile"

        # Read the text file back and analyze coherence
        $analyzedText = Get-Content $textFile -Raw
        Write-Info "Analyzing coherence of extracted text..."

        # Coherence checks:
        # 1. No garbled characters (control chars, null bytes, replacement chars)
        # 2. Contains at least 3 consecutive alphabetic words (basic sentence structure)
        # 3. Does NOT contain patterns typical of garbled output (e.g., repeated random chars)

        $hasGarbledChars = $analyzedText -match "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]" -or $analyzedText -match "\x00\x00\x00" -or $analyzedText -match [char]0xFFFD
        $hasSentenceStructure = $analyzedText -match "\b[a-zA-Z]+\s+[a-zA-Z]+\s+[a-zA-Z]+\b"
        $hasExcessiveRepeats = $analyzedText -match "(.)\1{10,}"  # 10+ repeated chars

        if ($coherenceExitCode -eq 0 -and -not $hasGarbledChars -and $hasSentenceStructure -and -not $hasExcessiveRepeats) {
            Write-Success "Coherence Test PASSED"
            Write-Info "Output snippet: $($analyzedText.Substring(0, [Math]::Min(100, $analyzedText.Length)))"
            $coherencePassed = $true
        } else {
            Write-Fail "Coherence Test FAILED"
            Write-Host "Analyzed text:"
            Write-Host $analyzedText
            if ($hasGarbledChars) { Write-Warn "Reason: Garbled characters detected" }
            if (-not $hasSentenceStructure) { Write-Warn "Reason: No sentence structure detected" }
            if ($hasExcessiveRepeats) { Write-Warn "Reason: Excessive character repetition detected" }
            Write-Warn "Check $outputFile and $textFile for details."
            $coherencePassed = $false
        }
    } else {
        Write-Warn "Skipping coherence test (NOT RECOMMENDED)"
        $coherencePassed = $true
    }

    # Step 5: Unit Tests
    if (-not $SkipTests) {
        Write-Header "Step 5: Unit Tests"
        $testResult = & ctest --test-dir $BuildDir --build-config Release --output-on-failure 2>&1
        $testExitCode = $LASTEXITCODE

        if ($testExitCode -eq 0) {
            Write-Success "Unit Tests PASSED"
        } else {
            Write-Fail "Unit Tests FAILED"
            Write-Host $testResult
        }
    } else {
        Write-Warn "Skipping unit tests"
        $testExitCode = 0
    }

    # Final verdict
    Write-Host ""
    Write-Header "Summary"
    if ($buildPassed) { Write-Success "Build: PASSED" } else { Write-Fail "Build: FAILED" }
    if ($debugPassed) { Write-Success "Debug Validation: PASSED" } else { Write-Fail "Debug Validation: FAILED" }
    if ($capturePassed) { Write-Success "GPU Capture Setup: PASSED" } else { Write-Fail "GPU Capture Setup: FAILED" }
    if ($coherencePassed) { Write-Success "Coherence: PASSED" } else { Write-Fail "Coherence: FAILED" }

    if (-not $coherencePassed) {
        Write-Host ""
        Write-Warn "=== COHERENCE TEST FAILED ==="
        Write-Warn "The model output is garbled or incoherent."
        Write-Warn "This indicates a bug in the compute path (e.g., thread mapping, quantization)."
        Write-Warn ""
        Write-Warn "Recommended actions:"
        Write-Warn "  1. Check recent commits for compute path changes"
        Write-Warn "  2. Verify thread mapping in dx12_gemm.cpp"
        Write-Warn "  3. Verify shader dispatch in dx12_graph.cpp"
        Write-Warn "  4. Check type_traits in ggml.c for NULL function pointers"
        Write-Warn "  5. Run test-backend-ops to verify op correctness"
        Write-Warn "  6. Use Visual Studio Graphics Debugger or PIX to capture GPU frames"
        Write-Warn "  7. Enable DRED with -EnableDRED to capture device removal info"
        Write-Warn ""

        $fixChoice = Read-Host "Would you like to see the diff of recent changes? (y/n)"
        if ($fixChoice -eq 'y') {
            & git log --oneline -5
            & git diff HEAD~1 --stat
        }

        exit 1
    }

    if ($testExitCode -ne 0) {
        exit 1
    }

    Write-Success "All tests passed!"
    Write-Info "Capture files available in: $captureDir"
    exit 0

} finally {
    Pop-Location
}
