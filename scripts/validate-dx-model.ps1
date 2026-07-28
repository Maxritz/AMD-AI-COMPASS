<#
.SYNOPSIS
    Validate DirectX AI model integrity and tensor data.

.DESCRIPTION
    This script validates that AI models (GGUF, ONNX, or other formats) are
    properly loaded and that tensor data is not corrupted. It uses the
    DirectX application to load the model and then inspects the output for
    signs of data corruption, zero-filled tensors, or invalid values.

    For GGUF models: Checks llama-cli output for coherence
    For ONNX models: Checks DirectML inference output for NaN/Inf values
    For custom formats: Runs the specified validation command

.PARAMETER AppPath
    Path to the AI application executable

.PARAMETER ModelPath
    Path to the model file

.PARAMETER ModelFormat
    Model format: "gguf" (default), "onnx", or "custom"

.PARAMETER ValidationCmd
    Custom validation command (required for -ModelFormat custom)

.PARAMETER TempDir
    Temporary directory for validation artifacts

.EXAMPLE
    .\validate-dx-model.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf"
    Validate a GGUF model using llama-cli.

.EXAMPLE
    .\validate-dx-model.ps1 -AppPath ".\my_onnx_app.exe" -ModelPath "model.onnx" -ModelFormat onnx
    Validate an ONNX model for NaN/Inf values.

.EXAMPLE
    .\validate-dx-model.ps1 -AppPath ".\my_app.exe" -ModelPath "model.bin" -ModelFormat custom -ValidationCmd "my_app --validate model.bin"
    Validate using a custom command.

.NOTES
    This script checks for:
    - Model loading success
    - Tensor data integrity (no NaN, Inf, or zero-filled tensors)
    - Semantic coherence of output (for text models)
    - Memory allocation success (no VRAM exhaustion)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$AppPath,

    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [ValidateSet("gguf", "onnx", "custom")]
    [string]$ModelFormat = "gguf",

    [string]$ValidationCmd = "",

    [string]$TempDir = ".\validation_temp\"
)

$ErrorActionPreference = "Stop"

function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Fail { Write-Host $args -ForegroundColor Red }
function Write-Warn { Write-Host $args -ForegroundColor Yellow }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

# Create temp directory
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

Write-Info "=== DirectX AI Model Validation ==="
Write-Info "App: $AppPath"
Write-Info "Model: $ModelPath"
Write-Info "Format: $ModelFormat"
Write-Info ""

# Check if app exists
if (-not (Test-Path $AppPath)) {
    Write-Fail "Application not found: $AppPath"
    exit 1
}

# Check if model exists
if (-not (Test-Path $ModelPath)) {
    Write-Fail "Model not found: $ModelPath"
    exit 1
}

$validationPassed = $true

switch ($ModelFormat) {
    "gguf" {
        Write-Info "Validating GGUF model..."

        # Run a simple prompt to check model loading and coherence
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $AppPath
        $psi.Arguments = "--model `"$ModelPath`" --prompt `"Hi`" --no-display-prompt --temp 0 --n-predict 15"
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $timeoutMs = 120000
        $exited = $proc.WaitForExit($timeoutMs)

        if (-not $exited) {
            $proc.Kill()
            Write-Fail "Model validation timed out"
            exit 1
        }

        $output = $proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd()
        $exitCode = $proc.ExitCode

        # Save output
        $outputFile = Join-Path $TempDir "gguf_validation_output.txt"
        $output | Out-File -FilePath $outputFile -Encoding utf8

        # Check for common issues
        $hasNaN = $output -match "(?i)(nan|inf|qnan|snan)"
        $hasZeroTensors = $output -match "(?i)(zero tensor|all zeros|empty tensor)"
        $hasVRAMExhaustion = $output -match "(?i)(out of memory|vram|allocation failed)"
        $hasLoadError = $output -match "(?i)(error|failed to load|cannot open)"

        # Check for garbled output
        $hasGarbledChars = $output -match "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
        $hasSentenceStructure = $output -match "\b[a-zA-Z]+\s+[a-zA-Z]+\s+[a-zA-Z]+\b"

        if ($exitCode -ne 0) {
            Write-Fail "Model failed to load/run (exit code: $exitCode)"
            $validationPassed = $false
        }

        if ($hasNaN) {
            Write-Fail "NaN or Inf values detected in output"
            $validationPassed = $false
        }

        if ($hasZeroTensors) {
            Write-Fail "Zero-filled tensors detected"
            $validationPassed = $false
        }

        if ($hasVRAMExhaustion) {
            Write-Fail "VRAM exhaustion detected"
            $validationPassed = $false
        }

        if ($hasLoadError) {
            Write-Fail "Model loading error detected"
            $validationPassed = $false
        }

        if ($hasGarbledChars) {
            Write-Fail "Garbled characters detected in output"
            $validationPassed = $false
        }

        if (-not $hasSentenceStructure) {
            Write-Fail "No coherent sentence structure in output"
            $validationPassed = $false
        }

        if ($validationPassed) {
            Write-Success "GGUF model validation PASSED"
        }

        # Extract and display output snippet
        $lines = $output -split "`n"
        $inResponse = $false
        $responseText = ""
        foreach ($line in $lines) {
            if ($line -match "^\s*>\s*Hi\s*$") {
                $inResponse = $true
                continue
            }
            if ($inResponse -and $line -match "^\s*>\s*$") { break }
            if ($inResponse -and $line -match "^\s*\[\s*Prompt:") { break }
            if ($inResponse) {
                $responseText += $line + "`n"
            }
        }
        $responseText = $responseText.Trim()
        if ($responseText) {
            Write-Info "Output: $($responseText.Substring(0, [Math]::Min(100, $responseText.Length)))"
        }
    }

    "onnx" {
        Write-Info "Validating ONNX model..."

        # Run the app with a simple inference to check for NaN/Inf
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $AppPath
        $psi.Arguments = "--model `"$ModelPath`" --validate"
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $timeoutMs = 120000
        $exited = $proc.WaitForExit($timeoutMs)

        if (-not $exited) {
            $proc.Kill()
            Write-Fail "ONNX validation timed out"
            exit 1
        }

        $output = $proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd()
        $exitCode = $proc.ExitCode

        $outputFile = Join-Path $TempDir "onnx_validation_output.txt"
        $output | Out-File -FilePath $outputFile -Encoding utf8

        $hasNaN = $output -match "(?i)(nan|inf|qnan|snan)"
        $hasZeroTensors = $output -match "(?i)(zero tensor|all zeros|empty tensor)"
        $hasLoadError = $output -match "(?i)(error|failed to load|cannot open|invalid model)"

        if ($exitCode -ne 0) {
            Write-Fail "ONNX model failed to load/run (exit code: $exitCode)"
            $validationPassed = $false
        }

        if ($hasNaN) {
            Write-Fail "NaN or Inf values detected in model output"
            $validationPassed = $false
        }

        if ($hasZeroTensors) {
            Write-Fail "Zero-filled tensors detected"
            $validationPassed = $false
        }

        if ($hasLoadError) {
            Write-Fail "Model loading error detected"
            $validationPassed = $false
        }

        if ($validationPassed) {
            Write-Success "ONNX model validation PASSED"
        }
    }

    "custom" {
        if (-not $ValidationCmd) {
            Write-Fail "ValidationCmd is required for custom model format"
            exit 1
        }

        Write-Info "Running custom validation command..."
        Write-Info "Command: $ValidationCmd"

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "cmd.exe"
        $psi.Arguments = "/c $ValidationCmd"
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $timeoutMs = 120000
        $exited = $proc.WaitForExit($timeoutMs)

        if (-not $exited) {
            $proc.Kill()
            Write-Fail "Custom validation timed out"
            exit 1
        }

        $output = $proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd()
        $exitCode = $proc.ExitCode

        $outputFile = Join-Path $TempDir "custom_validation_output.txt"
        $output | Out-File -FilePath $outputFile -Encoding utf8

        if ($exitCode -ne 0) {
            Write-Fail "Custom validation failed (exit code: $exitCode)"
            Write-Host $output
            $validationPassed = $false
        } else {
            Write-Success "Custom validation PASSED"
        }
    }
}

# Summary
Write-Host ""
Write-Info "=== Validation Summary ==="
if ($validationPassed) {
    Write-Success "Model validation: PASSED"
    Write-Info "Validation artifacts in: $TempDir"
    exit 0
} else {
    Write-Fail "Model validation: FAILED"
    Write-Info "Validation artifacts in: $TempDir"
    exit 1
}
