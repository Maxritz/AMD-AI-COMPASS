# AI-Compass-DX

DirectX AI development toolkit for building, testing, and debugging AI applications that use DirectX 12 and DirectML.

## Overview

AI-Compass-DX provides a comprehensive workflow for DirectX AI development, including:

1. **Build verification** - Ensures DirectX 12 backends compile correctly
2. **Coherence testing** - Validates model output is semantically correct (not garbled)
3. **Debug layer validation** - Checks D3D12 and DirectML debug layer availability
4. **GPU capture setup** - Configures PIX and Visual Studio Graphics Debugger for frame capture
5. **DRED support** - Enables Device Removed Extended Data for crash debugging
6. **Model validation** - Validates GGUF, ONNX, and custom model formats
7. **Unit test execution** - Runs ctest suite

## Scripts

### `toolkit-dx.ps1` - Main Development Toolkit

The primary toolkit for building and testing DirectX AI applications.

```powershell
# Full build + test + debug validation
.\scripts\toolkit-dx.ps1

# Skip build, run only coherence test with debug layers
.\scripts\toolkit-dx.ps1 -SkipBuild -EnableDRED

# Run with custom model and capture path
.\scripts\toolkit-dx.ps1 -ModelPath "my_model.gguf" -CapturePath ".\my_captures\"

# Skip everything except debug validation
.\scripts\toolkit-dx.ps1 -SkipBuild -SkipCoherence -SkipTests
```

**Parameters:**
- `-BuildDir` - Build directory (default: `build_dx12`)
- `-ModelPath` - GGUF model for coherence testing
- `-SkipBuild` - Skip the build step
- `-SkipCoherence` - Skip coherence test (NOT RECOMMENDED)
- `-SkipTests` - Skip unit tests
- `-SkipDebugValidation` - Skip debug layer validation
- `-SkipGPUCapture` - Skip GPU capture setup
- `-EnableDRED` - Enable Device Removed Extended Data
- `-CapturePath` - Directory for capture files
- `-Backend` - Backend for coherence check (`dx12` or `vulkan`)

### `run-dx-ai.ps1` - Generic AI Application Runner

Runs any DirectX AI application with proper debug configuration.

```powershell
# Run llama-cli with debug layers and DRED
.\scripts\run-dx-ai.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf" -EnableDebug -EnableDRED

# Run with PIX capture
.\scripts\run-dx-ai.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf" -Capture

# Run an ONNX model with DirectML
.\scripts\run-dx-ai.ps1 -AppPath ".\my_onnx_app.exe" -ModelPath "model.onnx" -Prompt "Hello" -EnableDebug
```

### `validate-dx-model.ps1` - Model Validation

Validates model integrity for different formats.

```powershell
# Validate GGUF model
.\scripts\validate-dx-model.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf"

# Validate ONNX model
.\scripts\validate-dx-model.ps1 -AppPath ".\my_onnx_app.exe" -ModelPath "model.onnx" -ModelFormat onnx

# Custom validation
.\scripts\validate-dx-model.ps1 -AppPath ".\my_app.exe" -ModelPath "model.bin" -ModelFormat custom -ValidationCmd "my_app --validate model.bin"
```

## Debugging Workflow

### 1. Enable Debug Layers

```powershell
# Set environment variables
$env:DX12_ENABLE_DEBUG_LAYER = "1"
$env:DX12_FORCE_DEBUG_LAYER = "1"
```

### 2. Enable DRED (for crash debugging)

```powershell
$env:DX12_ENABLE_DRED = "1"
$env:DX12_DRED_OUTPUT = ".\dred_log.txt"
```

### 3. GPU Capture with PIX

```powershell
# Using the toolkit
.\scripts\toolkit-dx.ps1 -CapturePath ".\captures\"

# Or using the runner
.\scripts\run-dx-ai.ps1 -AppPath ".\build_dx12\bin\Release\llama-cli.exe" -ModelPath "model.gguf" -Capture
```

### 4. GPU Capture with Visual Studio Graphics Debugger

1. Open Visual Studio 2022
2. Go to Debug → Graphics → Start Graphics Debugging
3. Run your AI application
4. Press ALT+F12 to capture frames
5. Inspect GPU resources and pipeline state

## Model Format Support

| Format | Description | Validation |
|--------|-------------|------------|
| GGUF | llama.cpp model format | Coherence test + tensor integrity |
| ONNX | Open Neural Network Exchange | NaN/Inf detection + tensor validation |
| Custom | Any other format | User-defined validation command |

## Requirements

- **Windows 10/11** with DirectX 12 support
- **DirectX 12 Agility SDK** - For latest D3D12 features
- **DXC** (DirectX Shader Compiler) - For HLSL compilation
- **Visual Studio 2022** with Graphics Debugging workload - For integrated GPU capture
- **Microsoft PIX** (optional) - For advanced GPU profiling and capture
- **CMake + MSVC** - For building DirectX backends

## Debug Layer Setup

For DirectML debug layers, ensure `dml.dll` is placed next to your application executable:

- Download from: https://learn.microsoft.com/en-us/windows/ai/directml/dml-debug-layer
- Set `DX12_ENABLE_DEBUG_LAYER=1` before running

## DRED Configuration

DRED (Device Removed Extended Data) captures breadcrumbs when the GPU device is removed:

```powershell
# In your application code:
ID3D12Debug1* debugController;
D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
debugController->SetEnableDRED(true);
debugController->SetDREDOutput(DXGI_DEBUG_DRED, DXGI_DRED_OUTPUT_FORMAT_TEXT);
```

See: https://devblogs.microsoft.com/directx/dred/

## License

MIT - same as upstream llama.cpp
