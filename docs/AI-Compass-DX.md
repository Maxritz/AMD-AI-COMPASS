# AI-Compass Unified Toolkit

Multi-backend AI development toolkit for building, testing, and debugging AI applications across HIP (ROCm), Vulkan, DirectX 12, and WinML (DirectML) compute backends.

## Overview

AI-Compass provides a unified workflow for multi-backend AI development, including:

1. **Build verification** - Ensures all backend compute paths compile correctly
2. **Coherence testing** - Validates model output is semantically correct (not garbled) across all backends
3. **Debug layer validation** - Checks D3D12, DirectML, Vulkan, and HIP debug layer availability
4. **GPU capture setup** - Configures PIX, RenderDoc, and other tools for frame capture
5. **DRED support** - Enables Device Removed Extended Data for crash debugging (DX12/WinML)
6. **Model validation** - Validates GGUF, ONNX, and custom model formats
7. **Cross-backend comparison** - Compares outputs across all available backends
8. **Unit test execution** - Runs ctest suite for each backend

## Supported Backends

| Backend | Description | Key Features |
|---------|-------------|--------------|
| HIP/ROCm | AMD GPU compute via HIP runtime | ROCm 7.x support, rocwmma v2+, RDNA4 optimizations |
| Vulkan | Cross-platform GPU compute via Vulkan | Vulkan SDK, GLSL compute shaders, RenderDoc capture |
| DirectX 12 | Microsoft GPU compute via D3D12 | DirectML, D3D12 debug layers, DRED, PIX capture |
| WinML | Windows ML via DirectML | ONNX model support, Windows 10/11 integration |

## Unified Scripts

### `toolkit-unified.ps1` - Unified Multi-Backend Toolkit

The primary unified entry point for building and testing AI applications across all backends.

```powershell
# Full test on all backends
.\scripts\toolkit-unified.ps1 -Backend all -Action test -ModelPath "model.gguf"

# Build all available backends
.\scripts\toolkit-unified.ps1 -Backend all -Action build

# Run coherence comparison across all backends
.\scripts\toolkit-unified.ps1 -Backend all -Action coherence -ModelPath "model.gguf" -Prompt "Hello"

# Debug mode with DRED on DX12 backend
.\scripts\toolkit-unified.ps1 -Backend dx12 -Action debug -ModelPath "model.gguf" -EnableDRED

# Validate system for all backends
.\scripts\toolkit-unified.ps1 -Backend all -Action validate

# GPU capture on all backends
.\scripts\toolkit-unified.ps1 -Backend all -Action capture -ModelPath "model.gguf"
```

### `validate-system.ps1` - System Validation

Validates that all AI compute backends are properly installed and configured.

```powershell
# Basic system validation
.\scripts\validate-system.ps1

# Detailed validation with version information
.\scripts\validate-system.ps1 -Detailed

# Attempt to fix common configuration issues
.\scripts\validate-system.ps1 -Fix
```

### `validate-model.ps1` - Model Validation

Validates AI models across all available backends and compares outputs.

```powershell
# Validate model on all backends
.\scripts\validate-model.ps1 -ModelPath "model.gguf"

# Validate on specific backends only
.\scripts\validate-model.ps1 -ModelPath "model.gguf" -Backends "hip,vulkan"

# Custom prompt and output directory
.\scripts\validate-model.ps1 -ModelPath "model.gguf" -Prompt "Explain quantum computing" -OutputDir ".\results\"
```

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
