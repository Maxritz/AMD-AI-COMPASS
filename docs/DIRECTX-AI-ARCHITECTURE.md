# DirectX AI System Architecture

## Overview

This document describes the complete architecture of a DirectX AI system running on Windows, covering all components from the OS level down to the GPU hardware. It serves as a reference for building, debugging, and optimizing DirectX-based AI applications.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  GGUF Model  │  │  ONNX Model  │  │ Custom D3D12 Compute│  │
│  │ (llama.cpp)  │  │ (DirectML)   │  │    Application      │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Runtime/API Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   llama.cpp  │  │  Windows ML  │  │    DirectML         │  │
│  │  DX12 Backend│  │  (ONNX RT)   │  │   (Low-level API)   │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   DirectX 12 Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ D3D12 Device │  │ Command Lists│  │  Resource Manager   │  │
│  │              │  │              │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Driver Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   GPU Driver │  │ DXGI Runtime │  │  Kernel Mode Driver │  │
│  │ (AMD/NVIDIA/ │  │              │  │                     │  │
│  │    Intel)    │  │              │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Hardware Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   GPU (VRAM) │  │   System     │  │    CPU (System)     │  │
│  │              │  │    RAM       │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Application Layer

#### GGUF Models (llama.cpp DX12 Backend)
- **Format**: GGUF (GPT-Generated Unified Format) - binary tensor format
- **Loading**: Direct file mapping, optional DirectStorage async loading
- **Tensor Flow**: 
  - Model weights loaded into D3D12 DEFAULT heaps (GPU-local memory)
  - Quantization handled in shader code (Q4_0, Q8_0, etc.)
  - Tensor operations dispatched as compute shaders
- **Compute Path**: 
  - `mmvq.cu` → `dx12_gemm.cpp` (matrix-vector quantization)
  - `vecdotq.cuh` → `dx12_quantize.cpp` (vector dot products)
  - Thread mapping: `threads_per_kblock = qi / vdr` (critical for correctness)

#### ONNX Models (DirectML)
- **Format**: ONNX (Open Neural Network Exchange) - protobuf-based
- **Loading**: ONNX Runtime with DirectML execution provider
- **Tensor Flow**:
  - Weights uploaded via D3D12 UPLOAD heaps → DEFAULT heaps
  - DirectML operators compiled to D3D12 compute shaders
  - Graph-based or layer-by-layer execution
- **Compute Path**:
  - DML operators → D3D12 compute dispatches
  - Automatic shader generation from ONNX graph
  - Resource binding via descriptor heaps

#### Custom D3D12 Compute
- **Format**: Any (custom model format)
- **Loading**: Application-specific
- **Tensor Flow**: Direct HLSL compute shaders
- **Compute Path**: Manual shader dispatch

### 2. Runtime/API Layer

#### DirectML
- **Role**: Low-level ML API for DirectX 12
- **Operators**: 50+ ML primitives (convolution, GEMM, activation, etc.)
- **Workflow**:
  1. Create D3D12 device + command queue
  2. Create DirectML device (wraps D3D12 device)
  3. Create operator instances (with specific parameters)
  4. Record binding + execution into command lists
  5. Execute on queue

#### Windows ML
- **Role**: High-level API for ONNX models
- **Backend**: Uses DirectML for GPU acceleration
- **Features**: Model loading, binding, evaluation
- **Limitations**: Less control than DirectML, higher overhead

#### llama.cpp DX12 Backend
- **Role**: Custom DX12 backend for GGUF models
- **Features**: 
  - Custom shader pipeline (LDS-tiled GEMM)
  - FlashAttention variants (single-query, multi-query, 2D-tiled)
  - MoE expert routing (MUL_MAT_ID)
  - DirectStorage async loading
  - VRAM budget management (DX12_MAX_VRAM_PCT)

### 3. DirectX 12 Layer

#### D3D12 Device
- **Creation**: `D3D12CreateDevice()` with feature level 11_1+
- **Debug Layer**: `ID3D12Debug::EnableDebugLayer()` (development only)
- **DRED**: Device Removed Extended Data for crash diagnostics

#### Command Lists
- **Types**: Direct, Bundle, Compute, Copy
- **Recording**: CPU-side command recording
- **Execution**: GPU-side command execution via command queues
- **Synchronization**: Fences, events, barriers

#### Resource Manager
- **Heaps**: DEFAULT (GPU-local), UPLOAD (CPU-write), READBACK (CPU-read)
- **Resources**: Buffers, textures, descriptor heaps
- **Lifetime**: Explicit management (no GC)
- **Barriers**: State transitions (D3D12_RESOURCE_STATE_*)

#### Descriptor Heaps
- **Types**: CBV_SRV_UAV, Sampler, RTV, DSV
- **Visibility**: Shader-visible vs. CPU-only
- **Allocation**: Manual or via descriptor heap allocator

### 4. Driver Layer

#### GPU Driver
- **AMD**: AMD Software: Adrenalin Edition
- **NVIDIA**: GeForce Game Ready / Studio Driver
- **Intel**: Intel Graphics Driver
- **Role**: Translates D3D12 calls to GPU-specific commands

#### DXGI Runtime
- **Role**: DirectX Graphics Infrastructure
- **Features**: Adapter enumeration, swap chain management, HDR
- **Components**: IDXGIFactory, IDXGIAdapter, IDXGISwapChain

#### Kernel Mode Driver (KMD)
- **Role**: Kernel-level GPU scheduling and memory management
- **TDR**: Timeout Detection and Recovery (default 2s GPU timeout)

### 5. Hardware Layer

#### GPU (VRAM)
- **Memory**: GDDR6/GDDR6X/HBM2e (varies by vendor)
- **Compute Units**: Stream processors / CUDA cores / Execution units
- **Cache Hierarchy**: L1/L2/L3 cache (varies by architecture)
- **Wavefronts**: AMD (64 threads), NVIDIA (32 threads), Intel (16/32 threads)

#### System RAM
- **Role**: CPU-side memory for model loading, preprocessing
- **Bandwidth**: DDR4/DDR5 (varies by system)
- **Transfer**: PCIe bus (GPU-CPU communication)

#### CPU
- **Role**: Command recording, scheduling, preprocessing
- **Architecture**: x86-64 or ARM64
- **Cores**: Varies by system

## Shader Compilation Pipeline

```
HLSL Source (.hlsl)
       │
       ▼
┌──────────────┐
│   DXC (DirectX Shader Compiler)   │
│   - HLSL → DXIL (DirectX Intermediate Language)   │
│   - Shader model 6.0+ support   │
│   - Wave-level operations   │
└──────────────┘
       │
       ▼
┌──────────────┐
│   DXIL → GPU Bytecode   │
│   - Vendor-specific compilation   │
│   - AMD: GCN/RDNA assembly   │
│   - NVIDIA: SASS assembly   │
│   - Intel: G4/G7 assembly   │
└──────────────┘
       │
       ▼
┌──────────────┐
│   GPU Execution   │
│   - Compute shader dispatch   │
│   - Thread groups (blocks)   │
│   - Shared memory (LDS)   │
└──────────────┘
```

### Shader Types
- **Compute Shaders**: Primary for ML operations (GEMM, quantization, etc.)
- **Vertex/Pixel Shaders**: For visualization (not used in pure ML)
- **Wave Shaders**: For wave-level operations (SM 6.0+)

### Compilation Tools
- **DXC**: DirectX Shader Compiler (modern, open-source)
- **FXC**: Legacy HLSL compiler (deprecated)
- **Integration**: CMake + compile_shaders.ps1

## Memory Management

### VRAM Hierarchy
```
┌─────────────────────────────────────┐
│  GPU Registers (fastest, smallest)  │
├─────────────────────────────────────┤
│  L1 Cache / LDS (shared memory)     │
├─────────────────────────────────────┤
│  L2 Cache                           │
├─────────────────────────────────────┤
│  VRAM (GDDR6/6X/HBM2e)              │
├─────────────────────────────────────┤
│  System RAM (via PCIe)              │
└─────────────────────────────────────┘
```

### Resource Allocation Patterns
1. **Weight Tensors**: D3D12_HEAP_TYPE_DEFAULT (GPU-local)
2. **Upload Buffers**: D3D12_HEAP_TYPE_UPLOAD (CPU-write, GPU-read)
3. **Readback Buffers**: D3D12_HEAP_TYPE_READBACK (GPU-write, CPU-read)
4. **Staging**: For CPU-GPU data transfer

### VRAM Budget Management
- **DX12_MAX_VRAM_PCT**: Configurable ceiling (default 92%)
- **FITT**: Automatic model fitting to available VRAM
- **CPU Split**: When model exceeds VRAM, some layers run on CPU

## Debugging and Profiling

### Debug Layers
1. **D3D12 Debug Layer**: Validates D3D12 API usage
2. **DirectML Debug Layer**: Validates DirectML API usage
3. **Graphics Tools FOD**: System component for debug layers

### DRED (Device Removed Extended Data)
- **Auto-Breadcrumbs**: Tracks GPU operations before TDR
- **Page Fault Reporting**: Identifies GPU memory access violations
- **Performance**: 2-5% overhead (off by default)

### GPU Capture Tools
1. **Microsoft PIX**: Primary GPU profiler
   - GPU captures, timing, memory analysis
   - Shader debugging, pipeline inspection
2. **Visual Studio Graphics Debugger**: Integrated in VS 2022
   - Frame capture, resource inspection
   - Pipeline state analysis
3. **Vendor Tools**:
   - AMD: Radeon GPU Profiler (RGP)
   - NVIDIA: Nsight Graphics
   - Intel: Graphics Performance Analyzers (GPA)

### TDR (Timeout Detection and Recovery)
- **Default Timeout**: 2 seconds
- **Configuration**: TdrLevel, TdrDelay in registry
- **AI Impact**: Large models can trigger TDR during long operations
- **Mitigation**: Chunking work, increasing TDR delay, using DRED

## Model Format Comparison

| Feature | GGUF (llama.cpp) | ONNX (DirectML) | Custom D3D12 |
|---------|------------------|-----------------|--------------|
| Format | Binary tensor | Protobuf | Any |
| Quantization | Built-in (Q4_0, Q8_0, etc.) | Limited | Manual |
| Loading | Direct file mapping | ONNX Runtime | Custom |
| Debug Layer | D3D12 only | D3D12 + DML | D3D12 only |
| VRAM Budget | Built-in (DX12_MAX_VRAM_PCT) | ONNX RT | Manual |
| TDR Mitigation | Built-in chunking | ONNX RT | Manual |
| Profiling | PIX, VS Graphics | PIX, VS Graphics | PIX, VS Graphics |
| Vendor Tools | All | All | All |

## System Requirements

### Minimum
- **OS**: Windows 10 1903+ or Windows 11
- **GPU**: DirectX 12 compatible
- **SDK**: Windows SDK 10.0.18362+
- **VRAM**: 8GB (minimum for small models)

### Recommended
- **OS**: Windows 11 22H2+
- **GPU**: RDNA2/RDNA3/NVIDIA RTX/Intel Arc
- **SDK**: Windows SDK 10.0.22621+ (Windows 11 SDK)
- **VRAM**: 16GB+ (for medium models)
- **CPU**: Modern x86-64 or ARM64

### Development
- **IDE**: Visual Studio 2022 17.0+
- **Workloads**: 
  - Desktop development with C++
  - Graphics debugging (for PIX/VS Graphics)
- **Tools**:
  - DirectX 12 Agility SDK
  - DXC (DirectX Shader Compiler)
  - Graphics Tools FOD (for debug layers)

## Common Issues and Solutions

### 1. Garbled Output
- **Cause**: Thread mapping bug in compute shaders
- **Check**: `threads_per_kblock = qi / vdr` (not `qk / qi`)
- **Debug**: Enable D3D12 debug layer, check shader constants

### 2. GPU Hangs (TDR)
- **Cause**: Long-running GPU operations
- **Check**: DRED auto-breadcrumbs, PIX capture
- **Fix**: Chunk work, increase TDR delay, use async submission

### 3. VRAM Exhaustion
- **Cause**: Model too large for GPU memory
- **Check**: VRAM budget ceiling, allocation tracking
- **Fix**: Use CPU split, reduce batch size, use smaller model

### 4. NaN/Inf in Output
- **Cause**: Invalid tensor data, overflow, division by zero
- **Check**: DirectML debug layer, shader validation
- **Fix**: Check tensor shapes, data types, normalization

### 5. Device Removal
- **Cause**: Driver crash, hardware fault, invalid API usage
- **Check**: DRED page fault reporting, GPU capture
- **Fix**: Validate API calls, check resource lifetimes

## Performance Optimization Guidelines

### 1. Shader Optimization
- Use LDS (shared memory) for tile-based GEMM
- Minimize thread divergence
- Use wave-level operations (SM 6.0+)
- Profile with PIX shader debugger

### 2. Memory Optimization
- Use appropriate heap types (DEFAULT for weights)
- Minimize CPU-GPU transfers
- Use resource aliasing for temporary buffers
- Implement VRAM budget management

### 3. Command List Optimization
- Bundle static commands
- Use multi-threaded command recording
- Minimize state changes
- Use command list re-use

### 4. Synchronization Optimization
- Minimize GPU stalls
- Use async compute where possible
- Proper barrier placement
- Fence-based synchronization

## References

- [DirectML Documentation](https://learn.microsoft.com/en-us/windows/ai/directml/dml)
- [DirectML Debug Layer](https://learn.microsoft.com/en-us/windows/ai/directml/dml-debug-layer)
- [DRED Blog Post](https://devblogs.microsoft.com/directx/dred/)
- [D3D12 Programming Guide](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide)
- [HLSL Documentation](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)
- [Windows AI Documentation](https://learn.microsoft.com/en-us/windows/ai/)
- [PIX Documentation](https://devblogs.microsoft.com/directx/pix/)
