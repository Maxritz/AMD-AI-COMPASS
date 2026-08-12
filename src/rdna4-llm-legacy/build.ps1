# RDNA4-LLM Build Script — Pure PowerShell + MSVC + Vulkan SDK
# Usage: .\build.ps1 [-Release] [-Debug] [-Clean]
param([switch]$Release, [switch]$Debug, [switch]$Clean)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir  = Join-Path $ScriptDir "build"
$GenDir    = Join-Path $BuildDir "generated"
$SpvDir    = Join-Path $BuildDir "spirv"
$ObjDir    = Join-Path $BuildDir "obj"
$OutExe    = Join-Path $BuildDir "rdna4-llm.exe"

# -- Detect Vulkan SDK -------------------------------------------------------
$VkRoot = $env:VULKAN_SDK
if (-not $VkRoot) { $VkRoot = (Get-ChildItem "C:\VulkanSDK" -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName }
if (-not $VkRoot) { Write-Error "Vulkan SDK not found. Install to C:\VulkanSDK\ or set VULKAN_SDK."; exit 1 }
$VkInc = Join-Path $VkRoot "Include"
$VkLib = Join-Path $VkRoot "Lib"
$Glslc = Join-Path $VkRoot "Bin\glslc.exe"
$SpvOpt = Join-Path $VkRoot "Bin\spirv-opt.exe"
Write-Host "[BUILD] Vulkan SDK: $VkRoot" -ForegroundColor Cyan

# -- Detect MSVC -------------------------------------------------------------
$VsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>$null
if (-not $VsPath) { Write-Error "Visual Studio not found."; exit 1 }
$MsvcVer = (Get-ChildItem "$VsPath\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
$MsvcBin = "$VsPath\VC\Tools\MSVC\$MsvcVer\bin\Hostx64\x64"
$WinSdk  = (Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\Include" -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } | Sort-Object Name -Descending | Select-Object -First 1).FullName
$WinLib  = (Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\Lib" -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } | Sort-Object Name -Descending | Select-Object -First 1).FullName
Write-Host "[BUILD] MSVC: $MsvcBin" -ForegroundColor Cyan

# Build config
if ($Debug) {
    $CfgName = "Debug"
    $CppFlags = "/std:c++17 /nologo /W3 /MTd /Od /Zi /RTC1"
    $LinkFlags = "/DEBUG:FULL"
} else {
    $CfgName = "Release"
    $CppFlags = "/std:c++17 /nologo /W3 /MT /O2 /arch:AVX2 /GL /GS- /fp:fast /DNDEBUG"
    $LinkFlags = "/OPT:REF /OPT:ICF /LTCG"
}

# -- Clean -------------------------------------------------------------------
if ($Clean) {
    Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
    Write-Host "[BUILD] Cleaned $BuildDir" -ForegroundColor Green
    exit 0
}

# -- Create directories ------------------------------------------------------
New-Item -ItemType Directory -Force -Path $BuildDir, $GenDir, $SpvDir, $ObjDir | Out-Null

# -- Helper -----------------------------------------------------------
$VcVarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $VcVarsAll)) { Write-Error "vcvarsall.bat not found at $VcVarsAll"; exit 1 }

# -- Compile shaders to SPIR-V -----------------------------------------------
Write-Host "[BUILD] Compiling shaders..." -ForegroundColor Yellow

$ShaderDir = Join-Path $ScriptDir "shaders"
$ShaderFiles = @{}
# Register all shader variants (name -> quant_defines, op_type, quant_type)
$ShaderRegistry = @(
    @{src="rms_norm";         def="";             op="OP_RMS_NORM";         qt="QUANT_FP16"},
    @{src="attn_qkv_fp16";    def="-DQUANT_FP16=1"; op="OP_ATTN_QKV";      qt="QUANT_FP16"},
    @{src="attn_qkv_q4_k";    def="-DQUANT_Q4_K=1"; op="OP_ATTN_QKV";      qt="QUANT_Q4_K"},
    @{src="attn_qkv_q6_k";    def="-DQUANT_Q6_K=1"; op="OP_ATTN_QKV";      qt="QUANT_Q6_K"},
    @{src="attn_qkv_q8_0";    def="-DQUANT_Q8_0=1"; op="OP_ATTN_QKV";      qt="QUANT_Q8_0"},
    @{src="attn_qkv_iq4_xs";  def="-DQUANT_IQ4_XS=1"; op="OP_ATTN_QKV";    qt="QUANT_IQ4_XS"},
    @{src="attn_compute";     def="";             op="OP_ATTN_COMPUTE";     qt="QUANT_FP16"},
    @{src="attn_output_fp16"; def="-DQUANT_FP16=1"; op="OP_ATTN_OUTPUT";   qt="QUANT_FP16"},
    @{src="attn_output_q4_k"; def="-DQUANT_Q4_K=1"; op="OP_ATTN_OUTPUT";   qt="QUANT_Q4_K"},
    @{src="attn_output_q6_k"; def="-DQUANT_Q6_K=1"; op="OP_ATTN_OUTPUT";   qt="QUANT_Q6_K"},
    @{src="attn_output_q8_0"; def="-DQUANT_Q8_0=1"; op="OP_ATTN_OUTPUT";   qt="QUANT_Q8_0"},
    @{src="attn_output_iq4_xs"; def="-DQUANT_IQ4_XS=1"; op="OP_ATTN_OUTPUT"; qt="QUANT_IQ4_XS"},
    @{src="ffn_gate_up_fp16"; def="-DQUANT_FP16=1"; op="OP_FFN_GATE_UP";   qt="QUANT_FP16"},
    @{src="ffn_gate_up_q4_k"; def="-DQUANT_Q4_K=1"; op="OP_FFN_GATE_UP";   qt="QUANT_Q4_K"},
    @{src="ffn_gate_up_q6_k"; def="-DQUANT_Q6_K=1"; op="OP_FFN_GATE_UP";   qt="QUANT_Q6_K"},
    @{src="ffn_gate_up_q8_0"; def="-DQUANT_Q8_0=1"; op="OP_FFN_GATE_UP";   qt="QUANT_Q8_0"},
    @{src="ffn_gate_up_iq4_xs"; def="-DQUANT_IQ4_XS=1"; op="OP_FFN_GATE_UP"; qt="QUANT_IQ4_XS"},
    @{src="ffn_down_fp16";    def="-DQUANT_FP16=1"; op="OP_FFN_DOWN";      qt="QUANT_FP16"},
    @{src="ffn_down_q4_k";    def="-DQUANT_Q4_K=1"; op="OP_FFN_DOWN";      qt="QUANT_Q4_K"},
    @{src="ffn_down_q6_k";    def="-DQUANT_Q6_K=1"; op="OP_FFN_DOWN";      qt="QUANT_Q6_K"},
    @{src="ffn_down_q8_0";    def="-DQUANT_Q8_0=1"; op="OP_FFN_DOWN";      qt="QUANT_Q8_0"},
    @{src="ffn_down_iq4_xs";  def="-DQUANT_IQ4_XS=1"; op="OP_FFN_DOWN";    qt="QUANT_IQ4_XS"},
    @{src="lm_head_fp16";     def="-DQUANT_FP16=1"; op="OP_LM_HEAD";       qt="QUANT_FP16"},
    @{src="lm_head_q4_k";     def="-DQUANT_Q4_K=1"; op="OP_LM_HEAD";       qt="QUANT_Q4_K"},
    @{src="lm_head_q6_k";     def="-DQUANT_Q6_K=1"; op="OP_LM_HEAD";       qt="QUANT_Q6_K"},
    @{src="lm_head_q8_0";     def="-DQUANT_Q8_0=1"; op="OP_LM_HEAD";       qt="QUANT_Q8_0"},
    @{src="lm_head_iq4_xs";   def="-DQUANT_IQ4_XS=1"; op="OP_LM_HEAD";     qt="QUANT_IQ4_XS"},
    @{src="token_embed";      def="";               op="OP_EMBEDDING_LOOKUP"; qt="QUANT_FP16"},
    @{src="token_embed_q8_0"; def="-DQUANT_Q8_0=1"; op="OP_EMBEDDING_LOOKUP"; qt="QUANT_Q8_0"},
    @{src="token_embed_q6_k"; def="-DQUANT_Q6_K=1"; op="OP_EMBEDDING_LOOKUP"; qt="QUANT_Q6_K"},
    @{src="token_embed_q4_k"; def="-DQUANT_Q4_K=1"; op="OP_EMBEDDING_LOOKUP"; qt="QUANT_Q4_K"},
    @{src="token_embed_iq4_xs"; def="-DQUANT_IQ4_XS=1"; op="OP_EMBEDDING_LOOKUP"; qt="QUANT_IQ4_XS"}
)

$WaveSizes = @(32, 64)
$ShaderHeaders = @()
$ShaderEntries = @()

foreach ($s in $ShaderRegistry) {
    foreach ($w in $WaveSizes) {
        $srcFile   = Join-Path $ShaderDir "$($s.src).comp"
        $spvFile   = Join-Path $SpvDir "$($s.src)_w${w}.spv"
        $headerFile = Join-Path $GenDir "$($s.src)_w${w}.h"
        $varName   = "shader_$($s.src)_w${w}"

        if (-not (Test-Path $srcFile)) {
            Write-Warning "Shader source not found: $srcFile"
            continue
        }

        $glslArgs = @(
            "-fshader-stage=compute",
            "--target-env=vulkan1.4",
            "--target-spv=spv1.4",
            "-DSUBGROUP_SIZE=$w",
            "-I", $ShaderDir
        )
        if ($s.def) { $glslArgs += $s.def.Split(" ") }

        $shaderName = "$($s.src)_w${w}"
        & $Glslc @glslArgs -o $spvFile $srcFile 2>&1 | ForEach-Object { Write-Host "  ${shaderName}: $_" }
        if ($LASTEXITCODE -ne 0) { Write-Error "glslc failed for $shaderName"; exit 1 }

        # Generate C header from SPIR-V
        $spvBytes = [System.IO.File]::ReadAllBytes($spvFile)
        $wordCount = [math]::Floor($spvBytes.Length / 4)
        $sb = [System.Text.StringBuilder]::new()
        [void]$sb.AppendLine("// Auto-generated from $($s.src)_w$w.spv")
        [void]$sb.AppendLine("// Words: $wordCount  Bytes: $($spvBytes.Length)")
        [void]$sb.AppendLine("#pragma once")
        [void]$sb.AppendLine("#include <stdint.h>")
        [void]$sb.AppendLine("#include <stddef.h>")
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("static const uint32_t ${varName}_data[] = {")
        for ($i = 0; $i -lt $spvBytes.Length; $i += 4) {
            $word = [BitConverter]::ToUInt32($spvBytes, $i)
            [void]$sb.Append("    0x$($word.ToString('X8')),")
            if (($i/4 + 1) % 16 -eq 0) { [void]$sb.AppendLine("") }
        }
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("};")
        [void]$sb.AppendLine("static const size_t ${varName}_size = $($spvBytes.Length);")
        [void]$sb.AppendLine("static const uint32_t ${varName}_word_count = $wordCount;")
        [System.IO.File]::WriteAllText($headerFile, $sb.ToString())

        $ShaderHeaders += $headerFile
        $ShaderEntries += @{
            name = "$($s.src)_w$w"
            var  = $varName
            op   = $s.op
            qt   = $s.qt
            wave = $w
            size = $spvBytes.Length
            words = $wordCount
        }
        Write-Host "  $($s.src)_w$w.h ($wordCount words)" -ForegroundColor DarkGray
    }
}

# Generate shader_registry.h
Write-Host "[BUILD] Generating shader_registry.h ($($ShaderEntries.Count) shaders)" -ForegroundColor Yellow
$regSb = [System.Text.StringBuilder]::new()
[void]$regSb.AppendLine("// Auto-generated shader registry -- do not edit")
[void]$regSb.AppendLine("#pragma once")
[void]$regSb.AppendLine("#include <stdint.h>")
[void]$regSb.AppendLine("#include <stddef.h>")
[void]$regSb.AppendLine("")
foreach ($h in $ShaderHeaders) {
    [void]$regSb.AppendLine("#include `"$(Split-Path $h -Leaf)`"")
}
[void]$regSb.AppendLine("")
[void]$regSb.AppendLine("struct embedded_shader_t {")
[void]$regSb.AppendLine("    const char*    name;")
[void]$regSb.AppendLine("    const uint32_t* data;")
[void]$regSb.AppendLine("    size_t          byte_size;")
[void]$regSb.AppendLine("    uint32_t        word_count;")
[void]$regSb.AppendLine("    uint32_t        subgroup_size;")
[void]$regSb.AppendLine("    int             quant_type;")
[void]$regSb.AppendLine("    int             op_type;")
[void]$regSb.AppendLine("};")
[void]$regSb.AppendLine("")
[void]$regSb.AppendLine("static const embedded_shader_t EMBEDDED_SHADERS[] = {")
foreach ($e in $ShaderEntries) {
    [void]$regSb.AppendLine("    { `"$($e.name)`", $($e.var)_data, $($e.var)_size, $($e.var)_word_count, $($e.wave), $($e.qt), $($e.op) },")
}
[void]$regSb.AppendLine("    { NULL, NULL, 0, 0, 0, 0, 0 }")
[void]$regSb.AppendLine("};")
$regFile = Join-Path $GenDir "shader_registry.h"
[System.IO.File]::WriteAllText($regFile, $regSb.ToString())
$ShaderHeaders += $regFile

# -- Compile C++ sources ------------------------------------------------------
Write-Host "[BUILD] Compiling sources ($CfgName)..." -ForegroundColor Yellow

$CppSources = @(
    "main.cpp",
    "vk_device.cpp",
    "vk_buffer.cpp",
    "vk_session.cpp",
    "vk_model.cpp",
    "vk_kv_cache.cpp",
    "vk_timeline.cpp",
    "vk_descriptor.cpp",
    "vk_validate.cpp",
    "gguf_parser.cpp",
    "sample.cpp"
)

$Defines = @(
    "VMA_STATIC_VULKAN_FUNCTIONS=0",
    "VMA_DYNAMIC_VULKAN_FUNCTIONS=1",
    "NOMINMAX"
)

$defineArgs = $Defines | ForEach-Object { "/D$_" }
$IncArgs = @("/I$VkInc", "/I$ScriptDir\third_party\vma", "/I$ScriptDir", "/I$GenDir")

# -- Generate compile/link batch file -----------------------------------------
Write-Host "[BUILD] Generating build script..." -ForegroundColor Yellow

$ObjFiles = @()
$buildBat = Join-Path $BuildDir "_build.bat"
$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("@echo off")
[void]$sb.AppendLine("call `"$VcVarsAll`" x64 >NUL 2>&1")
[void]$sb.AppendLine("if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%")
[void]$sb.AppendLine("")

foreach ($src in $CppSources) {
    $srcPath   = Join-Path $ScriptDir $src
    $objFile   = Join-Path $ObjDir "$([System.IO.Path]::GetFileNameWithoutExtension($src)).obj"
    [void]$sb.Append("cl.exe $CppFlags ")
    foreach ($ia in $IncArgs) { [void]$sb.Append("$ia ") }
    foreach ($da in $defineArgs) { [void]$sb.Append("$da ") }
    if ($src -eq "vk_buffer.cpp") { [void]$sb.Append("/DVMA_IMPLEMENTATION ") }
    [void]$sb.AppendLine("/c /Fo`"$objFile`" `"$srcPath`"")
    [void]$sb.AppendLine("if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%")
    $ObjFiles += $objFile
}

[void]$sb.AppendLine("")
[void]$sb.AppendLine("echo Linking...")
[void]$sb.Append("link.exe $LinkFlags /OUT:`"$OutExe`" `"$VkLib\vulkan-1.lib`" ")
foreach ($of in $ObjFiles) { [void]$sb.Append("`"$of`" ") }
[void]$sb.AppendLine("kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib")
[void]$sb.AppendLine("if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%")

[System.IO.File]::WriteAllText($buildBat, $sb.ToString())

# -- Run the build ------------------------------------------------------------
Write-Host "[BUILD] Compiling $CfgName..." -ForegroundColor Yellow
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$buildBat`"" -Wait -NoNewWindow -PassThru
Remove-Item $buildBat -ErrorAction SilentlyContinue

if ($proc.ExitCode -ne 0) { Write-Error "Build failed with code $($proc.ExitCode)"; exit 1 }

# Clean up .obj files  
Remove-Item -Force "$ObjDir\*.obj" -ErrorAction SilentlyContinue

# -- Report ------------------------------------------------------------------
$exeSize = if (Test-Path $OutExe) { "{0:N2} MB" -f ((Get-Item $OutExe).Length / 1MB) } else { "NOT FOUND" }
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESS ($CfgName)" -ForegroundColor Green
Write-Host "  Output : $OutExe" -ForegroundColor Green
Write-Host "  Size   : $exeSize" -ForegroundColor Green
Write-Host "  Shaders: $($ShaderEntries.Count) compiled" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
