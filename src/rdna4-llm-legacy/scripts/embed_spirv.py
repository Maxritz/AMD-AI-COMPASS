#!/usr/bin/env python3
"""Embeds SPIR-V binary as a C header (xxd -i style) and generates shader registry."""

import argparse
import os
import sys
import struct


def embed_spirv(input_path: str, output_path: str, var_name: str) -> None:
    """Read SPIR-V binary, write C header with static const uint32_t array."""
    with open(input_path, "rb") as f:
        data = f.read()

    if len(data) % 4 != 0:
        raise ValueError(f"SPIR-V file size {len(data)} is not a multiple of 4 bytes")

    word_count = len(data) // 4
    sanitized_name = var_name.replace(".", "_").replace("-", "_")

    lines = []
    lines.append(f"// Auto-generated from: {os.path.basename(input_path)}")
    lines.append(f"// Words: {word_count}  Bytes: {len(data)}")
    lines.append(f"#pragma once")
    lines.append(f"#include <stdint.h>")
    lines.append(f"#include <stddef.h>")
    lines.append(f"")
    lines.append(f"static const uint32_t {sanitized_name}_data[] = {{")

    for i in range(0, len(data), 4):
        word = struct.unpack_from("<I", data, i)[0]
        lines.append(f"    0x{word:08X},")

    lines.append(f"}};")
    lines.append(f"static const size_t {sanitized_name}_size = {len(data)};")
    lines.append(f"static const uint32_t {sanitized_name}_word_count = {word_count};")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Embedded {os.path.basename(input_path)} -> {os.path.basename(output_path)} "
          f"({word_count} words)")


QUANT_MAP = {
    "fp16":    "QUANT_FP16",
    "q4_k":    "QUANT_Q4_K",
    "q6_k":    "QUANT_Q6_K",
    "q8_0":    "QUANT_Q8_0",
    "iq4_xs":  "QUANT_IQ4_XS",
    "q4_0":    "QUANT_Q4_0",
}

OP_MAP = {
    "rms_norm":     "OP_RMS_NORM",
    "attn_qkv":     "OP_ATTN_QKV",
    "attn_compute": "OP_ATTN_COMPUTE",
    "attn_output":  "OP_ATTN_OUTPUT",
    "rope":         "OP_ROPE",
    "qk_norm":      "OP_QK_NORM",
    "ffn_gate_up":  "OP_FFN_GATE_UP",
    "ffn_down":     "OP_FFN_DOWN",
    "lm_head":      "OP_LM_HEAD",
    "token_embed":  "OP_EMBEDDING_LOOKUP",
}


def _parse_stem(stem: str):
    """Parse shader stem like 'attn_qkv_q4_k_w32' into (quant_type, op_type, wave_size)."""
    parts = stem.split("_")

    wave_size = 32
    for p in parts:
        if p.startswith("w") and p[1:].isdigit():
            wave_size = int(p[1:])
            break

    quant_type = "QUANT_FP16"
    op_type = "OP_UNKNOWN"

    for qkey in sorted(QUANT_MAP.keys(), key=len, reverse=True):
        if qkey in stem:
            quant_type = QUANT_MAP[qkey]
            break

    for okey in sorted(OP_MAP.keys(), key=len, reverse=True):
        if stem.startswith(okey):
            op_type = OP_MAP[okey]
            break

    return quant_type, op_type, wave_size


def generate_registry(output_path: str, shader_list: str) -> None:
    """Generate shader_registry.h that maps (op, quant, wave) -> embedded data ptr."""
    shader_paths = [p.strip() for p in shader_list.split(";") if p.strip()] if shader_list else []

    lines = []
    lines.append("// Auto-generated shader registry -- do not edit")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("")

    for path in shader_paths:
        basename = os.path.basename(path)
        lines.append(f'#include "{basename}"')

    lines.append("")
    lines.append("struct embedded_shader_t {")
    lines.append("    const char*    name;")
    lines.append("    const uint32_t* data;")
    lines.append("    size_t          byte_size;")
    lines.append("    uint32_t        word_count;")
    lines.append("    uint32_t        subgroup_size;")
    lines.append("    int             quant_type;")
    lines.append("    int             op_type;")
    lines.append("};")
    lines.append("")

    lines.append("static const embedded_shader_t EMBEDDED_SHADERS[] = {")

    for path in shader_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        quant_type, op_type, wave_size = _parse_stem(stem)
        var_name = f"shader_{stem}"

        lines.append(f"    {{ \"{stem}\", {var_name}_data, {var_name}_size, "
                     f"{var_name}_word_count, {wave_size}, {quant_type}, {op_type} }},")

    lines.append("    { NULL, NULL, 0, 0, 0, 0, 0 }  // sentinel")
    lines.append("};")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Generated registry: {os.path.basename(output_path)} "
          f"({len(shader_paths)} shaders)")


def main():
    parser = argparse.ArgumentParser(description="SPIR-V embed tool")
    parser.add_argument("--input", help="Input .spv file")
    parser.add_argument("--output", help="Output .h file")
    parser.add_argument("--name", help="C variable name prefix")
    parser.add_argument("--generate-registry", action="store_true",
                        help="Generate shader_registry.h")
    parser.add_argument("--shader-list", default="",
                        help="Semicolon-separated list of shader header paths")
    args = parser.parse_args()

    if args.generate_registry:
        generate_registry(args.output, args.shader_list)
    else:
        if not args.input or not args.output or not args.name:
            parser.error("--input, --output, and --name required for embed mode")
        embed_spirv(args.input, args.output, args.name)


if __name__ == "__main__":
    main()
