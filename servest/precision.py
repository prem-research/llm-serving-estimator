"""Quantization schemes: storage bits per weight (incl. scale overhead) and compute precision.

A scheme describes how *linear-layer* weights are stored and what precision the matmul runs
in. MoE checkpoints often quantize only the routed experts; `experts_only` schemes keep the
rest of the linear layers at `other_bits`/`other_act`. Embeddings and the LM head stay 16-bit.
"""
from __future__ import annotations
from dataclasses import dataclass

from .gpus import GPU


@dataclass(frozen=True)
class Scheme:
    name: str
    w_bits: float          # storage bits/weight for quantized linears, incl. scales
    act: str               # compute precision for those linears: bf16 | fp8 | int8 | fp4
    kernel: str            # kernel family, used for calibrated efficiency factors
    experts_only: bool = False
    other_bits: float = 16
    other_act: str = 'bf16'
    desc: str = ''


SCHEMES = {
    'bf16': Scheme('bf16', 16, 'bf16', 'bf16', desc='BF16/FP16 weights and activations'),
    'fp8': Scheme('fp8', 8.06, 'fp8', 'fp8', desc='W8A8 FP8 (per-tensor/channel/128-block scales)'),
    'mxfp8': Scheme('mxfp8', 8.25, 'fp8', 'fp8', desc='W8A8 MXFP8 (E8M0 scale per 32)'),
    'int8': Scheme('int8', 8.06, 'int8', 'int8', desc='W8A8 INT8 (SmoothQuant-style)'),
    'w8a16': Scheme('w8a16', 8.13, 'bf16', 'wonly', desc='8-bit weight-only (FP8/INT8 Marlin), BF16 compute'),
    'w4a16': Scheme('w4a16', 4.25, 'bf16', 'wonly', desc='4-bit weight-only (AWQ/GPTQ/Marlin, group 128), BF16 compute'),
    'w4a8': Scheme('w4a8', 4.25, 'fp8', 'w4a8', desc='4-bit weights, FP8 activations (e.g. FP4 experts on Hopper, QServe)'),
    'nvfp4': Scheme('nvfp4', 4.5, 'fp4', 'fp4', desc='W4A4 NVFP4 (E4M3 scale per 16)'),
    'mxfp4': Scheme('mxfp4', 4.25, 'fp4', 'fp4', desc='W4A4 MXFP4 (E8M0 scale per 32)'),
    # MoE-typical mixed checkpoints: 4-bit routed experts, FP8 everything else
    'fp4-experts': Scheme('fp4-experts', 4.5, 'fp4', 'fp4', True, 8.06, 'fp8',
                          'NVFP4/MXFP4 routed experts, FP8 attention/shared/dense'),
    'mxfp4-experts-bf16': Scheme('mxfp4-experts-bf16', 4.25, 'fp4', 'fp4', True, 16, 'bf16',
                                 'MXFP4 routed experts, BF16 rest (gpt-oss native)'),
    'w4a16-experts': Scheme('w4a16-experts', 4.25, 'bf16', 'wonly', True, 8.06, 'fp8',
                            '4-bit weight-only experts, FP8 rest'),
}
ALIASES = {
    'fp16': 'bf16', 'bfloat16': 'bf16', 'float16': 'bf16', 'w16a16': 'bf16', 'half': 'bf16',
    'w8a8': 'fp8', 'fp8-w8a8': 'fp8', 'w8a8-fp8': 'fp8', 'float8': 'fp8', 'e4m3': 'fp8',
    'w8a8-int8': 'int8', 'smoothquant': 'int8',
    'fp8-weight-only': 'w8a16', 'int8-weight-only': 'w8a16',
    'awq': 'w4a16', 'gptq': 'w4a16', 'int4': 'w4a16', 'w4a16-int4': 'w4a16', 'marlin': 'w4a16',
    'qserve': 'w4a8', 'w4a8-fp8': 'w4a8',
    'fp4': 'nvfp4', 'w4a4': 'nvfp4', 'nvfp4-w4a4': 'nvfp4', 'mxfp4-w4a4': 'mxfp4',
}


def get_scheme(name: str) -> Scheme:
    n = name.strip().lower()
    n = ALIASES.get(n, n)
    if n not in SCHEMES:
        raise KeyError(f'Unknown quantization {name!r}. Options: {", ".join(sorted(SCHEMES))} '
                       f'(aliases: {", ".join(sorted(ALIASES))})')
    return SCHEMES[n]


def adapt_to_gpu(s: Scheme, gpu: GPU) -> tuple[Scheme, list[str]]:
    """Downgrade compute precision the GPU can't run natively (storage is unchanged)."""
    notes = []

    def fix(act):
        if act == 'fp4' and not gpu.fp4_tflops:
            new = 'fp8' if gpu.fp8_tflops else 'bf16'
            notes.append(f'{gpu.key} has no FP4 tensor cores: 4-bit weights run as W4A{8 if new == "fp8" else 16}.')
            return new
        if act == 'fp8' and not gpu.fp8_tflops:
            notes.append(f'{gpu.key} has no FP8 tensor cores: FP8 weights run weight-only (Marlin) with BF16 compute.')
            return 'bf16'
        if act == 'int8' and not gpu.int8_tops:
            notes.append(f'{gpu.key} lacks INT8 tensor cores: BF16 compute.')
            return 'bf16'
        return act

    act, other = fix(s.act), fix(s.other_act)
    if act == s.act and other == s.other_act:
        return s, notes
    kernel = s.kernel
    if act != s.act:
        kernel = 'wonly' if act == 'bf16' else ('w4a8' if s.w_bits < 6 else 'fp8')
    return Scheme(s.name + f'@{act}', s.w_bits, act, kernel, s.experts_only, s.other_bits, other, s.desc), notes


KV_BYTES = {'bf16': 2.0, 'bfloat16': 2.0, 'float16': 2.0, 'fp16': 2.0, 'auto': 2.0, 'float8': 1.0, 'fp8_e4m3fn': 1.0, 'fp8': 1.0, 'fp8_e4m3': 1.0, 'fp8_e5m2': 1.0,
            'int8': 1.0, 'fp4': 0.5625, 'nvfp4': 0.5625}


def kv_bytes(kv_dtype: str) -> float:
    k = (kv_dtype or 'auto').lower()
    if k not in KV_BYTES:
        raise KeyError(f'Unknown KV-cache dtype {kv_dtype!r}: {", ".join(KV_BYTES)}')
    return KV_BYTES[k]
