"""GPU catalog: vendor spec-sheet peaks (dense, no sparsity), per GPU.

Units: memory GB (decimal), bandwidth GB/s, compute TFLOP/s dense. `link_gbs` is the
per-GPU unidirectional scale-up bandwidth used for collectives (NVLink / xGMI / PCIe).
FP8/FP4 = None means no native tensor-core support for that activation precision.
GeForce cards use the FP32-accumulate tensor rate, which is what serving kernels hit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class GPU:
    key: str
    vendor: str
    arch: str              # hopper, blackwell, blackwell-ultra, ampere, ada, volta, turing, cdna3, cdna4, ...
    vram_gb: float
    mem_bw_gbs: float
    bf16_tflops: float
    fp8_tflops: float | None
    fp4_tflops: float | None
    int8_tops: float | None
    interconnect: str      # nvlink | pcie | xgmi | unified
    link_gbs: float
    l2_mb: float | None = None
    aliases: tuple = field(default_factory=tuple)
    notes: str = ''


_G = [
    # ---- NVIDIA Hopper
    GPU('H100-SXM', 'nvidia', 'hopper', 80, 3350, 989, 1979, None, 1979, 'nvlink', 450, 50,
        ('h100', 'h100 sxm', 'h100-sxm5', 'h100 80gb', 'h100-80gb-hbm3')),
    GPU('H100-NVL', 'nvidia', 'hopper', 94, 3900, 835, 1671, None, 1671, 'pcie', 64, 50, ('h100 nvl',)),
    GPU('H100-PCIe', 'nvidia', 'hopper', 80, 2000, 756, 1513, None, 1513, 'pcie', 64, 50, ('h100 pcie', 'h100-pcie')),
    GPU('H200-SXM', 'nvidia', 'hopper', 141, 4800, 989, 1979, None, 1979, 'nvlink', 450, 50,
        ('h200', 'h200 sxm', 'h200 sxm 141gb', 'h200-141gb')),
    GPU('H200-NVL', 'nvidia', 'hopper', 141, 4800, 835, 1671, None, 1671, 'pcie', 64, 50, ('h200 nvl',)),
    GPU('H20', 'nvidia', 'hopper', 96, 4000, 148, 296, None, 296, 'nvlink', 450, 60, ('h20',)),
    GPU('GH200', 'nvidia', 'hopper', 96, 4000, 989, 1979, None, 1979, 'nvlink', 450, 50, ('gh200', 'gh200-96gb')),
    GPU('GH200-144GB', 'nvidia', 'hopper', 144, 4900, 989, 1979, None, 1979, 'nvlink', 450, 50, ('gh200-144gb',)),
    # ---- NVIDIA Blackwell
    GPU('B200', 'nvidia', 'blackwell', 180, 8000, 2250, 4500, 9000, 4500, 'nvlink', 900, 126,
        ('b200', 'b200 sxm', 'b200-sxm', 'b200 180gb', 'hgx b200')),
    GPU('GB200', 'nvidia', 'blackwell', 186, 8000, 2500, 5000, 10000, 5000, 'nvlink', 900, 126,
        ('gb200', 'gb200 nvl72', 'gb200-nvl72', 'gb200 192gb')),
    GPU('B300', 'nvidia', 'blackwell-ultra', 288, 8000, 2250, 4500, 13500, 300, 'nvlink', 900, 126,
        ('b300', 'b300 sxm', 'b300-sxm', 'hgx b300', 'b300 288gb')),
    GPU('GB300', 'nvidia', 'blackwell-ultra', 288, 8000, 2500, 5000, 15000, 330, 'nvlink', 900, 126,
        ('gb300', 'gb300 nvl72', 'gb300-nvl72')),
    GPU('RTX-PRO-6000', 'nvidia', 'blackwell-ws', 96, 1792, 503, 1007, 2015, 1007, 'pcie', 64, 128,
        ('rtx pro 6000', 'rtx pro 6000 blackwell', 'rtxpro6000', 'rtx6000pro', 'rtx pro 6000 workstation',
         'rtx pro 6000 server', 'rtx pro 6000 blackwell server edition'),
        'Server/Workstation (600 W). Max-Q edition clocks lower; see RTX-PRO-6000-MaxQ.'),
    GPU('RTX-PRO-6000-MaxQ', 'nvidia', 'blackwell-ws', 96, 1792, 438, 876, 1752, 876, 'pcie', 64, 128,
        ('rtx pro 6000 max-q', 'rtx pro 6000 maxq', 'rtx pro 6000 blackwell max-q')),
    GPU('GB10', 'nvidia', 'blackwell-ws', 128, 273, 125, 250, 500, 250, 'unified', 25, 24,
        ('gb10', 'dgx spark', 'nvidia gb10 128gb'), 'Unified LPDDR5x shared with CPU; ~110 GB usable.'),
    GPU('RTX-5090', 'nvidia', 'blackwell-ws', 32, 1792, 209, 419, 838, 838, 'pcie', 32, 96, ('5090', 'rtx 5090', 'geforce rtx 5090')),
    # ---- NVIDIA Ada / Ampere / older
    GPU('L40S', 'nvidia', 'ada', 48, 864, 362, 733, None, 733, 'pcie', 32, 96, ('l40s',)),
    GPU('L40', 'nvidia', 'ada', 48, 864, 181, 362, None, 362, 'pcie', 32, 96, ('l40',)),
    GPU('L4', 'nvidia', 'ada', 24, 300, 121, 242, None, 242, 'pcie', 32, 48, ('l4',)),
    GPU('RTX-6000-Ada', 'nvidia', 'ada', 48, 960, 364, 728, None, 728, 'pcie', 32, 96, ('rtx 6000 ada',)),
    GPU('RTX-4090', 'nvidia', 'ada', 24, 1008, 165, 330, None, 660, 'pcie', 32, 72, ('4090', 'rtx 4090', 'geforce rtx 4090')),
    GPU('A100-80GB', 'nvidia', 'ampere', 80, 2039, 312, None, None, 624, 'nvlink', 300, 40,
        ('a100', 'a100 80gb', 'a100-sxm4-80gb', 'a100 sxm', 'a100-80gb')),
    GPU('A100-40GB', 'nvidia', 'ampere', 40, 1555, 312, None, None, 624, 'nvlink', 300, 40, ('a100 40gb', 'a100-sxm4-40gb')),
    GPU('A100-PCIe-80GB', 'nvidia', 'ampere', 80, 1935, 312, None, None, 624, 'pcie', 32, 40, ('a100 pcie', 'a100-pcie-80gb')),
    GPU('A30', 'nvidia', 'ampere', 24, 933, 165, None, None, 330, 'pcie', 32, 24, ('a30',)),
    GPU('A10', 'nvidia', 'ampere', 24, 600, 125, None, None, 250, 'pcie', 32, 6, ('a10', 'a10g')),
    GPU('RTX-3090', 'nvidia', 'ampere', 24, 936, 71, None, None, 142, 'pcie', 32, 6, ('3090', 'rtx 3090', 'geforce rtx 3090')),
    GPU('V100-32GB', 'nvidia', 'volta', 32, 900, 125, None, None, None, 'nvlink', 150, 6, ('v100', 'v100-sxm2-32gb', 'v100 32gb', 'tesla v100')),
    GPU('T4', 'nvidia', 'turing', 16, 320, 65, None, None, 130, 'pcie', 16, 4, ('t4', 'tesla t4')),
    # ---- AMD Instinct
    GPU('MI300X', 'amd', 'cdna3', 192, 5300, 1307, 2615, None, 2615, 'xgmi', 448, 256, ('mi300x',)),
    GPU('MI325X', 'amd', 'cdna3', 256, 6000, 1307, 2615, None, 2615, 'xgmi', 448, 256, ('mi325x',)),
    GPU('MI350X', 'amd', 'cdna4', 288, 8000, 2307, 4614, 9228, 4614, 'xgmi', 538, 256, ('mi350x',)),
    GPU('MI355X', 'amd', 'cdna4', 288, 8000, 2516, 5033, 10066, 5033, 'xgmi', 538, 256, ('mi355x',)),
    GPU('MI250X', 'amd', 'cdna2', 128, 3277, 383, None, None, 383, 'xgmi', 200, 16, ('mi250x', 'mi250')),
    GPU('RX-7900XTX', 'amd', 'rdna3', 24, 960, 123, None, None, 123, 'pcie', 32, 6, ('7900xtx', 'rx 7900 xtx')),
]

CATALOG = {g.key: g for g in _G}
_ALIAS = {}
for g in _G:
    _ALIAS[g.key.lower()] = g.key
    for a in g.aliases:
        _ALIAS[a.lower()] = g.key


def _norm(s: str) -> str:
    s = s.lower().replace('_', ' ').strip()
    s = re.sub(r'^(nvidia|amd|instinct|tesla)[\s-]+', '', s)
    s = re.sub(r'^(nvidia|amd|instinct|tesla)[\s-]+', '', s)
    return re.sub(r'\s+', ' ', s)


def resolve_gpu(name: str) -> GPU:
    """Map a free-form GPU name (CLI input or benchmark label) to a catalog entry."""
    s = _norm(name)
    if s in _ALIAS:
        return CATALOG[_ALIAS[s]]
    s2 = s.replace(' ', '-')
    if s2 in _ALIAS:
        return CATALOG[_ALIAS[s2]]
    # heuristic matching on distinctive tokens, most specific first
    rules = [
        (r'gb300', 'GB300'), (r'gb200', 'GB200'), (r'b300', 'B300'), (r'b200', 'B200'), (r'gb10|dgx spark', 'GB10'),
        (r'pro 6000.*max', 'RTX-PRO-6000-MaxQ'), (r'pro.?6000', 'RTX-PRO-6000'), (r'6000 ada', 'RTX-6000-Ada'),
        (r'h200.*nvl', 'H200-NVL'), (r'h200', 'H200-SXM'), (r'h100.*nvl', 'H100-NVL'), (r'h100.*pcie', 'H100-PCIe'),
        (r'gh200.*144', 'GH200-144GB'), (r'gh200', 'GH200'), (r'h100', 'H100-SXM'), (r'\bh20\b', 'H20'),
        (r'a100.*40', 'A100-40GB'), (r'a100.*pcie', 'A100-PCIe-80GB'), (r'a100', 'A100-80GB'), (r'\ba30\b', 'A30'),
        (r'\ba10g?\b', 'A10'), (r'l40s', 'L40S'), (r'\bl40\b', 'L40'), (r'\bl4\b', 'L4'), (r'5090', 'RTX-5090'),
        (r'4090', 'RTX-4090'), (r'3090', 'RTX-3090'), (r'v100', 'V100-32GB'), (r'\bt4\b', 'T4'),
        (r'mi355', 'MI355X'), (r'mi350', 'MI350X'), (r'mi325', 'MI325X'), (r'mi300', 'MI300X'), (r'mi250', 'MI250X'),
        (r'7900', 'RX-7900XTX'),
    ]
    for pat, key in rules:
        if re.search(pat, s):
            return CATALOG[key]
    raise KeyError(f'Unknown GPU {name!r}. Known: {", ".join(sorted(CATALOG))}')


def peak_tflops(gpu: GPU, act: str) -> tuple[float, str]:
    """Dense peak for an activation/compute precision, falling back to what the GPU supports.
    Returns (tflops, compute precision actually used)."""
    if act == 'fp4' and gpu.fp4_tflops:
        return gpu.fp4_tflops, 'fp4'
    if act in ('fp4', 'fp8') and gpu.fp8_tflops:
        return gpu.fp8_tflops, 'fp8'
    if act == 'int8' and gpu.int8_tops:
        return gpu.int8_tops, 'int8'
    return gpu.bf16_tflops, 'bf16'
