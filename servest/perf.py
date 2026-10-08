"""Serving performance model.

Physics per replica (TP x PP GPUs; DP replicas share the load):
  decode step(b)  = smoothmax(HBM bytes / (BW*eta_bw), FLOPs / (peak*eta_dec)) + collectives + overhead
                    bytes = dense weights + routed experts touched by b*(1+d) tokens + KV/state reads
  prefill(S)      = smoothmax(FLOPs / (peak*eta_pre), chunks * weight bytes / BW) + per-chunk overhead
Steady state (chunked prefill time-shares the GPU with decode):
  TPOT = t_dec(b)/alpha + b * t_prefill / OSL          (decode steps + amortised prefill)
  lambda = b / (OSL * TPOT)                            (Little's law on the decode phase)
  TTFT = service * (1 + rho / (2(1-rho)))              (M/D/1 wait on the prefill path), rho = lambda*service
  closed loop: b + lambda*TTFT = users ; open loop: lambda given.
KV capacity caps b; users beyond capacity queue (TTFT grows).
Calibrated parameters are efficiencies and overheads only; everything else comes from the
architecture, the GPU spec sheet and the workload.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os

import numpy as np

from .arch import Arch
from .gpus import GPU, peak_tflops
from .precision import Scheme, adapt_to_gpu, kv_bytes

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, '..', 'data', 'calibration.json')

ENGINES = ['vllm', 'sglang', 'tensorrt-llm', 'atom', 'llama.cpp', 'other']
KERNELS = ['bf16', 'fp8', 'int8', 'wonly', 'fp4']   # w4a8 and w4a16 share the dequant-on-the-fly class 'wonly'
HW_CLASSES = ['hopper', 'blackwell', 'blackwell-ultra', 'cdna3', 'cdna4', 'ws', 'unified', 'ampere', 'ada', 'old']
LINKS = ['nvlink', 'pcie', 'xgmi', 'unified']
SPECS = ['none', 'mtp', 'eagle', 'ngram']


def hw_class(g: GPU) -> str:
    if g.arch in ('hopper', 'blackwell', 'blackwell-ultra', 'cdna3', 'cdna4', 'ampere', 'ada'):
        return g.arch
    if g.interconnect == 'unified':
        return 'unified'
    if g.arch == 'blackwell-ws':
        return 'ws'
    return 'old'


DEFAULT_PARAMS = {
    'eta_bw': {'hopper': 0.80, 'blackwell': 0.75, 'blackwell-ultra': 0.75, 'cdna3': 0.60, 'cdna4': 0.60,
               'ws': 0.75, 'unified': 0.70, 'ampere': 0.80, 'ada': 0.80, 'old': 0.70},
    'eta_dec': 0.45, 'eta_pre': 0.55,
    'kernel_bw': {k: 1.0 for k in KERNELS},
    'kernel_pre': {k: 1.0 for k in KERNELS},
    'engine_dec': {e: 1.0 for e in ENGINES},
    'engine_pre': {e: 1.0 for e in ENGINES},
    't_fixed_ms': 0.6, 't_layer_us': 6.0,
    'ar_lat_us': {'nvlink': 8.0, 'pcie': 25.0, 'xgmi': 12.0, 'unified': 5.0},
    'spec_alpha': {'none': 1.0, 'mtp': 1.9, 'eagle': 1.8, 'ngram': 1.3},
    'spec_draft': {'none': 0, 'mtp': 3, 'eagle': 3, 'ngram': 3},
    'link_eff': {'nvlink': 0.7, 'pcie': 0.7, 'xgmi': 0.7, 'unified': 0.7},
    'ttft_fixed_ms': {e: 20.0 for e in ENGINES}, 'moe_eff': 1.0, 'moe_comm_us': 15.0, 't_moe_layer_us': 10.0,
    'chunk': 8192.0, 'smooth_p': 4.0, 'ttft_q': {e: 1.0 for e in ENGINES}, 'mem_util': 0.92, 'overhead_gb': 1.5,
}


def load_params() -> dict:
    p = json.loads(json.dumps(DEFAULT_PARAMS))
    if os.path.exists(CALIB):
        c = json.load(open(CALIB))
        for k, v in c.get('params', {}).items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k].update(v)
            else:
                p[k] = v
    return p


@dataclass
class Deployment:
    arch: Arch
    gpu: GPU
    n_gpus: int
    scheme: Scheme
    kv_dtype: str = 'auto'
    tp: int | None = None
    pp: int = 1
    dp_attention: bool = False
    engine: str = 'vllm'
    spec: str = 'none'
    engram_in_hbm: bool = False
    mem_util: float | None = None
    max_num_seqs: int = 1024
    notes: list = field(default_factory=list)


@dataclass
class Workload:
    isl: float
    osl: float
    users: float | None = None          # closed-loop concurrency (no think time)
    rate: float | None = None           # open-loop requests/s
    prefix_hit: float = 0.0             # fraction of prompt tokens served from prefix cache
    think_s: float = 0.0                # closed loop: user pause between response and next request


def resolve_kv(dep: Deployment) -> str:
    if dep.kv_dtype and dep.kv_dtype != 'auto':
        return dep.kv_dtype
    return 'fp8' if dep.scheme.act in ('fp8', 'fp4') or dep.scheme.other_act == 'fp8' else 'bf16'


def _kv_shard(dep: Deployment, tp: int) -> float:
    """Ways one sequence's KV is split across a replica's GPUs (replicated beyond this)."""
    if dep.dp_attention:
        return float(tp)
    return float(min(tp, max(1, dep.arch.kv_heads)))


def featurize(dep: Deployment, wl: Workload, P: dict | None = None) -> dict:
    """Per-scenario physical quantities that do not depend on calibrated parameters."""
    P = P or load_params()
    a, g = dep.arch, dep.gpu
    sch, notes = adapt_to_gpu(dep.scheme, g)
    tp = dep.tp or dep.n_gpus
    rep = tp * dep.pp
    if dep.n_gpus % rep:
        raise ValueError(f'{dep.n_gpus} GPUs not divisible by TP*PP={rep}')
    R = dep.n_gpus // rep
    kvd = resolve_kv(dep)
    kvb = kv_bytes(kvd)
    w = a.weight_bytes(sch, dep.engram_in_hbm, include_mtp=dep.spec in ('mtp',))
    exp_bits = sch.w_bits
    each_bytes = a.expert_params_each * exp_bits / 8
    dense_bytes = w['dense_linear'] + a.params_lm_head * 2.0
    act_exp = a.top_k * a.expert_params_each * a.moe_layers if a.is_moe else 0.0
    act_dense = max(a.params_active_decode - act_exp, 0.0)
    pk_e, used_e = peak_tflops(g, sch.act)
    pk_d, used_d = peak_tflops(g, sch.other_act if (sch.experts_only and a.is_moe) else sch.act)
    pk_attn = g.bf16_tflops
    S = wl.isl * (1 - wl.prefix_hit)
    Lbar = wl.isl + wl.osl / 2
    shard = _kv_shard(dep, tp)
    kv_read_seq = a.decode_read_bytes(Lbar, kvb) / shard
    # KV capacity (per replica)
    mu = dep.mem_util or P['mem_util']
    cap_bytes = g.vram_gb * 1e9 * mu * rep - w['total'] - P['overhead_gb'] * 1e9 * rep
    per_seq = a.kv_bytes_per_seq(wl.isl + wl.osl, kvb) * (tp / shard)   # replicated copies count
    bmax = cap_bytes / per_seq if per_seq > 0 else 1e9
    flops_pre = 2 * a.params_active_prefill * S
    attn_pre = a.prefill_attn_flops(S, wl.isl - S)
    f = dict(
        R=R, rep=rep, tp=tp, n_layers=a.n_layers, hidden=a.hidden,
        bw=g.mem_bw_gbs * 1e9, pk_e=pk_e * 1e12, pk_d=pk_d * 1e12, pk_attn=pk_attn * 1e12,
        link=g.link_gbs * 1e9, link_cls=LINKS.index(g.interconnect), hw=HW_CLASSES.index(hw_class(g)),
        kernel=KERNELS.index('wonly' if sch.kernel == 'w4a8' else sch.kernel), engine=ENGINES.index(dep.engine if dep.engine in ENGINES else 'other'),
        spec=SPECS.index(dep.spec if dep.spec in SPECS else 'mtp'),
        dense_bytes=dense_bytes, each_bytes=each_bytes, E=float(a.n_experts or 1), k=float(a.top_k or 0),
        moe_layers=float(a.moe_layers if a.is_moe else 0), w_total=w['total'],
        act_exp=act_exp, act_dense=act_dense, act_pre=a.params_active_prefill,
        kv_read_seq=kv_read_seq, attn_dec=a.decode_attn_flops(Lbar) / tp,
        flops_pre=flops_pre, attn_pre=attn_pre, S=S, isl=wl.isl, osl=wl.osl,
        users=wl.users if wl.users else np.nan, rate=wl.rate if wl.rate else np.nan, think=wl.think_s,
        bmax=min(bmax, dep.max_num_seqs * (tp if dep.dp_attention else 1)), cap_bytes=cap_bytes, per_seq_bytes=per_seq,
        mtp_bytes=w['mtp'],
    )
    f['_notes'] = notes + dep.notes
    f['_scheme'] = sch.name
    f['_kv'] = kvd
    f['_weights'] = w
    f['_compute'] = (used_e, used_d)
    return f


# ----------------------------------------------------------------------------- vectorised core

def _arr(F, k):
    return np.asarray(F[k], dtype=float)


def _vec(F: dict) -> dict:
    """List-of-dicts or dict-of-scalars -> dict of numpy arrays."""
    if isinstance(F, list):
        keys = [k for k in F[0] if not k.startswith('_')]
        return {k: np.array([f[k] for f in F], dtype=float) for k in keys}
    return {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in F.items() if not k.startswith('_')}


def _pick(table: dict, order: list, idx: np.ndarray) -> np.ndarray:
    vals = np.array([table.get(k, 1.0) for k in order], dtype=float)
    return vals[idx.astype(int)]


def step_time(V: dict, P: dict, b: np.ndarray) -> np.ndarray:
    """Decode step seconds for b decoding sequences per replica (vectorised)."""
    d = _pick(P['spec_draft'], SPECS, V['spec'])
    T = np.maximum(b, 1e-9) * (1 + d)
    rep = V['rep']
    cover = np.where(V['moe_layers'] > 0, V['E'] * (1 - np.power(np.clip(1 - V['k'] / V['E'], 0, 1), T)), 0.0)
    wbytes = V['dense_bytes'] + P.get('moe_eff', 1.0) * V['moe_layers'] * cover * V['each_bytes'] + d * V['mtp_bytes'] / np.maximum(1, V['n_layers'])
    bytes_gpu = wbytes / rep + b * V['kv_read_seq']
    eta_bw = _pick(P['eta_bw'], HW_CLASSES, V['hw']) * _pick(P['kernel_bw'], KERNELS, V['kernel'])
    t_mem = bytes_gpu / (V['bw'] * eta_bw)
    fl = 2 * T * (V['act_exp'] / V['pk_e'] + V['act_dense'] / V['pk_d']) / rep + b * (1 + d) * V['attn_dec'] / V['pk_attn']
    t_comp = fl / P['eta_dec']
    p = P['smooth_p']
    t = np.power(np.power(t_mem, p) + np.power(t_comp, p), 1 / p)
    ar = _pick(P['ar_lat_us'], LINKS, V['link_cls']) * 1e-6
    leff = _pick(P.get('link_eff', {}), LINKS, V['link_cls'])
    comm = np.where(V['tp'] > 1, 2 * V['n_layers'] * (ar + T * V['hidden'] * 2 * 2 * (V['tp'] - 1) / V['tp'] / (V['link'] * leff)), 0.0)
    # expert dispatch/combine across ranks costs more than a plain all-reduce
    comm = comm + np.where(V['tp'] > 1, V['moe_layers'] * P.get('moe_comm_us', 0.0) * 1e-6, 0.0)
    over = P['t_fixed_ms'] * 1e-3 + V['n_layers'] * P['t_layer_us'] * 1e-6 + V['moe_layers'] * P.get('t_moe_layer_us', 0.0) * 1e-6
    return (t + comm + over) * _pick(P['engine_dec'], ENGINES, V['engine'])


def prefill_time(V: dict, P: dict) -> tuple[np.ndarray, np.ndarray]:
    """(GPU-seconds of prefill work for one request, number of chunks)."""
    rep = V['rep']
    S = np.maximum(V['S'], 1.0)
    chunks = np.ceil(S / P['chunk'])
    frac_e = np.where(V['act_exp'] + V['act_dense'] > 0, V['act_exp'] / (V['act_exp'] + V['act_dense']), 0)
    pk_lin = 1.0 / (frac_e / V['pk_e'] + (1 - frac_e) / V['pk_d'])
    eta = P['eta_pre'] * _pick(P['kernel_pre'], KERNELS, V['kernel'])
    t_c = (V['flops_pre'] / pk_lin + V['attn_pre'] / V['pk_attn']) / rep / eta
    eta_bw = _pick(P['eta_bw'], HW_CLASSES, V['hw'])
    tok_chunk = np.minimum(S, P['chunk'])
    cover = np.where(V['moe_layers'] > 0, V['E'] * (1 - np.power(np.clip(1 - V['k'] / V['E'], 0, 1), tok_chunk)), 0.0)
    wread = V['dense_bytes'] + V['moe_layers'] * cover * V['each_bytes']
    t_m = chunks * wread / rep / (V['bw'] * eta_bw)
    p = P['smooth_p']
    t = np.power(np.power(t_c, p) + np.power(t_m, p), 1 / p)
    ar = _pick(P['ar_lat_us'], LINKS, V['link_cls']) * 1e-6
    tok = np.minimum(S, P['chunk'])
    leff = _pick(P.get('link_eff', {}), LINKS, V['link_cls'])
    comm = np.where(V['tp'] > 1, chunks * 2 * V['n_layers'] * (ar + tok * V['hidden'] * 2 * 2 * (V['tp'] - 1) / V['tp'] / (V['link'] * leff)), 0.0)
    over = chunks * (P['t_fixed_ms'] * 1e-3 + V['n_layers'] * P['t_layer_us'] * 1e-6)
    return (t + comm + over) * _pick(P['engine_pre'], ENGINES, V['engine']), chunks


def solve(F, P: dict | None = None, iters: int = 60) -> dict:
    """Steady-state metrics for featurized scenarios (closed loop if users given, else open loop)."""
    P = P or load_params()
    V = _vec(F)
    alpha = _pick(P['spec_alpha'], SPECS, V['spec'])
    tpre, chunks = prefill_time(V, P)
    osl = np.maximum(V['osl'], 1.0)
    bmax = np.maximum(V['bmax'], 0.0)
    closed = ~np.isnan(V['users'])
    c = np.where(closed, V['users'] / V['R'], np.nan)
    lam_open = np.where(closed, np.nan, V['rate'] / V['R'])

    pack = np.minimum(1.0, np.maximum(V['S'], 1.0) / P['chunk'])      # fraction of a chunk one prompt fills
    tfix = _pick(P['ttft_fixed_ms'], ENGINES, V['engine']) * 1e-3 if isinstance(P['ttft_fixed_ms'], dict) else P['ttft_fixed_ms'] * 1e-3

    def metrics(b):
        td = step_time(V, P, b)
        tpot = td / alpha + b * tpre / osl
        lam = b / (osl * tpot)
        occ = tpre + np.maximum(chunks - 1 + pack, pack) * td            # prefill-path occupancy per request
        rho = np.clip(lam * occ, 0, 0.999)
        rho_q = np.where(closed, rho * np.clip((c - 1) / np.maximum(c, 1), 0, 1), rho)   # others only
        lat = tpre + chunks * td                                         # own prefill steps incl. decode riders
        tq = _pick(P['ttft_q'], ENGINES, V['engine']) if isinstance(P['ttft_q'], dict) else P['ttft_q']
        # Benchmarks with zero think time arrive in synchronised waves (calibrated burst factor);
        # users who pause between requests approach random (M/D/1) arrivals. Blend by think share.
        think = V.get('think', 0.0)
        cyc = lat + osl * tpot
        wgt = np.where(closed, cyc / (cyc + think), 0.5)
        tq = 1.0 + (tq - 1.0) * wgt ** 2
        ttft = lat + tq * rho_q / (2 * (1 - rho)) * occ + 0.5 * td + tfix
        return td, tpot, lam, ttft, rho

    # closed loop: bisection on b in (0, min(c, bmax)]
    hi = np.where(closed, np.minimum(c, bmax), np.minimum(bmax, 1e6))
    lo = np.zeros_like(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        td, tpot, lam, ttft, rho = metrics(np.maximum(mid, 1e-6))
        g_closed = mid + lam * (ttft + V.get('think', 0.0)) - c
        g_open = lam - lam_open
        g = np.where(closed, g_closed, g_open)
        lo = np.where(g < 0, mid, lo)
        hi = np.where(g >= 0, mid, hi)
    b = np.maximum(0.5 * (lo + hi), 1e-6)
    td, tpot, lam, ttft, rho = metrics(b)
    # closed loop beyond KV capacity: extra users wait in queue
    cap_bound = closed & (b >= np.minimum(c, bmax) * 0.999) & (c > bmax)
    ttft = np.where(cap_bound, np.maximum(ttft, c / lam - osl * tpot - V.get('think', 0.0)), ttft)
    overloaded = (~closed) & (lam < lam_open * 0.98)
    out = dict(
        tpot_s=tpot, ttft_s=ttft, decode_tps_per_user=1.0 / tpot,
        output_tps=lam * osl * V['R'], requests_per_s=lam * V['R'],
        batch_per_replica=b, prefill_util=rho, step_s=td, prefill_s=tpre, kv_max_seqs=bmax,
        kv_limited=cap_bound, overloaded=overloaded,
        e2e_s=ttft + osl * tpot,
    )
    out['input_tps'] = out['requests_per_s'] * V['isl']
    out['total_tps'] = out['output_tps'] + out['input_tps']
    return out
