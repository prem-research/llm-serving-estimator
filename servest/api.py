"""Programmatic API shared by the CLI and the browser (Pyodide) front-end.

Depends only on numpy + stdlib so it runs in WebAssembly. All public functions return
JSON-serialisable dicts.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import json
import math
import os
import re

import numpy as np

from . import arch as A
from . import residual
from . import tco
from .gpus import CATALOG, resolve_gpu
from .perf import Deployment, Workload, featurize, load_params, solve
from .precision import SCHEMES, ALIASES, get_scheme

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
TARGET_KEYS = {'tpot_ms': 'tpot_s', 'ttft_ms': 'ttft_s', 'output_tps': 'output_tps'}


def _load_json(name, default):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else default


FAMILIES = _load_json('families.json', {})
MODEL_TYPES = _load_json('model_types.json', {})
BANDS = _load_json('error_bands.json', {})
ORIGINS = _load_json('model_origins.json', {})
EFFECTS = residual.load()
KNOWN_FAMILIES = set(FAMILIES.values())
CAL_GPUS = set()
for _t in EFFECTS.values():
    CAL_GPUS |= set(_t.get('gpu', {}).keys())


@dataclass
class Query:
    model: str
    gpu: str
    gpus: int = 8
    users: float | None = None
    rate: float | None = None
    isl: float = 2048
    osl: float = 512
    quant: str = 'native'
    kv_dtype: str = 'auto'
    engine: str = 'vllm'
    tp: int | None = None
    dp_attention: bool = False
    spec: str = 'none'
    prefix_hit: float = 0.0
    think_time: float = 0.0
    mem_util: float | None = None
    engram_in_hbm: bool = False


def family_of(repo: str) -> str | None:
    if repo in FAMILIES:
        return FAMILIES[repo]
    k = residual.family_key(repo)
    for f in KNOWN_FAMILIES:
        if residual.family_key(f) == k:
            return f
    return None


def origin_of(repo: str, model_type: str = '') -> dict:
    """Publisher and country of the base model (repo-name rules first, then the config model_type)."""
    name, mt = str(repo).lower(), str(model_type or '').lower()
    for rules, text, basis in ((ORIGINS.get('name_rules', []), name, 'name'), (ORIGINS.get('type_rules', []), mt, 'architecture')):
        for pat, publisher, country in rules:
            if text and re.search(pat, text):
                return {'publisher': publisher, 'country': country,
                        'country_name': ORIGINS.get('countries', {}).get(country, country), 'basis': basis}
    return {'publisher': None, 'country': None, 'country_name': None, 'basis': None}


def load_arch(repo: str, cfg: dict | None = None, meta: dict | None = None, fetch: bool = True) -> A.Arch:
    return A.parse(repo, cfg=cfg, fetch=fetch, meta=meta)


def _deployment(ar, q: Query, n, tp, users):
    gpu = resolve_gpu(q.gpu)
    sch = get_scheme(ar.native_scheme if q.quant in (None, '', 'native', 'auto') else q.quant)
    dep = Deployment(arch=ar, gpu=gpu, n_gpus=n, scheme=sch, kv_dtype=q.kv_dtype, tp=tp, pp=1,
                     dp_attention=q.dp_attention, engine=q.engine, spec=q.spec,
                     engram_in_hbm=q.engram_in_hbm, mem_util=q.mem_util)
    wl = Workload(isl=q.isl, osl=q.osl, users=users, rate=q.rate if users is None else None,
                  prefix_hit=q.prefix_hit, think_s=q.think_time)
    return gpu, dep, wl


def evaluate(ar, gpu, dep, wl, P):
    f = featurize(dep, wl, P)
    o = {k: float(v[0]) if isinstance(v, np.ndarray) else v for k, v in solve([f], P).items()}
    fam = family_of(ar.repo)
    meta = {'family': fam or '?', 'gpu': gpu.key, 'engine': dep.engine, 'scheme': f['_scheme'], 'hw_class': gpu.arch}
    corr = {}
    for t in TARGET_KEYS:
        c, used = residual.lookup(meta, EFFECTS.get(t, {}))
        corr[t] = (float(np.clip(c, -0.7, 0.7)), used)
    o['tpot_s'] *= math.exp(corr['tpot_ms'][0])
    o['ttft_s'] *= math.exp(corr['ttft_ms'][0])
    o['output_tps'] *= math.exp(corr['output_tps'][0])
    o['decode_tps_per_user'] = 1.0 / o['tpot_s']
    o['e2e_s'] = o['ttft_s'] + wl.osl * o['tpot_s']
    o['requests_per_s'] = o['output_tps'] / wl.osl
    o['input_tps'] = o['requests_per_s'] * wl.isl
    o['total_tps'] = o['output_tps'] + o['input_tps']
    for k in ('kv_limited', 'overloaded'):
        o[k] = bool(o[k])
    return f, o, corr, fam


def candidate_tps(n, gpu):
    c = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= n and n % t == 0]
    if gpu.interconnect in ('nvlink', 'xgmi'):
        c = [t for t in c if t <= 8] or c        # TP beyond a node is rarely used; DP across nodes
    return c


def best_config(ar, q: Query, P=None, users=None, n_gpus=None):
    """Evaluate TP layouts for the GPU count. Returns (feasible sorted best-first, all)."""
    P = P or load_params()
    n = n_gpus or q.gpus
    gpu = resolve_gpu(q.gpu)
    u = q.users if users is None else users
    out = []
    for tp in ([q.tp] if q.tp else candidate_tps(n, gpu)):
        g, dep, wl = _deployment(ar, q, n, tp, u)
        try:
            f, o, corr, fam = evaluate(ar, g, dep, wl, P)
        except ValueError:
            continue
        out.append(dict(tp=tp, replicas=f['R'], f=f, o=o, corr=corr, fam=fam, dep=dep, wl=wl,
                        feasible=f['cap_bytes'] > 0 and f['bmax'] >= 1))
    feas = [r for r in out if r['feasible']]
    tight = lambda r: r['o']['kv_limited'] or r['o']['batch_per_replica'] > 0.85 * r['f']['bmax']
    feas.sort(key=lambda r: (tight(r), -r['o']['output_tps'], r['tp']))
    return feas, out


def band(target: str, regime: str):
    b = BANDS.get(regime, {}).get(target)
    return (b['p10'], b['p90']) if b else None


def regime_of(fam, gpu_key, isl, users, ar=None):
    tags = []
    if ar is not None and fam is None:
        novel = {s.kind for s in ar.attn} & {'csa', 'hca', 'blocksparse', 'sparse', 'linear', 'swa'}
        if MODEL_TYPES.get(ar.model_type, 0) < 50 or novel:
            tags.append(f"Architecture '{ar.model_type}'" + (f" uses {'/'.join(sorted(novel))} attention" if novel else '')
                        + ' and has little public benchmark data: early engine kernels for new architectures have measured'
                        ' up to 1.5-2x slower than this estimate.')
    known_f, known_g = fam is not None, gpu_key in CAL_GPUS
    r = 'known' if (known_f and known_g) else ('partial' if (known_f or known_g) else 'novel')
    if isl > 16384:
        tags.append(f'Prompt of {isl:,.0f} tokens is beyond most public measurements (few above 32k): the trend comes '
                    'from physics; validate on your hardware.')
    if users and users > 2048:
        tags.append('More than 2,048 concurrent users per deployment is beyond the calibration data.')
    return r, tags


def _rng(val, target, regime, inverse=False):
    b = band(target, regime)
    if not b:
        return None
    return [val / b[1], val / b[0]] if inverse else [val * b[0], val * b[1]]


def result_dict(ar, r, q: Query) -> dict:
    f, o, gpu = r['f'], r['o'], resolve_gpu(q.gpu)
    reg, tags = regime_of(r['fam'], gpu.key, q.isl, q.users, ar)
    w = f['_weights']
    users_rep = (q.users or 0) / r['replicas']
    return {
        'model': {'repo': ar.repo, 'type': ar.model_type, 'moe': ar.is_moe, 'experts': ar.n_experts, 'top_k': ar.top_k,
                  'layers': ar.n_layers, 'total_b': ar.params_total / 1e9, 'active_b': ar.params_active_decode / 1e9,
                  'active_prefill_b': ar.params_active_prefill / 1e9, 'native_scheme': ar.native_scheme,
                  'attention': {k: sum(1 for s in ar.attn if s.kind == k) for k in sorted({s.kind for s in ar.attn})},
                  'origin': origin_of(ar.repo, ar.model_type), 'notes': list(ar.notes)},
        'hardware': {'gpu': gpu.key, 'n_gpus': r['dep'].n_gpus, 'vram_gb': gpu.vram_gb, 'bw_tbs': gpu.mem_bw_gbs / 1000,
                     'interconnect': gpu.interconnect, 'tp': r['tp'], 'replicas': r['replicas'], 'engine': r['dep'].engine,
                     'spec': r['dep'].spec},
        'precision': {'scheme': f['_scheme'], 'desc': SCHEMES.get(f['_scheme'].split('@')[0], get_scheme('bf16')).desc,
                      'compute': list(f['_compute']), 'kv_dtype': f['_kv'], 'notes': list(f['_notes'])},
        'memory': {'weights_gb': w['total'] / 1e9, 'weights_per_gpu_gb': w['total'] / r['tp'] / 1e9,
                   'experts_gb': w['experts'] / 1e9, 'dense_gb': w['dense_linear'] / 1e9, 'embed_gb': w['embed_lm_head'] / 1e9,
                   'mtp_gb': w['mtp'] / 1e9, 'engram_gb': w['engram'] / 1e9,
                   'fits': f['cap_bytes'] > 0, 'kv_budget_gb': f['cap_bytes'] / 1e9, 'kv_per_user_mb': f['per_seq_bytes'] / 1e6,
                   'context_tokens': q.isl + q.osl, 'max_seqs_per_replica': f['bmax'], 'max_seqs_total': f['bmax'] * r['replicas'],
                   'kv_seqs_total': (f['cap_bytes'] / f['per_seq_bytes'] * r['replicas']) if f['per_seq_bytes'] > 0 else None,
                   'engine_cap_total': r['dep'].max_num_seqs * (r['tp'] if r['dep'].dp_attention else 1) * r['replicas'],
                   'users_per_replica': users_rep, 'kv_use': users_rep / max(f['bmax'], 1e-9) if f['cap_bytes'] > 0 else None},
        'perf': {'decode_tps_per_user': o['decode_tps_per_user'], 'decode_tps_per_user_range': _rng(o['decode_tps_per_user'], 'tpot_ms', reg, True),
                 'tpot_ms': o['tpot_s'] * 1e3, 'tpot_ms_range': _rng(o['tpot_s'] * 1e3, 'tpot_ms', reg),
                 'ttft_s': o['ttft_s'], 'ttft_s_range': _rng(o['ttft_s'], 'ttft_ms', reg),
                 'output_tps': o['output_tps'], 'output_tps_range': _rng(o['output_tps'], 'output_tps', reg),
                 'output_tps_per_gpu': o['output_tps'] / r['dep'].n_gpus, 'total_tps': o['total_tps'],
                 'requests_per_s': o['requests_per_s'], 'e2e_s': o['e2e_s'],
                 'step_ms': o['step_s'] * 1e3, 'batch_per_replica': o['batch_per_replica'], 'prefill_s': o['prefill_s'],
                 'prefill_share': min(o['prefill_util'], 1.0), 'kv_limited': o['kv_limited'], 'overloaded': o['overloaded']},
        'confidence': {'regime': reg, 'family': r['fam'], 'gpu_calibrated': gpu.key in CAL_GPUS, 'cautions': tags,
                       'corrections': {t: math.exp(c) for t, (c, _) in r['corr'].items() if abs(c) > 0.01}},
    }


def _layout_row(r):
    return {'tp': r['tp'], 'replicas': r['replicas'], 'decode_tps_per_user': r['o']['decode_tps_per_user'],
            'ttft_s': r['o']['ttft_s'], 'output_tps': r['o']['output_tps'], 'kv_limited': r['o']['kv_limited']}


def estimate(ar, q: Query, P=None) -> dict:
    P = P or load_params()
    feas, allr = best_config(ar, q, P)
    if not feas:
        if not allr:
            return {'error': f'No tensor-parallel layout divides {q.gpus} GPUs.'}
        r = max(allr, key=lambda r: r['tp'])
        d = result_dict(ar, r, q)
        d['feasible'] = False
        return d
    d = result_dict(ar, feas[0], q)
    d['feasible'] = True
    d['layouts'] = [_layout_row(r) for r in feas]
    return d


def sweep(ar, q: Query, users_list, P=None) -> list:
    P = P or load_params()
    rows = []
    for u in users_list:
        feas, _ = best_config(ar, q, P, users=u)
        if not feas:
            rows.append({'users': u, 'fits': False})
            continue
        r = feas[0]
        rows.append({'users': u, 'fits': True, **_layout_row(r),
                     'kv_use': u / r['replicas'] / max(r['f']['bmax'], 1e-9)})
    return rows


def _meets(o, min_tps, max_ttft):
    ok = not o['kv_limited'] and not o.get('overloaded')
    if min_tps:
        ok = ok and o['decode_tps_per_user'] >= min_tps
    if max_ttft:
        ok = ok and o['ttft_s'] <= max_ttft
    return ok


def capacity(ar, q: Query, min_tps=None, max_ttft=None, P=None) -> dict:
    """Largest number of concurrent users meeting the targets (central estimate)."""
    P = P or load_params()

    def ok_at(u):
        feas, _ = best_config(ar, q, P, users=u)
        good = [r for r in feas if _meets(r['o'], min_tps, max_ttft)]
        return good[0] if good else None

    lo, hi, best = 0.0, 1.0, None
    while hi < 2e5:
        r = ok_at(hi)
        if not r:
            break
        best, lo, hi = (hi, r), hi, hi * 2
    if best is None:
        return {'max_users': 0}
    for _ in range(12):
        mid = (lo + hi) / 2
        r = ok_at(mid)
        if r:
            lo, best = mid, (mid, r)
        else:
            hi = mid
    u, r = best
    return {'max_users': u, **_layout_row(r)}


def plan(ar, q: Query, min_tps=None, max_ttft=None, P=None) -> list:
    P = P or load_params()
    rows = []
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        feas, _ = best_config(ar, q, P, n_gpus=n)
        if not feas:
            rows.append({'gpus': n, 'fits': False})
            continue
        r = max(feas, key=lambda r: (_meets(r['o'], min_tps, max_ttft), r['o']['decode_tps_per_user']))
        m = _meets(r['o'], min_tps, max_ttft)
        rows.append({'gpus': n, 'fits': True, 'meets': m, **_layout_row(r)})
        if m:
            break
    return rows


SIZE_GPUS = ('H100-SXM', 'H200-SXM', 'B200', 'B300')


def _good_layout(ar, q, P, n, min_tps, max_ttft):
    feas, _ = best_config(ar, q, P, n_gpus=n)
    good = [r for r in feas if _meets(r['o'], min_tps, max_ttft)]
    return (good[0] if good else None), bool(feas)


def _headroom(ar, q, n, tp, min_tps, max_ttft, P, u0):
    """Most users the fleet (n GPUs, fixed TP layout) serves at the target, starting from a load known to meet it."""
    qn = replace(q, gpus=n, tp=tp)
    ok = lambda u: _good_layout(ar, replace(qn, users=u), P, n, min_tps, max_ttft)[0] is not None
    lo, hi = float(u0), None
    for _ in range(14):
        u = lo * 1.6
        if ok(u):
            lo = u
        else:
            hi = u
            break
    if hi is None:
        return lo
    for _ in range(7):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return math.floor(lo)


def size(ar, q: Query, min_tps=None, max_ttft=None, gpus=SIZE_GPUS, P=None, max_gpus: int = 512,
         headroom: bool = True) -> list:
    """Smallest GPU count per GPU type that serves the query's users (or request rate) at the target.
    Counts are 1, 2, 4, then whole 8-GPU nodes; nodes beyond the first add data-parallel replicas."""
    P = P or load_params()
    rows = []
    for g in gpus:
        gk = resolve_gpu(g).key
        qg = replace(q, gpu=gk, tp=None if (q.tp and q.tp > 8) else q.tp)
        small = [n for n in (1, 2, 4, 8) if n <= max_gpus]
        found, n_found, fits_at = None, None, None
        for n in small:
            r, fits = _good_layout(ar, qg, P, n, min_tps, max_ttft)
            fits_at = fits_at or (n if fits else None)
            if r:
                found, n_found = r, n
                break
        if not found:                                   # whole nodes: exponential then binary search
            lo, hi, r_hi = 1, None, None
            k = 2
            while k * 8 <= max_gpus:
                r, fits = _good_layout(ar, qg, P, k * 8, min_tps, max_ttft)
                fits_at = fits_at or (k * 8 if fits else None)
                if r:
                    hi, r_hi = k, r
                    break
                lo, k = k, k * 2
            if hi:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    r, _ = _good_layout(ar, qg, P, mid * 8, min_tps, max_ttft)
                    if r:
                        hi, r_hi = mid, r
                    else:
                        lo = mid
                found, n_found = r_hi, hi * 8
        if not found:
            rows.append({'gpu': gk, 'meets': False, 'fits_from_gpus': fits_at, 'max_gpus': max_gpus})
            continue
        o = found['o']
        row = {'gpu': gk, 'meets': True, 'n_gpus': n_found, 'nodes': math.ceil(n_found / 8), 'tp': found['tp'],
               'replicas': found['replicas'], 'decode_tps_per_user': o['decode_tps_per_user'], 'ttft_s': o['ttft_s'],
               'output_tps': o['output_tps'], 'requests_per_s': o['requests_per_s'], 'fits_from_gpus': fits_at,
               'scheme': found['f']['_scheme'], 'weights_gb': found['f']['_weights']['total'] / 1e9}
        if headroom and q.users:
            row['max_users'] = _headroom(ar, qg, n_found, found['tp'], min_tps, max_ttft, P, q.users)
        rows.append(row)
    return rows


def size_and_cost(ar, q: Query, min_tps=None, max_ttft=None, gpus=SIZE_GPUS, *, current_cloud=0.0, current_api=0.0,
                  api=None, overrides=None, size_from='users', acceptance=False, P=None) -> dict:
    """Fleet sizing plus own-vs-rent TCO. size_from='api_spend' converts the current API bill into a peak
    request rate (open loop) instead of using the query's concurrent users."""
    a = tco.assumptions(overrides)
    price = tco.api_price(api or a.get('api', 'claude-sonnet-5-5'))
    volume = None
    if size_from == 'api_spend':
        if not current_api or current_api <= 0:
            raise ValueError('Enter the current monthly API spend to size from it.')
        volume = tco.workload_from_api_spend(current_api, q.isl, q.osl, price, a['active_hours_month'], a['peak_to_average'])
        q = replace(q, users=None, rate=volume['peak_requests_per_s'], think_time=0.0)
    rows = size(ar, q, min_tps, max_ttft, gpus, P, headroom=size_from != 'api_spend')
    cost = tco.compare(rows, isl=q.isl, osl=q.osl, current_cloud=current_cloud, current_api=current_api, api=price,
                       overrides=overrides, acceptance=acceptance,
                       requests_per_s=volume['avg_requests_per_s'] if volume else None)
    return {'sizing': rows, 'cost': cost, 'volume': volume, 'size_from': size_from,
            'target': {'min_tps': min_tps, 'max_ttft': max_ttft}, 'origin': origin_of(ar.repo, ar.model_type),
            'model': {'repo': ar.repo, 'total_b': ar.params_total / 1e9, 'active_b': ar.params_active_decode / 1e9}}


def catalog() -> dict:
    """Static lists for building a UI."""
    return {
        'gpus': [{'key': g.key, 'vendor': g.vendor, 'vram_gb': g.vram_gb, 'bw_tbs': g.mem_bw_gbs / 1000,
                  'bf16': g.bf16_tflops, 'fp8': g.fp8_tflops, 'fp4': g.fp4_tflops, 'calibrated': g.key in CAL_GPUS}
                 for g in CATALOG.values()],
        'schemes': [{'key': k, 'desc': s.desc, 'bits': s.w_bits} for k, s in SCHEMES.items()],
        'aliases': ALIASES,
        'families': sorted(set(FAMILIES)),
        'prices': tco.load_prices(),
        'size_gpus': list(SIZE_GPUS),
    }
