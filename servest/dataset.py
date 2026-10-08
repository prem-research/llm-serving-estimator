"""Build the calibration dataset from an audited public-benchmark corpus (assembled 2026-10-08).

Corrections applied on top of that corpus:
  * InferenceX expert-parallel rows: the API's GPU count is TP*EP; real single-node GPU count is
    TP and aggregate throughput was inflated by EP (verified with Little's law).
  * Configured-upper-bound lengths (random range ratio r) use the mean (1+r)/2 x nominal.
  * Quantization is mapped from labels + checkpoint names to storage/compute schemes.
"""
from __future__ import annotations
import json
import math
import os
import re

import numpy as np
import pandas as pd

from . import arch as A
from .gpus import resolve_gpu
from .perf import Deployment, Workload, featurize, load_params
from .precision import get_scheme

HERE = os.path.dirname(os.path.abspath(__file__))
# Audited public-benchmark corpus (training.csv + external_holdout/); not redistributed in this repo.
STUDY = os.environ.get('SERVEST_CORPUS', os.path.join(HERE, '..', 'corpus'))
OUT = os.path.join(HERE, '..', 'data', 'calib_rows.pkl')


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def map_scheme(row, ar: A.Arch, gpu) -> str:
    q = str(row.get('quantization') or '').lower()
    from .precision import SCHEMES, ALIASES
    if q in SCHEMES or q in ALIASES:
        return ALIASES.get(q, q)
    if q in ('native', 'auto'):
        return ar.native_scheme
    repo = str(row.get('model') or '').lower()
    dt = str(row.get('dtype') or '').lower()
    wb = _f(row.get('weight_bits'))
    tag = q + ' ' + repo
    if wb is None:
        if 'fp8' in tag:
            wb = 8
        elif any(x in tag for x in ('awq', 'gptq', 'int4', 'fp4', '4bit')):
            wb = 4
    if wb is None:
        return ar.native_scheme
    if wb >= 16:
        return 'bf16'
    if wb >= 8:
        if ar.native_scheme == 'fp4-experts':          # DeepSeek-V4 "fp8" = FP4 experts + FP8 rest
            return 'fp4-experts'
        if 'int8' in tag or 'w8a8' in tag and 'int' in tag:
            return 'int8'
        if 'mxfp8' in tag:
            return 'mxfp8'
        return 'fp8'
    # 4-bit
    if any(x in tag for x in ('awq', 'gptq', 'int4', 'w4a16', 'autoround', 'auto-round')):
        return 'w4a16'
    if ar.native_scheme in ('mxfp4-experts-bf16', 'fp4-experts', 'w4a16') and 'nvfp4' not in tag:
        return ar.native_scheme
    if ar.is_moe:
        return 'fp4-experts'
    if 'mxfp4' in tag or gpu.vendor == 'amd':
        return 'mxfp4'
    return 'nvfp4'


def map_spec(x) -> str:
    s = str(x).lower()
    if s in ('none', 'false', 'nan', '', '0'):
        return 'none'
    if 'mtp' in s:
        return 'mtp'
    if 'ngram' in s:
        return 'ngram'
    return 'eagle'


def map_engine(x) -> str:
    s = str(x).lower()
    if 'sglang' in s:
        return 'sglang'
    if 'trt' in s or 'tensorrt' in s:
        return 'tensorrt-llm'
    if 'atom' in s:
        return 'atom'
    if 'llama.cpp' in s or 'llamacpp' in s:
        return 'llama.cpp'
    if 'vllm' in s:
        return 'vllm'
    return 'other'


def row_to_scenario(row, arch_cache: dict):
    repo = row['model']
    if repo not in arch_cache:
        try:
            arch_cache[repo] = A.parse(repo, fetch=False)
        except Exception as e:
            arch_cache[repo] = e
    ar = arch_cache[repo]
    if isinstance(ar, Exception):
        return None, f'arch: {ar}'
    try:
        gpu = resolve_gpu(str(row['gpu']))
    except KeyError as e:
        return None, f'gpu: {e}'
    n = _f(row.get('gpu_count'))
    tp = _f(row.get('tensor_parallel'))
    ep = _f(row.get('expert_parallel')) or 1
    pp = _f(row.get('pipeline_parallel')) or 1
    if not n or n < 1:
        return None, 'gpu_count'
    fixed_ep = False
    if row.get('source_id') == 'inferencex' and ep > 1 and tp and abs(n - tp * ep) < 0.5:
        n = tp
        fixed_ep = True
    tp = int(tp) if tp else int(n)
    pp = int(pp)
    if int(n) % (tp * pp):
        tp = int(n)
        pp = 1
    scheme = get_scheme(map_scheme(row, ar, gpu))
    dpa = bool(_f(row.get('dp_attention')) or 0)
    dep = Deployment(arch=ar, gpu=gpu, n_gpus=int(n), scheme=scheme, kv_dtype=str(row.get('kv_cache_dtype') or 'auto') if isinstance(row.get('kv_cache_dtype'), str) else 'auto',
                     tp=tp, pp=pp, dp_attention=dpa, engine=map_engine(row.get('engine')), spec=map_spec(row.get('speculative_decoding')))
    isl, osl = _f(row.get('input_tokens')), _f(row.get('output_tokens'))
    if not isl or not osl:
        return None, 'lengths'
    r = _f(row.get('documented_random_range_ratio')) or _f(row.get('random_range_ratio'))
    if str(row.get('input_tokens_kind')) == 'configured_upper_bound' and r:
        isl *= (1 + r) / 2
        osl *= (1 + r) / 2
    conc = _f(row.get('concurrency'))
    rate = _f(row.get('request_rate'))
    if conc:
        wl = Workload(isl=isl, osl=osl, users=conc)
    elif rate:
        wl = Workload(isl=isl, osl=osl, rate=rate)
    else:
        return None, 'load'
    if str(row.get('scenario', '')).startswith('prefill') and 'wiederholt' in str(row.get('scenario')):
        wl.prefix_hit = 0.98
    try:
        f = featurize(dep, wl)
    except Exception as e:
        return None, f'featurize: {e}'
    if f['cap_bytes'] <= 0:
        return None, 'weights do not fit (metadata/precision mismatch)'
    out_tps = _f(row.get('output_tps'))
    if out_tps and fixed_ep:
        out_tps /= ep
    ttft = _f(row.get('ttft_median_ms'))
    # physically impossible TTFT (faster than the prompt's FLOPs at 100% of peak) => prefix-cache hits
    lb_ms = (f['flops_pre'] + f['attn_pre']) / f['rep'] / max(f['pk_e'], f['pk_d']) * 1e3
    cached = ttft is not None and wl.prefix_hit == 0 and ttft < 1.0 * lb_ms
    lab = {'tpot_ms': _f(row.get('tpot_median_ms')), 'ttft_ms': None if cached else ttft, 'output_tps': out_tps}
    meta = {'row_id': row.get('row_id'), 'source': row.get('source_id'), 'model': repo, 'model_type': ar.model_type,
            'family': str(row.get('canonical_model') or repo), 'gpu': gpu.key, 'hw_class': gpu.arch, 'vendor': gpu.vendor,
            'engine': dep.engine, 'scheme': f['_scheme'], 'spec': dep.spec, 'n_gpus': int(n), 'tp': tp, 'ep_fixed': fixed_ep,
            'isl': isl, 'osl': osl, 'users': conc, 'ttft_cached': cached, 'rate': rate if not conc else None,
            'group': f"{row.get('canonical_model')}|{gpu.key}|{dep.engine}|{int(n)}"}
    return (f, lab, meta), None


LOCAL_CSV = os.path.join(HERE, '..', 'data', 'local_measurements.csv')
LOCAL_COLUMNS = ['model', 'gpu', 'gpu_count', 'tensor_parallel', 'engine', 'quantization', 'kv_cache_dtype',
                 'speculative_decoding', 'input_tokens', 'output_tokens', 'concurrency', 'request_rate',
                 'ttft_median_ms', 'tpot_median_ms', 'output_tps']


def build(source: str = 'primary', limit: int | None = None):
    if source == 'primary':
        df = pd.read_csv(os.path.join(STUDY, 'data', 'training.csv'), low_memory=False)
        if os.path.exists(LOCAL_CSV):            # your own benchmark runs, weighted like any other source
            loc = pd.read_csv(LOCAL_CSV)
            loc['source_id'] = 'local'
            loc['canonical_model'] = loc['model'].map(lambda m: m.split('/')[-1].lower())
            loc['row_id'] = ['local-%d' % i for i in range(len(loc))]
            df = pd.concat([df, loc], ignore_index=True)
    else:
        rows = [json.loads(l) for l in open(os.path.join(STUDY, 'data', 'external_holdout', 'features.jsonl'))]
        df = pd.DataFrame(rows)
        if 'canonical_model' not in df:
            df['canonical_model'] = df['model']
    if limit:
        df = df.head(limit)
    cache, data, rej = {}, [], {}
    for _, row in df.iterrows():
        res, err = row_to_scenario(row.to_dict(), cache)
        if res is None:
            key = err.split(':')[0]
            rej[key] = rej.get(key, 0) + 1
            continue
        data.append(res)
    return data, rej


if __name__ == '__main__':
    import pickle, sys
    for src in ('primary', 'external'):
        data, rej = build(src)
        print(src, len(data), 'rows; rejected', rej)
        pickle.dump(data, open(OUT.replace('.pkl', f'_{src}.pkl'), 'wb'))
