"""servest: "I have N x GPU, can I serve model M to C users, and what TPS/TTFT do I get?"

Examples
  servest estimate --model mistralai/Mistral-Small-4-119B-2603 --gpu H100 --gpus 4 --users 100 --isl 4096 --osl 1024
  servest sweep    --model openai/gpt-oss-120b --gpu H200 --gpus 1 --isl 1024 --osl 1024 --users 1,8,32,128,512
  servest capacity --model deepseek-ai/DeepSeek-V4-Flash --gpu B200 --gpus 8 --isl 8192 --osl 1024 --min-tps 30 --max-ttft 5
  servest plan     --model meta-llama/Llama-3.3-70B-Instruct --gpu H100 --users 200 --isl 2048 --osl 512 --min-tps 25
  servest model    deepseek-ai/DeepSeek-V4.1-Flash
  servest gpus
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np

from . import arch as A
from . import perf, residual
from .gpus import CATALOG, resolve_gpu
from .perf import Deployment, Workload, featurize, hw_class, load_params, solve
from .precision import SCHEMES, ALIASES, adapt_to_gpu, get_scheme

from .api import (Query, band, regime_of, family_of, load_arch, CAL_GPUS, FAMILIES, MODEL_TYPES, BANDS, EFFECTS)
from . import api


def query(args, users=None) -> Query:
    return Query(model=args.model, gpu=args.gpu, gpus=args.gpus, users=users if users is not None else getattr(args, 'users', None),
                 rate=args.rate, isl=args.isl, osl=args.osl, quant=args.quant, kv_dtype=args.kv_dtype, engine=args.engine,
                 tp=args.tp, dp_attention=args.dp_attention, spec=args.spec, prefix_hit=args.prefix_hit,
                 think_time=args.think_time, mem_util=args.mem_util, engram_in_hbm=args.engram_in_hbm)


def best_config(args, P, users=None, n_gpus=None):
    ar = load_arch(args.model, fetch=not args.offline)
    feas, out = api.best_config(ar, query(args, users), P, users=users, n_gpus=n_gpus)
    return feas, out, ar, resolve_gpu(args.gpu)


# ----------------------------------------------------------------------------- formatting

def gb(x):
    return f'{x / 1e9:,.1f} GB'


def report(res, args, P):
    f, o, ar, gpu, dep, wl = res['f'], res['o'], res['ar'], res['gpu'], res['dep'], res['wl']
    sch_used = f['_scheme']
    w = f['_weights']
    reg, tags = regime_of(res['fam'], gpu.key, wl.isl, wl.users, ar)
    lines = []
    lines.append(f"Model      {ar.repo}  [{ar.model_type}{', MoE %dx top-%d' % (ar.n_experts, ar.top_k) if ar.is_moe else ', dense'}, "
                 f"{ar.n_layers} layers]  total {ar.params_total/1e9:,.1f}B, active {ar.params_active_decode/1e9:,.1f}B"
                 + (f" (prefill {ar.params_active_prefill/1e9:,.1f}B)" if ar.prefill_active_scale != 1 else ''))
    lines.append(f"Hardware   {dep.n_gpus} x {gpu.key} ({gpu.vram_gb:g} GB, {gpu.mem_bw_gbs/1000:g} TB/s, {gpu.interconnect})  "
                 f"-> TP{res['tp']} x {res['replicas']} replica(s), engine {dep.engine}, spec-decode {dep.spec}")
    lines.append(f"Precision  weights {sch_used} ({SCHEMES.get(sch_used.split('@')[0], get_scheme('bf16')).desc}); compute {f['_compute'][0]}/{f['_compute'][1]}; KV cache {f['_kv']}")
    for n in f['_notes'] + ar.notes:
        lines.append(f'           note: {n}')
    rep = res['tp']
    per_gpu_w = w['total'] / rep
    lines.append('')
    lines.append(f"Memory     weights {gb(w['total'])} per replica = {gb(per_gpu_w)}/GPU "
                 f"(experts {gb(w['experts'])}, dense {gb(w['dense_linear'])}, emb/head {gb(w['embed_lm_head'])}"
                 + (f", MTP {gb(w['mtp'])}" if w['mtp'] else '') + (f", Engram {gb(w['engram'])}" if w['engram'] else '') + ')')
    ctx = wl.isl + wl.osl
    if f['cap_bytes'] <= 0:
        lines.append(f"           DOES NOT FIT: weights exceed {dep.mem_util or P['mem_util']:.0%} of {rep} x {gpu.vram_gb:g} GB. "
                     f"Use more GPUs per replica (TP), a smaller quantization, or a larger GPU.")
        return '\n'.join(lines)
    users_rep = (wl.users or 0) / res['replicas']
    lines.append(f"           KV/state budget {gb(f['cap_bytes'])}/replica; per user at {ctx:,.0f} tokens: {f['per_seq_bytes']/1e6:,.1f} MB"
                 f" -> max ~{f['bmax']:,.0f} concurrent sequences/replica ({f['bmax']*res['replicas']:,.0f} total)")
    if wl.users:
        fits = users_rep <= f['bmax']
        lines.append(f"           {'FITS' if fits else 'KV-LIMITED'}: {wl.users:,.0f} users -> {users_rep:,.0f}/replica "
                     f"({users_rep / max(f['bmax'], 1e-9):.0%} of KV capacity){'' if fits else '; extra users queue (TTFT grows)'}")
    lines.append('')
    load = (f"{wl.users:,.0f} concurrent users (closed loop" + (f", {wl.think_s:g} s think time)" if wl.think_s else ", no think time)")) if wl.users else f"{wl.rate} req/s (open loop)"
    lines.append(f"Performance  {load}, prompt {wl.isl:,.0f} tok, output {wl.osl:,.0f} tok" + (f", prefix-cache hit {wl.prefix_hit:.0%}" if wl.prefix_hit else ''))
    tps_u = o['decode_tps_per_user']
    bt = band('tpot_ms', reg)
    tps_band = f'[{tps_u / bt[1]:,.0f} – {tps_u / bt[0]:,.0f}]' if bt else ''
    tpot_band = f"[{o['tpot_s']*1e3*bt[0]:,.1f} – {o['tpot_s']*1e3*bt[1]:,.1f}]" if bt else ''
    bf = band('ttft_ms', reg)
    ttft_band = f"[{o['ttft_s']*bf[0]:,.2f} – {o['ttft_s']*bf[1]:,.2f}]" if bf else ''
    bo = band('output_tps', reg)
    out_band = f"[{o['output_tps']*bo[0]:,.0f} – {o['output_tps']*bo[1]:,.0f}]" if bo else ''
    lines.append(f"  per-user decode speed   {tps_u:9,.1f} tok/s   {tps_band:>22s}   (TPOT {o['tpot_s']*1e3:,.1f} ms {tpot_band})")
    lines.append(f"  time to first token     {o['ttft_s']:9,.2f} s       {ttft_band:>22s}   (typical/median request)")
    lines.append(f"  aggregate output        {o['output_tps']:9,.0f} tok/s   {out_band:>22s}   ({o['output_tps']/dep.n_gpus:,.0f} tok/s/GPU)")
    lines.append(f"  aggregate input+output  {o['total_tps']:9,.0f} tok/s   requests {o['requests_per_s']:,.2f}/s   end-to-end {o['e2e_s']:,.1f} s/request")
    td = o['step_s'] * 1e3
    lines.append(f"  internals: decode step {td:,.1f} ms at batch {o['batch_per_replica']:,.0f}/replica; prefill {o['prefill_s']:,.2f} s GPU-time/request; "
                 f"prefill share of GPU time {min(o['prefill_util'],1):.0%}")
    if o['kv_limited']:
        lines.append('  WARNING: users exceed KV-cache capacity; excess requests wait in queue.')
    if wl.rate and o['overloaded']:
        lines.append('  WARNING: offered request rate exceeds capacity (queue grows without bound).')
    lines.append('')
    corr_txt = ', '.join(f"{t.split('_')[0]} x{math.exp(c):.2f}" for t, (c, used) in res['corr'].items() if abs(c) > 0.01)
    lines.append(f"Confidence  regime '{reg}': model family {'seen ('+res['fam']+')' if res['fam'] else 'NOT in calibration data'}, "
                 f"GPU {'seen' if gpu.key in CAL_GPUS else 'NOT in calibration data'}" + (f"; data-driven correction {corr_txt}" if corr_txt else ''))
    lines.append("            bands = 10th–90th percentile of measured/predicted on held-out data for this regime")
    for t in tags:
        lines.append(f'  caution: {t}')
    return '\n'.join(lines)


def to_json(res):
    f, o = res['f'], res['o']
    return {'model': res['ar'].repo, 'gpu': res['gpu'].key, 'n_gpus': res['dep'].n_gpus, 'tp': res['tp'], 'replicas': res['replicas'],
            'scheme': f['_scheme'], 'kv_dtype': f['_kv'], 'weights_gb': f['_weights']['total'] / 1e9,
            'kv_budget_gb_per_replica': f['cap_bytes'] / 1e9, 'kv_per_user_mb': f['per_seq_bytes'] / 1e6,
            'max_concurrent_seqs': f['bmax'] * res['replicas'], 'feasible': res['feasible'],
            'decode_tps_per_user': o['decode_tps_per_user'], 'tpot_ms': o['tpot_s'] * 1e3, 'ttft_s': o['ttft_s'],
            'output_tps': o['output_tps'], 'total_tps': o['total_tps'], 'requests_per_s': o['requests_per_s'],
            'e2e_s': o['e2e_s'], 'kv_limited': bool(o['kv_limited']), 'model_family': res['fam']}


# ----------------------------------------------------------------------------- commands

def cmd_estimate(args):
    P = load_params()
    feas, allr, ar, gpu = best_config(args, P)
    if not feas:
        r = allr[-1] if allr else None
        if r is None:
            print('No valid TP configuration for this GPU count.')
            return 1
        r.update(ar=ar, gpu=gpu)
        print(report(r, args, P))
        return 1
    best = feas[0]
    best.update(ar=ar, gpu=gpu)
    if args.json:
        print(json.dumps(to_json(best), indent=1))
        return 0
    print(report(best, args, P))
    if len(feas) > 1 and not args.tp:
        print('\nOther parallel layouts for the same GPUs:')
        print(f"  {'TP':>3} {'replicas':>8} {'tok/s/user':>10} {'TTFT s':>8} {'agg tok/s':>10}")
        for r in feas:
            print(f"  {r['tp']:>3} {r['replicas']:>8} {r['o']['decode_tps_per_user']:>10.1f} {r['o']['ttft_s']:>8.2f} {r['o']['output_tps']:>10,.0f}")
    return 0


def _parse_list(s):
    return [float(x) for x in str(s).split(',') if x.strip()]


def cmd_sweep(args):
    P = load_params()
    users = _parse_list(args.users_list)
    print(f"{args.model} on {args.gpus} x {args.gpu}" + (f" (TP{args.tp})" if args.tp else " (best layout per load; fix with --tp)")
          + f", prompt {args.isl:.0f}, output {args.osl:.0f}"
          + (f", think time {args.think_time:g} s" if args.think_time else ''))
    print(f"{'users':>7} {'TP':>3} {'tok/s/user':>10} {'TTFT s':>8} {'agg out tok/s':>13} {'tok/s/GPU':>9} {'KV use':>7}")
    for u in users:
        feas, _, ar, gpu = best_config(args, P, users=u)
        if not feas:
            print(f'{u:>7.0f}  does not fit')
            continue
        r = feas[0]
        o, f = r['o'], r['f']
        print(f"{u:>7.0f} {r['tp']:>3} {o['decode_tps_per_user']:>10.1f} {o['ttft_s']:>8.2f} {o['output_tps']:>13,.0f} "
              f"{o['output_tps']/args.gpus:>9,.0f} {u / r['replicas'] / max(f['bmax'],1e-9):>6.0%}")
    return 0


def _meets(o, args):
    ok = True
    if args.min_tps:
        ok &= o['decode_tps_per_user'] >= args.min_tps
    if args.max_ttft:
        ok &= o['ttft_s'] <= args.max_ttft
    return ok and not o['kv_limited']


def cmd_capacity(args):
    P = load_params()
    lo, hi = 0.0, 1.0
    best = None
    while hi < 1e6:
        feas, _, ar, gpu = best_config(args, P, users=hi)
        ok = [r for r in feas if _meets(r['o'], args)]
        if not ok:
            break
        best = (hi, ok[0])
        lo, hi = hi, hi * 2
    if best is None:
        print('Even 1 user does not meet the target on this deployment.')
        return 1
    for _ in range(14):
        mid = (lo + hi) / 2
        feas, _, ar, gpu = best_config(args, P, users=mid)
        ok = [r for r in feas if _meets(r['o'], args)]
        if ok:
            lo, best = mid, (mid, ok[0])
        else:
            hi = mid
    u, r = best
    o = r['o']
    print(f"Max concurrent users meeting target (>= {args.min_tps or 0} tok/s/user, TTFT <= {args.max_ttft or 'inf'} s): "
          f"~{u:,.0f}  (TP{r['tp']} x {r['replicas']})")
    print(f"  at that load: {o['decode_tps_per_user']:.1f} tok/s/user, TTFT {o['ttft_s']:.2f} s, aggregate {o['output_tps']:,.0f} tok/s")
    print('  (central estimate; apply the error band from `estimate` for a conservative plan)')
    return 0


def cmd_plan(args):
    P = load_params()
    print(f"Smallest {args.gpu} deployment for {args.model}: {args.users:.0f} users, prompt {args.isl:.0f}, output {args.osl:.0f}, "
          f"target >= {args.min_tps or 0} tok/s/user, TTFT <= {args.max_ttft or 'inf'} s")
    print(f"{'GPUs':>5} {'TP':>3} {'fits':>5} {'tok/s/user':>10} {'TTFT s':>8} {'agg tok/s':>10}  meets")
    found = None
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        feas, allr, ar, gpu = best_config(args, P, users=args.users, n_gpus=n)
        if not feas:
            print(f'{n:>5}   -  no ')
            continue
        r = max(feas, key=lambda r: (_meets(r['o'], args), r['o']['decode_tps_per_user']))
        o = r['o']
        m = _meets(o, args)
        print(f"{n:>5} {r['tp']:>3} {'yes':>5} {o['decode_tps_per_user']:>10.1f} {o['ttft_s']:>8.2f} {o['output_tps']:>10,.0f}  {'YES' if m else 'no'}")
        if m:
            found = n
            break
    if not found:
        print('No configuration up to 128 GPUs meets the target.')
    return 0


def cmd_model(args):
    a = A.parse(args.model_repo, fetch=not args.offline)
    s = A.summary(a)
    for k, v in s.items():
        print(f'{k:28s} {v}')
    for L in (8192, 32768, 131072):
        print(f'KV+state per sequence @ {L:>6} tokens: bf16 {a.kv_bytes_per_seq(L, 2.0)/1e6:9.1f} MB   fp8 {a.kv_bytes_per_seq(L, 1.0)/1e6:9.1f} MB')
    for q in ('bf16', 'fp8', 'nvfp4', 'w4a16', 'fp4-experts'):
        print(f'weights as {q:12s}: {a.weight_bytes(get_scheme(q))["total"]/1e9:8.1f} GB')
    return 0


def cmd_gpus(args):
    print(f"{'key':20s} {'VRAM':>6} {'TB/s':>5} {'BF16':>6} {'FP8':>6} {'FP4':>6}  link")
    for g in CATALOG.values():
        print(f"{g.key:20s} {g.vram_gb:>6g} {g.mem_bw_gbs/1000:>5.2f} {g.bf16_tflops:>6g} {g.fp8_tflops or '-':>6} {g.fp4_tflops or '-':>6}  {g.interconnect}"
              + ('' if g.key in CAL_GPUS else '   (not in calibration data)'))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='servest', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, users=True):
        p.add_argument('--model', required=True, help='Hugging Face repo id (config fetched once and cached)')
        p.add_argument('--gpu', required=True, help='GPU type, e.g. H100, H200, B200, MI300X, L40S, RTX PRO 6000')
        p.add_argument('--gpus', type=int, default=8, help='number of GPUs available (default 8)')
        p.add_argument('--isl', type=float, default=2048, help='prompt tokens per request (default 2048)')
        p.add_argument('--osl', type=float, default=512, help='output tokens per request (default 512)')
        p.add_argument('--quant', default='native', help='native | ' + ' | '.join(sorted(SCHEMES)) + ' (aliases: awq, gptq, int4, w4a4, fp4, w8a8 ...)')
        p.add_argument('--kv-dtype', default='auto', help='auto | bf16 | fp8 | fp4 (auto = fp8 for fp8/fp4 weights, else bf16)')
        p.add_argument('--engine', default='vllm', choices=perf.ENGINES[:-1] + ['other'])
        p.add_argument('--tp', type=int, default=None, help='tensor-parallel size (default: try all and pick the best)')
        p.add_argument('--dp-attention', action='store_true', help='data-parallel attention (SGLang/vLLM DP+EP for MLA models)')
        p.add_argument('--spec', default='none', choices=perf.SPECS, help='speculative decoding (mtp/eagle add ~1.5-2x decode speed)')
        p.add_argument('--prefix-hit', type=float, default=0.0, help='fraction of prompt tokens served from prefix cache')
        p.add_argument('--mem-util', type=float, default=None, help='GPU memory fraction for weights+KV (default 0.92)')
        p.add_argument('--engram-in-hbm', action='store_true', help='keep DeepSeek Engram tables in GPU memory')
        p.add_argument('--offline', action='store_true', help='do not fetch configs from Hugging Face')
        p.add_argument('--json', action='store_true')
        p.add_argument('--rate', type=float, default=None, help='open-loop request rate (req/s) instead of users')
        p.add_argument('--think-time', type=float, default=0.0, help='seconds a user pauses between getting an answer and sending the next request (default 0 = continuous, benchmark-style worst case)')

    p = sub.add_parser('estimate', help='feasibility + TPS/TTFT for N GPUs, model, users')
    common(p)
    p.add_argument('--users', type=float, default=None, help='concurrent users (closed loop, no think time)')
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser('sweep', help='performance across user counts')
    common(p)
    p.add_argument('--users', dest='users_list', default='1,4,16,64,128,256,512')
    p.set_defaults(fn=cmd_sweep, users=None)
    p = sub.add_parser('capacity', help='max users meeting a per-user TPS / TTFT target')
    common(p)
    p.add_argument('--min-tps', type=float, default=None)
    p.add_argument('--max-ttft', type=float, default=None)
    p.set_defaults(fn=cmd_capacity, users=None)
    p = sub.add_parser('plan', help='smallest GPU count meeting a target for given users')
    common(p)
    p.add_argument('--users', type=float, required=True)
    p.add_argument('--min-tps', type=float, default=None)
    p.add_argument('--max-ttft', type=float, default=None)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser('model', help='architecture summary: params, KV per token, weight sizes')
    p.add_argument('model_repo')
    p.add_argument('--offline', action='store_true')
    p.set_defaults(fn=cmd_model)
    p = sub.add_parser('gpus', help='list GPU catalog')
    p.set_defaults(fn=cmd_gpus)
    args = ap.parse_args(argv)
    if getattr(args, 'cmd', None) == 'estimate' and args.users is None and args.rate is None:
        ap.error('estimate needs --users or --rate')
    try:
        return args.fn(args)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f'error: {e.args[0] if e.args else e}', file=sys.stderr)
        return 2
    except OSError as e:
        print(f'error fetching model metadata: {e} (use a cached model or check network / repo id)', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
