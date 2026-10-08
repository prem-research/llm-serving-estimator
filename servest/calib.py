"""Fit the efficiency/overhead parameters of the physics model to measured benchmarks and
evaluate generalisation with grouped hold-outs (model family, hardware class, source)."""
from __future__ import annotations
import copy
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import perf
from .perf import DEFAULT_PARAMS, ENGINES, HW_CLASSES, KERNELS, LINKS, SPECS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
TARGETS = ('tpot_ms', 'ttft_ms', 'output_tps')
TARGET_W = {'tpot_ms': 1.0, 'ttft_ms': 0.6, 'output_tps': 1.0}

# (name, path into P, init, prior sigma in log space)
def param_spec():
    s = []
    for h in HW_CLASSES:
        s.append((f'eta_bw.{h}', ('eta_bw', h), DEFAULT_PARAMS['eta_bw'][h], 0.5))
    s += [('eta_dec', ('eta_dec',), 0.45, 0.7), ('eta_pre', ('eta_pre',), 0.55, 0.7)]
    for k in KERNELS[1:]:
        s.append((f'kernel_bw.{k}', ('kernel_bw', k), 1.0, 0.4))
        s.append((f'kernel_pre.{k}', ('kernel_pre', k), 1.0, 0.4))
    for e in ENGINES[1:]:
        s.append((f'engine_dec.{e}', ('engine_dec', e), 1.0, 0.4))
        s.append((f'engine_pre.{e}', ('engine_pre', e), 1.0, 0.5))
    s += [('t_fixed_ms', ('t_fixed_ms',), 0.6, 1.0), ('t_layer_us', ('t_layer_us',), 6.0, 1.0)]
    for l in LINKS:
        s.append((f'ar_lat_us.{l}', ('ar_lat_us', l), DEFAULT_PARAMS['ar_lat_us'][l], 1.0))
        s.append((f'link_eff.{l}', ('link_eff', l), 0.7, 0.7))
    for sp in SPECS[1:]:
        s.append((f'spec_alpha.{sp}', ('spec_alpha', sp), DEFAULT_PARAMS['spec_alpha'][sp], 0.4))
    for e in ENGINES:
        s.append((f'ttft_fixed_ms.{e}', ('ttft_fixed_ms', e), 20.0, 1.5))
    for e in ENGINES:
        s.append((f'ttft_q.{e}', ('ttft_q', e), 1.0, 1.0))
    s += [
          ('moe_eff', ('moe_eff',), 1.0, 0.5), ('chunk', ('chunk',), 8192.0, 0.7), ('moe_comm_us', ('moe_comm_us',), 15.0, 1.2),
          ('t_moe_layer_us', ('t_moe_layer_us',), 10.0, 1.2)]
    return s


SPEC = param_spec()

PHYS_BOUNDS = {'eta_bw': (0.35, 0.95), 'link_eff': (0.45, 0.92), 'eta_pre': (0.15, 0.85), 'eta_dec': (0.15, 0.85),
               't_layer_us': (0.5, 80.0), 'ar_lat_us': (3.0, 40.0), 'moe_comm_us': (0.5, 200.0), 't_moe_layer_us': (0.5, 150.0), 'chunk': (2048.0, 16384.0), 'moe_eff': (0.7, 2.5)}


def bounds_arrays(prior):
    sig = np.array([s[3] for s in SPEC])
    lo, hi = prior - 4 * sig, prior + 4 * sig
    for i, (name, path, init, sg) in enumerate(SPEC):
        b = PHYS_BOUNDS.get(path[0])
        if b:
            lo[i], hi[i] = max(lo[i], np.log(b[0])), min(hi[i], np.log(b[1]))
    return lo, hi


def vec_to_params(x, base=None):
    P = copy.deepcopy(base or DEFAULT_PARAMS)
    P.setdefault('link_eff', {l: 0.7 for l in LINKS})
    if not isinstance(P.get('ttft_q'), dict):
        P['ttft_q'] = {e: 1.0 for e in ENGINES}
    if not isinstance(P.get('ttft_fixed_ms'), dict):
        P['ttft_fixed_ms'] = {e: 20.0 for e in ENGINES}
    P.setdefault('moe_eff', 1.0)
    P.setdefault('moe_comm_us', 15.0)
    P.setdefault('t_moe_layer_us', 10.0)
    for (name, path, init, sig), v in zip(SPEC, x):
        val = float(np.exp(v))
        if len(path) == 1:
            P[path[0]] = val
        else:
            P[path[0]][path[1]] = val
    return P


def params_to_vec(P):
    x = []
    for name, path, init, sig in SPEC:
        v = P.get(path[0]) if len(path) == 1 else (P.get(path[0]) or {}).get(path[1], init)
        x.append(np.log(v if v else init))
    return np.array(x)


def load(src='primary'):
    data = pickle.load(open(os.path.join(DATA, f'calib_rows_{src}.pkl'), 'rb'))
    F = [d[0] for d in data]
    Lb = pd.DataFrame([d[1] for d in data])
    M = pd.DataFrame([d[2] for d in data])
    return F, Lb, M


def predict(F, P):
    out = perf.solve(F, P)
    return pd.DataFrame({'tpot_ms': out['tpot_s'] * 1e3, 'ttft_ms': out['ttft_s'] * 1e3, 'output_tps': out['output_tps']})


def _row_weights(M, fam_w=None):
    g = M['group'].map(M['group'].value_counts())
    w = 1.0 / np.sqrt(g.values)
    if fam_w is not None:
        w = w * M['family'].map(fam_w).fillna(1.0).values
    return w


def make_residual_fn(F, Lb, M, x0, fam_w=None):
    V = perf._vec(F)
    w = _row_weights(M, fam_w)
    masks, ys, ws = {}, {}, {}
    for t in TARGETS:
        m = Lb[t].notna().values & (Lb[t].fillna(0).values > 0)
        masks[t] = m
        ys[t] = np.log(Lb[t].values[m])
        ws[t] = np.sqrt(TARGET_W[t] * w[m] / w[m].mean())
    sig = np.array([s[3] for s in SPEC])
    n_obs = sum(m.sum() for m in masks.values())
    prior_w = np.sqrt(n_obs / 400.0)

    def fn(x):
        P = vec_to_params(x)
        out = perf.solve(V, P)
        pred = {'tpot_ms': out['tpot_s'] * 1e3, 'ttft_ms': out['ttft_s'] * 1e3, 'output_tps': out['output_tps']}
        res = []
        for t in TARGETS:
            p = np.log(np.maximum(pred[t][masks[t]], 1e-9))
            r = (p - ys[t]) * ws[t]
            res.append(np.nan_to_num(r, nan=5.0, posinf=5.0, neginf=-5.0))
        res.append(prior_w * (x - x0) / sig)
        return np.concatenate(res)
    return fn


def family_weights(F, Lb, M, P, scale=0.25):
    """Down-weight whole families the physics cannot explain (e.g. immature kernels for a brand-new
    architecture) so they do not distort shared parameters; their offset goes to the residual layer."""
    pred = predict(F, P)
    meds = []
    for t in ('tpot_ms', 'output_tps'):
        m = Lb[t].notna() & (Lb[t] > 0)
        e = pd.Series(np.log(pred[t][m] / Lb[t][m]).values, index=M.index[m])
        meds.append(e.groupby(M['family'][m]).median())
    med = pd.concat(meds, axis=1).abs().max(axis=1)
    return (1.0 / (1.0 + (med / scale) ** 2)).to_dict()


def fit(F, Lb, M, x0=None, max_nfev=60, verbose=0, robust=True):
    prior = params_to_vec(vec_to_params(np.log([s[2] for s in SPEC])))
    x0 = prior if x0 is None else x0
    lo, hi = bounds_arrays(prior)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    stages = [None, 'fam'] if robust else [None]
    fam_w = None
    for st in stages:
        if st == 'fam':
            fam_w = family_weights(F, Lb, M, vec_to_params(r.x))
        fn = make_residual_fn(F, Lb, M, prior, fam_w)
        r = least_squares(fn, x0, bounds=(lo, hi), loss='soft_l1', f_scale=0.25,
                          max_nfev=max_nfev if st else max(10, max_nfev // 2), x_scale=1.0, diff_step=1e-3, verbose=verbose)
        x0 = r.x
    P = vec_to_params(r.x)
    P['_family_weights'] = fam_w
    return P, r


def metrics(pred, Lb, mask=None):
    rows = {}
    for t in TARGETS:
        m = Lb[t].notna() & (Lb[t] > 0) & np.isfinite(pred[t]) & (pred[t] > 0)
        if mask is not None:
            m &= mask
        if m.sum() == 0:
            continue
        ratio = pred[t][m] / Lb[t][m]
        ape = np.abs(ratio - 1) * 100
        le = np.abs(np.log(ratio))
        rows[t] = {'n': int(m.sum()), 'mdape': float(np.median(ape)), 'p90_ape': float(np.percentile(ape, 90)),
                   'within20': float((ape <= 20).mean() * 100), 'within50': float((ape <= 50).mean() * 100),
                   'within2x': float((le <= np.log(2)).mean() * 100), 'bias_x': float(np.exp(np.median(np.log(ratio))))}
    return rows


def holdout_eval(F, Lb, M, key, values, P_full, max_nfev=25):
    """Refit without each held-out value; score on it."""
    res = {}
    x_full = params_to_vec(P_full)
    preds = pd.DataFrame(index=M.index, columns=list(TARGETS), dtype=float)
    for v in values:
        te = (M[key] == v).values
        tr = ~te
        if te.sum() < 10:
            continue
        Ftr = [f for f, k in zip(F, tr) if k]
        Ptr, _ = fit(Ftr, Lb[tr].reset_index(drop=True), M[tr].reset_index(drop=True), x0=x_full, max_nfev=max_nfev)
        Fte = [f for f, k in zip(F, te) if k]
        p = predict(Fte, Ptr)
        p.index = M.index[te]
        preds.loc[te, list(TARGETS)] = p.values
        res[str(v)] = metrics(p, Lb[te].set_index(M.index[te]))
    return res, preds


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nfev', type=int, default=80)
    ap.add_argument('--cv', action='store_true')
    a = ap.parse_args()
    F, Lb, M = load('primary')
    t0 = time.time()
    P, r = fit(F, Lb, M, max_nfev=a.nfev, verbose=1)
    print(f'fit {time.time()-t0:.0f}s cost {r.cost:.1f} nfev {r.nfev}')
    pred = predict(F, P)
    print('in-sample', json.dumps(metrics(pred, Lb), indent=1))
    Fe, Le, Me = load('external')
    pe = predict(Fe, P)
    print('external', json.dumps(metrics(pe, Le), indent=1))
    json.dump({'params': P, 'fit': {'cost': float(r.cost), 'nfev': int(r.nfev), 'rows': len(F)}}, open(os.path.join(DATA, 'calibration.json'), 'w'), indent=1)
    from . import residual
    eff = {}
    for t in TARGETS:
        rr = np.log(Lb[t].values) - np.log(pred[t].values)
        eff[t] = residual.fit(M, rr)
    residual.save(eff)
    fam = {}
    for _, m in M.iterrows():
        fam[m['model']] = m['family']
    json.dump(fam, open(os.path.join(DATA, 'families.json'), 'w'), indent=0)
    print('saved calibration.json, residuals.json, families.json')
