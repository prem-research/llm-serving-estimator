"""Evaluation on the corpus's frozen grouped split + external campaigns,
plus leave-one-hardware-class-out and leave-one-family-out stress tests."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from servest import calib, residual  # noqa: E402

from servest.dataset import STUDY  # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(OUT, exist_ok=True)


def split_of(M):
    s = {}
    for name in ['train', 'validation', 'calibration', 'test']:
        for rid in pd.read_csv(os.path.join(STUDY, 'splits', f'{name}.csv'))['row_id']:
            s[rid] = name
    return M['row_id'].map(s)


def resid_targets(F, Lb, M, P):
    pred = calib.predict(F, P)
    return pred, {t: np.log(Lb[t].values) - np.log(pred[t].values) for t in calib.TARGETS}


def score(pred, Lb, eff=None, M=None):
    p = pred.copy()
    if eff is not None:
        for t in calib.TARGETS:
            p[t] = p[t] * np.exp(residual.apply(M, eff[t]))
    return calib.metrics(p, Lb), p


def subset(F, Lb, M, mask):
    idx = np.where(mask)[0]
    return [F[i] for i in idx], Lb.iloc[idx].reset_index(drop=True), M.iloc[idx].reset_index(drop=True)


def main(nfev=40, quick=False):
    F, Lb, M = calib.load('primary')
    Fe, Le, Me = calib.load('external')
    sp = split_of(M)
    res = {}
    # ---------------- 1. frozen grouped split: fit on train+validation, score on test
    tr = sp.isin(['train', 'validation']).values
    te = (sp == 'test').values
    Ftr, Ltr, Mtr = subset(F, Lb, M, tr)
    Fte, Lte, Mte = subset(F, Lb, M, te)
    t0 = time.time()
    P, _ = calib.fit(Ftr, Ltr, Mtr, max_nfev=nfev)
    print(f'fit on {tr.sum()} rows in {time.time()-t0:.0f}s')
    _, rtr = resid_targets(Ftr, Ltr, Mtr, P)
    eff = {t: residual.fit(Mtr, rtr[t]) for t in calib.TARGETS}
    pte = calib.predict(Fte, P)
    res['frozen_split_test'] = {'physics': score(pte, Lte)[0], 'physics+residual': score(pte, Lte, eff, Mte)[0],
                               'n_rows': int(te.sum())}
    pe = calib.predict(Fe, P)
    res['external'] = {'physics': score(pe, Le)[0], 'physics+residual': score(pe, Le, eff, Me)[0]}
    # by source on test
    bysrc = {}
    for s in Mte['source'].unique():
        m = (Mte['source'] == s).values
        if m.sum() >= 20:
            bysrc[s] = score(pte[m].reset_index(drop=True), Lte[m].reset_index(drop=True), eff, Mte[m].reset_index(drop=True))[0]
    res['frozen_split_test_by_source'] = bysrc
    for sec in ['frozen_split_test', 'external']:
        for var in ['physics', 'physics+residual']:
            print(f'{sec:17s} {var:17s}', '  '.join(f"{t}: MdAPE {v['mdape']:5.1f}% p90 {v['p90_ape']:6.1f}% <=20% {v['within20']:4.0f}% <=2x {v['within2x']:4.0f}% bias x{v['bias_x']:.2f}" for t, v in res[sec][var].items()))
    json.dump(res, open(os.path.join(OUT, 'evaluation_quick.json'), 'w'), indent=1)
    if quick:
        return res
    # ---------------- 2. leave-one-hardware-class-out (physics refit; GPU-level residuals unavailable)
    lohc = {}
    for hc in ['hopper', 'blackwell', 'blackwell-ultra', 'cdna3', 'cdna4', 'ampere', 'ada', 'blackwell-ws']:
        m = (M['hw_class'] == hc).values
        if m.sum() < 50:
            continue
        Fa, La, Ma = subset(F, Lb, M, ~m)
        Fb, Lbb, Mb = subset(F, Lb, M, m)
        Ph, _ = calib.fit(Fa, La, Ma, max_nfev=max(15, nfev // 2))
        _, ra = resid_targets(Fa, La, Ma, Ph)
        ea = {t: residual.fit(Ma, ra[t]) for t in calib.TARGETS}
        pb = calib.predict(Fb, Ph)
        lohc[hc] = {'n': int(m.sum()), 'physics': score(pb, Lbb)[0], 'physics+residual': score(pb, Lbb, ea, Mb)[0]}
        print('LOHC', hc, json.dumps({k: {t: round(v[t]['mdape'], 1) for t in v} for k, v in lohc[hc].items() if k != 'n'}))
    res['leave_hw_class_out'] = lohc
    # ---------------- 3. leave-one-family-out (largest families)
    lofo = {}
    top = M['family'].value_counts()
    for fam in list(top.index[:10]):
        m = (M['family'] == fam).values
        Fa, La, Ma = subset(F, Lb, M, ~m)
        Fb, Lbb, Mb = subset(F, Lb, M, m)
        Pf, _ = calib.fit(Fa, La, Ma, max_nfev=max(15, nfev // 2))
        _, ra = resid_targets(Fa, La, Ma, Pf)
        ea = {t: residual.fit(Ma, ra[t]) for t in calib.TARGETS}
        pb = calib.predict(Fb, Pf)
        lofo[fam] = {'n': int(m.sum()), 'physics': score(pb, Lbb)[0], 'physics+residual': score(pb, Lbb, ea, Mb)[0]}
        print('LOFO', fam, json.dumps({k: {t: round(v[t]['mdape'], 1) for t in v} for k, v in lofo[fam].items() if k != 'n'}))
    res['leave_family_out'] = lofo
    json.dump(res, open(os.path.join(OUT, 'evaluation.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    main(nfev=int(sys.argv[1]) if len(sys.argv) > 1 else 40, quick='--quick' in sys.argv)
