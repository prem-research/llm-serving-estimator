"""Empirical error bands (measured / predicted quantiles) per support regime, from held-out data.

known   : frozen grouped split, test rows whose model family AND GPU appear in the fit data
partial : leave-one-family-out over the largest families (family unseen, GPU seen)
novel   : external campaigns (new publishers/hardware/models; never used for fitting)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from servest import calib, residual  # noqa: E402
from evaluate import split_of, subset  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


def corrected(F, M, P, eff):
    p = calib.predict(F, P)
    for t in calib.TARGETS:
        p[t] = p[t] * np.exp(np.clip(residual.apply(M, eff[t]), -0.7, 0.7))
    return p


def fit_with_resid(F, Lb, M, nfev):
    P, _ = calib.fit(F, Lb, M, max_nfev=nfev)
    pr = calib.predict(F, P)
    eff = {t: residual.fit(M, np.log(Lb[t].values) - np.log(pr[t].values)) for t in calib.TARGETS}
    return P, eff


def ratios(pred, Lb):
    out = {}
    for t in calib.TARGETS:
        m = Lb[t].notna() & (Lb[t] > 0) & np.isfinite(pred[t]) & (pred[t] > 0)
        out[t] = (Lb[t][m] / pred[t][m]).values
    return out


def main(nfev=30, n_fam=8):
    F, Lb, M = calib.load('primary')
    Fe, Le, Me = calib.load('external')
    sp = split_of(M)
    tr = sp.isin(['train', 'validation']).values
    Ftr, Ltr, Mtr = subset(F, Lb, M, tr)
    P, eff = fit_with_resid(Ftr, Ltr, Mtr, nfev)
    te = (sp == 'test').values & M['family'].isin(set(Mtr['family'])).values & M['gpu'].isin(set(Mtr['gpu'])).values
    Fte, Lte, Mte = subset(F, Lb, M, te)
    R = {'known': ratios(corrected(Fte, Mte, P, eff), Lte)}
    R['novel'] = ratios(corrected(Fe, Me, P, eff), Le)
    part = {t: [] for t in calib.TARGETS}
    for fam in M['family'].value_counts().index[:n_fam]:
        m = (M['family'] == fam).values
        Fa, La, Ma = subset(F, Lb, M, ~m)
        Fb, Lbb, Mb = subset(F, Lb, M, m)
        Pf, ef = fit_with_resid(Fa, La, Ma, max(12, nfev // 2))
        r = ratios(corrected(Fb, Mb, Pf, ef), Lbb)
        for t in calib.TARGETS:
            # equal weight per family: subsample to at most 300 rows
            v = r[t]
            if len(v) > 300:
                v = np.random.default_rng(0).choice(v, 300, replace=False)
            part[t].append(v)
        print('family done', fam, {t: round(float(np.median(np.abs(np.log(r[t])))), 3) for t in r}, flush=True)
    R['partial'] = {t: np.concatenate(v) for t, v in part.items()}
    bands = {}
    for reg, rr in R.items():
        bands[reg] = {}
        for t, v in rr.items():
            if len(v) < 10:
                continue
            ape = np.abs(v - 1) * 100
            bands[reg][t] = {'n': int(len(v)), 'p10': float(np.percentile(v, 10)), 'p50': float(np.percentile(v, 50)),
                             'p90': float(np.percentile(v, 90)), 'mdape': float(np.median(np.abs(1 / v - 1) * 100)),
                             'within2x': float(np.mean(np.abs(np.log(v)) <= np.log(2)) * 100)}
    json.dump(bands, open(os.path.join(DATA, 'error_bands.json'), 'w'), indent=1)
    print(json.dumps(bands, indent=1))


if __name__ == '__main__':
    main()
