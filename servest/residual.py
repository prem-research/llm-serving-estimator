"""Hierarchical shrinkage correction on top of the physics model.

log(measured) = log(physics) + sum of shrunken effects over the levels that apply:
  family, gpu, kernel x hw, family x gpu, family x gpu x engine
Effects are fitted by backfitting with ridge-style shrinkage (n / (n + lam)), so sparse
levels stay near zero and unseen combinations fall back to physics plus broader levels.
"""
from __future__ import annotations
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'data', 'residuals.json')
LEVELS = [('family',), ('gpu',), ('kernel', 'hw_class'), ('family', 'gpu'), ('family', 'gpu', 'engine')]
LAM = {('family',): 40.0, ('gpu',): 80.0, ('kernel', 'hw_class'): 80.0, ('family', 'gpu'): 25.0,
       ('family', 'gpu', 'engine'): 25.0}


def family_key(name: str) -> str:
    s = str(name).lower().split('/')[-1]
    s = re.sub(r'[-_.](fp8|nvfp4|mxfp4|mxfp8|fp4|awq.*|gptq.*|int4.*|int8.*|w4a16|w8a8|bf16|fp16|instruct.*|chat|it|base|hf)$', '', s)
    s = re.sub(r'[-_.](fp8|nvfp4|mxfp4|mxfp8|fp4|awq.*|gptq.*|int4.*|instruct.*|chat|it)$', '', s)
    return s


def _keys(M, lv):
    return M[list(lv)].astype(str).agg('|'.join, axis=1)


def fit(M, resid: np.ndarray, iters: int = 8) -> dict:
    import pandas as pd
    M = M.copy()
    M['kernel'] = M['scheme'].astype(str).str.split('@').str[0]
    m = np.isfinite(resid)
    r = resid.copy()
    keys = {lv: _keys(M, lv) for lv in LEVELS}
    eff = {lv: {} for lv in LEVELS}
    for _ in range(iters):
        for lv in LEVELS:
            other = np.zeros(len(M))
            for lv2 in LEVELS:
                if lv2 != lv:
                    other += keys[lv2].map(eff[lv2]).fillna(0).values
            part = pd.Series(r - other)[m].groupby(keys[lv][m].values)
            s, n = part.sum(), part.count()
            eff[lv] = (s / (n + LAM[lv])).to_dict()
    return {'|'.join(lv): v for lv, v in eff.items()}


def apply(M, eff: dict) -> np.ndarray:
    M = M.copy()
    M['kernel'] = M['scheme'].astype(str).str.split('@').str[0]
    out = np.zeros(len(M))
    for lv in LEVELS:
        e = eff.get('|'.join(lv), {})
        out += _keys(M, lv).map(e).fillna(0).values
    return out


def save(effects: dict):
    json.dump(effects, open(PATH, 'w'))


def load() -> dict:
    return json.load(open(PATH)) if os.path.exists(PATH) else {}


def lookup(meta: dict, effects_for_target: dict) -> tuple[float, list]:
    """Correction (log) for one scenario + which levels contributed (no pandas: runs in the browser)."""
    m = dict(meta)
    m['kernel'] = str(m.get('scheme', '')).split('@')[0]
    tot, used = 0.0, []
    for lv in LEVELS:
        k = '|'.join(str(m.get(c)) for c in lv)
        v = effects_for_target.get('|'.join(lv), {}).get(k)
        if v is not None:
            tot += v
            used.append(('x'.join(lv), round(float(np.exp(v)), 3)))
    return tot, used
