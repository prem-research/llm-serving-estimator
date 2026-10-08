"""Assemble the static GitHub Pages site in ./site from the package + calibration data.

The page runs the *same* Python package in the browser via Pyodide, so web and CLI agree exactly.
"""
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
SITE = os.path.join(ROOT, 'site')
WEB = os.path.join(ROOT, 'web')          # hand-written front-end sources
PY_MODULES = ['__init__.py', 'gpus.py', 'precision.py', 'arch.py', 'perf.py', 'residual.py', 'api.py']
DATA_FILES = ['calibration.json', 'error_bands.json', 'families.json', 'model_types.json', 'model_overrides.json',
              'residuals.json']


def main():
    if os.path.exists(SITE):
        shutil.rmtree(SITE)
    os.makedirs(os.path.join(SITE, 'py', 'servest'))
    os.makedirs(os.path.join(SITE, 'py', 'data'))
    for f in os.listdir(WEB):
        shutil.copy(os.path.join(WEB, f), SITE)
    for m in PY_MODULES:
        shutil.copy(os.path.join(ROOT, 'servest', m), os.path.join(SITE, 'py', 'servest', m))
    for d in DATA_FILES:
        shutil.copy(os.path.join(ROOT, 'data', d), os.path.join(SITE, 'py', 'data', d))
    # cached HF configs + parameter totals, so common models work offline / for gated repos
    cache = {}
    cdir, mdir = os.path.join(ROOT, 'data', 'configs'), os.path.join(ROOT, 'data', 'hf_meta')
    for f in sorted(os.listdir(cdir)):
        if not f.endswith('.json') or '__' not in f:
            continue
        repo = f[:-5].replace('__', '/', 1)
        meta_p = os.path.join(mdir, f)
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        cache[repo] = {'config': json.load(open(os.path.join(cdir, f))),
                       'meta': {'safetensors': meta.get('safetensors')} if meta.get('safetensors') else {}}
    json.dump(cache, open(os.path.join(SITE, 'model_cache.json'), 'w'), separators=(',', ':'))
    from servest import api
    cat = api.catalog()
    cat['cached_models'] = sorted(cache)
    cat['py_modules'] = PY_MODULES
    cat['data_files'] = DATA_FILES
    json.dump(cat, open(os.path.join(SITE, 'catalog.json'), 'w'), indent=0)
    open(os.path.join(SITE, '.nojekyll'), 'w').close()
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(SITE) for fn in fns)
    print(f'site built: {SITE} ({size/1e6:.1f} MB, {len(cache)} cached models)')


if __name__ == '__main__':
    main()
