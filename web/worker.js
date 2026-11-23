// Runs the servest Python package inside Pyodide (WebAssembly), off the main thread.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(PYODIDE + 'pyodide.js');

let ready = null;

async function boot(files) {
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: 'status', text: 'Loading numpy…' });
  await py.loadPackage('numpy');
  postMessage({ type: 'status', text: 'Loading model…' });
  py.FS.mkdirTree('/sv/servest');
  py.FS.mkdirTree('/sv/data');
  await Promise.all(files.py.map(async (m) => {
    const t = await (await fetch('py/servest/' + m)).text();
    py.FS.writeFile('/sv/servest/' + m, t);
  }));
  await Promise.all(files.data.map(async (d) => {
    const t = await (await fetch('py/data/' + d)).text();
    py.FS.writeFile('/sv/data/' + d, t);
  }));
  py.runPython(`
import sys, json, traceback
sys.path.insert(0, '/sv')
from servest import api, tco

_ARCH = {}

def _arch(repo, cfg_json, meta_json):
    key = (repo, hash(cfg_json), hash(meta_json))
    if key not in _ARCH:
        _ARCH[key] = api.load_arch(repo, cfg=json.loads(cfg_json), meta=json.loads(meta_json), fetch=False)
    return _ARCH[key]

def run(req_json, cfg_json, meta_json):
    try:
        req = json.loads(req_json)
        ar = _arch(req['model'], cfg_json, meta_json)
        q = api.Query(**{k: v for k, v in req.items() if k in api.Query.__dataclass_fields__})
        out = {'estimate': api.estimate(ar, q)}
        u = q.users or 1
        grid = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, int(round(u))})
        grid = [g for g in grid if g <= max(64, u * 4)]
        out['sweep'] = api.sweep(ar, q, grid)
        if req.get('min_tps') or req.get('max_ttft'):
            out['capacity'] = api.capacity(ar, q, req.get('min_tps'), req.get('max_ttft'))
        return json.dumps(out, default=float)
    except Exception as e:
        return json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]})

def run_tco(req_json, cfg_json, meta_json):
    try:
        req = json.loads(req_json)
        ar = _arch(req['model'], cfg_json, meta_json)
        q = api.Query(**{k: v for k, v in req.items() if k in api.Query.__dataclass_fields__})
        t = req['tco']
        out = api.size_and_cost(ar, q, req.get('min_tps') or 20, req.get('max_ttft') or 3, t['gpus'],
                                current_cloud=t.get('current_cloud') or 0.0, current_api=t.get('current_api') or 0.0,
                                api=t.get('api'), overrides=t.get('overrides'), size_from=t.get('size_from', 'users'),
                                acceptance=bool(t.get('acceptance')))
        c = out['cost']
        out['series'] = {o['gpu']: tco.series(o, c['months'], c['current_month']) for o in c['options'] if o['available']}
        return json.dumps(out, default=float)
    except Exception as e:
        return json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]})
`);
  return py;
}

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type === 'init') {
    ready = boot(msg.files).then((py) => { postMessage({ type: 'ready' }); return py; })
      .catch((e) => { postMessage({ type: 'fatal', text: String(e) }); throw e; });
    return;
  }
  if (msg.type === 'run') {
    const py = await ready;
    const t0 = performance.now();
    const out = py.globals.get(msg.kind === 'tco' ? 'run_tco' : 'run')(msg.req, msg.cfg, msg.meta);
    postMessage({ type: 'result', id: msg.id, json: out, ms: performance.now() - t0 });
  }
};
