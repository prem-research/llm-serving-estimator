'use strict';
// LLM Serving Estimator — front-end. Inference runs in worker.js (Pyodide + the servest package).

const $ = (id) => document.getElementById(id);
const FIELDS = ['model', 'gpu', 'gpus', 'users', 'think', 'isl', 'osl', 'quant', 'kv', 'engine-sel', 'tp', 'spec', 'prefix', 'dpa', 'min_tps', 'max_ttft'];
const EXAMPLES = [
  { model: 'meta-llama/Llama-3.3-70B-Instruct', gpu: 'H100-SXM', gpus: 8, users: 100, think: 10, isl: 2048, osl: 512, quant: 'fp8' },
  { model: 'openai/gpt-oss-120b', gpu: 'H200-SXM', gpus: 1, users: 64, think: 0, isl: 1024, osl: 1024, quant: 'native' },
  { model: 'deepseek-ai/DeepSeek-V4-Flash', gpu: 'B200', gpus: 8, users: 100, think: 30, isl: 32768, osl: 1024, quant: 'native' },
];

let catalog = null;
let cachePromise = null;
const fetched = {};
let worker = null;
let engineReady = false;
let pending = null;
let reqId = 0;
const waiters = {};

// ---------------------------------------------------------------- formatting
const nf = (x, d = 0) => (x == null || !isFinite(x)) ? '–' : x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
function sig(x) {
  if (x == null || !isFinite(x)) return '–';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1000) return nf(x, 0);
  if (a >= 100) return nf(x, 0);
  if (a >= 10) return nf(x, 1);
  if (a >= 1) return nf(x, 2);
  return nf(x, a >= 0.1 ? 2 : 3);
}
const range = (r, unit = '') => r ? `${sig(r[0])}–${sig(r[1])}${unit}` : '';
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function secs(s) {
  if (s == null || !isFinite(s)) return '–';
  if (s < 1) return `${nf(s * 1000, 0)} ms`;
  if (s < 60) return `${sig(s)} s`;
  return `${nf(s / 60, 1)} min`;
}

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    catalog = await (await fetch('catalog.json')).json();
  } catch (e) {
    setEngine('error', 'Could not load catalog');
    return;
  }
  populate();
  cachePromise = fetch('model_cache.json').then((r) => r.json()).catch(() => ({}));
  worker = new Worker('worker.js');
  worker.onmessage = onWorker;
  worker.onerror = (e) => setEngine('error', 'Engine failed: ' + (e.message || 'see console'));
  worker.postMessage({ type: 'init', files: { py: catalog.py_modules, data: catalog.data_files } });
  readHash() || applyValues({ model: 'meta-llama/Llama-3.3-70B-Instruct', gpu: 'H100-SXM', gpus: 8, users: 100, think: 10, isl: 2048, osl: 512, quant: 'fp8' });
  if (location.hash.length > 1) submit();
}

function setEngine(state, text) {
  const el = $('engine');
  el.className = 'pill ' + state;
  $('engine-text').textContent = text;
}

function onWorker(ev) {
  const m = ev.data;
  if (m.type === 'status') setEngine('loading', m.text);
  else if (m.type === 'ready') {
    engineReady = true;
    setEngine('ready', 'Engine ready');
    if (pending) { const p = pending; pending = null; p(); }
  } else if (m.type === 'fatal') setEngine('error', 'Engine failed to load: ' + m.text);
  else if (m.type === 'result') { const w = waiters[m.id]; delete waiters[m.id]; if (w) w(m); }
}

function populate() {
  const gpuSel = $('gpu');
  const groups = { nvidia: document.createElement('optgroup'), amd: document.createElement('optgroup') };
  groups.nvidia.label = 'NVIDIA'; groups.amd.label = 'AMD';
  for (const g of catalog.gpus) {
    const o = document.createElement('option');
    o.value = g.key;
    o.textContent = `${g.key} · ${g.vram_gb} GB · ${g.bw_tbs.toFixed(2)} TB/s${g.calibrated ? '' : ' · uncalibrated'}`;
    (groups[g.vendor] || groups.nvidia).appendChild(o);
  }
  gpuSel.append(groups.nvidia, groups.amd);
  const q = $('quant');
  q.innerHTML = '<option value="native">native (as published)</option>' +
    catalog.schemes.map((s) => `<option value="${s.key}">${s.key} — ${esc(s.desc)}</option>`).join('');
  $('models').innerHTML = catalog.cached_models.map((m) => `<option value="${esc(m)}">`).join('');
  document.querySelectorAll('[data-example]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    applyValues(EXAMPLES[+a.dataset.example]);
    submit();
  }));
}

// ---------------------------------------------------------------- form state <-> URL
function values() {
  const v = {};
  for (const f of FIELDS) {
    const el = $(f);
    v[f] = el.type === 'checkbox' ? el.checked : el.value.trim();
  }
  return v;
}
function applyValues(v) {
  if (v.quant && catalog && catalog.aliases[String(v.quant).toLowerCase()]) v = { ...v, quant: catalog.aliases[String(v.quant).toLowerCase()] };
  for (const [k, val] of Object.entries(v)) {
    const el = $(k === 'engine' ? 'engine-sel' : k);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = val === true || val === 'true' || val === '1';
    else el.value = val;
  }
  if (v.min_tps || v.max_ttft) $('target').open = true;
}
function writeHash(v) {
  const p = new URLSearchParams();
  for (const [k, val] of Object.entries(v)) {
    if (val === '' || val === false) continue;
    p.set(k === 'engine-sel' ? 'engine' : k, val === true ? '1' : val);
  }
  history.replaceState(null, '', '#' + p.toString());
}
function readHash() {
  if (location.hash.length < 2) return false;
  const p = new URLSearchParams(location.hash.slice(1));
  const v = {};
  for (const [k, val] of p) v[k] = val;
  applyValues(v);
  return true;
}

// ---------------------------------------------------------------- model metadata
async function modelData(repo) {
  const paste = $('cfgpaste').value.trim();
  if (paste) {
    let cfg;
    try { cfg = JSON.parse(paste); } catch (e) { throw new Error('The pasted config.json is not valid JSON.'); }
    const pt = parseFloat($('ptotal').value);
    return { cfg, meta: isFinite(pt) && pt > 0 ? { safetensors: { total: pt * 1e9 } } : {}, source: 'pasted config' };
  }
  const cache = await cachePromise;
  if (cache[repo]) return { cfg: cache[repo].config, meta: cache[repo].meta || {}, source: 'cached' };
  if (fetched[repo]) return fetched[repo];
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error('Enter a Hugging Face repo id like "org/model".');
  const r = await fetch(`https://huggingface.co/${repo}/resolve/main/config.json`);
  if (r.status === 401 || r.status === 403) throw new Error(`${repo} is gated or private on Hugging Face. Paste its config.json under "Gated or private model?".`);
  if (!r.ok) throw new Error(`Could not read ${repo}/config.json from Hugging Face (HTTP ${r.status}). Check the repo id.`);
  const cfg = await r.json();
  let meta = {};
  try {
    const m = await fetch(`https://huggingface.co/api/models/${repo}`);
    if (m.ok) { const d = await m.json(); if (d.safetensors) meta = { safetensors: d.safetensors }; }
  } catch (e) { /* parameter total is optional */ }
  return (fetched[repo] = { cfg, meta, source: 'huggingface.co' });
}

function request(v) {
  const num = (x) => (x === '' || x == null ? null : Number(x));
  return {
    model: v.model, gpu: v.gpu, gpus: num(v.gpus), users: num(v.users), think_time: num(v.think) || 0,
    isl: num(v.isl), osl: num(v.osl), quant: v.quant || 'native', kv_dtype: v.kv || 'auto', engine: v['engine-sel'],
    tp: v.tp ? num(v.tp) : null, spec: v.spec, prefix_hit: (num(v.prefix) || 0) / 100, dp_attention: !!v.dpa,
    min_tps: num(v.min_tps), max_ttft: num(v.max_ttft),
  };
}

function validate(req) {
  if (!req.model) return 'Enter a model.';
  for (const [k, label] of [['gpus', 'Number of GPUs'], ['users', 'Concurrent users'], ['isl', 'Prompt tokens'], ['osl', 'Output tokens']]) {
    if (!(req[k] >= 1)) return `${label} must be at least 1.`;
  }
  if (req.tp && req.gpus % req.tp) return `Tensor parallel ${req.tp} must divide the ${req.gpus} GPUs.`;
  return null;
}

// ---------------------------------------------------------------- run
async function submit() {
  const v = values();
  const req = request(v);
  const err = validate(req);
  $('form-error').hidden = !err;
  if (err) { $('form-error').textContent = err; return; }
  writeHash(v);
  $('go').disabled = true;
  showLoading(engineReady ? 'Computing…' : 'Starting the in-browser engine (first load downloads ~10 MB, then it is cached)…');
  try {
    const md = await modelData(req.model);
    if (!engineReady) await new Promise((res) => { pending = res; });
    const id = ++reqId;
    const msg = await new Promise((res) => {
      waiters[id] = res;
      worker.postMessage({ type: 'run', id, req: JSON.stringify(req), cfg: JSON.stringify(md.cfg), meta: JSON.stringify(md.meta) });
    });
    const out = JSON.parse(msg.json);
    if (out.error) throw new Error(out.error);
    render(out, req, md, msg.ms);
    if (window.innerWidth < 980) $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    $('go').disabled = false;
  }
}

function showLoading(text) {
  $('results').innerHTML = `<div class="card skeleton">${esc(text)}</div>`;
}
function showError(text) {
  $('results').innerHTML = `<div class="verdict bad"><div><div class="big">Couldn't estimate</div><div class="sub">${esc(text)}</div></div></div>`;
}

// ---------------------------------------------------------------- render
function render(out, req, md, ms) {
  const e = out.estimate;
  if (e.error) return showError(e.error);
  const p = e.perf, mem = e.memory, hw = e.hardware, mdl = e.model, pr = e.precision, conf = e.confidence;
  const html = [];

  // verdict
  if (!e.feasible) {
    html.push(`<div class="verdict bad"><div><div class="big">Does not fit</div><div class="sub">Weights need ${nf(mem.weights_gb, 0)} GB but ${hw.n_gpus} × ${esc(hw.gpu)} leave too little memory at this precision. Add GPUs, use a smaller quantization, or a larger GPU.</div></div></div>`);
  } else {
    const active = activeSeqs(p, req, hw);
    const use = active / Math.max(mem.max_seqs_total, 1e-9);
    const kvSeqs = mem.kv_seqs_total || mem.max_seqs_total;
    const capTxt = kvSeqs > mem.engine_cap_total * 1.01
      ? `KV cache holds ~${nf(kvSeqs)} sequences of ${nf(mem.context_tokens)} tokens (engine runs up to ${nf(mem.engine_cap_total)} at once)`
      : `KV cache holds ~${nf(kvSeqs)} sequences of ${nf(mem.context_tokens)} tokens`;
    const tight = p.kv_limited || use > 0.95;
    const cls = tight ? 'warn' : 'good';
    const big = tight ? 'Fits, but KV cache is the bottleneck' : 'Fits';
    const sub = tight
      ? `~${nf(active)} users are active at once but only ~${nf(mem.max_seqs_total)} sequences of ${nf(mem.context_tokens)} tokens fit in KV cache; extra requests queue, so time to first token grows. Add GPUs, use FP8 KV cache or a smaller quantization.`
      : `${nf(req.users)} users, ~${nf(active)} active at once. ${capTxt} — ${nf(use * 100)}% in use.`;
    html.push(`<div class="verdict ${cls}"><div><div class="big">${big}</div><div class="sub">${sub}</div></div></div>`);
  }

  if (e.feasible) {
    html.push(`<div class="kpis">
      ${kpi('Per-user speed', sig(p.decode_tps_per_user), 'tok/s', `range ${range(p.decode_tps_per_user_range)} · TPOT ${sig(p.tpot_ms)} ms`)}
      ${kpi('Time to first token', secs(p.ttft_s).split(' ')[0], secs(p.ttft_s).split(' ')[1] || '', `range ${p.ttft_s_range ? secs(p.ttft_s_range[0]) + '–' + secs(p.ttft_s_range[1]) : '–'}`)}
      ${kpi('Total output', sig(p.output_tps), 'tok/s', `range ${range(p.output_tps_range)} · ${sig(p.output_tps_per_gpu)}/GPU`)}
      ${kpi('Requests', sig(p.requests_per_s), '/s', `${secs(p.e2e_s)} per request end to end`)}
    </div>`);
  }

  // capacity
  if (out.capacity) {
    const c = out.capacity;
    const t = [req.min_tps ? `≥ ${req.min_tps} tok/s per user` : '', req.max_ttft ? `TTFT ≤ ${req.max_ttft} s` : ''].filter(Boolean).join(' and ');
    html.push(`<div class="card panel"><h3>Max users at your target <span class="badge">${esc(t)}</span></h3>
      <div class="kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">
        ${kpi('Concurrent users', c.max_users ? nf(c.max_users) : '0', '', c.max_users ? `with ${req.think_time || 0} s think time` : 'target not reachable on this setup')}
        ${c.max_users ? kpi('Per-user speed there', sig(c.decode_tps_per_user), 'tok/s', `TTFT ${secs(c.ttft_s)}`) : ''}
        ${c.max_users ? kpi('Layout', 'TP' + c.tp, '', `× ${c.replicas} replica${c.replicas > 1 ? 's' : ''}`) : ''}
      </div><p class="muted" style="margin:8px 0 0;font-size:12px">Central estimate. For a conservative plan, apply the lower end of the ranges above.</p></div>`);
  }

  // memory
  html.push(memoryPanel(e, req));

  // charts
  if (e.feasible && out.sweep && out.sweep.length > 1) html.push(chartsPanel(out.sweep, req));

  // layouts
  if (e.layouts && e.layouts.length > 1) {
    html.push(`<div class="card panel"><h3>Parallel layouts for ${hw.n_gpus} × ${esc(hw.gpu)}</h3><div class="tablewrap"><table>
      <thead><tr><th>Layout</th><th>Per-user tok/s</th><th>TTFT</th><th>Total output tok/s</th><th></th></tr></thead><tbody>
      ${e.layouts.map((l, i) => `<tr class="${i === 0 ? 'chosen' : ''}"><td>TP${l.tp} × ${l.replicas}</td><td>${sig(l.decode_tps_per_user)}</td><td>${secs(l.ttft_s)}</td><td>${nf(l.output_tps)}</td><td>${i === 0 ? 'chosen' : (l.kv_limited ? 'KV-limited' : '')}</td></tr>`).join('')}
      </tbody></table></div><p class="muted" style="margin:8px 0 0;font-size:12px">"Auto" picks the layout with the highest total throughput that keeps active users in KV cache. Set tensor parallel to compare.</p></div>`);
  }

  // details
  const notes = [...pr.notes, ...mdl.notes].map((n) => `<li>${esc(n)}</li>`);
  const cautions = conf.cautions.map((c) => `<li class="caution">${esc(c)}</li>`);
  const regimeText = { known: 'this model family and GPU are both in the calibration data', partial: conf.family ? 'model family measured, GPU not' : 'model family not measured; GPU is', novel: 'neither the model family nor the GPU is in the calibration data' }[conf.regime];
  const corr = Object.entries(conf.corrections || {}).map(([k, v]) => `${k.split('_')[0]} ×${v.toFixed(2)}`).join(', ');
  html.push(`<div class="card panel"><h3>Details</h3>
    <div class="facts">
      <div><span>Model</span><span>${esc(mdl.repo)}</span></div>
      <div><span>Architecture</span><span>${esc(mdl.type)}${mdl.moe ? `, MoE ${mdl.experts}×top-${mdl.top_k}` : ', dense'}, ${mdl.layers} layers</span></div>
      <div><span>Parameters</span><span>${sig(mdl.total_b)}B total · ${sig(mdl.active_b)}B active${Math.abs(mdl.active_prefill_b - mdl.active_b) > 0.01 ? ` (${sig(mdl.active_prefill_b)}B prefill)` : ''}</span></div>
      <div><span>Attention layers</span><span>${Object.entries(mdl.attention).map(([k, n]) => `${n} ${k}`).join(', ')}</span></div>
      <div><span>Weights</span><span>${esc(pr.scheme)} · compute ${esc(pr.compute.join('/'))}</span></div>
      <div><span>KV cache</span><span>${esc(pr.kv_dtype)} · ${sig(mem.kv_per_user_mb)} MB per user</span></div>
      <div><span>Layout</span><span>TP${hw.tp} × ${hw.replicas} · ${esc(hw.engine)}${hw.spec !== 'none' ? ' · spec ' + esc(hw.spec) : ''}</span></div>
      <div><span>Decode step</span><span>${sig(p.step_ms)} ms at batch ${nf(p.batch_per_replica)}/replica</span></div>
      <div><span>Prefill</span><span>${secs(p.prefill_s)} of GPU time per request · ${nf(p.prefill_share * 100)}% of GPU time</span></div>
      <div><span>Model config</span><span>${esc(md.source)}</span></div>
    </div>
    <h3 style="margin-top:16px">Confidence <span class="badge">${esc(conf.regime)}</span></h3>
    <ul class="notes"><li>Ranges are the 10th–90th percentile of measured ÷ predicted on held-out data where ${esc(regimeText)}.</li>
      ${corr ? `<li>Data-driven correction applied from similar measured deployments: ${esc(corr)}.</li>` : ''}
      ${cautions.join('')}${notes.join('')}</ul>
    <div class="actions" style="margin-top:14px"><button type="button" id="copylink">Copy share link</button><button type="button" id="copycli">Copy CLI command</button></div>
    <pre class="cli" id="cli">${esc(cliCommand(req))}</pre>
    <p class="muted" style="font-size:12px;margin:8px 0 0">Computed in your browser in ${nf(ms)} ms.</p>
  </div>`);

  $('results').innerHTML = html.join('');
  $('copylink').onclick = () => copy(location.href, 'copylink');
  $('copycli').onclick = () => copy(cliCommand(req), 'copycli');
}

function kpi(k, v, unit, r) {
  return `<div class="card kpi"><div class="k">${esc(k)}</div><div class="v">${v}<small>${esc(unit)}</small></div><div class="r">${r}</div></div>`;
}

function activeSeqs(p, req, hw) {
  const cyc = p.ttft_s + req.osl * p.tpot_ms / 1000;
  return req.users * cyc / (cyc + (req.think_time || 0));
}

function memoryPanel(e, req) {
  const mem = e.memory, hw = e.hardware;
  const total = hw.vram_gb * hw.tp;
  const w = mem.weights_gb;
  const budget = Math.max(mem.kv_budget_gb, 0);
  const active = e.feasible ? Math.min(activeSeqs(e.perf, req, hw) / hw.replicas, mem.max_seqs_per_replica) : 0;
  const kvUsed = Math.min(active * mem.kv_per_user_mb / 1000, budget);
  const reserved = Math.max(total - w - budget, 0);
  const pct = (x) => `${Math.max(0, Math.min(100, x / total * 100)).toFixed(2)}%`;
  const over = w > total;
  return `<div class="card panel memwrap"><h3>Memory per replica · ${hw.tp} × ${esc(hw.gpu)} = ${nf(total)} GB</h3>
    <div class="bar" role="img" aria-label="weights ${nf(w)} GB, KV in use ${nf(kvUsed)} GB, KV free ${nf(budget - kvUsed)} GB">
      <div class="seg" style="width:${over ? '100%' : pct(w)};background:${over ? 'var(--bad)' : 'var(--mem-weights)'}"></div>
      ${over ? '' : `<div class="seg" style="width:${pct(kvUsed)};background:var(--mem-kv)"></div>
      <div class="seg" style="width:${pct(budget - kvUsed)};background:var(--mem-free)"></div>
      <div class="seg" style="width:${pct(reserved)};background:var(--mem-res)"></div>`}
    </div>
    <div class="legend">
      <span><i style="background:var(--mem-weights)"></i>Weights ${nf(w, 1)} GB (${mem.experts_gb > 0.05 ? `experts ${nf(mem.experts_gb, 1)}, ` : ''}${mem.experts_gb > 0.05 ? 'attention/dense' : 'layers'} ${nf(mem.dense_gb, 1)}, embeddings ${nf(mem.embed_gb, 1)}${mem.mtp_gb > 0 ? `, MTP ${nf(mem.mtp_gb, 1)}` : ''}${mem.engram_gb > 0 ? `, Engram ${nf(mem.engram_gb, 1)}` : ''})</span>
      ${over ? '' : `<span><i style="background:var(--mem-kv)"></i>KV in use ${nf(kvUsed, 1)} GB</span>
      <span><i style="background:var(--mem-free)"></i>KV free ${nf(budget - kvUsed, 1)} GB</span>
      <span><i style="background:var(--mem-res)"></i>Reserved ${nf(reserved, 1)} GB</span>`}
    </div></div>`;
}

// ---------------------------------------------------------------- charts (inline SVG)
function chartsPanel(rows, req) {
  const ok = rows.filter((r) => r.fits);
  const xs = ok.map((r) => r.users);
  return `<div class="charts">
    ${lineChart('Per-user speed (tok/s)', xs, ok.map((r) => r.decode_tps_per_user), req.users, 'var(--series-1)', false, ok)}
    ${lineChart('Time to first token (s)', xs, ok.map((r) => r.ttft_s), req.users, 'var(--series-3)', true, ok)}
    ${lineChart('Total output (tok/s)', xs, ok.map((r) => r.output_tps), req.users, 'var(--series-2)', false, ok)}
  </div>${ok.some((r) => r.kv_limited) ? '<p class="muted" style="margin:-6px 0 0;font-size:12px">Orange points: KV cache is full at that load — extra users queue.</p>' : ''}`;
}

function lineChart(title, xs, ys, mark, color, logY, rows) {
  const W = 300, H = 170, L = 44, R = 10, T = 10, B = 26;
  const lx = xs.map((x) => Math.log10(x));
  const x0 = Math.min(...lx), x1 = Math.max(...lx);
  const vals = ys.filter((y) => y > 0 && isFinite(y));
  let y0 = logY ? Math.log10(Math.min(...vals)) : 0;
  let y1 = logY ? Math.log10(Math.max(...vals)) : Math.max(...vals) * 1.08;
  if (logY && y1 - y0 < 0.3) { y0 -= 0.15; y1 += 0.15; }
  if (!(y1 > y0)) y1 = y0 + 1;
  const px = (x) => L + (Math.log10(x) - x0) / Math.max(x1 - x0, 1e-9) * (W - L - R);
  const py = (y) => T + (1 - ((logY ? Math.log10(Math.max(y, 1e-9)) : y) - y0) / (y1 - y0)) * (H - T - B);
  const pts = xs.map((x, i) => `${px(x).toFixed(1)},${py(ys[i]).toFixed(1)}`).join(' ');
  const xt = xs.filter((x) => [1, 4, 16, 64, 256, 1024, 4096].includes(x));
  const yt = [];
  if (logY) {
    for (let e = Math.floor(y0); e <= Math.ceil(y1); e++) for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e), lv = Math.log10(v);
      if (lv >= y0 - 1e-9 && lv <= y1 + 1e-9) yt.push(v);
    }
    while (yt.length > 6) yt.splice(1, 1);
  } else { const step = niceStep((y1 - y0) / 4); for (let v = 0; v <= y1 + 1e-9; v += step) yt.push(v); }
  const mi = xs.indexOf(mark);
  const kvl = rows.map((r, i) => r.kv_limited ? `<circle cx="${px(xs[i]).toFixed(1)}" cy="${py(ys[i]).toFixed(1)}" r="2.5" fill="var(--warn)"/>` : '').join('');
  return `<div class="card chart"><h4>${esc(title)}</h4>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)} versus concurrent users">
      ${yt.map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${py(v).toFixed(1)}" y2="${py(v).toFixed(1)}"/><text class="tick" x="${L - 6}" y="${(py(v) + 3).toFixed(1)}" text-anchor="end">${tick(v)}</text>`).join('')}
      <line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
      ${xt.map((x) => `<text class="tick" x="${px(x).toFixed(1)}" y="${H - B + 14}" text-anchor="middle">${nf(x)}</text>`).join('')}
      <text class="tick" x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">concurrent users</text>
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>
      ${kvl}
      ${mi >= 0 ? `<circle cx="${px(xs[mi]).toFixed(1)}" cy="${py(ys[mi]).toFixed(1)}" r="4.5" fill="${color}" stroke="var(--card)" stroke-width="2"/>` : ''}
    </svg></div>`;
}
function tick(v) {
  if (v === 0) return '0';
  if (v >= 1000) return (v / 1000).toLocaleString('en-US', { maximumFractionDigits: 1 }) + 'k';
  if (v >= 1) return v.toLocaleString('en-US', { maximumFractionDigits: 1 });
  return v.toLocaleString('en-US', { maximumSignificantDigits: 2 });
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

// ---------------------------------------------------------------- misc
function cliCommand(req) {
  const a = ['servest estimate', `--model ${req.model}`, `--gpu ${req.gpu}`, `--gpus ${req.gpus}`, `--users ${req.users}`,
    `--isl ${req.isl}`, `--osl ${req.osl}`];
  if (req.think_time) a.push(`--think-time ${req.think_time}`);
  if (req.quant && req.quant !== 'native') a.push(`--quant ${req.quant}`);
  if (req.kv_dtype !== 'auto') a.push(`--kv-dtype ${req.kv_dtype}`);
  if (req.engine !== 'vllm') a.push(`--engine ${req.engine}`);
  if (req.tp) a.push(`--tp ${req.tp}`);
  if (req.spec !== 'none') a.push(`--spec ${req.spec}`);
  if (req.dp_attention) a.push('--dp-attention');
  if (req.prefix_hit) a.push(`--prefix-hit ${req.prefix_hit}`);
  return a.join(' ');
}
async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); flash(btn, 'Copied'); } catch (e) { flash(btn, 'Copy failed'); }
}
function flash(id, text) {
  const b = $(id); const old = b.textContent; b.textContent = text; setTimeout(() => { b.textContent = old; }, 1400);
}

$('form').addEventListener('submit', (e) => { e.preventDefault(); submit(); });
$('model').addEventListener('change', () => { $('cfgpaste').value = ''; });
window.addEventListener('hashchange', () => { if (readHash()) submit(); });
boot();
