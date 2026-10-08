# servest — LLM serving capacity & performance estimator

Answers the recurring question **"I have N × GPU — can I serve model M to C users, and what TPS / TTFT will they get?"**
for any Hugging Face model, any common datacenter/workstation GPU, any quantization, at any prompt/output length.

It is a **physics model** of LLM serving (HBM roofline for decode, FLOP roofline for prefill, MoE expert
coverage, per-architecture KV cache, tensor-parallel collectives, chunked-prefill time sharing and queueing),
whose ~50 efficiency/overhead constants are **calibrated on 15,169 public benchmark measurements**
(InferenceX, WattGPU, SGLang cookbook, community runs; 110 models, 21 GPU families), plus a small
**data-driven correction** for model/GPU combinations that have been measured. Because the trend comes
from physics rather than from a lookup table, it extrapolates sensibly to unseen models, GPUs and
context lengths — and it says how far it is extrapolating.

## Web app

`web/` is a static page that runs this exact Python package in the browser through Pyodide (WebAssembly):
form → feasibility, memory, per-user speed, TTFT, throughput with error bands, load-sweep charts, layout comparison,
max-users-at-target, share links. Model configs come from a bundled cache of ~180 models or are fetched live from
huggingface.co (gated repos: paste `config.json`). Nothing is sent anywhere else.

```bash
.venv/bin/python scripts/build_site.py              # assembles ./site
python3 -m http.server 8765 --directory site        # open http://localhost:8765
```

GitHub Pages: `.github/workflows/pages.yml` builds `site/` on every push to `main` and deploys it
(repo Settings → Pages → Source: GitHub Actions). After recalibrating, just push — the page picks up the new
`data/*.json`.

## Quick start

```bash
cd serving-estimator
uv venv .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/servest estimate --model mistralai/Mistral-Small-4-119B-2603 --gpu H100 --gpus 4 --users 100 --isl 4096 --osl 1024
```

Any public HF repo works (its `config.json` and parameter count are fetched once and cached in `data/`).

| Command | Question it answers |
|---|---|
| `servest estimate --model M --gpu G --gpus N --users C --isl I --osl O` | Does it fit? Memory breakdown, KV capacity, per-user tok/s, TTFT, aggregate tok/s, with error bands; compares TP/DP layouts |
| `servest sweep ... --users 1,16,64,256` | How do per-user speed / TTFT / throughput change with load? |
| `servest capacity ... --min-tps 25 --max-ttft 3` | How many users can this deployment carry at an SLA? |
| `servest plan ... --users 200 --min-tps 25` | Smallest GPU count that meets the SLA |
| `servest model <repo>` | Params (total/active), attention types, KV bytes per token, weight size per quantization |
| `servest gpus` | GPU catalog (spec-sheet values used) and which GPUs are in the calibration data |

Useful options: `--quant` (see below), `--kv-dtype auto|bf16|fp8|fp4`, `--engine vllm|sglang|tensorrt-llm|atom`,
`--tp`, `--dp-attention` (DP attention for MLA models), `--spec mtp|eagle` (speculative decoding),
`--think-time S` (users pause S seconds between answers and their next request; 0 = benchmark-style
continuous load, the worst case), `--rate R` (open-loop requests/s instead of users), `--prefix-hit F`,
`--mem-util`, `--json`.

### Quantization (`--quant`)

`native` (default) reads the checkpoint's own scheme from `config.json`. Each scheme sets weight bytes
(memory + bandwidth), the compute precision of the matmuls (which tensor-core peak applies), and a calibrated
kernel-efficiency class. Unsupported combinations fall back automatically (e.g. FP4 on Hopper → W4A8,
FP8 on Ampere → weight-only) and the output says so.

| `--quant` | storage bits/weight | compute | typical |
|---|---|---|---|
| `bf16` (`fp16`) | 16 | BF16 | unquantized |
| `fp8` (`w8a8`) | 8.06 | FP8 | FP8 checkpoints, DeepSeek/MiniMax native |
| `mxfp8` | 8.25 | FP8 | MX block scaling |
| `int8` | 8.06 | INT8 | SmoothQuant W8A8 |
| `w8a16` | 8.13 | BF16 | 8-bit weight-only |
| `w4a16` (`awq`, `gptq`, `int4`) | 4.25 | BF16 | AWQ/GPTQ/Marlin, Kimi-K2.5 native INT4 |
| `w4a8` | 4.25 | FP8 | 4-bit weights + FP8 activations |
| `nvfp4` (`fp4`, `w4a4`) | 4.5 | FP4 | NVFP4 W4A4 (Blackwell) |
| `mxfp4` | 4.25 | FP4 | MXFP4 W4A4 (Blackwell, MI355X) |
| `fp4-experts` | experts 4.5, rest 8 | FP4/FP8 | MoE NVFP4 checkpoints, DeepSeek-V4 native |
| `mxfp4-experts-bf16` | experts 4.25, rest 16 | FP4/BF16 | gpt-oss native |
| `w4a16-experts` | experts 4.25, rest 8 | BF16/FP8 | 4-bit weight-only experts |

KV-cache precision is separate (`--kv-dtype`; `auto` = FP8 when weights are FP8/FP4, else BF16).

## How it works

Per replica (TP GPUs; extra GPUs become data-parallel replicas):

* **Memory**: weights by component (routed experts / dense linears / embeddings / MTP heads / optional Engram
  tables) at the scheme's bits; KV + state per sequence computed per layer type — GQA/MHA, sliding window,
  MLA latent (replicated across TP ranks unless DP-attention), DeepSeek sparse attention + indexer keys,
  DeepSeek-V4 compressed attention (CSA/HCA + sliding window), V4.1 cross-layer KV sharing, MiniMax
  block-sparse, and linear-attention recurrent state (Gated DeltaNet/KDA). → feasibility and max
  concurrent sequences at the requested context.
* **Decode step** (batch *b*): `smoothmax(bytes / (BW·η_bw), FLOPs / (peak·η))` + TP all-reduce latency/bandwidth
  + per-step and per-layer overheads. Bytes = dense weights + **experts actually touched by b·(1+drafts)
  tokens** (`E·(1-(1-k/E)^T)`) + KV/state reads at the current context.
* **Prefill**: FLOPs (2·active·S + causal attention per layer type, incl. sparse indexers) at the compute
  precision's peak, vs. chunked weight reads; TP collectives per chunk.
* **Steady state**: chunked prefill time-shares the GPU with decode:
  `TPOT = step(b)/α + b·prefill/OSL`, `λ = b/(OSL·TPOT)`, TTFT = own prefill + M/D/1 queueing for the
  prefill path (finite-population corrected), closed loop `b + λ·(TTFT + think) = users`, capped by KV capacity.
  α = accepted tokens per step with speculative decoding.
* **Calibration** (`servest/calib.py`): ~50 constants (bandwidth efficiency per hardware class, decode/prefill
  compute efficiency, kernel class, engine, per-layer overhead, collective latency/efficiency, spec-decode
  acceptance, TTFT overhead and burst factor per engine) fitted by robust least squares on log errors, with
  physical bounds and priors, equal weight per deployment, and a second pass that down-weights whole model
  families physics can't explain (software immaturity) so they don't distort shared constants.
* **Correction layer** (`servest/residual.py`): hierarchical shrinkage effects (family, GPU, kernel×hardware,
  family×GPU, family×GPU×engine). Applies only where data exists; unseen combinations fall back to physics.

## Accuracy (held-out)

See `reports/evaluation.json` and `data/error_bands.json` (regenerate with `scripts/evaluate.py`, `scripts/bands.py`).
Median absolute % error; "external" = 152 measurements from three campaigns never used for fitting
(DGX Spark / RTX PRO 6000 / RTX 3090, new models, speculative decoding).

| Test | per-user TPOT | TTFT | aggregate output TPS |
|---|---|---|---|
| Frozen grouped split, test (unseen deployment groups; fit on train+val only) | 15.2% | 32.5% | 16.3% |
| External campaigns (95% DGX Spark/GB10 — compute-bound decode never calibrated there) | 16.5% | 59% | 49% |
| Leave-one-family-out (8 largest families; family never seen) | 25% | 45% | 23% |

For comparison, tuned gradient-boosted regressors (XGBoost/CatBoost) trained on the same corpus score 14.6% / 19.5% / 11.2%
on that test (they memorise known model×GPU pairs) but ~33% / 92% / 48% on the external campaigns.

The CLI prints a 10th–90th percentile band for the support regime of your query:
**known** (family and GPU measured), **partial** (one of them unseen), **novel**.

## Known limits — read before quoting numbers to a customer

* **TTFT is the least certain output.** Public benchmarks fire requests in synchronised waves; real users with
  think time see far less queueing. With `--think-time > 0` the calibrated burst factor fades toward random-arrival
  (M/D/1) queueing — a modelling assumption, not a measurement.
* **Long context (> 16k)**: only ~280 public rows above 8k, 75 above 32k. Physics carries the trend (attention FLOPs,
  KV growth, sparse/compressed attention), but no band is validated there. Measure your own 32k–200k runs.
* **Brand-new architectures** (DeepSeek-V4 on Hopper/AMD measured ~1.5–2× slower than physics): early engine
  kernels lag. The CLI warns when the architecture has little benchmark data.
* **AMD**: calibrated bandwidth efficiency is markedly lower (CDNA3 ≈ 0.47, CDNA4 ≈ 0.60 of peak) — software-stack
  dependent; re-measure on current ROCm builds.
* Not modelled: disaggregated prefill/decode, multi-node pipeline parallelism, CPU offload, tail latency (p99).
* Deployment knobs that public data rarely records (memory utilisation, max running requests, chunk size,
  prefix caching) change results; pass them where you know them.

## Improve it with your own measurements

Put benchmark results in `data/local_measurements.csv` (columns in `data/local_measurements.example.csv`:
model, gpu, gpu_count, tensor_parallel, engine, quantization, kv_cache_dtype, speculative_decoding,
input_tokens, output_tokens, concurrency, request_rate, ttft_median_ms, tpot_median_ms, output_tps), then:

```bash
.venv/bin/python -m servest.dataset && .venv/bin/python -m servest.calib && .venv/bin/python scripts/bands.py
```

Your rows enter calibration and the correction layer (a measured model×GPU becomes "known").

## Data provenance and corrections

Calibration rows come from an audited public-benchmark corpus assembled on 2026-10-08 (InferenceX public API
history, WattGPU/WattCounts, SGLang cookbook, community benchmark repos). The corpus is not redistributed here;
point `SERVEST_CORPUS` at it to recalibrate. Corrections applied on top of it: InferenceX expert-parallel rows had
GPU count = TP×EP and throughput inflated by EP (fixed; verified with Little's law, ratio 0.98 for corrected and
clean rows alike); configured-upper-bound lengths use their mean; TTFT labels faster than the prompt's FLOPs at
100% of peak are treated as prefix-cache hits and dropped; model metadata is recomputed from raw configs.
