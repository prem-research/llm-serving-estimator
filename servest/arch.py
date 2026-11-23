"""Model architecture from a Hugging Face config.json: parameters, per-layer attention
geometry, KV-cache/state bytes and attention cost as a function of context length.

Handles dense GQA/MHA, sliding-window, MLA (DeepSeek-V2/V3, Kimi), DeepSeek sparse
attention (V3.2/GLM-5 DSA), DeepSeek-V4 compressed attention (CSA/HCA + SWA), V4.1
cross-layer KV sharing, MiniMax block-sparse attention, linear attention (Gated DeltaNet,
KDA, lightning), MoE geometry, MTP heads and Engram memory tables.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, '..', 'data', 'configs')
META_DIR = os.path.join(HERE, '..', 'data', 'hf_meta')
OVERRIDES = os.path.join(HERE, '..', 'data', 'model_overrides.json')


@dataclass
class AttnSpec:
    kind: str                 # full | sliding | sparse | csa | hca | swa | blocksparse | linear
    q_heads: int = 0
    kv_elems: float = 0.0     # cached elements per stored entry (K+V, or MLA latent)
    d_qk_pre: float = 0.0     # per-head dims used in prefill attention math
    d_v_pre: float = 0.0
    d_qk_dec: float = 0.0     # per-head dims used in decode (MLA weight absorption)
    d_v_dec: float = 0.0
    window: int | None = None
    compress: float = 1.0     # one stored entry per `compress` tokens
    topk: int | None = None   # sparse: entries attended per query
    swa: int = 0              # extra local window attended/stored (DeepSeek-V4)
    swa_elems: float = 0.0
    idx_heads: int = 0        # sparse indexer
    idx_dim: int = 0
    idx_bytes: float = 1.0
    idx_compute: bool = True  # False if indices are reused from an earlier layer
    state_elems: float = 0.0  # linear-attention recurrent state per sequence
    state_bytes_per_elem: float = 4.0
    store_kv: bool = True     # False when the layer reuses another layer's KV
    kv_bytes_fixed: float | None = None  # bytes per stored entry, overrides kv dtype (e.g. FP4 KV)

    # ---- per-sequence storage
    def stored_entries(self, L: float) -> float:
        if not self.store_kv or self.kind in ('linear', 'swa'):
            return 0.0
        if self.kind == 'sliding':
            return min(L, self.window or L)
        return L / self.compress

    def cache_bytes(self, L: float, kvb: float) -> float:
        b = self.stored_entries(L) * self.kv_elems * (self.kv_bytes_fixed or kvb)
        if self.swa:
            b += min(L, self.swa) * self.swa_elems * max(kvb, 1.0)
        if self.idx_heads and self.idx_compute:
            b += (L / self.compress) * self.idx_dim * self.idx_bytes
        if self.kind == 'linear':
            b += self.state_elems * self.state_bytes_per_elem
        return b

    # ---- per-query attention work
    def attended(self, L: float) -> float:
        k = self.kind
        if k == 'full':
            return L
        if k == 'sliding':
            return min(L, self.window or L)
        if k in ('sparse', 'blocksparse'):
            return min(L, self.topk or L)
        if k == 'csa':
            return min(L / self.compress, self.topk or L) + min(L, self.swa)
        if k == 'hca':
            return L / self.compress + min(L, self.swa)
        if k == 'swa':
            return min(L, self.swa)
        return 0.0

    def decode_read_bytes(self, L: float, kvb: float) -> float:
        """HBM bytes read by one decoding query of context L."""
        if self.kind == 'linear':
            return 2 * self.state_elems * self.state_bytes_per_elem
        main = self.attended(L) - (min(L, self.swa) if self.swa else 0.0)
        b = main * self.kv_elems * (self.kv_bytes_fixed or kvb)
        if self.swa:
            b += min(L, self.swa) * self.swa_elems * max(kvb, 1.0)
        if self.idx_heads and self.idx_compute:
            b += (L / self.compress) * self.idx_dim * self.idx_bytes
        return b

    def decode_flops(self, L: float) -> float:
        if self.kind == 'linear':
            return 4 * self.state_elems
        f = 2 * self.q_heads * (self.d_qk_dec + self.d_v_dec) * self.attended(L)
        if self.idx_heads and self.idx_compute:
            f += 2 * self.idx_heads * self.idx_dim * L / self.compress
        return f

    def prefill_flops(self, S: float, start: float = 0.0) -> float:
        """Attention FLOPs to prefill tokens (start, start+S] (causal)."""
        if self.kind == 'linear':
            return 6 * self.state_elems * S
        hd = self.q_heads * (self.d_qk_pre + self.d_v_pre)
        mid = start + S / 2.0                      # mean query position
        k = self.kind
        if k == 'full':
            att = mid
        elif k == 'sliding':
            att = min(mid, self.window or mid)
        elif k in ('sparse', 'blocksparse'):
            att = min(mid, self.topk or mid)
        elif k == 'csa':
            att = min(mid / self.compress, self.topk or mid) + min(mid, self.swa)
        elif k == 'hca':
            att = mid / self.compress + min(mid, self.swa)
        elif k == 'swa':
            att = min(mid, self.swa)
        else:
            att = 0
        f = 2 * hd * att * S
        if self.idx_heads and self.idx_compute:
            f += 2 * self.idx_heads * self.idx_dim * (mid / self.compress) * S
        return f


@dataclass
class Arch:
    repo: str
    model_type: str
    n_layers: int
    hidden: int
    vocab: int
    attn: list
    is_moe: bool
    n_experts: int = 0
    top_k: int = 0
    moe_layers: int = 0
    expert_params_each: float = 0.0      # one routed expert in one layer
    params_expert_total: float = 0.0     # all routed experts (backbone)
    params_dense_linear: float = 0.0     # attention + dense FFN + shared experts + router (backbone)
    params_embed: float = 0.0
    params_lm_head: float = 0.0
    params_mtp: float = 0.0              # MTP / draft heads (memory; used only with spec decode)
    params_other: float = 0.0            # vision tower etc. (memory only)
    params_engram: float = 0.0           # host-offloadable lookup tables
    params_total_hf: float | None = None
    active_decode_override: float | None = None
    prefill_active_scale: float = 1.0
    native_scheme: str = 'bf16'
    native_kv: str = 'auto'
    max_context: int | None = None
    kv_heads: int = 1                     # KV heads of full-attention layers (1 for MLA / latent KV)
    notes: list = field(default_factory=list)

    @property
    def params_lm_head_compute(self) -> float:
        """Output-projection weights read every decode step; a tied head reuses the embedding matrix."""
        return self.params_lm_head or self.params_embed

    @property
    def params_active_decode(self) -> float:
        if self.active_decode_override:
            return self.active_decode_override
        return (self.params_dense_linear + self.top_k * self.expert_params_each * self.moe_layers
                + self.params_lm_head_compute)

    @property
    def params_active_prefill(self) -> float:
        # prefill computes logits only for the last prompt token, so the output projection is excluded
        return max(self.params_active_decode - self.params_lm_head_compute, 0.0) * self.prefill_active_scale

    @property
    def params_total(self) -> float:
        return (self.params_expert_total + self.params_dense_linear + self.params_embed + self.params_lm_head
                + self.params_mtp + self.params_other + self.params_engram)

    def kv_bytes_per_seq(self, L: float, kvb: float) -> float:
        return sum(a.cache_bytes(L, kvb) for a in self.attn)

    def kv_bytes_per_token(self, kvb: float, L: float = 32768) -> float:
        """Marginal cache growth per token at context L (excludes constant state/windows)."""
        return (self.kv_bytes_per_seq(L + 1024, kvb) - self.kv_bytes_per_seq(L, kvb)) / 1024

    def decode_read_bytes(self, L: float, kvb: float) -> float:
        return sum(a.decode_read_bytes(L, kvb) for a in self.attn)

    def decode_attn_flops(self, L: float) -> float:
        return sum(a.decode_flops(L) for a in self.attn)

    def prefill_attn_flops(self, S: float, start: float = 0.0) -> float:
        return sum(a.prefill_flops(S, start) for a in self.attn)

    def weight_bytes(self, scheme, engram_in_hbm: bool = False, include_mtp: bool = True) -> dict:
        """Bytes resident on GPUs for a quantization scheme, by component."""
        exp_bits = scheme.w_bits
        other_bits = scheme.other_bits if (scheme.experts_only and self.is_moe) else scheme.w_bits
        d = {
            'experts': self.params_expert_total * exp_bits / 8,
            'dense_linear': self.params_dense_linear * other_bits / 8,
            'embed_lm_head': (self.params_embed + self.params_lm_head) * 2.0,
            'mtp': self.params_mtp * other_bits / 8 if include_mtp else 0.0,
            'other': self.params_other * 2.0,
            'engram': self.params_engram * 1.0 if engram_in_hbm else 0.0,
        }
        d['total'] = sum(d.values())
        return d


# --------------------------------------------------------------------------- config access

def _cache_name(repo: str) -> str:
    return repo.replace('/', '__') + '.json'


def load_config(repo: str, fetch: bool = True) -> dict:
    p = os.path.join(CFG_DIR, _cache_name(repo))
    if os.path.exists(p):
        return json.load(open(p))
    if not fetch:
        raise FileNotFoundError(f'No cached config for {repo}; run with network access to fetch it.')
    url = f'https://huggingface.co/{repo}/resolve/main/config.json'
    with urllib.request.urlopen(url, timeout=30) as r:
        cfg = json.loads(r.read())
    os.makedirs(CFG_DIR, exist_ok=True)
    json.dump(cfg, open(p, 'w'))
    return cfg


def load_hf_meta(repo: str, fetch: bool = True) -> dict:
    """Safetensors parameter totals from the HF API (cached)."""
    p = os.path.join(META_DIR, _cache_name(repo))
    if os.path.exists(p):
        return json.load(open(p))
    if not fetch:
        return {}
    try:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}', timeout=30) as r:
            d = json.loads(r.read())
        meta = {'safetensors': d.get('safetensors'), 'base_model': (d.get('cardData') or {}).get('base_model')}
    except Exception as e:  # network or gated repo
        meta = {'error': str(e)}
    os.makedirs(META_DIR, exist_ok=True)
    json.dump(meta, open(p, 'w'))
    return meta


def _g(c, *keys, default=None):
    for k in keys:
        v = c.get(k)
        if v is not None:
            return v
    return default


# --------------------------------------------------------------------------- native quantization

def _group_scheme(g: dict) -> str | None:
    """One compressed-tensors / ModelOpt config group -> scheme name."""
    w = g.get('weights') or {}
    a = g.get('input_activations') or {}
    fmt = str(g.get('format') or '').lower()
    wb, wt = w.get('num_bits'), w.get('type', 'int')
    ab = a.get('num_bits') if a else None
    if 'nvfp4' in fmt or (wb == 4 and wt == 'float' and ab == 4 and w.get('group_size') == 16):
        return 'nvfp4'
    if 'mxfp4' in fmt or (wb == 4 and wt == 'float'):
        return 'mxfp4'
    if 'mxfp8' in fmt:
        return 'mxfp8'
    if wb == 4:
        return 'w4a8' if ab == 8 else 'w4a16'
    if wb == 8:
        if ab == 8:
            return 'fp8' if wt == 'float' else 'int8'
        return 'w8a16'
    return None


def _groups_scheme(q: dict, is_moe: bool) -> str | None:
    """Mixed checkpoints list one group per precision; the 4-bit group decides, experts first."""
    groups = [g for g in (q.get('config_groups') or {}).values() if isinstance(g, dict)]
    found = [(s, 'experts' in json.dumps(g.get('targets') or []).replace('shared_expert', ''))
             for g in groups for s in [_group_scheme(g)] if s]
    if not found:
        return None
    four = [(s, e) for s, e in found if s in ('nvfp4', 'mxfp4', 'w4a16', 'w4a8')]
    eight = [s for s, _ in found if s in ('fp8', 'mxfp8', 'int8', 'w8a16')]
    if four:
        s, on_experts = next(((s, e) for s, e in four if e), four[0])
        if is_moe and on_experts and eight:              # 4-bit routed experts, 8-bit everything else
            return 'w4a16-experts' if s == 'w4a16' else 'fp4-experts'
        if is_moe and s in ('nvfp4', 'mxfp4'):
            # language-model attention excluded from the 4-bit group => only the experts are 4-bit;
            # the rest keeps the base precision (FP8 for fp8-method checkpoints, else BF16)
            ign = ' '.join(str(x) for x in (q.get('ignore') or q.get('exclude_modules') or [])
                           if not re.search(r'vision|visual|audio|image|patch|multi_modal|mm_proj', str(x)))
            if re.search(r'self_attn|[.*]attn[.*]', ign) or str(q.get('moe_quant_algo') or '').upper() in ('NVFP4', 'MXFP4'):
                base_fp8 = str(q.get('quant_method', '')).lower() == 'fp8'
                return 'mxfp4-experts-bf16' if (s == 'mxfp4' and not base_fp8) else 'fp4-experts'
        return s
    return eight[0] if eight else found[0][0]


NAME_SCHEMES = [(r'nvfp4|[-_]fp4\b', 'nvfp4'), (r'mxfp4', 'mxfp4'), (r'mxfp8', 'mxfp8'), (r'[-_]fp8\b|fp8[-_]dynamic', 'fp8'),
                (r'awq|gptq|[-_]int4|w4a16|4bit', 'w4a16'), (r'[-_]int8|w8a8', 'int8')]


def scheme_from_name(repo: str, is_moe: bool) -> str | None:
    """Fallback for checkpoints whose config.json carries no quantization_config (e.g. NVIDIA ModelOpt
    repos that keep it in hf_quant_config.json)."""
    name = repo.lower().split('/')[-1]
    for pat, s in NAME_SCHEMES:
        if re.search(pat, name):
            return 'fp4-experts' if (is_moe and s in ('nvfp4', 'mxfp4')) else s
    return None


def native_scheme(cfg: dict, tc: dict, is_moe: bool, repo: str = '', notes: list | None = None) -> str:
    q = cfg.get('quantization_config') or tc.get('quantization_config') or {}
    expert_fp4 = (cfg.get('expert_dtype') or tc.get('expert_dtype') or q.get('expert_dtype')) == 'fp4'
    m = str(q.get('quant_method', '')).lower()
    if not q:
        s = scheme_from_name(repo, is_moe)
        if s:
            if notes is not None:
                notes.append(f'config.json has no quantization_config; weights assumed {s} from the repo name.')
            return s
        return 'bf16'
    if q.get('config_groups'):
        s = _groups_scheme(q, is_moe)
        if s:
            return s
    if m == 'fp8':
        experts4 = str(q.get('routed_experts_quant_method') or '').lower() in ('mxfp4', 'nvfp4')
        return 'fp4-experts' if (expert_fp4 or experts4) else 'fp8'
    if m == 'mxfp8':
        return 'mxfp8'
    if m == 'mxfp4':
        keep = ' '.join(q.get('modules_to_not_convert') or [])
        return 'mxfp4-experts-bf16' if 'attn' in keep and is_moe else 'mxfp4'
    if m in ('awq', 'gptq', 'auto-round', 'autoround', 'bitsandbytes', 'hqq'):
        bits = q.get('bits') or q.get('w_bit') or 4
        return 'w4a16' if bits <= 4 else 'w8a16'
    if m == 'modelopt':
        algo = str(q.get('quant_algo') or (q.get('quantization') or {}).get('quant_algo') or '').upper()
        if 'FP4' in algo:
            return 'fp4-experts' if is_moe else 'nvfp4'
        if 'FP8' in algo:
            return 'fp8'
    if m in ('compressed-tensors', 'compressed_tensors'):
        groups = q.get('config_groups') or {}
        for g in groups.values():
            w = g.get('weights') or {}
            a = g.get('input_activations') or {}
            wb, wt = w.get('num_bits', 16), w.get('type', 'int')
            ab, at = (a.get('num_bits'), a.get('type')) if a else (None, None)
            if wb == 4 and wt == 'float' and ab == 4:
                ign = ' '.join(q.get('ignore') or [])
                return 'fp4-experts' if is_moe and ('self_attn' in ign) else 'nvfp4'
            if wb == 4 and ab == 8:
                return 'w4a8'
            if wb == 4:
                return 'w4a16'
            if wb == 8 and ab == 8:
                return 'fp8' if wt == 'float' else 'int8'
            if wb == 8:
                return 'w8a16'
    if 'quant_method' in q:
        return 'w4a16' if '4' in json.dumps(q) else 'fp8'
    return 'bf16'


# --------------------------------------------------------------------------- parser

NON_GATED = {'gpt2', 'gptj', 'gpt_neox', 'phi', 'falcon', 'bloom', 'opt', 'gpt_bigcode', 'starcoder2', 'mpt', 'nemotron'}
SQUARED_RELU = {'relu2', 'relu_squared', 'squared_relu'}     # Nemotron / Nemotron-H: up + down only


def _ffn_mats(model_type, act):
    if model_type in NON_GATED or str(act).lower() in SQUARED_RELU:
        return 2
    return 3


def _mamba2(tc: dict, H: int) -> tuple:
    """Mamba-2 mixer (Nemotron-H / Granite-4.0-H style keys): recurrent-state spec and parameter count."""
    expand = float(_g(tc, 'mamba_expand', 'expand', default=2))
    nh = int(_g(tc, 'mamba_num_heads', 'mamba_n_heads', default=0) or 0)
    hd = int(_g(tc, 'mamba_head_dim', 'mamba_d_head', default=0) or 0)
    if not nh and hd:
        nh = int(expand * H) // hd
    if not hd and nh:
        hd = int(expand * H) // nh
    if not nh:
        hd = 64
        nh = int(expand * H) // hd
    ds = int(_g(tc, 'ssm_state_size', 'mamba_d_state', 'state_size', default=128))
    ng = int(_g(tc, 'n_groups', 'mamba_n_groups', default=1))
    ck = int(_g(tc, 'conv_kernel', 'mamba_d_conv', default=4))
    d_inner = nh * hd
    conv_dim = d_inner + 2 * ng * ds
    sb = 4.0 if str(_g(tc, 'mamba_ssm_cache_dtype', 'mamba_ssm_dtype', default='float32')) in ('float32', 'fp32') else 2.0
    spec = AttnSpec('linear', q_heads=nh, state_elems=nh * hd * ds + (ck - 1) * conv_dim * (2 / sb), state_bytes_per_elem=sb)
    params = H * (2 * d_inner + 2 * ng * ds + nh) + conv_dim * ck + d_inner * H
    return spec, params


def parse(repo: str, cfg: dict | None = None, fetch: bool = True, meta: dict | None = None) -> Arch:
    cfg = cfg if cfg is not None else load_config(repo, fetch)
    tc = cfg.get('text_config') or cfg.get('llm_config') or cfg
    mt = str(tc.get('model_type') or cfg.get('model_type') or '')
    L = int(_g(tc, 'num_hidden_layers', 'n_layer', 'num_layers'))
    H = int(_g(tc, 'hidden_size', 'n_embd', 'd_model'))
    V = int(_g(tc, 'vocab_size', default=32000))
    nh = int(_g(tc, 'num_attention_heads', 'n_head'))
    kvh = int(_g(tc, 'num_key_value_heads', 'num_kv_heads', 'multi_query_group_num', default=nh) or nh)
    if tc.get('multi_query') and not tc.get('num_key_value_heads'):
        kvh = 1
    hd = tc.get('head_dim') or H // nh
    if not hd:
        hd = H // nh
    hd = int(hd)
    # transformers omits tie_word_embeddings from config.json when it equals the library default (True)
    tied = bool(_g(tc, 'tie_word_embeddings', 'tie_embedding',
                   default=_g(cfg, 'tie_word_embeddings', 'tie_embedding', default=True)))
    act = _g(tc, 'hidden_act', 'activation_function', 'mlp_hidden_act', 'hidden_activation')
    mats = _ffn_mats(mt, act)
    notes = []
    # Nemotron-H style: one block per layer, M = Mamba-2, * = attention, - = MLP, E = MoE
    pattern = tc.get('hybrid_override_pattern')
    blocks = pattern[:L] if isinstance(pattern, str) and len(pattern) >= L else None

    # ---------------- MoE geometry
    E = int(_g(tc, 'n_routed_experts', 'num_local_experts', 'num_experts', 'moe_num_experts', default=0) or 0)
    k = int(_g(tc, 'num_experts_per_tok', 'experts_per_token', 'num_experts_per_token', 'moe_topk', 'top_k',
               'top_k_experts', 'moe_top_k', default=0) or 0)
    if tc.get('enable_moe_block') is False:      # Gemma 4: expert fields present but the MoE block is off
        E = k = 0
    is_moe = E > 1 and k > 0
    dense_inter = _g(tc, 'dense_intermediate_size', 'intermediate_size_mlp', 'intermediate_size', 'ffn_hidden_size', 'n_inner')
    if dense_inter is None:
        dense_inter = 4 * H
    if tc.get('block_auto_adjust_ff_dim'):       # LFM2: SwiGLU width is 2/3 of block_ff_dim, rounded up
        ff = int(2 * dense_inter / 3)
        if tc.get('block_ffn_dim_multiplier') is not None:
            ff = int(tc['block_ffn_dim_multiplier'] * ff)
        mo = int(tc.get('block_multiple_of') or 1)
        dense_inter = mo * ((ff + mo - 1) // mo)
    moe_inter = _g(tc, 'moe_intermediate_size', 'expert_intermediate_size')
    if is_moe and moe_inter is None:
        moe_inter = tc.get('intermediate_size')
    expert_in = _g(tc, 'routed_expert_hidden_size', 'moe_latent_size', default=H)   # latent MoE: experts run in a smaller space
    moe_mask = [False] * L
    if is_moe:
        fk = _g(tc, 'first_k_dense_replace', 'num_dense_layers', default=None)
        freq = tc.get('moe_layer_freq')
        mlt = tc.get('mlp_layer_types')
        only = tc.get('mlp_only_layers') or []
        step = tc.get('decoder_sparse_step') or 1
        moe_list = tc.get('moe_layers') if isinstance(tc.get('moe_layers'), list) else None
        inter_step = int(tc.get('interleave_moe_layer_step') or 1)
        for i in range(L):
            m = True
            if blocks is not None:
                m = blocks[i] == 'E'
            elif moe_list is not None:
                m = i in moe_list
            elif isinstance(mlt, list) and len(mlt) >= L:
                m = mlt[i] in ('sparse', 'moe')
            elif isinstance(freq, list) and len(freq) >= L:
                m = bool(freq[i])
            else:
                if fk is not None and i < int(fk):
                    m = False
                if isinstance(freq, int) and freq > 1 and (i % freq) != freq - 1:
                    m = False
                if inter_step > 1 and (i + 1) % inter_step != 0:     # Llama 4: MoE every other layer
                    m = False
                if i in only or (step > 1 and (i + 1) % step != 0):
                    m = False
            moe_mask[i] = m
    moe_layers = sum(moe_mask)
    dense_ffn_layers = blocks.count('-') if blocks is not None else L - moe_layers
    n_shared = _g(tc, 'n_shared_experts', 'num_shared_experts', default=None)
    if is_moe and n_shared is None and mt.startswith('llama4'):
        n_shared = 1                                # Llama 4 always adds one shared expert
    sh_inter = _g(tc, 'shared_expert_intermediate_size', 'shared_intermediate_size',
                  'moe_shared_expert_intermediate_size', default=None)
    if sh_inter:
        shared_params = mats * H * sh_inter * (n_shared or 1)
    elif n_shared and moe_inter:
        shared_params = mats * H * moe_inter * n_shared
    else:
        shared_params = 0
    if is_moe and tc.get('enable_moe_block'):
        # Gemma 4 MoE layers run a dense MLP (intermediate_size) beside the routed experts
        shared_params += mats * H * (tc.get('intermediate_size') or 0)
    expert_each = mats * expert_in * (moe_inter or 0) if is_moe else 0.0
    if is_moe and expert_in != H:
        shared_params += 2 * H * expert_in      # latent MoE: per-layer down/up projection around the routed experts
    router = H * E if is_moe else 0

    # ---------------- attention layer kinds
    kinds = ['full'] * L
    lt = tc.get('layer_types')
    lin_cfg = tc.get('linear_attn_config') or {}
    sparse_cfg = tc.get('sparse_attention_config') or {}
    if blocks is not None:
        kinds = [{'M': 'mamba', '*': 'full'}.get(c, 'none') for c in blocks]     # '-' / 'E' blocks have no mixer
    elif isinstance(lt, list) and len(lt) >= L:
        mp = {'full_attention': 'full', 'sliding_attention': 'sliding', 'linear_attention': 'linear',
              'deepseek_sparse_attention': 'sparse', 'chunked_attention': 'sliding', 'attention': 'full',
              'mamba': 'mamba', 'hybrid': 'full', 'conv': 'shortconv'}
        kinds = [mp.get(str(x), 'full') for x in lt[:L]]
    elif isinstance(tc.get('full_attention_layers'), list) and (tc.get('linear_attention_type') or tc.get('linear_num_value_heads')):
        fa = set(tc['full_attention_layers'])
        kinds = ['full' if i in fa else 'linear' for i in range(L)]
    elif tc.get('attention_chunk_size') and isinstance(tc.get('no_rope_layers'), list):
        nrl = tc['no_rope_layers']                  # Llama 4: RoPE layers use chunked local attention, NoPE layers global
        kinds = ['sliding' if (i < len(nrl) and nrl[i]) else 'full' for i in range(L)]
    elif lin_cfg.get('kda_layers') or lin_cfg.get('full_attn_layers'):
        kda = set(lin_cfg.get('kda_layers') or [])
        one_based = max(kda | set(lin_cfg.get('full_attn_layers') or [0])) >= L
        kinds = ['linear' if (i + (1 if one_based else 0)) in kda else 'full' for i in range(L)]
    elif isinstance(tc.get('attn_type_list'), list):
        kinds = ['full' if x == 1 else 'linear' for x in tc['attn_type_list'][:L]]
    elif tc.get('full_attention_interval'):
        fi = int(tc['full_attention_interval'])
        kinds = ['full' if (i + 1) % fi == 0 else 'linear' for i in range(L)]
    elif tc.get('sliding_window') and tc.get('use_sliding_window', mt not in ('qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe')):
        if mt in ('gemma2',):
            kinds = ['sliding' if i % 2 == 0 else 'full' for i in range(L)]
        elif mt in ('gemma3', 'gemma3_text'):
            kinds = ['full' if (i + 1) % 6 == 0 else 'sliding' for i in range(L)]
        elif mt in ('mistral', 'mixtral') and tc.get('sliding_window') and tc['sliding_window'] < 65536:
            kinds = ['sliding'] * L
    if sparse_cfg.get('use_sparse_attention') and isinstance(sparse_cfg.get('sparse_attention_freq'), list):
        kinds = ['blocksparse' if f else kd for f, kd in zip(sparse_cfg['sparse_attention_freq'][:L], kinds)]
    is_dsa = 'index_topk' in tc and mt not in ('deepseek_v4', 'deepseek_v41_text')
    if is_dsa:
        kinds = ['sparse' if kd in ('full', 'sparse') else kd for kd in kinds]

    window = tc.get('sliding_window') or tc.get('attention_chunk_size')
    mla = bool(tc.get('kv_lora_rank'))
    kv_lora = tc.get('kv_lora_rank') or 0
    rope = tc.get('qk_rope_head_dim') or 0
    nope = tc.get('qk_nope_head_dim') or 0
    vhd = tc.get('v_head_dim') or hd
    q_lora = tc.get('q_lora_rank')
    gating = str(tc.get('gating') or '').replace('_', '-')        # Laguna: per-element / per-head output gate
    attn_gate = bool(tc.get('attn_output_gate') or tc.get('attention_output_gate')) or gating == 'per-element'
    head_gate = gating == 'per-head'
    heads_per_layer = tc.get('num_attention_heads_per_layer') if isinstance(tc.get('num_attention_heads_per_layer'), list) else None
    # Gemma 3n / Gemma 4: the last num_kv_shared_layers reuse earlier layers' KV (no K/V projections, no cache)
    n_kv_shared = int(tc.get('num_kv_shared_layers') or 0)
    first_kv_shared = L - n_kv_shared if 0 < n_kv_shared < L else L

    attn = []
    dense_params = 0.0
    idx_types = tc.get('indexer_types')
    for i, kd in enumerate(kinds):
        if kd == 'none':
            continue
        if kd == 'mamba':
            spec, p = _mamba2(tc, H)
            attn.append(spec)
            dense_params += p
            continue
        if kd == 'shortconv':                       # LFM2 gated short convolution: in_proj H->3H, out_proj H->H
            lc = int(_g(tc, 'conv_L_cache', default=3))
            cd = int(_g(tc, 'conv_dim', default=H))
            attn.append(AttnSpec('linear', q_heads=0, state_elems=cd * lc, state_bytes_per_elem=2.0))
            dense_params += 3 * H * cd + cd * H + cd * lc
            continue
        if kd == 'linear':
            nv = int(_g(lin_cfg, 'num_heads', default=None) or _g(tc, 'linear_num_value_heads', default=nh))
            nk = int(_g(tc, 'linear_num_key_heads', default=nv))
            dk = int(_g(lin_cfg, 'head_dim', default=None) or _g(tc, 'linear_key_head_dim', default=hd))
            dv = int(_g(lin_cfg, 'head_dim', default=None) or _g(tc, 'linear_value_head_dim', default=hd))
            conv = int(_g(tc, 'linear_conv_kernel_dim', default=None) or _g(lin_cfg, 'short_conv_kernel_size', default=4))
            sb = 4.0 if str(tc.get('mamba_ssm_dtype', 'float32')) in ('float32', 'fp32') else 2.0
            if mt.startswith('minimax'):        # lightning attention: per-head d x d state
                nv, dk, dv = nh, hd, hd
            spec = AttnSpec('linear', q_heads=nv, state_elems=nv * dk * dv + (conv - 1) * (2 * nk * dk + nv * dv) * (2 / sb),
                            state_bytes_per_elem=sb)
            attn.append(spec)
            dense_params += H * (2 * nk * dk + 2 * nv * dv) + nv * dv * H + 2 * H * nv
            continue
        if mla:
            hq = nh
            ent = kv_lora + rope
            spec = AttnSpec(kd if kd != 'sliding' else 'full', q_heads=hq, kv_elems=ent,
                            d_qk_pre=nope + rope, d_v_pre=vhd, d_qk_dec=kv_lora + rope, d_v_dec=kv_lora)
            qp = (H * q_lora + q_lora * hq * (nope + rope)) if q_lora else H * hq * (nope + rope)
            dense_params += qp + H * (kv_lora + rope) + kv_lora * hq * (nope + vhd) + hq * vhd * H
        else:
            hdl, kvl = hd, kvh
            nhl = int(heads_per_layer[i]) if heads_per_layer and i < len(heads_per_layer) else nh
            if kd == 'full' and tc.get('global_head_dim'):
                hdl, kvl = int(tc['global_head_dim']), int(tc.get('num_global_key_value_heads') or kvh)
            kv_mult = 1 if (kd == 'full' and tc.get('attention_k_eq_v')) else 2
            spec = AttnSpec(kd, q_heads=nhl, kv_elems=kv_mult * kvl * hdl, d_qk_pre=hdl, d_v_pre=hdl, d_qk_dec=hdl, d_v_dec=hdl,
                            window=window if kd == 'sliding' else None)
            kv_proj = kv_mult * H * kvl * hdl
            if i >= first_kv_shared:
                spec.store_kv = False
                kv_proj = 0
            dense_params += (H * nhl * hdl + kv_proj + nhl * hdl * H + (H * nhl * hdl if attn_gate else 0)
                             + (H * nhl if head_gate else 0))
        if kd == 'sparse':
            spec.topk = int(tc['index_topk'])
            spec.idx_heads = int(tc.get('index_n_heads', 0))
            spec.idx_dim = int(tc.get('index_head_dim', 128))
            spec.idx_bytes = 1.0 + 4.0 / spec.idx_dim
            if isinstance(idx_types, list) and i < len(idx_types):
                spec.idx_compute = idx_types[i] == 'full'
            elif tc.get('index_topk_freq'):
                spec.idx_compute = (i % int(tc['index_topk_freq'])) == 0
            dense_params += (q_lora or H) * spec.idx_heads * spec.idx_dim + H * spec.idx_dim + H * spec.idx_heads
        if kd == 'blocksparse':
            spec.topk = int(sparse_cfg.get('sparse_topk_blocks', 16) * sparse_cfg.get('sparse_block_size', 128)
                            + sparse_cfg.get('sparse_local_block', 1) * sparse_cfg.get('sparse_block_size', 128))
            spec.idx_heads = int(sparse_cfg.get('sparse_num_index_heads', 0))
            spec.idx_dim = int(sparse_cfg.get('sparse_index_dim', 128))
            spec.compress = 1.0
            spec.idx_bytes = 2.0
        attn.append(spec)

    # ---------------- DeepSeek-V4 / V4.1 compressed attention
    prefill_scale = 1.0
    engram = 0.0
    if mt in ('deepseek_v4', 'deepseek_v41_text') or 'compress_ratios' in tc:
        ratios = list(tc.get('compress_ratios') or [])[:L]
        ent = int(tc.get('head_dim', 512))
        nhq = nh
        idx_h, idx_d, topk = int(tc.get('index_n_heads', 64)), int(tc.get('index_head_dim', 128)), int(tc.get('index_topk', 512))
        swa = int(tc.get('sliding_window', 128))
        kv_src = set(tc.get('kv_source_layer_ids') or [])
        idx_src = set(tc.get('index_source_layer_ids') or [])
        v41 = bool(kv_src)
        attn = []
        for i, r in enumerate(ratios + [0] * (L - len(ratios))):
            if r == 0:
                s = AttnSpec('swa', q_heads=nhq, d_qk_pre=ent, d_v_pre=ent, d_qk_dec=ent, d_v_dec=ent, swa=swa, swa_elems=ent)
            elif v41:
                s = AttnSpec('csa', q_heads=nhq, kv_elems=ent, d_qk_pre=ent, d_v_pre=ent, d_qk_dec=ent, d_v_dec=ent,
                             compress=float(r), topk=topk, swa=swa, swa_elems=ent, idx_heads=int(tc.get('index_n_heads', 32)),
                             idx_dim=idx_d, idx_bytes=0.5625, idx_compute=i in idx_src, store_kv=i in kv_src,
                             kv_bytes_fixed=0.5625)
            elif r <= 8:
                s = AttnSpec('csa', q_heads=nhq, kv_elems=ent, d_qk_pre=ent, d_v_pre=ent, d_qk_dec=ent, d_v_dec=ent,
                             compress=float(r), topk=topk, swa=swa, swa_elems=ent, idx_heads=idx_h, idx_dim=idx_d,
                             idx_bytes=0.5625, kv_bytes_fixed=1.0)
            else:
                s = AttnSpec('hca', q_heads=nhq, kv_elems=ent, d_qk_pre=ent, d_v_pre=ent, d_qk_dec=ent, d_v_dec=ent,
                             compress=float(r), swa=swa, swa_elems=ent, kv_bytes_fixed=1.0)
            attn.append(s)
        dense_params = None  # derived from the HF parameter total below
        if v41:
            prefill_scale = 0.5   # causal encoder-decoder: prompt runs through the encoder half only
            n_emb = tc.get('engram_num_embeddings') or []
            engram = float(sum(n_emb)) * float(tc.get('engram_head_dim', 256)) if n_emb else 0.0
            notes.append('Engram tables are host-memory resident by default (prefetched over RDMA).')

    # ---------------- parameter totals
    ffn_dense = dense_ffn_layers * mats * H * dense_inter
    if tc.get('use_double_wide_mlp') and n_kv_shared and not is_moe:
        ffn_dense += min(n_kv_shared, L) * mats * H * dense_inter     # Gemma 3n/4: KV-shared layers use a 2x MLP
    ffn_shared = moe_layers * shared_params + moe_layers * router
    expert_total = E * expert_each * moe_layers
    embed = V * H
    lm_head = 0.0 if tied else V * H
    n_mtp = int(_g(tc, 'num_nextn_predict_layers', 'mtp_num_hidden_layers', 'num_mtp_modules', default=0) or 0)
    if tc.get('mtp_transformer_layers') and tc.get('num_mtp_modules'):
        n_mtp = int(tc['num_mtp_modules']) * int(tc['mtp_transformer_layers'])
    arch = Arch(repo=repo, model_type=mt, n_layers=L, hidden=H, vocab=V, attn=attn, is_moe=is_moe,
                n_experts=E, top_k=k, moe_layers=moe_layers, expert_params_each=expert_each,
                params_expert_total=expert_total, params_embed=embed, params_lm_head=lm_head,
                params_engram=engram, prefill_active_scale=prefill_scale, notes=notes,
                kv_heads=1 if (mla or 'compress_ratios' in tc) else kvh,
                max_context=_g(tc, 'max_position_embeddings', 'max_sequence_length', 'seq_length'))
    if dense_params is not None:
        arch.params_dense_linear = dense_params + ffn_dense + ffn_shared
        per_layer = (arch.params_dense_linear + expert_total) / max(L, 1)
        arch.params_mtp = n_mtp * per_layer
    ple = int(tc.get('hidden_size_per_layer_input') or 0)
    if ple:     # Gemma 3n / 4 per-layer embedding tables: looked up, not multiplied; memory only
        arch.params_other += int(tc.get('vocab_size_per_layer_input') or V) * ple * L + 3 * H * ple * L
    arch.native_scheme = native_scheme(cfg, tc, is_moe, repo, arch.notes)
    _reconcile_with_hf(arch, repo, ffn_dense + ffn_shared, n_mtp, fetch, meta)
    _apply_overrides(arch)
    return arch


def _hf_total(meta: dict) -> float | None:
    st = (meta or {}).get('safetensors') or {}
    return float(st['total']) if st.get('total') else None


def _reconcile_with_hf(arch: Arch, repo: str, ffn_nonexpert: float, n_mtp: int, fetch: bool, meta: dict | None = None):
    """Use the published safetensors parameter total to correct the non-expert/aux budget."""
    meta = meta if meta is not None else load_hf_meta(repo, fetch)
    total = _hf_total(meta)
    arch.params_total_hf = total
    dtypes = set((((meta or {}).get('safetensors') or {}).get('parameters') or {}))
    unpacked = bool(dtypes) and dtypes <= {'BF16', 'F16', 'F32', 'F8_E4M3', 'F8_E5M2'}   # one stored element per parameter
    if total and unpacked and arch.params_expert_total and arch.params_total > 1.25 * total:
        # expert sharing the config does not describe (e.g. one expert pool reused across layers):
        # the checkpoint's parameter count is authoritative for memory
        cfg_total = arch.params_total
        arch.params_expert_total = max(arch.params_expert_total - (cfg_total - total), 0.05 * arch.params_expert_total)
        arch.notes.append(f'Config implies {cfg_total/1e9:,.0f}B parameters but the checkpoint holds {total/1e9:,.1f}B; '
                          'routed-expert memory was scaled to the checkpoint (expert layout not recognised).')
    if arch.params_dense_linear == 0 and total:      # exotic attention: back out non-expert params
        rest = total - arch.params_expert_total - arch.params_embed - arch.params_lm_head - arch.params_engram
        per_layer_exp = arch.params_expert_total / max(arch.n_layers, 1)
        mtp = n_mtp * per_layer_exp * 1.05
        arch.params_mtp = mtp
        arch.params_dense_linear = max(rest - mtp, 0.02 * total)
        return
    if arch.params_dense_linear == 0:
        arch.params_dense_linear = ffn_nonexpert
        arch.notes.append('Non-expert parameter count approximated (no HF total available).')
        return
    if total:
        diff = total - arch.params_total
        if diff > 0.01 * total:
            arch.params_other += diff          # vision tower, extra heads, etc.: memory only
        elif diff < -0.08 * total:
            arch.notes.append(f'Config-derived params exceed HF safetensors total by {-diff/1e9:.1f}B '
                              '(packed low-bit tensors or shared weights); using config-derived counts.')


def _apply_overrides(arch: Arch):
    if not os.path.exists(OVERRIDES):
        return
    ov = json.load(open(OVERRIDES)).get(arch.repo) or json.load(open(OVERRIDES)).get(_base_name(arch.repo))
    if not ov:
        return
    for k, v in ov.items():
        if k.startswith('_'):
            continue
        setattr(arch, k, v)


def _base_name(repo: str) -> str:
    return re.sub(r'[-_](FP8|NVFP4|MXFP4|MXFP8|AWQ.*|GPTQ.*|INT4|INT8|W4A16|BF16)$', '', repo, flags=re.I)


def summary(a: Arch) -> dict:
    return {
        'repo': a.repo, 'model_type': a.model_type, 'layers': a.n_layers, 'hidden': a.hidden,
        'moe': a.is_moe, 'experts': a.n_experts, 'top_k': a.top_k,
        'params_total_b': round(a.params_total / 1e9, 2),
        'params_total_hf_b': round(a.params_total_hf / 1e9, 2) if a.params_total_hf else None,
        'active_decode_b': round(a.params_active_decode / 1e9, 2),
        'active_prefill_b': round(a.params_active_prefill / 1e9, 2),
        'attention_kinds': {k: sum(1 for s in a.attn if s.kind == k) for k in sorted({s.kind for s in a.attn})},
        'kv_bytes_per_token_bf16@32k': round(a.kv_bytes_per_token(2.0), 1),
        'kv_bytes_per_token_fp8@32k': round(a.kv_bytes_per_token(1.0), 1),
        'native_scheme': a.native_scheme,
        'engram_b': round(a.params_engram / 1e9, 1),
        'notes': a.notes,
    }
