"""Total cost of ownership: buy vs rent the GPUs a sizing needs, against current cloud + model-API spend.

Money is USD per month unless a name says otherwise. Defaults come from data/prices.json (dated planning
prices with sources); every value can be overridden. Pure stdlib so it runs in the browser (Pyodide).

Own   = servers bought outright (whole 8-GPU nodes) + networking/storage uplift, depreciated straight-line;
        monthly running cost = power (average draw x PUE) + colocation per kW + support contract + hardware ops staff.
Rent  = GPU-hours at a per-GPU rate, billed around the clock (reserved/committed) or only in active hours.
API   = tokens the same workload sends, at a per-million-token list price.
"""
from __future__ import annotations
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES_PATH = os.path.join(HERE, '..', 'data', 'prices.json')
HOURS_PER_MONTH = 730.0

_PRICES = None


def load_prices() -> dict:
    global _PRICES
    if _PRICES is None:
        _PRICES = json.load(open(PRICES_PATH)) if os.path.exists(PRICES_PATH) else {'gpus': {}, 'apis': [], 'defaults': {}}
    return _PRICES


def assumptions(overrides: dict | None = None) -> dict:
    """Defaults merged with overrides. overrides['gpus'][key] may replace any per-GPU price field."""
    p = load_prices()
    a = dict(p.get('defaults', {}))
    a['gpus'] = {k: {f: v for f, v in g.items() if not f.startswith('_')} for k, g in p.get('gpus', {}).items()}
    for k, v in (overrides or {}).items():
        if k == 'gpus' and isinstance(v, dict):
            for g, fields in v.items():
                a['gpus'].setdefault(g, {}).update({f: x for f, x in (fields or {}).items() if x is not None})
        elif v is not None:
            a[k] = v
    return a


def api_price(key_or_prices) -> dict:
    """{'in': $/M input tokens, 'out': $/M output tokens, 'name': ...} from a preset key or explicit prices."""
    if isinstance(key_or_prices, dict):
        return {'name': key_or_prices.get('name', 'custom'), 'in': float(key_or_prices['in']), 'out': float(key_or_prices['out'])}
    for api in load_prices().get('apis', []):
        if api['key'] == key_or_prices:
            return {'name': api['name'], 'in': float(api['in']), 'out': float(api['out'])}
    raise KeyError(f'Unknown API price preset {key_or_prices!r}')


def blended_usd_per_mtok(isl: float, osl: float, price: dict) -> float:
    """Average $ per million tokens (input + output) for a request shape."""
    return (isl * price['in'] + osl * price['out']) / max(isl + osl, 1e-9)


def workload_from_api_spend(spend_month: float, isl: float, osl: float, price: dict, active_hours: float,
                            peak_to_average: float) -> dict:
    """Token volume implied by a monthly API bill, and the peak request rate the fleet must carry."""
    usd_per_request = (isl * price['in'] + osl * price['out']) / 1e6
    requests_month = spend_month / max(usd_per_request, 1e-12)
    avg_rps = requests_month / max(active_hours * 3600.0, 1.0)
    return {'requests_month': requests_month, 'tokens_month': requests_month * (isl + osl),
            'avg_requests_per_s': avg_rps, 'peak_requests_per_s': avg_rps * peak_to_average}


def own_costs(n_gpus: int, gpu: dict, a: dict) -> dict:
    nodes = math.ceil(n_gpus / gpu.get('node_gpus', 8))
    hardware = nodes * gpu['server_usd']
    capex = hardware * (1 + a['infra_uplift_pct'] / 100.0)
    kw = nodes * gpu['node_kw']
    power = kw * a['power_draw_pct'] / 100.0 * a['pue'] * HOURS_PER_MONTH * a['usd_per_kwh']
    colo = kw * a['colo_usd_kw_month']
    support = hardware * a['support_pct_year'] / 100.0 / 12.0
    ops = a['ops_fte'] * a['usd_per_fte_year'] / 12.0
    opex = power + colo + support + ops
    depreciation = capex / (a['depreciation_years'] * 12.0)
    return {'nodes': nodes, 'gpus_bought': nodes * gpu.get('node_gpus', 8), 'capex': capex, 'monthly_opex': opex,
            'monthly_depreciation': depreciation, 'monthly_amortized': opex + depreciation, 'kw': kw,
            'breakdown': {'power': power, 'colocation': colo, 'support': support, 'operations': ops}}


def rent_costs(n_gpus: int, gpu: dict, a: dict) -> dict:
    hours = a['active_hours_month'] if a.get('rent_active_hours_only') else HOURS_PER_MONTH
    return {'gpus': n_gpus, 'hours': hours, 'usd_gpu_h': gpu['rent_usd_gpu_h'], 'monthly': n_gpus * gpu['rent_usd_gpu_h'] * hours}


def _breakeven(capex: float, monthly_saving: float) -> float | None:
    """Months until buying pays back against an alternative that costs `monthly_saving` more per month."""
    return capex / monthly_saving if monthly_saving > 0 else None


def compare(sizing: list, *, isl: float, osl: float, requests_per_s: float | None = None, current_cloud: float = 0.0,
            current_api: float = 0.0, api: str | dict | None = None, overrides: dict | None = None,
            acceptance: bool = False) -> dict:
    """Cost every sized GPU option. `sizing` rows come from api.size(); `requests_per_s` is the average request
    rate during active hours (defaults to each row's rate at the sized load)."""
    a = assumptions(overrides)
    price = api_price(api or a.get('api', 'claude-sonnet-5-5'))
    months = int(round(a['horizon_years'] * 12))
    current = (current_cloud or 0.0) + (current_api or 0.0)
    options = []
    for row in sizing:
        gpu = a['gpus'].get(row['gpu'])
        if not row.get('meets') or gpu is None:
            options.append({'gpu': row['gpu'], 'available': False,
                            'reason': 'no price data for this GPU' if gpu is None else 'target not met within the GPU limit'})
            continue
        n = int(row['n_gpus'])
        acc = int(row.get('tp') or 1) if acceptance else 0
        own = own_costs(n, gpu, a)
        if acc and own['gpus_bought'] - n < acc:          # no spare GPUs on the production nodes: buy another node
            own = own_costs(n + acc, gpu, a)
        rent = rent_costs(n + acc, gpu, a)
        rps = requests_per_s if requests_per_s is not None else row.get('requests_per_s', 0.0)
        reqs = rps * a['active_hours_month'] * 3600.0
        mtok = reqs * (isl + osl) / 1e6
        api_month = reqs * (isl * price['in'] + osl * price['out']) / 1e6
        own_cash = own['capex'] + own['monthly_opex'] * months
        options.append({
            'gpu': row['gpu'], 'available': True, 'n_gpus': n, 'acceptance_gpus': acc,
            'own': {**own, 'horizon_total': own_cash, 'usd_per_mtok': own['monthly_amortized'] / mtok if mtok else None},
            'rent': {**rent, 'horizon_total': rent['monthly'] * months, 'usd_per_mtok': rent['monthly'] / mtok if mtok else None},
            'api_equivalent_month': api_month, 'tokens_month_m': mtok, 'requests_month': reqs,
            'breakeven_own_vs_rent_months': _breakeven(own['capex'], rent['monthly'] - own['monthly_opex']),
            'breakeven_own_vs_current_months': _breakeven(own['capex'], current - own['monthly_opex']) if current else None,
            'saving_vs_current_year': {'own': (current - own['monthly_amortized']) * 12 if current else None,
                                       'rent': (current - rent['monthly']) * 12 if current else None},
        })
    avail = [o for o in options if o['available']]
    best_own = min(avail, key=lambda o: o['own']['horizon_total'], default=None)
    best_rent = min(avail, key=lambda o: o['rent']['horizon_total'], default=None)
    return {
        'assumptions': {k: v for k, v in a.items() if k != 'gpus'}, 'gpu_prices': a['gpus'], 'api': price,
        'months': months, 'current_month': current, 'current_cloud': current_cloud, 'current_api': current_api,
        'options': options,
        'best': {'own': best_own['gpu'] if best_own else None, 'rent': best_rent['gpu'] if best_rent else None},
        'as_of': load_prices().get('as_of'), 'sources': load_prices().get('sources', []),
    }


def series(option: dict, months: int, current_month: float = 0.0) -> dict:
    """Cumulative cash spend per month 0..months for charting (buying is paid up front)."""
    m = list(range(months + 1))
    out = {'month': m,
           'own': [option['own']['capex'] + option['own']['monthly_opex'] * i for i in m],
           'rent': [option['rent']['monthly'] * i for i in m],
           'api': [option['api_equivalent_month'] * i for i in m]}
    if current_month:
        out['current'] = [current_month * i for i in m]
    return out
