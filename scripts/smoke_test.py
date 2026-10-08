"""Smoke tests: architecture parsing vs official figures, and a few measured anchor points."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from servest import arch  # noqa: E402
from servest.cli import main  # noqa: E402
import json, io, contextlib  # noqa: E402

PARAMS = {  # repo: (total B, active B) from model cards
    'openai/gpt-oss-120b': (116.8, 5.1), 'deepseek-ai/DeepSeek-R1-0528': (685, 37), 'meta-llama/Llama-3.3-70B-Instruct': (70.6, 69.5),
    'Qwen/Qwen3.5-397B-A17B': (403, 17), 'zai-org/GLM-5': (754, 40), 'moonshotai/Kimi-K2.5': (1027, 32),
    'deepseek-ai/DeepSeek-V4-Flash': (291, 13), 'deepseek-ai/DeepSeek-V4-Pro': (1599, 49), 'Qwen/Qwen3-235B-A22B': (235, 22),
}
ok = True
for repo, (tot, act) in PARAMS.items():
    a = arch.parse(repo, fetch=False)
    et, ea = abs(a.params_total / 1e9 / tot - 1), abs(a.params_active_decode / 1e9 / act - 1)
    flag = 'OK ' if et < 0.06 and ea < 0.12 else 'BAD'
    ok &= flag == 'OK '
    print(f'{flag} {repo:40s} total {a.params_total/1e9:7.1f} (ref {tot}) active {a.params_active_decode/1e9:5.1f} (ref {act})')

# measured anchors: InferenceX gpt-oss-120b, 1xH200, vLLM, 1k/1k (upper-bound lengths -> mean 922)
ANCHORS = [(4, 6.4), (16, 12.7), (64, 25.2)]   # users, measured median TPOT ms
for users, tpot in ANCHORS:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(['estimate', '--model', 'openai/gpt-oss-120b', '--gpu', 'H200', '--gpus', '1', '--users', str(users),
              '--isl', '922', '--osl', '922', '--json', '--offline'])
    pred = json.loads(buf.getvalue())['tpot_ms']
    flag = 'OK ' if abs(pred / tpot - 1) < 0.25 else 'BAD'
    ok &= flag == 'OK '
    print(f'{flag} gpt-oss-120b 1xH200 vLLM users={users:3d}: TPOT pred {pred:5.1f} ms vs measured {tpot} ms')
print('ALL OK' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
