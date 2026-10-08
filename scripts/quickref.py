"""Quick-reference grid: popular models x 8-GPU nodes, 100 users, 2k prompt / 512 output, 10 s think time."""
import io, json, contextlib, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from servest.cli import main
MODELS = [('meta-llama/Llama-3.3-70B-Instruct', 'fp8'), ('Qwen/Qwen3-235B-A22B', 'fp8'), ('openai/gpt-oss-120b', 'native'),
          ('mistralai/Mistral-Small-4-119B-2603', 'native'), ('deepseek-ai/DeepSeek-R1-0528', 'native'),
          ('deepseek-ai/DeepSeek-V4-Flash', 'native'), ('moonshotai/Kimi-K2.5', 'native'), ('zai-org/GLM-5', 'fp8')]
GPUS = ['H100', 'H200', 'B200', 'MI300X']
ISL, OSL, USERS, THINK = int(sys.argv[1]) if len(sys.argv) > 1 else 2048, 512, 100, 10
print(f'_{USERS} users, {ISL} prompt / {OSL} output tokens, {THINK} s think time, vLLM, auto layout_\n')
print(f'| model (quantization) | ' + ' | '.join(f'8x{g}' for g in GPUS) + ' |')
print('|---|' + '---|' * len(GPUS))
for m, q in MODELS:
    cells = []
    for g in GPUS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = main(['estimate', '--model', m, '--gpu', g, '--gpus', '8', '--users', str(USERS), '--isl', str(ISL),
                       '--osl', str(OSL), '--think-time', str(THINK), '--quant', q, '--json', '--offline'])
        try:
            d = json.loads(buf.getvalue())
            cells.append(f"{d['decode_tps_per_user']:.0f} tok/s/user · TTFT {d['ttft_s']:.2f}s · {d['output_tps']:,.0f} tok/s · TP{d['tp']}" + (' · KV-limited' if d['kv_limited'] else ''))
        except Exception:
            cells.append('does not fit')
    print(f'| {m.split("/")[-1]} ({q}) | ' + ' | '.join(cells) + ' |')
