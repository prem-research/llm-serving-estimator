_100 users, 2048 prompt / 512 output tokens, 10 s think time, vLLM, auto layout_

| model (quantization) | 8xH100 | 8xH200 | 8xB200 | 8xMI300X |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct (fp8) | 52 tok/s/user · TTFT 0.18s · 2,498 tok/s · TP8 | 60 tok/s/user · TTFT 0.14s · 2,759 tok/s · TP4 | 92 tok/s/user · TTFT 0.08s · 3,245 tok/s · TP8 | 51 tok/s/user · TTFT 0.27s · 2,762 tok/s · TP8 |
| Qwen3-235B-A22B (fp8) | 34 tok/s/user · TTFT 0.16s · 1,913 tok/s · TP8 | 38 tok/s/user · TTFT 0.12s · 2,014 tok/s · TP8 | 56 tok/s/user · TTFT 0.08s · 2,666 tok/s · TP8 | 25 tok/s/user · TTFT 0.18s · 1,595 tok/s · TP8 |
| gpt-oss-120b (native) | 177 tok/s/user · TTFT 0.05s · 3,763 tok/s · TP8 | 194 tok/s/user · TTFT 0.07s · 3,954 tok/s · TP1 | 356 tok/s/user · TTFT 0.03s · 5,320 tok/s · TP1 | 141 tok/s/user · TTFT 0.08s · 4,110 tok/s · TP1 |
| Mistral-Small-4-119B-2603 (native) | 100 tok/s/user · TTFT 0.05s · 3,162 tok/s · TP8 | 136 tok/s/user · TTFT 0.05s · 3,334 tok/s · TP1 | 209 tok/s/user · TTFT 0.04s · 4,097 tok/s · TP1 | 69 tok/s/user · TTFT 0.09s · 2,743 tok/s · TP1 |
| DeepSeek-R1-0528 (native) | does not fit | 24 tok/s/user · TTFT 0.17s · 1,502 tok/s · TP8 | 37 tok/s/user · TTFT 0.14s · 1,958 tok/s · TP8 | 14 tok/s/user · TTFT 0.23s · 1,027 tok/s · TP8 |
| DeepSeek-V4-Flash (native) | 68 tok/s/user · TTFT 0.10s · 2,425 tok/s · TP8 | 79 tok/s/user · TTFT 0.08s · 2,502 tok/s · TP8 | 142 tok/s/user · TTFT 0.04s · 3,484 tok/s · TP8 | 48 tok/s/user · TTFT 0.10s · 2,116 tok/s · TP8 |
| Kimi-K2.5 (native) | 41 tok/s/user · TTFT 41.30s · 727 tok/s · TP8 · KV-limited | 24 tok/s/user · TTFT 0.25s · 1,517 tok/s · TP8 | 45 tok/s/user · TTFT 0.08s · 2,455 tok/s · TP8 | 9 tok/s/user · TTFT 0.51s · 630 tok/s · TP8 |
| GLM-5 (fp8) | does not fit | 18 tok/s/user · TTFT 0.24s · 1,173 tok/s · TP8 | 28 tok/s/user · TTFT 0.11s · 1,618 tok/s · TP8 | 11 tok/s/user · TTFT 0.36s · 825 tok/s · TP8 |
