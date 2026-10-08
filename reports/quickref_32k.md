_100 users, 32768 prompt / 512 output tokens, 10 s think time, vLLM, auto layout_

| model (quantization) | 8xH100 | 8xH200 | 8xB200 | 8xMI300X |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct (fp8) | 10 tok/s/user · TTFT 48.69s · 350 tok/s · TP2 | 12 tok/s/user · TTFT 42.94s · 375 tok/s · TP2 | 18 tok/s/user · TTFT 26.51s · 766 tok/s · TP1 | 10 tok/s/user · TTFT 53.17s · 500 tok/s · TP1 |
| Qwen3-235B-A22B (fp8) | 12 tok/s/user · TTFT 93.00s · 365 tok/s · TP4 · KV-limited | 8 tok/s/user · TTFT 43.16s · 410 tok/s · TP4 | 16 tok/s/user · TTFT 13.50s · 955 tok/s · TP2 | 7 tok/s/user · TTFT 26.41s · 451 tok/s · TP2 |
| gpt-oss-120b (native) | 64 tok/s/user · TTFT 6.59s · 2,169 tok/s · TP1 | 66 tok/s/user · TTFT 5.27s · 2,345 tok/s · TP1 | 211 tok/s/user · TTFT 0.71s · 4,629 tok/s · TP1 | 60 tok/s/user · TTFT 3.49s · 2,654 tok/s · TP1 |
| Mistral-Small-4-119B-2603 (native) | 31 tok/s/user · TTFT 4.72s · 1,584 tok/s · TP2 | 49 tok/s/user · TTFT 4.32s · 1,887 tok/s · TP1 | 110 tok/s/user · TTFT 1.01s · 3,298 tok/s · TP1 | 36 tok/s/user · TTFT 2.97s · 1,802 tok/s · TP1 |
| DeepSeek-R1-0528 (native) | does not fit | 6 tok/s/user · TTFT 104.59s · 216 tok/s · TP8 · KV-limited | 7 tok/s/user · TTFT 41.37s · 444 tok/s · TP8 | 3 tok/s/user · TTFT 46.51s · 218 tok/s · TP8 |
| DeepSeek-V4-Flash (native) | 20 tok/s/user · TTFT 14.97s · 988 tok/s · TP4 | 26 tok/s/user · TTFT 10.37s · 1,157 tok/s · TP2 | 71 tok/s/user · TTFT 1.52s · 2,611 tok/s · TP2 | 24 tok/s/user · TTFT 6.57s · 1,249 tok/s · TP1 |
| Kimi-K2.5 (native) | 65 tok/s/user · TTFT 520.94s · 89 tok/s · TP8 · KV-limited | 8 tok/s/user · TTFT 158.72s · 189 tok/s · TP8 · KV-limited | 23 tok/s/user · TTFT 55.77s · 530 tok/s · TP4 · KV-limited | 4 tok/s/user · TTFT 228.17s · 134 tok/s · TP4 · KV-limited |
| GLM-5 (fp8) | does not fit | 12 tok/s/user · TTFT 143.31s · 216 tok/s · TP8 · KV-limited | 12 tok/s/user · TTFT 44.14s · 465 tok/s · TP8 · KV-limited | 5 tok/s/user · TTFT 97.54s · 229 tok/s · TP8 · KV-limited |
