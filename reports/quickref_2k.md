_100 users, 2048 prompt / 512 output tokens, 10 s think time, vLLM, auto layout_

| model (quantization) | 8xH100 | 8xH200 | 8xB200 | 8xMI300X |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct (fp8) | 52 tok/s/user · TTFT 0.18s · 2,490 tok/s · TP8 | 60 tok/s/user · TTFT 0.14s · 2,759 tok/s · TP4 | 91 tok/s/user · TTFT 0.08s · 3,258 tok/s · TP8 | 51 tok/s/user · TTFT 0.27s · 2,758 tok/s · TP8 |
| Qwen3-235B-A22B (fp8) | 34 tok/s/user · TTFT 0.16s · 1,916 tok/s · TP8 | 38 tok/s/user · TTFT 0.12s · 2,015 tok/s · TP8 | 56 tok/s/user · TTFT 0.08s · 2,672 tok/s · TP8 | 25 tok/s/user · TTFT 0.18s · 1,598 tok/s · TP8 |
| gpt-oss-120b (native) | 177 tok/s/user · TTFT 0.05s · 3,752 tok/s · TP8 | 192 tok/s/user · TTFT 0.07s · 3,939 tok/s · TP1 | 351 tok/s/user · TTFT 0.03s · 5,308 tok/s · TP1 | 139 tok/s/user · TTFT 0.08s · 4,099 tok/s · TP1 |
| Mistral-Small-4-119B-2603 (native) | 101 tok/s/user · TTFT 0.05s · 3,163 tok/s · TP8 | 135 tok/s/user · TTFT 0.05s · 3,325 tok/s · TP1 | 206 tok/s/user · TTFT 0.04s · 4,093 tok/s · TP1 | 69 tok/s/user · TTFT 0.06s · 2,743 tok/s · TP8 |
| DeepSeek-R1-0528 (native) | does not fit | 24 tok/s/user · TTFT 0.17s · 1,508 tok/s · TP8 | 37 tok/s/user · TTFT 0.14s · 1,966 tok/s · TP8 | 14 tok/s/user · TTFT 0.23s · 1,031 tok/s · TP8 |
| DeepSeek-V4-Flash (native) | 68 tok/s/user · TTFT 0.10s · 2,426 tok/s · TP8 | 79 tok/s/user · TTFT 0.07s · 2,502 tok/s · TP8 | 143 tok/s/user · TTFT 0.04s · 3,483 tok/s · TP8 | 48 tok/s/user · TTFT 0.10s · 2,116 tok/s · TP8 |
| Kimi-K2.5 (native) | 41 tok/s/user · TTFT 41.57s · 730 tok/s · TP8 · KV-limited | 24 tok/s/user · TTFT 0.25s · 1,518 tok/s · TP8 | 45 tok/s/user · TTFT 0.08s · 2,452 tok/s · TP8 | 9 tok/s/user · TTFT 0.51s · 630 tok/s · TP8 |
| GLM-5 (fp8) | does not fit | 18 tok/s/user · TTFT 0.24s · 1,178 tok/s · TP8 | 28 tok/s/user · TTFT 0.11s · 1,622 tok/s · TP8 | 11 tok/s/user · TTFT 0.36s · 828 tok/s · TP8 |
