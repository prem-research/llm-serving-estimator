_100 users, 32768 prompt / 512 output tokens, 10 s think time, vLLM, auto layout_

| model (quantization) | 8xH100 | 8xH200 | 8xB200 | 8xMI300X |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct (fp8) | 10 tok/s/user · TTFT 49.39s · 347 tok/s · TP2 | 12 tok/s/user · TTFT 43.57s · 370 tok/s · TP2 | 19 tok/s/user · TTFT 26.93s · 755 tok/s · TP1 | 10 tok/s/user · TTFT 54.07s · 495 tok/s · TP1 |
| Qwen3-235B-A22B (fp8) | 12 tok/s/user · TTFT 93.98s · 362 tok/s · TP4 · KV-limited | 8 tok/s/user · TTFT 44.78s · 405 tok/s · TP4 | 16 tok/s/user · TTFT 14.00s · 946 tok/s · TP2 | 7 tok/s/user · TTFT 27.27s · 447 tok/s · TP2 |
| gpt-oss-120b (native) | 64 tok/s/user · TTFT 6.68s · 2,176 tok/s · TP1 | 67 tok/s/user · TTFT 5.38s · 2,352 tok/s · TP1 | 216 tok/s/user · TTFT 0.70s · 4,657 tok/s · TP1 | 61 tok/s/user · TTFT 3.53s · 2,660 tok/s · TP1 |
| Mistral-Small-4-119B-2603 (native) | 31 tok/s/user · TTFT 4.68s · 1,589 tok/s · TP2 | 50 tok/s/user · TTFT 4.24s · 1,908 tok/s · TP1 | 112 tok/s/user · TTFT 0.99s · 3,317 tok/s · TP1 | 36 tok/s/user · TTFT 2.91s · 1,814 tok/s · TP1 |
| DeepSeek-R1-0528 (native) | does not fit | 6 tok/s/user · TTFT 105.66s · 213 tok/s · TP8 · KV-limited | 7 tok/s/user · TTFT 42.77s · 436 tok/s · TP8 | 3 tok/s/user · TTFT 48.01s · 215 tok/s · TP8 |
| DeepSeek-V4-Flash (native) | 19 tok/s/user · TTFT 15.94s · 968 tok/s · TP4 | 25 tok/s/user · TTFT 11.16s · 1,133 tok/s · TP2 | 70 tok/s/user · TTFT 1.57s · 2,599 tok/s · TP2 | 24 tok/s/user · TTFT 7.01s · 1,234 tok/s · TP1 |
| Kimi-K2.5 (native) | 64 tok/s/user · TTFT 520.84s · 88 tok/s · TP8 · KV-limited | 7 tok/s/user · TTFT 160.04s · 185 tok/s · TP8 · KV-limited | 22 tok/s/user · TTFT 56.03s · 520 tok/s · TP4 · KV-limited | 4 tok/s/user · TTFT 229.02s · 131 tok/s · TP4 · KV-limited |
| GLM-5 (fp8) | does not fit | 12 tok/s/user · TTFT 145.24s · 214 tok/s · TP8 · KV-limited | 12 tok/s/user · TTFT 44.58s · 460 tok/s · TP8 · KV-limited | 5 tok/s/user · TTFT 98.33s · 226 tok/s · TP8 · KV-limited |
