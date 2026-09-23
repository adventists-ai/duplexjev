# Latency and cost benchmarks

Reproduces the latency, SLO-capacity and prefix-sharing numbers in the paper (1× H200, bf16, Qwen3-32B).

| file | experiment |
|---|---|
| `cascade_prep.py`, `cascade_bench.py` | cascade baseline: Qwen3-ASR transcript → Qwen3-32B generates a JSON with 10 decisions |
| `engine_bench.py` | same vLLM engine for both methods: single-token readout vs JSON generation, per-event latency |
| `engine_slo.py` | events/s served within a 0.25 / 0.5 / 2 s budget |
| `prefix_share.py`, `prefix_share_m3.py` | N questions over a shared context: sequential vs KV copy vs one packed row (block-diagonal mask) |

Paths refer to our cluster; see [docs/PATHS.md](../docs/PATHS.md).
