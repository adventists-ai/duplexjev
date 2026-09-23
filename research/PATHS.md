# Paths in research/

These scripts are the exact versions used for the paper and still reference our cluster layout:

| prefix in code | meaning |
|---|---|
| `/data/exp01/exp03_train` | project root (checkouts of this repo + Ultravox fork, data, artifacts) |
| `/data/exp01/models/Qwen3-32B`, `.../Qwen3-ASR-0.6B-hf` | frozen LLM / ASR from the original model hubs |
| `.../data/ml_asr/official` | local copy of the Ultravox v0.6 training mixture |
| `.../data/ml_asr/packs100/pNNN` | 100-pack split produced by `data_pipeline/split100.py` |
| `.../artifacts/...` | outputs (eval JSON, latency, continuation store) |

A cleaned package with configurable paths will replace this folder. Until then, create the same layout or
edit the path constants at the top of each script.
