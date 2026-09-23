# Paths used by the research code

The scripts in `duplexjev/research/`, `evaluation/`, `training/`, `benchmarks/` and `examples/` are the exact versions used
for the paper. They still reference our cluster layout:

| name in code | meaning |
|---|---|
| `jev_qwen` | this repository's `duplexjev/research/` (internal name) |
| `/data/exp01/exp03_train` | project root (checkout of this repo, the Ultravox fork, data, artifacts) |
| `/data/exp01/models/Qwen3-32B`, `.../Qwen3-ASR-0.6B-hf` | frozen LLM and ASR model, downloaded from Hugging Face |
| `.../ultravox` | [Ultravox](https://github.com/fixie-ai/ultravox) checkout, patched by `training/ultravox_patches/` |
| `.../data/ml_asr/official` | local copy of the Ultravox v0.6 training mixture |
| `.../data/ml_asr/packs100/pNNN` | 100-pack split produced by `training/data/split100.py` |
| `.../artifacts/...` | outputs (evaluation JSON, latency, continuation store) |

A cleaned, installable package with configurable paths will replace this layout. Until then, recreate it or edit
the path constants at the top of each script.
