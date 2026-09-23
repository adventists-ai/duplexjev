# Evaluation

All benchmarks are read with the same typed single-token protocol: options under letters, answer = next-token
distribution over those letters, 0 decode steps.

| benchmark | what it measures | size | where |
|---|---|---|---|
| **qa100** | speech path vs oracle transcript on spoken multiple-choice (zh + en, logic + fact) | 100 | 🤗 [adventists-ai/qa100](https://huggingface.co/datasets/adventists-ai/qa100) |
| **ZJU audio-gender-benchmark — `gender`** | speaker gender from audio (zh + en, real + TTS) | 100 | [Vsky-morigen/audio-gender-benchmark](https://github.com/Vsky-morigen/audio-gender-benchmark) |
| **ZJU audio-gender-benchmark — `main_language`** | spoken multiple-choice with real speech (fact, math, logic) | 100 | same repository |

The ZJU sets are maintained by their authors and are not redistributed here; cite the version you run
(`VERSION` file and commit).

Each run reports three conditions with one prompt: **audio** (question as speech), **text** (oracle transcript,
upper bound) and **options-only** (no question, lower bound).

| file | purpose |
|---|---|
| `build_qa100.py` | build an evaluation pack (JSON + audio) from a benchmark release |
| `run_qa100.py` | run a checkpoint on a pack; `--task qa` for multiple-choice, `--task gender` for gender |

Results for the released checkpoints will be listed in the main README.
Paths refer to our cluster; see [docs/PATHS.md](../docs/PATHS.md).
