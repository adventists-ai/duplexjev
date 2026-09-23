# Training

The projector (and, for variant A, the encoder fusion block) is trained with a patched
[Ultravox](https://github.com/fixie-ai/ultravox) on the Ultravox v0.6 data mixture.
The ASR encoder (Qwen3-ASR-0.6B) and the LLM (Qwen3-32B) stay frozen.

| | variant A (`configs/v2_xattn_sf2.yaml`) | variant B (`configs/v1_h18_sf2.yaml`) |
|---|---|---|
| encoder read-out | cross-attention fusion, Q = h18, K = h14, V = h9 (8 heads, zero-init out-proj, residual) | last layer h18 |
| trainable params | 43.1 M | 39.9 M |
| stack factor | 2 (12.5 Hz → 6.25 tokens/s, 160 ms per token) | 2 |
| steps × global batch | 32,000 × 16 (4 GPUs × 4) | 32,000 × 16 |
| lr / warm-up | 2e-4 / 1,000 | 2e-4 / 1,000 |
| max audio | 25 s | 25 s |

## Data recipe

140 training sets from the Ultravox v0.6 mixture (the four Swedish sets are excluded). Most corpora appear twice:
as **transcription** and as **continuation** (the LLM's own continuation of the transcript, generated once with
Qwen3-32B). Transcription and continuation are 45.7 % each of the weighted mixture; the rest is CoVoST 2 speech
translation and MUSAN noise.

| source | share (weight × samples) | link |
|---|---:|---|
| WenetSpeech | 42.8 % | [wenet-e2e/WenetSpeech](https://github.com/wenet-e2e/WenetSpeech) |
| GigaSpeech | 24.2 % | [SpeechColab/GigaSpeech](https://github.com/SpeechColab/GigaSpeech) |
| Common Voice (non-English) | 14.9 % | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) |
| CoVoST 2 | 7.3 % | [facebookresearch/covost](https://github.com/facebookresearch/covost) |
| People's Speech | 4.4 % | [MLCommons](https://mlcommons.org/datasets/peoples-speech/) |
| Common Voice (English) | 3.2 % | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) |
| MUSAN (noise) | 1.2 % | [OpenSLR 17](https://www.openslr.org/17/) |
| LibriSpeech | 0.8 % | [OpenSLR 12](https://www.openslr.org/12/) |
| Common Voice + MUSAN | 0.6 % | — |
| Multilingual LibriSpeech | 0.4 % | [OpenSLR 94](https://www.openslr.org/94/) |

We do not redistribute audio. Download each corpus from its source (or the Ultravox dataset mirrors); licenses differ by corpus, and some are non-commercial.

## 100 packs

The mixture is split into 100 disjoint packs that each keep the official ratio, so a run can use a few packs and a
later run can continue on unseen ones.

* Pack of a row = `blake2b(key) % 100`, key = `audio.path` → `path` → `file` → `id`; WenetSpeech groups by source recording.
* Subsets with fewer than 6,400 rows are shared by all packs.
* One pack ≈ 268k distinct clips ≈ 680k weighted samples; one run (32,000 × 16 = 512k samples) uses ~75 % of a pack.
* `verify100.py` checks row conservation, hash ownership, per-pack ratios and what the training loader actually reads.

## Files

| path | purpose |
|---|---|
| `configs/` | Ultravox training configs for A and B |
| `run_sf2_ab.sh` | launch A and B on one pack (`PACK=0 bash run_sf2_ab.sh`) |
| `data/split100.py`, `data/verify100.py`, `data/data_audit_lib.py` | 100-pack split and verification |
| `data/official_local.py`, `data/wenet_lhotse.py` | map the Ultravox dataset configs to local copies |
| `data/cont_coverage.py` | refuse to train when fewer than 95 % of a pack's rows have a continuation |
| `continuation/` | generate and look up Qwen3-32B continuations |
| `ultravox_patches/` | patches to Ultravox: stack factor, continuation lookup, pack filter |

Paths refer to our cluster; see [docs/PATHS.md](../docs/PATHS.md).
