# qa100 datasheet

**Purpose.** Measure how much a speech path loses against reading the oracle transcript when the answer is read as a
single constrained token. Not a general knowledge benchmark.

**Composition.** 100 four-way multiple-choice items: zh-logic 25, zh-fact 25, en-logic 25, en-fact 25.
Gold letters balanced (A/B/C/D = 25 each). Option order is fixed per item (deterministic hash shuffle).

**Audio.** Question stem only, synthesised with VoxCPM2 (default voice, cfg 2.0, 10 steps). Every clip was
transcribed by an independent ASR (Qwen3-ASR-1.7B); clips without an exact normalised match were re-synthesised
(`clean` field records the check). Options are given as text.

**Conditions.** `audio` (stem as audio), `text` (oracle transcript; upper bound), `options_only` (no stem; lower bound).
Reference scores with Qwen3-32B: text 91, options-only 36, Ultravox v0.6 Whisper speech path 89 (/100).

**Limitations.** Synthetic speech only; single voice; 100 items (95% CI roughly ±6 points). For real speech see the
ZJU main-language benchmark (evaluation only, not redistributed here).

**License.** CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Questions were written by the authors.

**Files.** `qa100.json` (items; `audio` paths are relative to this folder), `audio/*.wav`.
