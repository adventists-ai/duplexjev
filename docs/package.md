# The `duplexjev` package in detail

## Choosing a model

`Decider.from_pretrained(...)` (CLI: `--model`) takes a Hugging Face repo id or a local path of a **speech checkpoint in
Ultravox format**. A checkpoint is an ASR encoder plus the connector trained for it, and it names the frozen LLM it was
trained against. So you pick one name; the encoder comes with it, and the LLM is fetched from the id in its config.

```python
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B")      # download everything
d = Decider.from_pretrained("adventists-ai/DuplexJev-B-Qwen3-ASR-0.6B-Qwen3-32B",
                            text_model="/data/models/Qwen3-32B")                        # reuse a local LLM copy
d = Decider.from_pretrained("/data/models/my-checkpoint", device="auto")                # local path, shard over GPUs
```

| checkpoint | encoder (inside) | frozen LLM | status |
|---|---|---|---|
| `adventists-ai/DuplexJev-*` (see the [README leaderboard](../README.md#3-models)) | as named | as named | released, tested with 0.2.2 |
| `fixie-ai/ultravox-v0_6-qwen-3-32b` | Whisper-large-v3-turbo | Qwen3-32B | tested (qa100 0.89) |
| `fixie-ai/ultravox-v0_6-gemma-3-27b` | Whisper-large-v3-turbo | Gemma-3-27B (licence acceptance needed) | same format |

- **The ASR encoder is not a separate choice.** The connector only works with the encoder it was trained on.
  `audio_model=` / `text_model=` exist only to point at a *local copy of the same* encoder or LLM (offline machines,
  shared model folders); a different encoder or LLM loads without error but gives meaningless answers.
- Only the encoder's hidden states are used: nothing is transcribed.
- `device="cuda:0"` (default: first GPU, bf16) or `device="auto"` to shard a large LLM over several GPUs.
  A 32B LLM in bf16 needs one 80 GB GPU.
- Standard Hugging Face variables apply: `HF_TOKEN` (gated models), `HF_HOME` (cache location),
  `HF_ENDPOINT` (mirror), `HF_HUB_OFFLINE=1` (use the local cache only).
- Speech checkpoints in Ultravox format need transformers 4.51–4.55, which the `speech` extra installs. The SenseVoice
  encoder also needs `torchaudio`.

## What the package guarantees

Unit tests in [`tests/`](../tests), checked on Qwen3-32B:

- **0 decode steps.** Each answer is the next-token softmax over its option letters; options appear under permuted
  letters (`n_perm` averages several orders).
- **Exact prefix sharing** (`mode="packed"`, default): the context and audio of an item are encoded once and all its
  questions are packed into one row under a block-diagonal mask. The packing engine, checked in fp32 on Qwen3-32B,
  matches one-row-per-question to within 2e-5; in bf16 the mean difference is 0.001–0.002, with at most 1/100 answers
  changed.
- **Recurrent LLMs.** LLMs with recurrent or convolutional layers (Falcon-H1 and other Mamba-style hybrids, LFM2) carry
  state along the row, so packing is not valid for them; since 0.2.2 the package detects them and uses
  `mode="batch"` (one row per question) automatically.
- **Batch invariance.** An item's answers do not depend on which other items share the pass (fp32: max difference
  1e-5). For speech this requires aligning every clip to whole audio tokens; the package pads silence at the end of each
  clip (batched Ultravox inference otherwise leaks padding into the last audio token of shorter clips). In bf16, GPU
  kernels depend on batch shape, so running an item alone vs. in a batch changed 1/100 answers.
- **Checked accuracy.** Through the package (default prompts), every released connector is within 0–5 points of the
  paper protocol (see each model card).
- **Speed.** Plain PyTorch, one A800, bf16, Qwen3-32B: 8 questions for 64 calls in 9 s (about 140 ms per call); 48
  concurrent server requests are answered in one tick. The paper's latency numbers use a vLLM engine; a vLLM backend for
  the package is on the roadmap.

More in [`examples/`](../examples): tick-batch timing, server client, speech quick start.
