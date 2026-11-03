# Hearing the whole clip: the 6.5 vs 6.25 tokens-per-second bug and DuplexJev-32B-Para v1.5

*2026-11-03 · [中文](2026-11-full-length_zh.md) · model: [DuplexJev-32B-Para v1.5](https://huggingface.co/adventists-ai/DuplexJev-32B-Para)*

## The bug

DuplexJev feeds speech to the LLM through the Qwen3-ASR-0.6B encoder and a connector that stacks two encoder frames
into one audio token. Our processors (the Ultravox-derived training code, the `duplexjev` package and the vLLM plugin)
computed the number of audio tokens as `ceil(mel_frames / 16)`, i.e. 6.25 tokens per second, the value one gets
from an 8× downsampling encoder.

Qwen3-ASR does not downsample by a flat 8×. It encodes audio in 1 s windows of 100 mel frames and emits **13** frames
per window, so the connector produces **6.5** audio tokens per second. With 6.25 tokens reserved, the last ~4 % of
every 30 s chunk (about 0.5 s of every 12 s, more on long clips) never reached the LLM. For most decisions this cost
little, but the end of an utterance is exactly where end-of-turn, a final stressed word or a trailing laugh lives.

## The fix

- The processors now compute the token count from the encoder's own output length
  (`_get_feat_extract_output_lengths`), in training and in inference.
- v1.5 was retrained on the true length from v1.4: two stages (2,000 + 1,500 steps, connector + a new rank-16 LoRA
  each, merged), replaying the full v1.4 mix, plus transcription items with native-language instructions and padded
  silence (FLEURS dev split and the v1.4 weak-language speech; the FLEURS test split was never used) and a light replay
  of all emotion data. The training log's length check went from 0.96 to 1.00.
- Simply switching an old checkpoint to the full length without retraining did not help (content unchanged, turn
  detection slightly worse): the connector had learned to live with the truncated input, so the fix needs retraining.
- Compatibility: v1.5 sets `"audio_full_length": true` in `config.json`. `duplexjev-vllm>=0.4` and `duplexjev>=0.7`
  read it; older checkpoints keep the old count and behave exactly as before (checked: v1.4 through the new plugin
  gives the same vLLM acceptance numbers). `DJ_FULL_LEN=1/0` overrides the flag.

## Results (v1.4 → v1.5)

| | v1.4 | v1.5 |
|---|---:|---:|
| qa100 / ZJU-ML (vLLM) | 97 / 90 | **98 / 91** |
| gender / emotion (vLLM) | 92.1 / 89.2 | **92.6 / 90.0** |
| Easy-Turn (vLLM) | 94.1 | **95.4** |
| Smart Turn v3.2 test set, all / English / Chinese | 93.3 / 93.5 / 91.4 | **94.4 / 94.4 / 94.6** |
| eot-bench false cut-offs at 300 / 600 ms, English (fast mode, assistant's previous message) | 20.4 / 8.9% | **17.7 / 7.0%** |
| eot-bench, Chinese | 42.3 / 13.8% | **41.0 / 13.3%** |
| StressTest | 81.7 | **85.3** |
| StressPresso (which meaning the stress implies) | 88.6 | **89.1** |
| DEBATE stressed word / pause | 89.8 / 79.8 | **90.7 / 81.4** |
| LISTEN emotion / non-verbal (audio only) | 32.4 / 48.8 | **33.2 / 51.6** |

Both columns use the same harness. Smart Turn's training set and Easy-Turn's training split are in our mix
(in-distribution); eot-bench audio is not.

## If you built on our processors

The same counting rule appears wherever the Qwen3-ASR audio representation is produced or consumed. If you used our
processors in your own training or inference code, update to `duplexjev>=0.7` / `duplexjev-vllm>=0.4`, or compute the
token count from the encoder's output length as above.
