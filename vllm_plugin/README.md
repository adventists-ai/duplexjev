# duplexjev-vllm

vLLM plugin for [DuplexJev](https://github.com/adventists-ai/duplexjev) speech-decision models, such as
[`adventists-ai/DuplexJev-4B-Para`](https://huggingface.co/adventists-ai/DuplexJev-4B-Para).

```bash
pip install "vllm[audio]>=0.29" "duplexjev-vllm>=0.4"
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096
```

The plugin registers the architectures `DuplexJevForConditionalGeneration` (all Qwen-based DuplexJev models: 32B-Para, 4B-Para, …) and,
since 0.2.0, `DuplexJevGemmaForConditionalGeneration` (DuplexJev-Gemma-31B: audio embeddings scaled ×3 and wrapped in
Gemma's `<|audio> … <audio|>` markers), and since 0.3.0 `DuplexJevVisionForConditionalGeneration` (DuplexJev-32B-Vision,
speech and an image in one request; see below) through vLLM's `vllm.general_plugins` entry point, so nothing needs to be
imported by hand. It adds the Qwen3-ASR audio encoder and the matching
preprocessing (128-bin log-mel, clips padded to whole audio tokens) to vLLM's Ultravox implementation; the connector
and the LLM run on vLLM's own code.

Each question is one request with `max_tokens=1`; the answer is the next-token distribution over the option letters.
A client that asks several questions about one clip concurrently is in
[`examples/vllm_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_client.py).

## Full-length audio tokens (0.4.0)

The Qwen3-ASR encoder emits 13 frames per second (100 mel frames), i.e. 6.5 audio tokens per second after stacking.
Up to 0.3.x the plugin counted `ceil(mel_frames / 16)` tokens, 6.25 per second, so the last ~4 % of every 30 s chunk
was dropped. Checkpoints trained on the true length (DuplexJev-32B-Para v1.5+) carry `"audio_full_length": true` in
`config.json`; 0.4.0 reads it and feeds the whole clip. Older checkpoints keep the old count, so their behaviour does
not change. `DJ_FULL_LEN=1` / `DJ_FULL_LEN=0` force either count for any checkpoint.

## Speech + image (DuplexJev-32B-Vision, 0.3.0)

```bash
vllm serve adventists-ai/DuplexJev-32B-Vision --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1, "audio": 1, "video": 0}'
```

vLLM's Qwen3-VL implementation plus the DuplexJev audio encoder and connector (loaded from `audio/` in the checkpoint).
In a chat request, put `<|audio|>` in the text where the speech goes, an `image_url` part where the picture goes, and
the clip as an `input_audio` part. Audio tokens take plain text positions in Qwen3-VL's 3-D RoPE, as in training.
Client: [`examples/vllm_vision_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_vision_client.py).

`modeling_qwen3_asr_encoder.py` is a standalone port of the Qwen3-ASR audio encoder from Hugging Face `transformers`
(Apache-2.0, Copyright the Qwen team, Alibaba Cloud, and the HuggingFace team). The rest of this package is
Apache-2.0, Copyright Adventists.ai.
