# duplexjev-vllm

vLLM plugin for [DuplexJev](https://github.com/adventists-ai/duplexjev) speech-decision models, such as
[`adventists-ai/DuplexJev-4B`](https://huggingface.co/adventists-ai/DuplexJev-4B).

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm
vllm serve adventists-ai/DuplexJev-4B --max-model-len 4096
```

The plugin registers the architecture `DuplexJevForConditionalGeneration` through vLLM's `vllm.general_plugins`
entry point, so nothing needs to be imported by hand. It adds the Qwen3-ASR audio encoder and the matching
preprocessing (128-bin log-mel, clips padded to whole audio tokens) to vLLM's Ultravox implementation; the connector
and the LLM run on vLLM's own code.

Each question is one request with `max_tokens=1`; the answer is the next-token distribution over the option letters.
A client that asks several questions about one clip concurrently is in
[`examples/vllm_client.py`](https://github.com/adventists-ai/duplexjev/blob/main/examples/vllm_client.py).

`modeling_qwen3_asr_encoder.py` is a standalone port of the Qwen3-ASR audio encoder from Hugging Face `transformers`
(Apache-2.0, Copyright the Qwen team, Alibaba Cloud, and the HuggingFace team). The rest of this package is
Apache-2.0, Copyright Adventists.ai.
