"""vLLM plugin for DuplexJev speech-decision models (https://github.com/adventists-ai/duplexjev).

Registered as a `vllm.general_plugins` entry point: `pip install duplexjev-vllm`, then
`vllm serve adventists-ai/DuplexJev-4B`.
"""
__version__ = "0.1.0"

ARCH = "DuplexJevForConditionalGeneration"


def register():
    from transformers import AutoConfig

    from .configuration_qwen3_asr_encoder import Qwen3ASREncoderConfig

    try:
        AutoConfig.register(Qwen3ASREncoderConfig.model_type, Qwen3ASREncoderConfig)
    except ValueError:
        pass  # already registered in this process

    from vllm import ModelRegistry

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "duplexjev_vllm.model:DuplexJevForConditionalGeneration")
