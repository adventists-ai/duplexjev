"""Model code for the complete DuplexJev models (encoder + connector + LLM in one repo, e.g. DuplexJev-4B-Para).

The files are the Ultravox remote code shipped with the DuplexJev connector checkpoints and the Qwen3-ASR encoder,
vendored here so that complete models load with plain PyTorch without trust_remote_code downloads.
Ultravox is MIT-licensed (fixie-ai/ultravox); Qwen3-ASR is Apache-2.0.
"""
import transformers

from .configuration_qwen3_asr_encoder import Qwen3ASREncoderConfig
from .modeling_qwen3_asr_encoder import Qwen3ASREncoder
from .ultravox_config import UltravoxConfig
from .ultravox_model import UltravoxModel
from .ultravox_processing import UltravoxProcessor

try:
    transformers.AutoConfig.register("qwen3_asr_encoder_portable", Qwen3ASREncoderConfig)
except ValueError:  # already registered (e.g. by a connector checkpoint's remote code)
    pass
try:
    transformers.AutoModel.register(Qwen3ASREncoderConfig, Qwen3ASREncoder)
except ValueError:
    pass

__all__ = ["Qwen3ASREncoderConfig", "Qwen3ASREncoder", "UltravoxConfig", "UltravoxModel", "UltravoxProcessor"]
