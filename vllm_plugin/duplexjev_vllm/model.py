"""vLLM model for DuplexJev checkpoints: Qwen3-ASR audio encoder + Ultravox-style projector + a vLLM text model.

Everything except the audio tower and the preprocessing is vLLM's own Ultravox implementation.
"""
from functools import lru_cache

from transformers import AutoTokenizer, WhisperFeatureExtractor

import vllm.model_executor.models.ultravox as U
from vllm.multimodal import MULTIMODAL_REGISTRY

import torch
import torch.nn.functional as F

from .modeling_qwen3_asr_encoder import Qwen3ASREncoder, _get_feat_extract_output_lengths
from .processing import DuplexJevProcessor


class DuplexJevAudioTower(Qwen3ASREncoder):
    """Qwen3-ASR encoder with the (features, lens) -> hidden_states call used by vLLM's Ultravox."""

    def forward(self, input_features, audio_lens=None):
        h = super().forward(input_features, audio_len=audio_lens).last_hidden_state
        # The encoder pads its output to the longest valid clip in the call; vLLM concatenates several calls, so
        # pad every call to the length implied by the (shared) padded mel length instead.
        n = int(_get_feat_extract_output_lengths(torch.tensor(input_features.shape[-1]), self.n_window))
        if h.shape[1] < n:
            h = F.pad(h, (0, 0, 0, n - h.shape[1]))
        return h[:, :n]


def _make_tower(*, vllm_config, prefix: str = "", enable_pp: bool = False):
    return DuplexJevAudioTower(vllm_config.model_config.hf_config)


@lru_cache(maxsize=8)
def _processor(model: str, revision, trust_remote_code: bool, stack_factor: int, ds: int):
    fe = WhisperFeatureExtractor.from_pretrained(model, revision=revision)
    tok = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
    return DuplexJevProcessor(fe, tok, stack_factor=stack_factor, encoder_ds_factor=ds)


class DuplexJevProcessingInfo(U.UltravoxProcessingInfo):
    def get_hf_processor(self, **kwargs):
        mc = self.ctx.model_config
        cfg = mc.hf_config
        p = _processor(mc.model, mc.revision, mc.trust_remote_code,
                       cfg.stack_factor, int(getattr(cfg.audio_config, "encoder_ds_factor", 8)))
        p.audio_token_replacement = U._AUDIO_PLACEHOLDER_OVERRIDE
        p.audio_replacement_token_id = cfg.audio_token_index
        return p


@MULTIMODAL_REGISTRY.register_processor(U.UltravoxMultiModalProcessor, info=DuplexJevProcessingInfo,
                                        dummy_inputs=U.UltravoxDummyInputsBuilder)
class DuplexJevForConditionalGeneration(U.UltravoxModel):
    def __init__(self, *, vllm_config, prefix: str = ""):
        orig = U.UltravoxWhisperEncoder
        U.UltravoxWhisperEncoder = _make_tower
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            U.UltravoxWhisperEncoder = orig
