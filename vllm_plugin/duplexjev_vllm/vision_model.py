"""vLLM model for DuplexJev-32B-Vision: speech and an image in one request.

Qwen3-VL (vision encoder + language model, vLLM's own implementation) plus the DuplexJev audio path (Qwen3-ASR encoder
+ Ultravox-style projector, loaded from ``audio/`` in the checkpoint). Audio soft tokens sit on ``<|audio|>`` and take
plain text positions in Qwen3-VL's 3-D RoPE, exactly as in training.

Prompt (OpenAI chat API): put ``<|audio|>`` in the text where the speech goes and an ``image_url`` part where the picture
goes, e.g. ``[text "The user said: <|audio|>\\n\\nPicture: ", image_url, text "\\n\\n<question>", input_audio]``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchFeature, WhisperFeatureExtractor

import vllm.model_executor.models.qwen3_vl as Q
import vllm.model_executor.models.ultravox as U
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm.multimodal.inputs import MultiModalFieldConfig

from .model import DuplexJevAudioTower
from .processing import DuplexJevProcessor

AUDIO_TOKEN = "<|audio|>"
AUDIO_KEYS = ("audio_features", "audio_embeds", "audio_lens", "audio_token_len", "audio_num_chunks")
SR = 16000
CHUNK_S = 30          # Qwen3-ASR context per chunk (3000 mel frames)
MAX_AUDIO_CHUNKS = 4  # longest clip budgeted for profiling: 120 s


def _audio_file(model: str, revision, name: str) -> str:
    if os.path.isdir(model):
        return os.path.join(model, "audio", name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, f"audio/{name}", revision=revision)


@lru_cache(maxsize=4)
def _audio_cfg(model: str, revision):
    from vllm.transformers_utils.configs.ultravox import UltravoxConfig

    from . import register

    register()  # makes the Qwen3-ASR encoder config known to transformers
    c = json.load(open(_audio_file(model, revision, "config.json")))
    c.pop("architectures", None)
    c.update(text_model_id=None, audio_model_id=None)
    return UltravoxConfig(**c)


@lru_cache(maxsize=4)
def _audio_processor(model: str, revision, stack_factor: int):
    fe = WhisperFeatureExtractor.from_pretrained(os.path.dirname(_audio_file(model, revision, "preprocessor_config.json")))
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    return DuplexJevProcessor(fe, tok, stack_factor=stack_factor, encoder_ds_factor=8)


def _tokens_per_chunk(stack_factor: int) -> int:
    return -(-CHUNK_S * 100 // (8 * stack_factor))


# --------------------------------------------------------------------------- processing
class DuplexJevVisionProcessingInfo(Q.Qwen3VLProcessingInfo):
    def _mc(self):
        return self.ctx.model_config

    def get_audio_config(self):
        return _audio_cfg(self._mc().model, self._mc().revision)

    def get_audio_processor(self) -> DuplexJevProcessor:
        return _audio_processor(self._mc().model, self._mc().revision, int(self.get_audio_config().stack_factor))

    def get_audio_token_id(self) -> int:
        return int(self.get_tokenizer().convert_tokens_to_ids(AUDIO_TOKEN))

    def get_data_parser(self):
        return Q.Qwen3VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
            target_sr=SR,
            target_channels=1,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None, "audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        d = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        d["audio"] = MAX_AUDIO_CHUNKS * _tokens_per_chunk(int(self.get_audio_config().stack_factor))
        return d


class DuplexJevVisionDummyInputsBuilder(Q.Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_TOKEN * mm_counts.get("audio", 0) + super().get_dummy_text(mm_counts)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        vis = {k: v for k, v in mm_counts.items() if k != "audio"}
        d = dict(super().get_dummy_mm_data(seq_len, vis, mm_options))
        n = mm_counts.get("audio", 0)
        if n:
            d["audio"] = self._get_dummy_audios(length=SR * CHUNK_S * MAX_AUDIO_CHUNKS, num_audios=n,
                                                overrides=mm_options.get("audio"))
        return d


class DuplexJevVisionMultiModalProcessor(Q.Qwen3VLMultiModalProcessor):
    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs) -> BatchFeature:
        counts = mm_items.get_all_counts()
        vis_keys = {k for k, c in counts.items() if k != "audio" and c > 0}
        out = (super()._apply_hf_processor_main(mm_items.select(vis_keys), hf_processor_mm_kwargs)
               if vis_keys else BatchFeature({}))
        if counts.get("audio", 0) > 0:
            proc_data, passthrough = self._get_hf_mm_data(mm_items.select({"audio"}))
            audios = proc_data.get("audios") or proc_data.get("audio") or []
            if not isinstance(audios, (list, tuple)):
                audios = [audios]
            if audios:
                x = self.info.get_audio_processor()(audio=list(audios), sampling_rate=SR,
                                                    include_audio_num_chunks=True, return_tensors="pt")
                x["audio_features"] = x.pop("audio_values")
                x.pop("audio_batch_size", None)
                out.update(dict(x))
            out.update(passthrough)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        d = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        nch = hf_inputs.get("audio_num_chunks", torch.zeros(0))
        d.update(
            audio_features=MultiModalFieldConfig.flat_from_sizes("audio", nch),
            audio_token_len=MultiModalFieldConfig.flat_from_sizes("audio", nch),
            audio_lens=MultiModalFieldConfig.flat_from_sizes("audio", nch, keep_on_cpu=True),
            audio_num_chunks=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )
        return d

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[Any]:
        ups = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        aid = self.info.get_audio_token_id()

        def repl(i: int):
            n = int(out_mm_kwargs["audio"][i]["audio_token_len"].data.sum())
            return PromptUpdateDetails.select_token_id([aid] * n, aid)

        ups.append(PromptReplacement(modality="audio", target=[aid], replacement=repl))
        return ups


# --------------------------------------------------------------------------- model
@MULTIMODAL_REGISTRY.register_processor(DuplexJevVisionMultiModalProcessor, info=DuplexJevVisionProcessingInfo,
                                        dummy_inputs=DuplexJevVisionDummyInputsBuilder)
class DuplexJevVisionForConditionalGeneration(Q.Qwen3VLForConditionalGeneration):
    # audio path borrowed unchanged from vLLM's Ultravox
    _audio_features_to_embeddings = U.UltravoxModel._audio_features_to_embeddings
    _parse_and_validate_audio_input = U.UltravoxModel._parse_and_validate_audio_input
    _process_audio_input = U.UltravoxModel._process_audio_input

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_TOKEN
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        mc = vllm_config.model_config
        ucfg = _audio_cfg(mc.model, mc.revision)
        self.pad_audio_to_max_context = False
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = DuplexJevAudioTower(ucfg.audio_config)
            self.multi_modal_projector = U.UltravoxFeedForwardProjector(
                ucfg, quant_config=None, prefix=maybe_prefix(prefix, "multi_modal_projector"))

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # audio soft tokens are ordinary text positions (as in training)
        return super().get_mrope_input_positions(input_tokens, [f for f in mm_features if f.modality != "audio"])

    def embed_multimodal(self, **kwargs: object):
        ak = {k: kwargs.pop(k) for k in AUDIO_KEYS if k in kwargs}
        out = list(super().embed_multimodal(**kwargs) or ()) if kwargs else []
        a_in = self._parse_and_validate_audio_input(**ak) if ak else None
        if a_in is not None:
            embs = self._process_audio_input(a_in)
            if self.use_deepstack:  # no deep-stack features for audio tokens
                embs = [F.pad(e, (0, self.multiscale_dim)) for e in embs]
            out.extend(embs)
        return tuple(out)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from safetensors.torch import load_file

        loaded = set(super().load_weights(weights))
        params = dict(self.named_parameters())
        sd = load_file(_audio_file(self.model_config.model, self.model_config.revision, "model.safetensors"))
        for k, v in sd.items():
            if k not in params:
                continue
            p = params[k]
            getattr(p, "weight_loader", default_weight_loader)(p, v)
            loaded.add(k)
        missing = [k for k in params if k.startswith(("audio_tower.", "multi_modal_projector.")) and k not in loaded]
        if missing:
            raise ValueError(f"audio weights missing from audio/model.safetensors: {missing[:5]}")
        return loaded
