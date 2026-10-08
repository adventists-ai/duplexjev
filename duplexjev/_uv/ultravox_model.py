import contextlib
import logging
import os
import re
from typing import Any, Dict, Generator, Optional, Set, Tuple, TypeVar, Union

import accelerate
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models
from transformers.generation.utils import GenerationMixin
from transformers.models.whisper import modeling_whisper as whisper

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_config import LossConfig
from .ultravox_config import LossFunction
from .ultravox_config import UltravoxConfig

FROM_PRETRAINED_KWARGS = {}
SHARED_PRETRAINED_KWARGS = [
    "tp_plan",
    "device_map",
    "torch_dtype",
    "attn_implementation",
    "use_flash_attention_2",
]


def _is_diffusion_gemma(config: UltravoxConfig) -> bool:
    return getattr(config.text_config, "model_type", None) == "diffusion_gemma"


def _init_weights_enabled() -> bool:
    # transformers < 5 exposed a module-level bool, toggled by the internal
    # `no_init_weights()` context manager during `PreTrainedModel.from_pretrained()`,
    # that ultravox reused here to detect "are we being constructed as a nested
    # skeleton for an outer from_pretrained() (e.g. loading a saved, already-merged
    # ultravox checkpoint, where audio/text_model_id are None) vs a normal fresh
    # construction with real model ids". transformers >= 5 removed the flag (see
    # transformers/initialization.py's `no_init_weights`); there is no longer a
    # public way to query it.
    #
    # Correction (found while loading a saved dgemma-a1 checkpoint for inference,
    # not just training-time fresh construction as originally assumed here):
    # `UltravoxModel.from_pretrained(<saved checkpoint>)` IS a real, used path --
    # our checkpoints only save the trained diff (projector + fusion.*, via
    # diff_state_dict()), not the frozen base weights, so loading one for
    # inference still needs this branch to fetch the real audio/text base
    # weights fresh, same as a training-time construction. Default True is
    # still the right answer in both cases here; see `_real_device_context()`
    # below for the actual new problem this surfaced (a meta-device clash, not
    # a "should we skip loading" question).
    return getattr(transformers.modeling_utils, "_init_weights", True)


def _copy_meta_buffers(target: nn.Module, source: nn.Module):
    """`nn.Module.state_dict()` (and therefore `load_state_dict(..., assign=True)`)
    only includes *persistent* buffers by design -- `persistent=False` buffers are
    deliberately excluded, e.g. because they're cheap to recompute and shouldn't be
    checkpointed. DiffusionGemma registers several this way (RoPE `inv_freq`/
    `original_inv_freq` per layer type, `embed_scale` for its scaled word
    embeddings -- see modeling_diffusion_gemma.py), so `_repair_meta_base_weights`'s
    `load_state_dict()` call never touches them.

    Despite the name, this does NOT only fix *meta* buffers: `embed_scale`
    specifically came back as a real (non-meta) bf16 CUDA tensor after the nested
    from_pretrained() in __init__, but with garbage content (~2.9e-39, nowhere near
    the real sqrt(hidden_size)~53), silently collapsing every scaled embedding
    lookup toward zero (std=0.0) and poisoning the whole encoder forward with NaN --
    worse than an outright meta tensor, since `device.type == "meta"` detection
    (used elsewhere in this repair, e.g. the `any(p.device.type == "meta" ...)`
    guards) doesn't catch it at all. So: copy *every* buffer target and source share
    unconditionally, regardless of the target's current device/value -- re-copying
    an already-correct buffer is harmless, and this is the only reliable way to
    catch both the meta case (RoPE inv_freq) and this real-but-garbage case
    (embed_scale). Same underlying cause as the ASR encoder's non-persistent
    sinusoidal position buffer (modeling_qwen3_asr_encoder.py's
    `SinusoidsPositionEmbedding.get_embedding`) -- transformers >= 5's fast-init
    doesn't reliably materialize `persistent=False` buffers at all, not just
    "leaves them meta".
    """
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    n_fixed = 0
    for name in target_buffers:
        if name not in source_buffers:
            continue
        parts = name.split(".")
        mod = target
        for p in parts[:-1]:
            mod = getattr(mod, p)
        setattr(mod, parts[-1], source_buffers[name].to(target_buffers[name].dtype))
        n_fixed += 1
    logging.warning(
        f"_copy_meta_buffers({type(target).__name__}): copied {n_fixed} of "
        f"{len(target_buffers)} target buffers from a fresh reference load"
    )


def _real_device_context():
    """transformers >= 5's `from_pretrained()` constructs the model under an
    ambient `torch.device("meta")` context for fast-init (verified: not the
    case in 4.51.3), then materializes real tensors from the checkpoint's
    state_dict afterward. That's fine for parameters transformers itself
    fills in -- but `_create_audio_tower`/`_create_language_model` make their
    *own* nested `from_pretrained()` calls to fetch the frozen base weights
    (not present in our checkpoints at all, see `_init_weights_enabled()`),
    and transformers 5.x hard-errors if a `from_pretrained()` call happens
    while a meta-device context is ambient ("anti-pattern", checked via
    `get_torch_context_manager_or_global_device()` in
    transformers/integrations/accelerate.py). Escape to a real device for
    just these nested calls; no-op when there's no ambient meta context
    (e.g. plain training-time construction, or transformers 4.51.3).
    """
    if torch.empty(0).device.type == "meta":
        return torch.device("cpu")
    return contextlib.nullcontext()


class UltravoxModel(transformers.LlamaPreTrainedModel, GenerationMixin):
    """
    The Ultravox model which consists of an audio encoder and a language model.

    Audio input is processed by the audio encoder, then every `stack_factor` frames are stacked together and
    projected to the language model's embedding space using a few linear layers.
    The text is embedded by the language model as usual and then the audio and text embeddings are merged together.

    A special token `<|audio|>` is used to indicate the start of the audio embeddings in the merged embeddings.

    Parameters:
        config: Model configuration class with all the parameters of the model.
    """

    config_class = UltravoxConfig
    config: UltravoxConfig  # for type hinting
    # Usually we load encoder and LLM weights from a pretrained model separately, so they are allowed to be missing
    _keys_to_ignore_on_load_missing = ["audio_tower.*", "language_model.*"]
    # Since we have kwargs in forward, we need to set this to False, otherwise grad_accum_steps will cause incorrect train loss to be reported
    # see https://github.com/huggingface/transformers/issues/35856 and https://github.com/huggingface/trl/pull/2615/files
    accepts_loss_kwargs = False

    def __init__(self, config: UltravoxConfig):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        if not config.llm_only_training:
            self.audio_tower = self._create_audio_tower(config)
            self.multi_modal_projector = self._create_multi_modal_projector(config)
            self.audio_tower_context_length = self.audio_tower.max_context_length

        self.language_model = self._create_language_model(config)

        # DiffusionGemma is an encoder-decoder block-diffusion model, not a causal LM:
        # it has no `labels`/loss convention of its own (transformers ships the architecture
        # + inference sampler only), and its tokenizer's mask token id / canvas length aren't
        # on the model config, so we resolve and cache them once here for forward()'s custom
        # training-loss path (see _forward_diffusion_gemma).
        self.is_diffusion_gemma = _is_diffusion_gemma(config)
        if self.is_diffusion_gemma and config.text_model_id is not None:
            _dgemma_tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.text_model_id, trust_remote_code=True
            )
            self.diffusion_mask_token_id = _dgemma_tokenizer.mask_token_id
            self.diffusion_pad_token_id = self.language_model.config.text_config.pad_token_id
            self.diffusion_canvas_length = self.language_model.config.canvas_length

        if self.language_model._tied_weights_keys is not None:
            # transformers 4.51.3 uses a list of key strings here (implicit tie to
            # the input embedding); transformers 5.x (see DiffusionGemmaForBlockDiffusion,
            # which sets `{"lm_head.weight": "model.decoder.embed_tokens.weight"}`)
            # uses an explicit dict of {target: source}. `_get_tied_weight_keys`
            # in the new transformers calls `.keys()` on this, so a bare
            # re-prefixed list (the old code here) crashes on save_pretrained.
            _sub_tied = self.language_model._tied_weights_keys
            if isinstance(_sub_tied, dict):
                self._tied_weights_keys = {
                    f"language_model.{k}": f"language_model.{v}"
                    for k, v in _sub_tied.items()
                }
            else:
                self._tied_weights_keys = [
                    f"language_model.{k}" for k in _sub_tied
                ]

        # Determine no_split_modules dynamically to use with FSDP auto_wrap policy.
        # This would be something like ["LlamaDecoderLayer"] as we don't split audio encoder layers.
        # FSDP throws an error if some of the layer types are not found in the model, and they need to be filted out.
        # 1. Get the names the language model *wants* to keep intact
        candidate_names = set(
            getattr(self.language_model, "_no_split_modules", []) or []
        )
        # 2. Names that actually exist in the current model
        present_names = {m.__class__.__name__ for m in self.modules()}
        # 3. Keep only those that are both requested and present
        self._no_split_modules = list(candidate_names & present_names)

        self.loss_config = LossConfig()
        self.post_init()

    def _init_weights(self, module):
        if module is self:
            if self.config.text_model_id is not None:
                self.language_model = self._create_language_model(self.config)
            if self.config.audio_model_id is not None:
                self.audio_tower = self._create_audio_tower(self.config)
        elif module in self.language_model.modules():
            pass
        elif module in self.audio_tower.modules():
            pass
        else:
            super()._init_weights(module)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        global FROM_PRETRAINED_KWARGS
        FROM_PRETRAINED_KWARGS = {
            k: v for k, v in kwargs.items() if k in SHARED_PRETRAINED_KWARGS
        }
        model = super().from_pretrained(*args, **kwargs)
        FROM_PRETRAINED_KWARGS = {}
        model._repair_meta_base_weights()
        return model

    def _repair_meta_base_weights(self):
        """transformers >= 5's from_pretrained() builds this model's __init__
        under a meta-device context for fast-init, then materializes real
        tensors only for keys present in *this checkpoint's* state_dict. Our
        checkpoints only ever save the trained diff (multi_modal_projector.* +
        audio_tower.fusion.*, see diff_state_dict()) -- never the frozen
        audio/text base weights, which are meant to come from
        config.audio_model_id/text_model_id via the *nested* from_pretrained
        calls in _create_audio_tower/_create_language_model instead. Under
        transformers 4.51.3 that nested call fetched them for real; under 5.x
        it silently comes back with every key "missing" (randomly/meta
        initialized) instead of erroring -- verified against the 25B
        DiffusionGemma base, and neither escaping the ambient meta context
        (`_real_device_context()`) nor `low_cpu_mem_usage=False` on the nested
        call fixed it. Repair here instead: reload each frozen base for real,
        then `assign=True` (a parameter swap, not an in-place `copy_()` that
        would fail on a meta tensor: "aten::equal: ... Meta tensors") to
        materialize. No-op when nothing came back meta (transformers 4.51.3,
        or a fresh `from_config()` build).
        """
        if not any(p.device.type == "meta" for p in self.language_model.parameters()):
            return

        logging.warning(
            "UltravoxModel.from_pretrained: language_model/audio_tower came back "
            "meta-initialized from the nested from_pretrained() call in __init__ "
            "(transformers >= 5 fast-init quirk); reloading their frozen base "
            "weights for real."
        )

        if self.config.text_model_id is not None:
            base_lm = type(self.language_model).from_pretrained(
                self.config.text_model_id,
                trust_remote_code=True,
                torch_dtype=self.config.torch_dtype,
            )
            self.language_model.load_state_dict(base_lm.state_dict(), assign=True)
            _copy_meta_buffers(self.language_model, base_lm)
            del base_lm

        if self.config.audio_model_id is not None and any(
            p.device.type == "meta" for p in self.audio_tower.parameters()
        ):
            base_audio = transformers.AutoModel.from_pretrained(
                self.config.audio_model_id,
                trust_remote_code=True,
                torch_dtype=self.config.torch_dtype,
            )
            unfreeze = list(
                (self.config.audio_model_lora_config or {}).get("unfreeze_layers")
                or []
            )
            trained_keys = {
                name
                for name, _ in base_audio.named_parameters()
                if any(re.match(p, name) for p in unfreeze)
            }
            base_sd = {
                k: v
                for k, v in base_audio.state_dict().items()
                if k not in trained_keys
            }
            self.audio_tower.load_state_dict(base_sd, strict=False, assign=True)
            _copy_meta_buffers(self.audio_tower, base_audio)
            del base_audio

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self, *args, **kwargs):
        # transformers >= 5 calls `self.tie_weights(recompute_mapping=False)` from
        # `init_weights()`; transformers 4.51.3 (the Qwen3-32B venv) calls it with no
        # args. Forward whatever we got -- `self.language_model` is itself a
        # `PreTrainedModel` on both versions, so its own `tie_weights()` accepts the
        # same signature transformers just called us with.
        try:
            return self.language_model.tie_weights(*args, **kwargs)
        except NotImplementedError as e:
            # transformers >= 5's tie_weights() now *compares* the source/target
            # tensors' values (torch.equal(...)) before tying, which fails on meta
            # tensors ("aten::equal: ... Meta tensors"). This fires mid-way through
            # `UltravoxModel.from_pretrained()`'s own internal finalization -- our
            # checkpoints never save language_model's frozen base weights (see
            # `_repair_meta_base_weights`), so at this exact point they're still
            # meta and there's nothing meaningful to compare yet. Skip tying here;
            # `_repair_meta_base_weights()` fully replaces language_model's
            # parameters right after `from_pretrained()` returns, at which point
            # they're real tensors from a source that was already correctly tied.
            if "Meta tensors" not in str(e):
                raise
            logging.warning(
                "UltravoxModel.tie_weights: skipping (language_model still "
                "meta-initialized at this point, see _repair_meta_base_weights)"
            )

    def set_loss_config(self, loss_config: LossConfig):
        self.loss_config = loss_config

    def _setup_cache(
        self, cache_cls, max_batch_size: int, max_cache_len: Optional[int] = None
    ):
        self.language_model._setup_cache(cache_cls, max_batch_size, max_cache_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _get_prediction_mask(
        self, labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get boolean masks for positions where we want to compute KL divergence.

        For each label position, we want the position before it since that's where
        the model makes the prediction for that label.

        Additionally, we want to identify the position right before the EOT token
        (the last token with label != -100).

        Args:
            labels: Tensor of shape (B, T) where B is batch size and T is sequence length,
                   with -100 for masked positions and token ids for label positions

        Returns:
            Tuple containing:
            - pred_mask: Boolean tensor of shape (B, T) that's True for positions where we want to compute KL divergence
            - eot_mask: Boolean tensor of shape (B, T) that's True only for the last prediction position in each sequence
        """
        if labels is None:
            raise ValueError("labels must be provided")

        # Shift the label mask right by 1 along the sequence dimension
        # This gives us positions where we make predictions for the next token
        label_mask = labels != -100
        pred_mask = torch.zeros_like(label_mask)
        pred_mask[:, :-1] = label_mask[
            :, 1:
        ]  # shift right by 1 along sequence dimension

        # Create EOT mask - identify only the last prediction position in each sequence
        eot_mask = torch.zeros_like(pred_mask)
        batch_size = labels.shape[0]

        for i in range(batch_size):
            # Find positions where we make predictions
            pred_positions = torch.where(pred_mask[i])[0]
            if len(pred_positions) > 0:
                # Only mark the last prediction position
                eot_mask[i, pred_positions[-1]] = True

        return pred_mask, eot_mask

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        _ce_loss = None
        if __import__("os").environ.get("ML_ASR_MIXED_LOSS") == "1" and alt_labels is not None:
            _mcq = (labels != -100).any(-1) & ~(alt_labels != -100).any(-1)
            if _mcq.any():
                _lg = lm_output.logits[_mcq][:, :-1].float()
                _lb = labels[_mcq][:, 1:]
                _ce_loss = F.cross_entropy(_lg.reshape(-1, _lg.size(-1)), _lb.reshape(-1), ignore_index=-100)
            labels = labels.clone()
            labels[_mcq] = -100
            if not (labels != -100).any():
                return _ce_loss
        # disable gradient computation for the teacher model
        with torch.no_grad():
            # compute the teacher (text-only) model's distribution
            alt_inputs_embeds = self.get_input_embeddings().forward(alt_input_ids)
            alt_lm_output = self.language_model.forward(
                inputs_embeds=alt_inputs_embeds,
                labels=alt_labels,
                attention_mask=alt_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # Get prediction masks for regular tokens and EOT tokens
        pred_mask, eot_mask = self._get_prediction_mask(labels)
        alt_pred_mask, alt_eot_mask = self._get_prediction_mask(alt_labels)

        # compute the KL divergence loss between the two models for regular tokens
        kl_loss = F.kl_div(
            F.log_softmax(
                lm_output.logits[pred_mask] / self.loss_config.kl_temperature,
                dim=-1,
            ),
            F.softmax(
                alt_lm_output.logits[alt_pred_mask] / self.loss_config.kl_temperature,
                dim=-1,
            ),
            reduction="batchmean",
        )

        # Compute the KL divergence loss for EOT token positions if any exist
        if self.loss_config.eot_loss_weight > 0:
            eot_loss = F.kl_div(
                F.log_softmax(
                    lm_output.logits[eot_mask] / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                F.softmax(
                    alt_lm_output.logits[alt_eot_mask]
                    / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                reduction="batchmean",
            )
            kl_loss += self.loss_config.eot_loss_weight * eot_loss

        if _ce_loss is not None:
            kl_loss = kl_loss + _ce_loss
        return kl_loss

    def _audio_iter(
        self, audio_batch_size: torch.Tensor
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Iterate over the audio batch size and yield the batch index and audio index of each audio item.

        Args:
            audio_batch_size: A tensor of shape (B,) where B is the batch size.

        Returns:
            A generator that yields a tuple of (start index, length) for each audio item.
        """
        audio_index = 0
        for i_b, batch_count in enumerate(audio_batch_size):
            for _ in range(batch_count):
                yield i_b, audio_index
                audio_index += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        # the alt_* fields are needed for KL divergence loss
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        """
        Forward pass for the Ultravox model.

        `input_ids` are the tokenized text input. They are embedded by the language model as usual.
        `audio_values` are processed by the audio encoder and then every `stack_factor` frames are stacked together and
        projected to the language model's embedding space using a few linear layers.
        The audio and text embeddings are merged together. A special token `<|audio|>` is used to indicate the start
        of the audio embeddings in the merged embeddings.

        Args:
            input_ids: The tokenized text input.
            audio_values: The processed audio values.
            inputs_embeds: The embeddings for the input tokens.
            labels: The tokenized text labels.
            attention_mask: The attention mask for the input.
            position_ids: The position ids for the input.
            past_key_values: The past key value cache for the language model attention layers.
            **kwargs: Additional keyword arguments. Passed directly to the language model.
        """
        if self.is_diffusion_gemma:
            return self._forward_diffusion_gemma(
                input_ids=input_ids,
                audio_values=audio_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
                audio_token_start_idx=audio_token_start_idx,
                audio_lens=audio_lens,
                audio_token_len=audio_token_len,
                audio_batch_size=audio_batch_size,
            )

        if inputs_embeds is None:
            # B x T  ->  B x T x D
            inputs_embeds = self.get_input_embeddings().forward(input_ids)

        if audio_values is not None and len(audio_values) > 0:
            inputs_embeds = self._prepare_audio_embeds(
                inputs_embeds=inputs_embeds,
                audio_values=audio_values,
                audio_token_start_idx=audio_token_start_idx,
                audio_lens=audio_lens,
                audio_token_len=audio_token_len,
                audio_batch_size=audio_batch_size,
            )

        lm_output = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.training:
            if self.loss_config.loss_function == LossFunction.CrossEntropy:
                pass
            elif self.loss_config.loss_function == LossFunction.KL_Divergence:
                lm_output.loss = self._compute_kl_loss(
                    lm_output=lm_output,
                    labels=labels,
                    past_key_values=past_key_values,
                    alt_input_ids=alt_input_ids,
                    alt_attention_mask=alt_attention_mask,
                    alt_labels=alt_labels,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported loss function: {self.loss_config.loss_function}"
                )
        return lm_output

    def _prepare_audio_embeds(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            inputs_embeds is not None
            and audio_values is not None
            and audio_token_start_idx is not None
            and audio_token_len is not None
            and audio_lens is not None
            and audio_batch_size is not None
        ), "inputs_embeds/audio_values/audio_token_start_idx/audio_token_len/audio_lens/audio_batch_size must be provided."
        assert (
            len(audio_token_start_idx)
            == len(audio_token_len)
            == len(audio_lens)
            == len(audio_values)
        ), "audio_token_start_idx/audio_token_len/audio_lens/audio_values must have the same batch size."
        assert len(audio_batch_size) == len(
            inputs_embeds
        ), "audio_batch_size and inputs_embeds must have the same batch size."

        if os.environ.get("DGEMMA_DEBUG"):
            print(
                "[DGEMMA_DEBUG audio_in]",
                "audio_values shape=", tuple(audio_values.shape),
                "isnan=", torch.isnan(audio_values).any().item(),
                "isinf=", torch.isinf(audio_values).any().item(),
                "min/max=", audio_values.min().item(), audio_values.max().item(),
                "audio_lens=", audio_lens.tolist(),
                "audio_token_start_idx=", audio_token_start_idx.tolist(),
                "audio_token_len=", audio_token_len.tolist(),
                "audio_batch_size=", audio_batch_size.tolist(),
                flush=True,
            )

        # B x A/3200 x (D=max-audio-length-in-batch)
        audio_tower_output = self.audio_tower.forward(
            audio_values.to(self.audio_tower.dtype),
            audio_len=audio_lens,
        ).last_hidden_state
        audio_tower_output = audio_tower_output.to(inputs_embeds.dtype)
        audio_embeds = self.multi_modal_projector.forward(audio_tower_output)

        if os.environ.get("DGEMMA_DEBUG"):
            print(
                "[DGEMMA_DEBUG audio]",
                "audio_tower_output isnan=", torch.isnan(audio_tower_output).any().item(),
                "min/max=", audio_tower_output.min().item(), audio_tower_output.max().item(),
                "audio_embeds isnan=", torch.isnan(audio_embeds).any().item(),
                "min/max=", audio_embeds.min().item(), audio_embeds.max().item(),
                "abs_mean=", audio_embeds.abs().mean().item(),
                flush=True,
            )

        _n = getattr(self, "_len_check_n", 0)
        if _n < 3:
            self._len_check_n = _n + 1
            _ratio = float(audio_token_len.max()) / max(1, audio_embeds.shape[1])
            print(f"[stack-check-data] max audio_token_len={int(audio_token_len.max())} "
                  f"projector_frames={audio_embeds.shape[1]} ratio={_ratio:.2f} (expect ~1.0)", flush=True)

        # combine audio and text embeddings
        for i_b, i_a in self._audio_iter(audio_batch_size):
            start_idx = audio_token_start_idx[i_a]
            token_len = audio_token_len[i_a]
            item_embedding = audio_embeds[i_a][:token_len]
            inputs_embeds[i_b][start_idx : start_idx + token_len] = item_embedding

        return inputs_embeds

    def _forward_diffusion_gemma(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        """
        DiffusionGemma (`DiffusionGemmaForBlockDiffusion`) is an encoder-decoder block-diffusion
        model, not a causal LM: the encoder autoregressively reads the prompt into a KV cache,
        and a separate bidirectional decoder denoises a fixed-length `canvas` of answer tokens
        against that cache. `transformers` ships only the architecture and the *inference*
        entropy-bounded sampler (see `generation_diffusion_gemma.py`) -- there is no packaged
        training loss, since Google hasn't published one. What follows is a standard
        absorbing-state / masked discrete-diffusion training objective (Austin et al. 2021
        D3PM; Sahoo et al. 2024 MDLM; Nie et al. 2025 LLaDA use the same 1/t-weighted masked-CE
        ELBO estimator) -- a reasonable and well-documented choice, but not a claim that it
        matches Google's actual (unpublished) recipe.

        The upstream `UltravoxModel.forward()` (Qwen-32B path) concatenates prompt+assistant
        into one causal sequence and lets the LM's own shifted-label loss find the boundary.
        Here the prompt (encoder, KV cache) and the assistant answer (decoder canvas) must be
        physically separate tensors, so we re-derive that boundary from `labels` (first non
        `ignore_index` position, i.e. where the collator's loss mask turns on) instead of
        needing a second, diffusion-specific data pipeline.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings().forward(input_ids)

        if audio_values is not None and len(audio_values) > 0:
            inputs_embeds = self._prepare_audio_embeds(
                inputs_embeds=inputs_embeds,
                audio_values=audio_values,
                audio_token_start_idx=audio_token_start_idx,
                audio_lens=audio_lens,
                audio_token_len=audio_token_len,
                audio_batch_size=audio_batch_size,
            )

        ignore_index = self.config.ignore_index
        batch_size, _seq_len = input_ids.shape
        device = input_ids.device
        canvas_length = self.diffusion_canvas_length
        mask_token_id = self.diffusion_mask_token_id
        pad_token_id = self.diffusion_pad_token_id

        if os.environ.get("DGEMMA_DEBUG"):
            print(
                "[DGEMMA_DEBUG embeds]",
                "inputs_embeds isnan=", torch.isnan(inputs_embeds).any().item(),
                "isinf=", torch.isinf(inputs_embeds).any().item(),
                "min=", inputs_embeds.min().item(),
                "max=", inputs_embeds.max().item(),
                "abs_mean=", inputs_embeds.abs().mean().item(),
                flush=True,
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        real_len = attention_mask.sum(dim=1)

        if labels is not None:
            is_target = labels != ignore_index
            has_target = is_target.any(dim=1)
            # argmax on an all-False row returns 0, which would put the whole sequence in the
            # canvas; fall back to real_len (empty canvas) for rows with no assistant span.
            prompt_len = torch.where(has_target, is_target.float().argmax(dim=1), real_len)
        else:
            prompt_len = real_len

        max_prompt_len = max(int(prompt_len.max().item()), 1)
        prompt_embeds = torch.zeros(
            batch_size,
            max_prompt_len,
            inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype,
            device=device,
        )
        prompt_attention_mask = torch.zeros(
            batch_size, max_prompt_len, dtype=torch.long, device=device
        )
        clean_canvas = torch.full(
            (batch_size, canvas_length), pad_token_id, dtype=torch.long, device=device
        )
        canvas_valid = torch.zeros(batch_size, canvas_length, dtype=torch.bool, device=device)

        for i in range(batch_size):
            p_len = int(prompt_len[i].item())
            r_len = int(real_len[i].item())
            prompt_embeds[i, :p_len] = inputs_embeds[i, :p_len]
            prompt_attention_mask[i, :p_len] = 1
            tgt_len = min(max(r_len - p_len, 0), canvas_length)
            if tgt_len > 0:
                clean_canvas[i, :tgt_len] = input_ids[i, p_len : p_len + tgt_len]
                canvas_valid[i, :tgt_len] = True

        encoder_outputs = self.language_model.model.encoder(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_attention_mask,
        )

        decoder_attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(batch_size, canvas_length, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # Sample a per-example timestep t ~ U[eps, 1] (masking probability) and corrupt only
        # the real (non-pad) canvas positions with the absorbing <mask> token.
        eps = 0.03
        t = torch.rand(batch_size, device=device) * (1 - eps) + eps
        rand = torch.rand(batch_size, canvas_length, device=device)
        corrupt_mask = canvas_valid & (rand < t.unsqueeze(1))
        decoder_input_ids = torch.where(corrupt_mask, mask_token_id, clean_canvas)

        if os.environ.get("DGEMMA_DEBUG"):
            print(
                "[DGEMMA_DEBUG]",
                "real_len=", real_len.tolist(),
                "prompt_len=", prompt_len.tolist(),
                "canvas_valid.sum(dim=1)=", canvas_valid.sum(dim=1).tolist(),
                "corrupt_mask.sum(dim=1)=", corrupt_mask.sum(dim=1).tolist(),
                "t=", t.tolist(),
                flush=True,
            )

        decoder_outputs = self.language_model.model.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=encoder_outputs.past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )

        logits = self.language_model.lm_head(decoder_outputs.last_hidden_state)
        logits = logits.to(torch.float32)
        softcap = self.language_model.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if labels is not None:
            # Unbiased ELBO estimator for absorbing-state diffusion: weight each masked
            # position's CE by 1/t, average over masked positions, then over the batch.
            per_token_ce = F.cross_entropy(
                logits.transpose(1, 2), clean_canvas, reduction="none"
            )
            weight = corrupt_mask.float() / t.unsqueeze(1)
            denom = corrupt_mask.float().sum(dim=1).clamp(min=1.0)
            per_example_loss = (per_token_ce * weight).sum(dim=1) / denom
            loss = per_example_loss.mean()

            if os.environ.get("DGEMMA_DEBUG"):
                print(
                    "[DGEMMA_DEBUG loss]",
                    "logits.isnan.any=", torch.isnan(logits).any().item(),
                    "logits.isinf.any=", torch.isinf(logits).any().item(),
                    "per_token_ce isnan/isinf=", torch.isnan(per_token_ce).any().item(),
                    torch.isinf(per_token_ce).any().item(),
                    "per_example_loss=", per_example_loss.tolist(),
                    "loss=", loss.item(),
                    flush=True,
                )

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=encoder_outputs.past_key_values,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings().forward(input_ids)

        if audio_values is not None and len(audio_values) > 0:
            inputs_embeds = self._prepare_audio_embeds(
                inputs_embeds=inputs_embeds,
                audio_values=audio_values,
                audio_token_start_idx=audio_token_start_idx,
                audio_lens=audio_lens,
                audio_token_len=audio_token_len,
                audio_batch_size=audio_batch_size,
            )

        if self.is_diffusion_gemma:
            # Eval-time only: `input_ids` here is prompt-only (no assistant answer
            # appended yet, since that's what we're generating), so there's no
            # boundary to re-derive like in _forward_diffusion_gemma -- inputs_embeds
            # is already the full prompt.
            #
            # DiffusionGemmaGenerationMixin.generate() doesn't actually support
            # inputs_embeds for the prompt (verified: it does `batch_size, cur_len =
            # input_ids.shape` unconditionally near the top, and never once reads
            # inputs_embeds for the encoder side -- only its own multimodal path of
            # input_ids + pixel_values, which the encoder merges internally). So we
            # do that merge step ourselves (reusing the same encoder call
            # _forward_diffusion_gemma uses) to get a real prompt KV cache, then hand
            # generate() an empty (batch, 0) input_ids placeholder + that cache --
            # matching its own documented past_key_values contract ("input_ids...
            # must correspond to uncached data only").
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
                )
            encoder_outputs = self.language_model.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            empty_input_ids = torch.zeros(
                inputs_embeds.shape[0], 0, dtype=torch.long, device=inputs_embeds.device
            )
            return self.language_model.generate(
                input_ids=empty_input_ids,
                past_key_values=encoder_outputs.past_key_values,
                **kwargs,
            )

        return self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @classmethod
    def _create_multi_modal_projector(
        cls, config: UltravoxConfig
    ) -> "UltravoxProjector":
        projector = UltravoxProjector(config)
        dtype = config.torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        projector.to(dtype)
        return projector

    @classmethod
    def _create_audio_tower(
        cls, config: UltravoxConfig
    ) -> Union[transformers.Wav2Vec2Model, "ModifiedWhisperEncoder"]:
        # We probably don't want to pass tp_plan or device_map to the audio tower
        # But potentially other kwargs can be passed in. TODO
        kwargs = {"torch_dtype": config.torch_dtype}
        if _init_weights_enabled() and config.audio_model_id is not None:
            # low_cpu_mem_usage=False: transformers >= 5 defaults to fast-init
            # (build skeleton on meta device, materialize from the checkpoint
            # after). `_real_device_context()` above only stops the *outer*
            # anti-pattern crash; without this flag the nested from_pretrained
            # call still meta-inits its own skeleton internally and every key
            # silently comes back "missing" (verified: the base model's own
            # LOAD REPORT showed the entire 25B DiffusionGemma as missing --
            # not an error, just silently random-initialized). This forces the
            # old eager real-tensor-from-the-start path for these nested loads.
            with _real_device_context():
                if "whisper" in config.audio_model_id.lower():
                    audio_tower = ModifiedWhisperEncoder.from_pretrained(
                        config.audio_model_id, low_cpu_mem_usage=False, **kwargs
                    )
                    audio_tower.init_latency_mask(
                        config.audio_latency_block_size, dtype=config.torch_dtype
                    )
                    audio_tower.init_latency_mask(
                        config.audio_latency_block_size, dtype=config.torch_dtype
                    )
                else:
                    assert config.audio_latency_block_size in (
                        None,
                        0,
                    ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                    audio_tower = transformers.AutoModel.from_pretrained(
                        config.audio_model_id,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                        **kwargs,
                    )
        else:
            with accelerate.init_empty_weights():
                if "whisper" in config.audio_config._name_or_path.lower():
                    audio_tower = ModifiedWhisperEncoder(config.audio_config)
                    audio_tower.init_latency_mask(
                        config.audio_latency_block_size,
                        dtype=config.torch_dtype,
                    )
                else:
                    assert config.audio_latency_block_size in (
                        None,
                        0,
                    ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                    # we only ever use from_config if the weights are retrained, hence initializing is not
                    # required. This makes the model quite creation faster since init on CPU is quite slow.
                    audio_tower = transformers.AutoModel.from_config(
                        config.audio_config, trust_remote_code=True, **kwargs
                    )

        if isinstance(
            audio_tower,
            (transformers.Wav2Vec2BertModel, transformers.WhisperModel),
        ):
            # For these models we only need the encoder part
            # Wav2Vec2BertModel -> Wav2Vec2BertEncoder
            # WhisperModel -> WhisperEncoder
            audio_tower = audio_tower.encoder

        audio_tower = apply_lora(audio_tower, config.audio_model_lora_config)
        return audio_tower

    @classmethod
    def _create_language_model(
        cls, config: UltravoxConfig
    ) -> transformers.LlamaForCausalLM:
        model_cls = (
            transformers.DiffusionGemmaForBlockDiffusion
            if _is_diffusion_gemma(config)
            else transformers.AutoModelForCausalLM
        )
        if _init_weights_enabled() and config.text_model_id is not None:
            # See the matching comment in _create_audio_tower: low_cpu_mem_usage=False
            # is required, not just _real_device_context(), or this nested load
            # silently comes back with every key "missing" (randomly initialized)
            # instead of erroring -- verified against the 25B DiffusionGemma base.
            with _real_device_context():
                language_model = model_cls.from_pretrained(
                    config.text_model_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    **{
                        "attn_implementation": config.text_config._attn_implementation,
                        "torch_dtype": config.torch_dtype,
                        **FROM_PRETRAINED_KWARGS,
                    },
                )
        else:
            with accelerate.init_empty_weights():
                # we only ever use from_config if the weights are retrained, hence initializing is not
                # required. This makes the model quite creation faster since init on CPU is quite slow.
                language_model = model_cls.from_config(
                    config.text_config,
                    attn_implementation=config.text_config._attn_implementation,
                    torch_dtype=config.torch_dtype,
                )

        language_model = apply_lora(language_model, config.text_model_lora_config)
        return language_model

    def merge_and_unload(self):
        if isinstance(self.language_model, peft.PeftModel):
            self.language_model = self.language_model.merge_and_unload()
            # no need to download base language model weights anymore, so we can remove the id
            self.config.text_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"language_model.{name}"
                        for name, _ in self.language_model.named_parameters()
                    ]
                )
            )

        if hasattr(self, "audio_tower") and isinstance(
            self.audio_tower, peft.PeftModel
        ):
            self.audio_tower = self.audio_tower.merge_and_unload()
            # no need to download base audio model weights anymore, so we can remove the id
            self.config.audio_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"audio_tower.{name}"
                        for name, _ in self.audio_tower.named_parameters()
                    ]
                )
            )

        for param in ["text_model_lora_config", "audio_model_lora_config"]:
            if hasattr(self.config, param):
                delattr(self.config, param)

    def push_to_hub(self, *args, **kwargs):
        self.merge_and_unload()
        return super().push_to_hub(*args, **kwargs)

    def diff_state_dict(
        self, state_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if state_dict is None:
            state_dict = super().state_dict()

        trainable_params = {k for k, v in self.named_parameters() if v.requires_grad}
        # normalize the keys to match the original model
        # Example: audio_tower.base_model.model.layers.0._fsdp_wrapped_module.self_attn.k_proj.lora_B.default.weight
        trainable_params = {
            k.replace("_fsdp_wrapped_module.", "") for k in trainable_params
        }

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params or k in trainable_params
        }

        return state_dict

    def save_pretrained(
        self, *args, state_dict: Optional[Dict[str, Any]] = None, **kwargs
    ):
        state_dict = self.diff_state_dict(state_dict)

        super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def _pre_load_state_dict_hook(self, state_dict: Dict[str, Any], *args, **kwargs):
        self.keep_params.update(set(state_dict.keys()))

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model (reuses Peft model's method)
        """
        count_params = peft.peft_model.PeftModel.get_nb_trainable_parameters

        trainable_params, all_param = count_params(self)

        logging.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param:.1f}%"
        )

        lm_trainable_params, lm_all_params = count_params(self.language_model)
        if hasattr(self, "audio_tower") and self.audio_tower is not None:
            audio_trainable_params, audio_all_params = count_params(self.audio_tower)
        else:
            audio_trainable_params, audio_all_params = 0, 0

        projector_trainable_params = (
            trainable_params - lm_trainable_params - audio_trainable_params
        )
        projector_all_params = all_param - lm_all_params - audio_all_params

        # Calculate percentages only if the total parameters are non-zero
        audio_percent = (
            0.0
            if audio_all_params == 0
            else 100 * audio_trainable_params / audio_all_params
        )
        projector_percent = (
            0.0
            if projector_all_params == 0
            else 100 * projector_trainable_params / projector_all_params
        )

        logging.info(
            f"Trainable%:   "
            f" LLM: {100 * lm_trainable_params / lm_all_params:.1f}%"
            f" || Audio Encoder: {audio_percent:.1f}%"
            f" || Projector: {projector_percent:.1f}%"
        )


def get_checkpoint_files(
    model_id: str,
) -> tuple[list[str], dict | None, list[str]]:
    resolved_archive_file = transformers.utils.cached_file(
        model_id,
        transformers.utils.SAFE_WEIGHTS_NAME,
        _raise_exceptions_for_missing_entries=False,
    )

    if resolved_archive_file is not None:
        # not sharded
        sharded_metadata = None
        state_dict = transformers.modeling_utils.load_state_dict(resolved_archive_file)
        loaded_state_dict_keys = list(state_dict.keys())
    else:
        # sharded
        resolved_archive_file = transformers.utils.cached_file(
            model_id, transformers.utils.SAFE_WEIGHTS_INDEX_NAME
        )
        resolved_archive_file, sharded_metadata = (
            transformers.modeling_utils.get_checkpoint_shard_files(
                model_id,
                resolved_archive_file,
            )
        )
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]

    if isinstance(resolved_archive_file, str):
        resolved_archive_file = [resolved_archive_file]

    return resolved_archive_file, sharded_metadata, loaded_state_dict_keys


# TODO: refactor common parts to a shared module
def is_cache_empty(
    past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]],
) -> bool:
    """
    Check if the cache is empty.
    """
    if past_key_values is None:
        return True
    if isinstance(past_key_values, tuple):
        return all(len(c) == 0 for c in past_key_values)
    return past_key_values.get_seq_length() == 0


T = TypeVar("T", bound=torch.nn.Module)


def apply_lora(model: T, lora_config: dict) -> T:
    """
    Applies LoRA finetuning to the model. If the `r` parameter is set to 0, the model is frozen instead.
    """
    # _create_audio_tower/_create_language_model can run more than once for the
    # same config (e.g. UltravoxModel._init_weights rebuilds audio_tower/
    # language_model a second time during post_init()). lora_config here is the
    # same dict object stored on the model config, shared across those calls, so
    # popping from it destructively drops unfreeze_layers on the second call and
    # silently freezes everything. Copy before popping.
    lora_config = dict(lora_config or {})
    unfreeze_layers = lora_config.pop("unfreeze_layers", None)
    lora_config = peft.LoraConfig(**lora_config)

    if lora_config.r == 0:
        # freeze the model entirely, except for the specified layers
        for name, param in model.named_parameters():
            if not unfreeze_layers or not any(
                re.match(layer, name) for layer in unfreeze_layers
            ):
                param.requires_grad = False
            else:
                logging.info(f"Unfreezing layer: {name} with #{param.numel()} params")
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


class StackAudioFrames(nn.Module):
    """
    Stack the audio embedding frames to reduce the sequence length by a factor
    of `stack_factor`.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds


class RMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):
    def __init__(self, hidden_size: int, init: float = 1, eps: float = 1e-6):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight.data.fill_(init)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class UltravoxProjector(nn.Module):
    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim_in = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim_in, init=config.norm_init)
        self.linear_1 = nn.Linear(dim_in, self.hidden_dim, bias=False)
        dim_mid = self.hidden_dim
        self.act = transformers.activations.get_activation(config.projector_act)
        dim_mid = dim_mid // 2 if config.projector_act == "swiglu" else dim_mid
        dim_out = config.text_config.hidden_size
        self.linear_2 = nn.Linear(dim_mid, dim_out, bias=False)

        # Ultravox v0.4.1 and below uses layer_norm after the second linear layer,
        # while v0.5.0 and above uses layer_norm after the first linear layer.
        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid, init=config.norm_init)
            self.ln_post: nn.Module = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out, init=config.norm_init)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Takes in audio features from the audio tower and projects them to the text model's embedding space.
        It reduces the number of frames by a factor of `stack_factor` and increases the number of channels by the same factor.
        If the number of audio frames are not a multiple of the stack factor, the last few frames will be padded with zeros.

        Input shape:
            audio_features: B, T*S, C
        Output shape:
            hidden_states: B, T, D
        Where:
            B: batch size
            F: number of frames in the audio tower
            T: number of output embeddings
                T = ceil(F / S)
            S: stack factor
            C: number of channels out of the encoder (aka audio tower)
            H: hidden size of the projector (config.hidden_size)
            D: dimension of the text model (config.text_config.hidden_size)

        """
        # B, F, C -> B, T, C*S
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        # B, T, C*S -> B, T, H
        hidden_states = self.linear_1(audio_features)
        # B, T, H -> B, T, H/2 (assuming swiglu)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_mid(hidden_states)
        # B, T, H/2 -> B, T, D
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class ModifiedWhisperEncoder(
    whisper.WhisperEncoder, transformers.modeling_utils.ModuleUtilsMixin
):
    """
    Encoder portion of OpenAI's Whisper model.

    This implementation is a slightly modified version of HF Transformers' Whisper Encoder, with only a few fixes:
    1. base_model_prefix updated to allow for doing `.from_pretrained` directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    """

    base_model_prefix = "model.encoder"
    _no_split_modules = ["WhisperEncoderLayer"]
    _keys_to_ignore_on_load_unexpected = ["model.decoder.*"]

    def __init__(self, config: transformers.WhisperConfig):
        super().__init__(config)
        self.config.is_decoder = False

    @property
    def max_context_length(self):
        return (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )

    def init_latency_mask(
        self, audio_latency_block_size: int | None, dtype: torch.dtype
    ):
        if audio_latency_block_size is None:
            self.audio_streaming_mask = None
            return

        # Use max_context_length directly in the calculation
        max_seqlen = self.max_context_length
        assert (
            max_seqlen > 0
        ), f"maximum sequence length must be positive, got {max_seqlen}"
        assert (
            max_seqlen % audio_latency_block_size == 0
        ), f"audio_latency_block_size {audio_latency_block_size} must divide {max_seqlen} evenly."
        # Given the block size, we calculate number of blocks.
        audio_latency_nblocks = max_seqlen // audio_latency_block_size
        audio_streaming_mask = (
            torch.tril(
                torch.ones(audio_latency_nblocks, audio_latency_nblocks),
                diagonal=0,
            )
            .repeat_interleave(audio_latency_block_size, dim=0)
            .repeat_interleave(audio_latency_block_size, dim=1)
        )
        audio_streaming_mask = (1.0 - audio_streaming_mask) * torch.finfo(dtype).min
        audio_streaming_mask = audio_streaming_mask[None, None, :, :]
        self.register_buffer(
            "audio_streaming_mask", audio_streaming_mask, persistent=False
        )

    def forward(
        self,
        input_features,
        audio_len=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = self.max_context_length
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length} or less, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Create attention mask based on audio lengths to mask out padding tokens
        # For each sample in batch:
        # - Convert raw audio length to feature length after convolutions
        # - Create boolean mask that is True for valid positions and False for padding
        # - Convert to extended attention mask format expected by transformer layers
        #   (1.0 for positions to attend to, large negative for positions to ignore)
        # This masking ensures consistent behavior between training and inference
        # by preventing the model from attending to padding tokens in both cases
        attention_mask = None
        if audio_len is not None:
            audio_feature_len = self._get_feat_extract_output_lengths(audio_len)
            max_seq_len = hidden_states.shape[1]
            attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
                None, :
            ].lt(audio_feature_len.view(-1, 1))
            attention_mask = self.get_extended_attention_mask(
                attention_mask,
                None,
                dtype=hidden_states.dtype,
            )

        if self.audio_streaming_mask is not None:
            seqlen = hidden_states.size(-2)
            if attention_mask is not None:
                attention_mask = torch.minimum(
                    self.audio_streaming_mask[:, :, :seqlen, :seqlen], attention_mask
                )  # merge
            else:
                attention_mask = self.audio_streaming_mask[:, :, :seqlen, :seqlen]
            attention_mask = attention_mask.to(hidden_states.dtype)

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


UltravoxConfig.register_for_auto_class()
UltravoxModel.register_for_auto_class()

try:
    transformers.AutoConfig.register("ultravox", UltravoxConfig)
except ValueError:  # already registered by remote code
    pass
try:
    transformers.AutoModel.register(UltravoxConfig, UltravoxModel)
except ValueError:  # already registered by remote code
    pass

transformers.activations.ACT2FN["swiglu"] = SwiGLU
