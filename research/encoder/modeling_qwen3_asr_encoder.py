"""
Portable, standalone reimplementation of the Qwen3-ASR audio encoder
(transformers.models.qwen3_asr.modeling_qwen3_asr.Qwen3ASREncoder), rewritten to avoid
any transformers-5.x-only internal utilities (ALL_ATTENTION_FUNCTIONS, capture_outputs,
torch_compilable_check, GradientCheckpointingLayer, etc.) so it loads cleanly via
trust_remote_code on older transformers installs (verified against 4.51.3).

Forward-pass math is unchanged from the original: same conv frontend, same windowed
(non-causal, block-local) attention via cu_seqlens chunk splitting, same sinusoidal
position embedding. Only the plumbing around attention-backend selection and
gradient-checkpointing wrappers was stripped out (this encoder is used frozen /
fine-tuned without gradient checkpointing here).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_qwen3_asr_encoder import Qwen3ASREncoderConfig


def _get_feat_extract_output_lengths(input_lengths, n_window=50):
    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13


def get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window):
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens, n_window)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths, n_window)
    max_len_after_cnn = feature_lens_after_cnn.max().item()

    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio

    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]

    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        position_embedding = self.compute_default_singular_positional_embedding()
        self.register_buffer("positional_embedding", position_embedding, persistent=False)

    def compute_default_singular_positional_embedding(self):
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def get_embedding(self, seqlen: int, device=None, dtype=None):
        # Some transformers versions (verified: 5.17.0, not 4.51.3) construct
        # PreTrainedModel subclasses under a meta-device context during
        # from_pretrained() for faster loading. Real (checkpoint) parameters
        # get correctly materialized afterward from the state_dict, but this
        # buffer is persistent=False (deliberately excluded from the
        # checkpoint -- it is pure derived math, nothing to load) and never
        # gets that materialization pass, leaving it as uninitialized memory
        # under that loading path. Detect and recompute rather than trusting
        # the cached buffer; this is a tiny (length x channels) tensor so
        # recomputing is effectively free.
        pe = self.positional_embedding
        if pe.device.type == "meta" or pe.numel() == 0 or torch.isnan(pe).any():
            pe = self.compute_default_singular_positional_embedding()
        if device is not None or dtype is not None:
            pe = pe.to(device=device, dtype=dtype)
        return pe[:seqlen, :]

    def forward(self, seqlen: int):
        return self.get_embedding(seqlen)


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-headed attention, block-local (non-causal within each cu_seqlens chunk)."""

    def __init__(self, config: Qwen3ASREncoderConfig, *args, **kwargs):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """hidden_states: (seq_length, embed_dim) -- a single packed sequence."""
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        # (seq, heads, head_dim) -> (1, heads, seq, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(query_states, lengths, dim=2)
        k_chunks = torch.split(key_states, lengths, dim=2)
        v_chunks = torch.split(value_states, lengths, dim=2)

        attn_outputs = []
        for q, k, v in zip(q_chunks, k_chunks, v_chunks):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=False,
            )
            attn_outputs.append(out)
        attn_output = torch.cat(attn_outputs, dim=2)

        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class CrossAttentionFusion(nn.Module):
    """Fuses an early/mid-layer hidden state (acoustic-leaning: pitch, timbre,
    prosody not yet abstracted away) with the final-layer hidden state
    (phonetic/semantic-leaning) via chunk-local cross-attention, so paralinguistic
    detail that later layers discard has a path back into the encoder output.
    Query = final-layer (semantic) hidden state; Key/Value = mid-layer (acoustic)
    hidden state. Residual-connected onto the semantic stream so this only adds
    information rather than replacing what already works for transcription.
    Uses the same per-chunk (cu_seqlens) splitting as Qwen3ASRAudioAttention so it
    cannot leak information across chunk/sample boundaries."""

    def __init__(self, config: Qwen3ASREncoderConfig, *args, **kwargs):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.fusion_num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError("fusion embed_dim must be divisible by fusion_num_heads")
        self.scaling = self.head_dim**-0.5
        # Q, K, V are drawn from three different depths (see config docstring:
        # Q=H18 final/semantic, K=H14 semantic-leaning retrieval index, V=H9
        # mid-layer/acoustic-leaning content), each with its own statistics, so
        # each gets its own pre-attention LayerNorm rather than sharing one.
        self.ln_q = nn.LayerNorm(self.embed_dim)
        self.ln_k = nn.LayerNorm(self.embed_dim)
        self.ln_v = nn.LayerNorm(self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.ln_out = nn.LayerNorm(self.embed_dim)
        # out_proj is zero-initialized in Qwen3ASREncoder._init_weights (not here):
        # under HF's meta-device loading path (from_pretrained with missing keys),
        # a plain nn.init call here targets a meta tensor with no real storage and
        # is silently ineffective; _init_weights is what actually runs for
        # newly-added parameters missing from the checkpoint.

    def forward(self, q_hidden: torch.Tensor, k_hidden: torch.Tensor, v_hidden: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """q_hidden (H18), k_hidden (H14), v_hidden (H9): (seq_length, embed_dim)
        -- packed sequences, same length and same chunk boundaries (all captured
        from the same forward pass, just at different layer depths)."""
        seq_length, _ = q_hidden.size()

        q = self.q_proj(self.ln_q(q_hidden)).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(self.ln_k(k_hidden)).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(self.ln_v(v_hidden)).reshape(seq_length, self.num_heads, -1)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(q, lengths, dim=2)
        k_chunks = torch.split(k, lengths, dim=2)
        v_chunks = torch.split(v, lengths, dim=2)

        attn_outputs = []
        for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
            out = F.scaled_dot_product_attention(
                qc, kc, vc, attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False,
            )
            attn_outputs.append(out)
        attn_output = torch.cat(attn_outputs, dim=2)

        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        # Normalize the (currently zero, until trained) acoustic correction term
        # before adding it, rather than re-normalizing the whole sum afterward.
        # The semantic (Q) stream is already ln_post-normalized and works;
        # renormalizing it a second time here would needlessly rescale it and
        # amplify whatever tiny pre-existing per-frame noise it has (e.g.
        # feature-extraction boundary effects when batched with padding).
        fused = q_hidden + self.ln_out(attn_output)
        return fused


class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASREncoderConfig, *args, **kwargs):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class Qwen3ASREncoder(PreTrainedModel):
    config_class = Qwen3ASREncoderConfig
    base_model_prefix = "qwen3_asr_encoder"
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]

    def __init__(self, config: Qwen3ASREncoderConfig, *args, **kwargs):
        super().__init__(config)
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_position_embeddings, embed_dim)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.fusion_k_layer_idx = config.fusion_k_layer_idx
        self.fusion_v_layer_idx = config.fusion_v_layer_idx
        self.fusion = CrossAttentionFusion(config)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.post_init()

    @property
    def max_context_length(self) -> int:
        # Matches the WhisperFeatureExtractor-compatible preprocessor_config.json
        # (chunk_length=30s -> nb_max_frames=3000) used alongside this encoder.
        return 3000

    def _init_weights(self, module):
        # Called by HF's loading machinery for any parameter missing from the
        # checkpoint (i.e. the newly-added fusion module; every other parameter
        # here is always loaded from the pretrained checkpoint, never randomly
        # initialized in practice). Standard init for Linear/LayerNorm/Conv2d,
        # plus a zero-init override for the fusion block's output projection so
        # it starts as a no-op (fused == semantic) and has to learn to pull in
        # acoustic detail rather than perturbing the working semantic path.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, CrossAttentionFusion):
            module.out_proj.weight.data.zero_()
            module.out_proj.bias.data.zero_()

    @staticmethod
    def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
        return lengths

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor = None, attention_mask: torch.Tensor = None, audio_len: torch.Tensor = None, **kwargs):
        if input_features_mask is None:
            input_features_mask = attention_mask
        if input_features_mask is None and audio_len is not None:
            # Ultravox's real call site (UltravoxModel.forward -> audio_tower.forward)
            # passes audio_len: a per-sample valid length in mel-FRAME units (matching
            # UltravoxProcessor's audio_lens output), not a full mask and not a raw
            # waveform-sample count. This mirrors ModifiedWhisperEncoder's own
            # audio_len handling so this encoder is a drop-in replacement.
            max_seq_len = input_features.shape[-1]
            positions = torch.arange(max_seq_len, device=input_features.device)[None, :]
            input_features_mask = positions.lt(audio_len.to(input_features.device).view(-1, 1)).to(torch.int32)
        if input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[0], input_features.shape[-1], dtype=torch.int32, device=input_features.device)
        batch_size = input_features.shape[0]
        if batch_size > 1:
            # The chunked local-attention grouping (get_audio_cu_seqlens) and the
            # conv2d batch-folding below are numerically batch-shape-sensitive: with
            # true multi-sample batching, different per-sample padding amounts change
            # how chunks/attention blocks are grouped and measurably leak between
            # samples (verified empirically: batched vs per-sample outputs diverged
            # by up to ~4 on a LayerNorm-scaled hidden state, far beyond bf16 noise).
            # Looping per-sample reuses the single-sample path unchanged, which is
            # the one verified to match a standalone (batch_size=1) call exactly.
            outs = []
            for i in range(batch_size):
                len_i = int(input_features_mask[i].sum().item())
                feat_i = input_features[i : i + 1, :, :len_i]
                mask_i = input_features_mask[i : i + 1, :len_i]
                out_i = self._forward_single(feat_i, mask_i).last_hidden_state[0]
                outs.append(out_i)
            max_len = max(o.shape[0] for o in outs)
            out = input_features.new_zeros(batch_size, max_len, outs[0].shape[-1])
            for i, o in enumerate(outs):
                out[i, : o.shape[0]] = o
            return BaseModelOutputWithPooling(last_hidden_state=out)
        return self._forward_single(input_features, input_features_mask)

    def _forward_single(self, input_features: torch.Tensor, input_features_mask: torch.Tensor):
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2

        if padded_feature_length % chunk_len != 0:
            # Callers (e.g. a generic batch collator) may not know about this
            # encoder's chunk-alignment requirement. Pad defensively to the next
            # chunk boundary instead of erroring out; the extra frames are marked
            # invalid via input_features_mask and dropped before the transformer
            # layers, so they do not affect the output.
            pad_amount = chunk_len - (padded_feature_length % chunk_len)
            input_features = torch.nn.functional.pad(input_features, (0, pad_amount))
            input_features_mask = torch.nn.functional.pad(input_features_mask, (0, pad_amount))
            padded_feature_length = input_features.shape[-1]

        num_chunks = padded_feature_length // chunk_len
        feature_lens = input_features_mask.sum(-1).to(torch.long)
        chunk_lengths = (
            input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        )
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)

        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )

        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out = conv_out + self.positional_embedding.get_embedding(time_steps, device=conv_out.device, dtype=conv_out.dtype)

        chunk_post_cnn_lens = self._post_cnn_length(
            input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        )
        valid_mask = torch.arange(time_steps, device=input_features.device) < chunk_post_cnn_lens.unsqueeze(1)
        valid_indices = valid_mask.flatten().nonzero().squeeze(-1)
        hidden_states = torch.index_select(conv_out.reshape(-1, conv_out.shape[-1]), 0, valid_indices)

        k_hidden = None
        v_hidden = None
        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, cu_seqlens)
            if layer_idx == self.fusion_v_layer_idx:
                # H9: still mid-depth, pitch/timbre/prosody detail not yet
                # abstracted away -- this is the content the fusion retrieves.
                v_hidden = hidden_states
            if layer_idx == self.fusion_k_layer_idx:
                # H14: high-level but not final -- used as the retrieval index,
                # decoupled from H9 (what gets retrieved) and H18 (the query).
                k_hidden = hidden_states

        hidden_states = self.ln_post(hidden_states)
        if k_hidden is not None and v_hidden is not None:
            hidden_states = self.fusion(hidden_states, k_hidden, v_hidden, cu_seqlens)

        # hidden_states is a packed (total_valid_frames, C) sequence (no batch dim).
        # Downstream code (UltravoxProjector etc.) expects a standard padded batch
        # tensor (B, T, C), matching what a Whisper-style encoder would return.
        # Unpack using each sample's own post-CNN frame count.
        per_sample_len = _get_feat_extract_output_lengths(feature_lens, self.n_window)
        max_len = int(per_sample_len.max().item()) if batch_size > 0 else 0
        out = hidden_states.new_zeros(batch_size, max_len, hidden_states.shape[-1])
        offset = 0
        for i in range(batch_size):
            n = int(per_sample_len[i].item())
            out[i, :n] = hidden_states[offset : offset + n]
            offset += n

        return BaseModelOutputWithPooling(last_hidden_state=out)
