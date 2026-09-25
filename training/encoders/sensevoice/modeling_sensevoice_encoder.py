"""SenseVoice-Small audio encoder (FunAudioLLM/SenseVoiceSmall) as a standalone transformers model for DuplexJev.

Input: 10 ms frames of raw 16 kHz samples (see feature_extraction_sensevoice.py). Inside: SenseVoice's frontend
(Kaldi fbank 80, dither 0, LFR m=7 n=6, CMVN from am.mvn), the four task queries SenseVoice prepends (language=auto,
event, emotion, text-norm=woitn), and the 50 + 20 SANM encoder blocks. The query frames are dropped from the output,
so one output frame = 60 ms of audio. Weights are unchanged from SenseVoiceSmall (encoder + query embeddings)."""
import torch
import torchaudio
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_sensevoice_encoder import SenseVoiceEncoderConfig
from .sensevoice_sanm import SenseVoiceEncoderSmall


def apply_lfr(x, m, n):
    T = x.shape[0]
    T_lfr = -(-T // n)
    x = torch.cat([x[:1].repeat((m - 1) // 2, 1), x], 0)
    need = (T_lfr - 1) * n + m
    if x.shape[0] < need:
        x = torch.cat([x, x[-1:].repeat(need - x.shape[0], 1)], 0)
    idx = (torch.arange(T_lfr, device=x.device)[:, None] * n + torch.arange(m, device=x.device)[None, :])
    return x[idx].reshape(T_lfr, -1)


class SenseVoiceEncoder(PreTrainedModel):
    config_class = SenseVoiceEncoderConfig
    base_model_prefix = "sensevoice_encoder"
    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = SenseVoiceEncoderSmall(input_size=config.input_size, output_size=config.output_size,
                                              attention_heads=config.attention_heads, linear_units=config.linear_units,
                                              num_blocks=config.num_blocks, tp_blocks=config.tp_blocks, dropout_rate=0.0,
                                              positional_dropout_rate=0.0, attention_dropout_rate=0.0, input_layer="pe",
                                              normalize_before=True, kernel_size=config.kernel_size, sanm_shfit=config.sanm_shfit,
                                              selfattention_layer_type="sanm")
        self.embed = nn.Embedding(config.num_query_embeddings, config.input_size)
        self.register_buffer("cmvn", torch.zeros(2, config.input_size), persistent=True)
        self.post_init()

    @property
    def max_context_length(self):
        return 3000

    def _init_weights(self, module):
        pass

    def _frontend(self, wave):  # wave: 1-D float tensor in [-1, 1]
        c = self.config
        x = (wave.float() * 32768.0).unsqueeze(0)
        mat = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=c.n_mels, frame_length=c.frame_length, frame_shift=c.frame_shift,
                                                dither=0.0, energy_floor=0.0, window_type="hamming", sample_frequency=16000,
                                                snip_edges=True)
        mat = apply_lfr(mat, c.lfr_m, c.lfr_n)
        return (mat + self.cmvn[0].float()) * self.cmvn[1].float()

    def forward(self, input_features, attention_mask=None, audio_len=None, input_features_mask=None, **kwargs):
        B, H, F = input_features.shape
        if audio_len is None:
            m = attention_mask if attention_mask is not None else input_features_mask
            audio_len = m.sum(-1) if m is not None else torch.full((B,), F, device=input_features.device)
        wave = input_features.float().transpose(1, 2).reshape(B, F * H)
        feats = []
        for i in range(B):
            n = max(int(audio_len[i]) * H, 400)
            feats.append(self._frontend(wave[i, :n]))
        lens = torch.tensor([f.shape[0] for f in feats], device=input_features.device)
        x = nn.utils.rnn.pad_sequence(feats, batch_first=True).to(self.embed.weight.dtype)
        dev = x.device
        q_lang = self.embed(torch.full((B, 1), self.config.language_id, device=dev, dtype=torch.long))
        q_ev = self.embed(torch.tensor([[1, 2]], device=dev).repeat(B, 1))
        q_tn = self.embed(torch.full((B, 1), self.config.textnorm_id, device=dev, dtype=torch.long))
        x = torch.cat([q_lang, q_ev, q_tn, x], 1)
        out, _ = self.encoder(x, lens + 4)
        out = out[:, 4:]
        # The processor predicts ceil(frames / lfr_n) encoder frames; Kaldi fbank with snip_edges drops up to two
        # 10 ms frames, so LFR can come out one frame short. Pad at the end so every predicted token has a frame.
        need = int(-(-int(audio_len.max()) // self.config.lfr_n))
        if out.shape[1] < need:
            out = nn.functional.pad(out, (0, 0, 0, need - out.shape[1]))
        return BaseModelOutput(last_hidden_state=out)
