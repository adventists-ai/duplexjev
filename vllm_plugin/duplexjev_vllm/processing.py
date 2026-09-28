"""Audio + text preprocessing for DuplexJev checkpoints inside vLLM.

Same maths as the Ultravox processor the connectors were trained with (Whisper-style 128-bin log-mel, 30 s chunks,
audio_token_len = ceil(mel_frames / (encoder_ds_factor * stack_factor))), plus the right-padding the duplexjev
package applies: every clip is padded with silence to a whole number of audio tokens, so a clip reads the same
whatever it is batched with.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers

HOP = 160


class DuplexJevProcessor:
    def __init__(self, feature_extractor, tokenizer, *, stack_factor: int = 2, encoder_ds_factor: int = 8,
                 audio_context_size: int = 3000, audio_placeholder: str = "<|audio|>"):
        self.audio_processor = feature_extractor  # vLLM reads the WhisperFeatureExtractor from here
        self.tokenizer = tokenizer
        self.stack_factor = stack_factor
        self.encoder_ds_factor = encoder_ds_factor
        self.audio_context_size = audio_context_size
        self.audio_placeholder = audio_placeholder
        self.vocab = tokenizer.get_vocab()
        self.audio_token_replacement = tokenizer.eos_token  # vLLM overrides this and audio_replacement_token_id
        self.audio_replacement_token_id = None

    def _align(self, a: np.ndarray) -> np.ndarray:
        step = HOP * self.encoder_ds_factor * self.stack_factor
        n = max(len(a), 2 * HOP)
        n = -(-n // step) * step
        return np.pad(a, (0, n - len(a))) if n > len(a) else a

    def _chunk(self, values: torch.Tensor, lens: torch.Tensor, include_audio_num_chunks: bool) -> dict:
        ctx = self.audio_context_size or values.shape[-1]
        chunks: List[torch.Tensor] = []
        clens: List[int] = []
        cont: List[bool] = []
        nch: List[int] = []
        for i in range(values.shape[0]):
            L = int(lens[i])
            nch.append(int(np.ceil(L / ctx)))
            for off in range(0, L, ctx):
                c = values[i, :, off:off + ctx]
                if off > 0 and c.shape[-1] < ctx:
                    c = F.pad(c, (0, ctx - c.shape[-1]))
                chunks.append(c)
                clens.append(min(L - off, ctx))
                cont.append(off > 0)
        d = {"audio_values": torch.stack(chunks, 0), "audio_lens": torch.tensor(clens, dtype=torch.int64),
             "audio_is_continuation": torch.tensor(cont, dtype=torch.bool),
             "audio_batch_size": torch.tensor([len(chunks)])}
        if include_audio_num_chunks:
            d["audio_num_chunks"] = torch.tensor(nch, dtype=torch.int64)
        return d

    def __call__(self, text: Optional[str] = None, audio=None, audios=None, sampling_rate: Optional[int] = None,
                 return_tensors="pt", include_audio_num_chunks: bool = False, **kwargs) -> transformers.BatchFeature:
        if audio is not None:
            audios = audio if isinstance(audio, list) else [audio]
        audios = [a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a) for a in (audios or [])]
        data, is_cont = {}, []
        if audios:
            audios = [self._align(a.astype(np.float32)) for a in audios]
            x = self.audio_processor(audios, sampling_rate=sampling_rate or self.audio_processor.sampling_rate,
                                     padding="longest", pad_to_multiple_of=HOP, truncation=False,
                                     return_attention_mask=True, return_tensors="pt")
            data.update(self._chunk(torch.as_tensor(x.input_features), torch.as_tensor(x.attention_mask).sum(-1),
                                    include_audio_num_chunks))
            is_cont = data.pop("audio_is_continuation")
            data["audio_token_len"] = torch.ceil(
                data["audio_lens"] / (self.encoder_ds_factor * self.stack_factor)).to(torch.int)
        if text is not None:
            parts = self.tokenizer(text.split(self.audio_placeholder), add_special_tokens=False)["input_ids"]
            rep = self.vocab[self.audio_token_replacement]
            ids: List[int] = []
            starts: List[int] = []
            p = -1
            for i, n in enumerate(data.get("audio_token_len", [])):
                if not is_cont[i]:
                    p += 1
                    if p >= len(parts):
                        raise ValueError("Text contains too few audio placeholders.")
                    ids.extend(parts[p])
                starts.append(len(ids))
                ids.extend([rep] * int(n))
            p += 1
            if p != len(parts) - 1:
                raise ValueError("Text contains too many audio placeholders.")
            ids.extend(parts[p])
            if "audio_token_len" in data:
                data["audio_token_start_idx"] = torch.as_tensor(starts)
            data["input_ids"] = [ids]
            data["attention_mask"] = [[1] * len(ids)]
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)
