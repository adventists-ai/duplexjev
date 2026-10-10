"""DuplexJev-32B-Vision (research preview): decisions about speech and an image together, in one forward pass.

    from duplexjev.vision import VisionDecider
    vd = VisionDecider.from_pretrained("adventists-ai/DuplexJev-32B-Vision")       # one 80 GB GPU
    vd.decide(audio="clip.wav", image="face.jpg",
              questions=[{"id": "match", "text": "Does the face in the picture show the same emotion as the "
                          "speaker's voice?", "options": ["Yes", "No"]}])
    vd.chat(audio="clip.wav", image="photo.jpg", text="What do you notice?")

The model is Qwen3-VL-32B's vision encoder and the DuplexJev-32B-Para language model with one more LoRA (merged), plus
the DuplexJev-32B-Para audio encoder and connector (in ``audio/``). Audio and image are both optional. Answers are read
as one constrained token, averaged over option orders, like ``duplexjev.Decider``. Needs transformers >= 4.57.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np
import torch

from .decider import SAMPLE_RATE, _local_dir, load_audio
from .question import LETTERS, as_questions

INSTR = "Answer with only the letter of the correct option."


class VisionDecider:
    def __init__(self, model, image_processor, tok, audio_tower, projector, audio_processor, device):
        self.model, self.ip, self.tok = model, image_processor, tok
        self.tower, self.proj, self.aproc, self.device = audio_tower, projector, audio_processor, device
        cfg = model.config
        self.IMG = cfg.image_token_id
        self.VS, self.VE, self.AUD, self.END = [tok.convert_tokens_to_ids(x) for x in
                                                ("<|vision_start|>", "<|vision_end|>", "<|audio|>", "<|im_end|>")]
        self.merge = cfg.vision_config.spatial_merge_size
        self.letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS[:10]]

    @classmethod
    def from_pretrained(cls, name_or_path: str = "adventists-ai/DuplexJev-32B-Vision", *, device: str = "cuda",
                        dtype=torch.bfloat16):
        import transformers
        from safetensors.torch import load_file

        from . import _uv
        from ._uv.ultravox_model import UltravoxProjector

        path = _local_dir(name_or_path) if os.path.isdir(name_or_path) else _download(name_or_path)
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map={"": device})
        model.eval()
        ip = transformers.AutoImageProcessor.from_pretrained(path)
        tok = transformers.AutoTokenizer.from_pretrained(path)
        ad = os.path.join(path, "audio")
        c = json.load(open(os.path.join(ad, "config.json")))
        c.pop("architectures", None)
        c.update(text_model_id=None, audio_model_id=None)
        ucfg = _uv.UltravoxConfig(**c)
        tower = _uv.Qwen3ASREncoder(ucfg.audio_config)  # the vendored encoder (transformers may map a different one)
        proj = UltravoxProjector(ucfg)
        sd = load_file(os.path.join(ad, "model.safetensors"))
        tower.load_state_dict({k[len("audio_tower."):]: v for k, v in sd.items() if k.startswith("audio_tower.")})
        proj.load_state_dict({k[len("multi_modal_projector."):]: v for k, v in sd.items()
                              if k.startswith("multi_modal_projector.")})
        tower.to(device, dtype).eval()
        proj.to(device, dtype).eval()
        fe = transformers.WhisperFeatureExtractor.from_pretrained(ad)
        aproc = _uv.UltravoxProcessor(audio_processor=transformers.WhisperProcessor(feature_extractor=fe, tokenizer=tok),
                                      tokenizer=tok, stack_factor=ucfg.stack_factor, encoder_ds_factor=8,
                                      audio_context_size=3000, audio_padding="longest")
        return cls(model, ip, tok, tower, proj, aproc, device)

    # ------------------------------------------------------------------ inputs
    @torch.no_grad()
    def _audio_emb(self, audio: Any) -> torch.Tensor:
        a = load_audio(audio)
        hop = 160 * 8 * int(self.aproc.stack_factor)
        r = (-len(a)) % hop
        if r:
            a = np.concatenate([a, np.zeros(r, dtype=np.float32)])
        p = self.aproc(text="<|audio|>", audio=a, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        h = self.tower(p["audio_values"].to(self.device, self.tower.dtype),
                       audio_len=p["audio_lens"].to(self.device)).last_hidden_state
        return self.proj(h)[0, : int(p["audio_token_len"][0])]

    def _image(self, image: Any):
        from PIL import Image

        im = image if isinstance(image, Image.Image) else Image.open(image)
        x = self.ip(images=[im.convert("RGB")], return_tensors="pt")
        return x["pixel_values"], x["image_grid_thw"]

    def _build(self, body_text: str, audio=None, image=None, system=None):
        parts = []
        if audio is not None:
            parts.append(("The user said: ", "AUDIO"))
        if image is not None:
            parts.append(("Picture: ", "IMAGE"))
        body = "\n\n".join(p + "\u0000" + k for p, k in parts)
        if body_text:
            body = (body + "\n\n" if body else "") + body_text
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": body}]
        full = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids, ap, pv, grid = [], None, None, None
        for seg in full.split("\u0000"):
            if seg.startswith("AUDIO"):
                e = self._audio_emb(audio)
                ap = (len(ids), e)
                ids += [self.AUD] * e.shape[0]
                seg = seg[5:]
            elif seg.startswith("IMAGE"):
                pv, grid = self._image(image)
                n = int(grid.prod()) // self.merge ** 2
                ids += [self.VS] + [self.IMG] * n + [self.VE]
                seg = seg[5:]
            ids += self.tok.encode(seg, add_special_tokens=False)
        t = torch.tensor([ids], device=self.device)
        x = self.model.get_input_embeddings()(t)
        if ap:
            x = x.clone()
            x[0, ap[0]: ap[0] + ap[1].shape[0]] = ap[1].to(x.dtype)
        att = torch.ones_like(t)
        pos, _ = self.model.model.get_rope_index(t, (t == self.IMG).int(),
                                                 image_grid_thw=grid.to(self.device) if grid is not None else None,
                                                 attention_mask=att)
        kw = dict(pixel_values=pv.to(self.device, torch.bfloat16), image_grid_thw=grid.to(self.device)) \
            if pv is not None else {}
        return dict(inputs_embeds=x, attention_mask=att, position_ids=pos, **kw)

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def decide(self, audio=None, image=None, questions=None, *, n_perm: int = 2) -> dict:
        """``{question_id: {"answer", "confidence", "probs"}}``; each answer is one constrained token."""
        out = {}
        for q in as_questions(questions if isinstance(questions, (list, tuple)) else [questions]):
            n = len(q.options)
            P = np.zeros(n)
            for s in range(n_perm):
                opts = [q.options[(j + s) % n] for j in range(n)]
                text = q.text + "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) \
                    + "\n\n" + INSTR
                lg = self.model(**self._build(text, audio, image), logits_to_keep=1).logits[0, -1].float()
                p = lg[self.letter_ids[:n]].softmax(-1).cpu().numpy()
                for j in range(n):
                    P[(j + s) % n] += p[j] / n_perm
            probs = {o: float(P[i]) for i, o in enumerate(q.options)}
            best = max(probs, key=probs.get)
            out[q.id] = {"answer": best, "confidence": probs[best], "probs": probs}
        return out

    @torch.no_grad()
    def chat(self, audio=None, image=None, text: str = "", *, system: str | None = None,
             max_new_tokens: int = 128) -> str:
        """Greedy free-form reply to speech and/or an image and/or text."""
        kw = self._build(text, audio, image, system)
        o = self.model(**kw, use_cache=True, logits_to_keep=1)
        pkv, pos = o.past_key_values, kw["position_ids"]
        nxt = int(o.logits[0, -1].argmax())
        out, last, L = [], int(pos.max()), kw["attention_mask"].shape[1]
        for i in range(max_new_tokens):
            if nxt == self.END:
                break
            out.append(nxt)
            t = torch.tensor([[nxt]], device=self.device)
            p = torch.full((pos.shape[0], 1, 1), last + 1 + i, device=self.device, dtype=pos.dtype)
            att = torch.ones((1, L + i + 1), device=self.device, dtype=kw["attention_mask"].dtype)
            o = self.model(inputs_embeds=self.model.get_input_embeddings()(t), position_ids=p, attention_mask=att,
                           past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
        return self.tok.decode(out)


def _download(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja", "LICENSE*"])
