"""Typed decisions about speech, answered from one forward pass with exact prefix sharing.

Input is audio plus one or more Jev-style option groups (``Question``); output is a probability for every option.
``Decider.decide`` answers the option groups for one clip; ``Decider.decide_batch`` answers many option groups about
many clips at once, each group naming the clip(s) it is about.

Two layouts give the same answers (up to floating-point noise):

* ``mode="packed"`` (default): for every item, the shared prefix (chat template, context, audio) is encoded once and
  all of that item's question suffixes are packed into one row under a block-diagonal mask, with position ids
  restarting at the prefix length. All items of a call are processed together: one prefill for the prefixes, one
  forward for all suffixes. KV memory grows with ``P + sum(L_i)`` per item instead of ``N * (P + L)``.
* ``mode="batch"``: every (item, question) pair is its own row. Simpler, slower; kept as a reference.

The answer to a question is the next-token softmax restricted to its option letters at the last prompt position.
No tokens are generated.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .question import Question, as_questions

SAMPLE_RATE = 16000
_SENTINEL = "⁣QUESTION⁣"

PROMPTS = {
    "en": {"context": "Context:\n{c}", "audio": "The user said: <|audio|>"},
    "zh": {"context": "上下文：\n{c}", "audio": "用户说：<|audio|>"},
}


# ---------------------------------------------------------------------------------------------------------- inputs
def load_audio(x: Any) -> np.ndarray:
    """Accept a 16 kHz float array, a (array, sample_rate) tuple or a file path; return 16 kHz mono float32."""
    sr = SAMPLE_RATE
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        import soundfile as sf

        x, sr = sf.read(x, dtype="float32")
    elif isinstance(x, tuple):
        x, sr = x
    a = np.asarray(x, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1) if a.shape[1] <= 8 else a.mean(axis=0)
    if sr != SAMPLE_RATE:  # polyphase resampling: fast and without a one-off JIT warm-up
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SAMPLE_RATE)
        a = resample_poly(a, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return a


# ---------------------------------------------------------------------------------------------------------- engine
class Decider:
    """Answer Jev-style option groups about speech with zero decode steps.

    Create with :meth:`from_pretrained` from a speech checkpoint in Ultravox format: the DuplexJev adapters
    (ASR encoder + trained connector + frozen LLM) or the released Ultravox models.
    """

    def __init__(self, model, processor, *, device=None):
        self.model = model.eval()
        self.processor = processor
        self.tok = processor.tokenizer
        tokenizer = self.tok
        if device is None or str(device) == "auto":  # sharded model: inputs go where the embeddings live
            device = model.language_model.get_input_embeddings().weight.device
        self.device = torch.device(device)
        self.lm = model.language_model
        self.base = getattr(self.lm, self.lm.base_model_prefix)
        self.head = self.lm.get_output_embeddings()
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._letter_ids: dict[str, int] = {}
        self._tail_cache: dict[tuple, tuple[str, str]] = {}
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        text_model: str | None = None,
        audio_model: str | None = None,
        **kwargs,
    ) -> "Decider":
        """Load a speech checkpoint (Ultravox format: encoder + connector, pointing to its frozen LLM).

        ``device="auto"`` shards the model over all visible GPUs (needs ``accelerate``).
        ``text_model`` / ``audio_model`` point the checkpoint at a local copy of *the same* LLM or encoder it was
        trained with (e.g. ``/data/models/Qwen3-32B``). They are not a way to swap models: the connector only works
        with its own encoder and LLM. Remaining kwargs go to ``from_pretrained`` of the model.
        """
        import transformers

        dkw = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") or device == "auto" else torch.float32
        dmap = "auto" if device == "auto" else {"": device}
        cfg = transformers.AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        if getattr(cfg, "model_type", "") != "ultravox":
            raise ValueError(
                f"{name_or_path!r} is not a speech checkpoint. duplexjev answers questions about audio and needs an "
                "encoder + connector checkpoint in Ultravox format, e.g. adventists-ai/DuplexJev-A-Qwen3-ASR-0.6B-Qwen3-32B "
                "or fixie-ai/ultravox-v0_6-qwen-3-32b."
            )
        if True:
            if dkw == "dtype":
                import warnings

                warnings.warn(
                    "Ultravox-format checkpoints ship remote code written for transformers 4.51-4.55; with "
                    f"transformers {transformers.__version__} loading can be extremely slow (weights built on CPU). "
                    'Install with `pip install "duplexjev[speech]"` to get a compatible version.'
                )
            if text_model:
                cfg.text_model_id = text_model
            if audio_model:
                cfg.audio_model_id = audio_model
            model = transformers.AutoModel.from_pretrained(
                name_or_path, config=cfg, trust_remote_code=True, device_map=dmap, **{dkw: dtype}, **kwargs
            )
            processor = transformers.AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
            if not hasattr(processor, "audio_processor"):  # e.g. a stray preprocessor_config.json took precedence
                from transformers.dynamic_module_utils import get_class_from_dynamic_module

                cls_ = get_class_from_dynamic_module(cfg.auto_map["AutoProcessor"], name_or_path)
                processor = cls_.from_pretrained(name_or_path)
            ap = processor.audio_processor
            if isinstance(ap, transformers.WhisperFeatureExtractor) and not hasattr(ap, "feature_extractor"):
                processor.audio_processor = transformers.WhisperProcessor(
                    feature_extractor=ap, tokenizer=processor.tokenizer
                )
            return cls(model, processor, device=None if device == "auto" else device)

    # ------------------------------------------------------------------ prompt pieces
    def _head_tail(self, context: str | None, content: str, lang: str) -> tuple[str, str]:
        """Split the chat-formatted prompt into the shared head (up to the question) and the fixed tail."""
        key = (context, content, lang)
        if key in self._tail_cache:
            return self._tail_cache[key]
        parts = []
        if context:
            parts.append(PROMPTS[lang]["context"].format(c=context))
        parts.append(content)
        parts.append(_SENTINEL)
        user = "\n\n".join(parts)
        if getattr(self.tok, "chat_template", None):
            full = self.tok.apply_chat_template(
                [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        else:
            full = user + "\n\nAnswer: "
        head, tail = full.split(_SENTINEL)
        if len(self._tail_cache) < 4096:
            self._tail_cache[key] = (head, tail)
        return head, tail

    def _letter_id(self, letter: str) -> int:
        if letter not in self._letter_ids:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"option letter {letter!r} is not a single token for this tokenizer: {ids}")
            self._letter_ids[letter] = ids[0]
        return self._letter_ids[letter]

    def _align(self, a: np.ndarray) -> np.ndarray:
        """Left-pad with silence to a whole number of audio tokens.

        The projector stacks ``stack_factor`` encoder frames per LLM token. If the last group is partial, it is filled
        with whatever follows in the padded batch tensor, so the same clip would read differently depending on its
        batch neighbours. Aligning every clip to full groups (160 ms for Whisper and Qwen3-ASR encoders at stack 8
        and 2) makes every answer independent of batch composition.
        """
        hop = 160 * int(getattr(self.processor, "encoder_ds_factor", 2)) * int(getattr(self.processor, "stack_factor", 8))
        r = (-len(a)) % hop
        return np.concatenate([np.zeros(r, dtype=np.float32), a]) if r else a

    def _encode_prefix(self, item: dict, lang: str) -> dict:
        head, tail = self._head_tail(item.get("context"), PROMPTS[lang]["audio"], lang)
        p = self.processor(text=head, audio=self._align(load_audio(item["audio"])), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        aud = {k: p[k] for k in ("audio_values", "audio_lens", "audio_token_len", "audio_token_start_idx", "audio_batch_size") if k in p}
        if "audio_batch_size" not in aud:
            aud["audio_batch_size"] = torch.tensor([p["audio_values"].shape[0]])
        return {"ids": p["input_ids"][0].tolist(), "audio": aud, "tail": tail}

    def _suffixes(self, qs: list[Question], tail: str, n_perm: int, seed: int):
        """Token ids of every (question, permutation) suffix, with the letter ids to read."""
        out = []
        for qi, q in enumerate(qs):
            for k in range(n_perm):
                perm = q.permutation(seed + k)
                ids = self.tok(q.render(perm) + tail, add_special_tokens=False)["input_ids"]
                letters = [self._letter_id(c) for c in q.letters()]
                out.append({"q": qi, "perm": perm, "ids": ids, "letters": letters})
        return out

    # ------------------------------------------------------------------ collation
    def _collate(self, rows: list[tuple[list[int], dict]]) -> dict:
        """Left-pad token rows; stack audio (right-padded) and shift audio start indices by the left padding."""
        L = max(len(r[0]) for r in rows)
        ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), L), dtype=torch.long)
        for i, (r, _) in enumerate(rows):
            ids[i, L - len(r):] = torch.tensor(r)
            mask[i, L - len(r):] = 1
        batch = {"input_ids": ids, "attention_mask": mask}
        if True:
            vals, lens, tlen, start, bs = [], [], [], [], []
            for r, a in rows:
                shift = L - len(r)
                vals += list(a["audio_values"])
                lens.append(a["audio_lens"])
                tlen.append(a["audio_token_len"])
                start.append(a["audio_token_start_idx"] + shift)
                bs.append(a["audio_batch_size"].view(-1))
            T = max(v.shape[-1] for v in vals)
            batch["audio_values"] = torch.stack([torch.nn.functional.pad(v, (0, T - v.shape[-1])) for v in vals])
            batch["audio_lens"] = torch.cat(lens)
            batch["audio_token_len"] = torch.cat(tlen)
            batch["audio_token_start_idx"] = torch.cat(start)
            batch["audio_batch_size"] = torch.cat(bs)
        out = {k: v.to(self.device) for k, v in batch.items()}
        if "audio_values" in out:
            out["audio_values"] = out["audio_values"].to(self.model.dtype)
        return out

    def _letter_logits(self, h: torch.Tensor, letters: Sequence[int]) -> torch.Tensor:
        """Logits of the option letters only, from final hidden states ``h`` [..., d]."""
        W = self.head.weight[list(letters)]
        z = h.to(W.dtype) @ W.T
        if getattr(self.head, "bias", None) is not None:
            z = z + self.head.bias[list(letters)]
        return z.float()

    # ------------------------------------------------------------------ public API
    def decide(
        self,
        audio: Any,
        questions: Question | dict | Sequence[Question | dict],
        *,
        context: str | None = None,
        lang: str | None = None,
        **options,
    ) -> dict:
        """Answer one or more option groups about one audio clip.

        Args:
            audio: a file path, a 16 kHz float array, or an ``(array, sample_rate)`` tuple.
            questions: one ``Question`` (option group) or a list of them.
            context: optional dialogue context or state, shared by all questions.
            lang: language of the fixed prompt words (``"en"``/``"zh"``); defaults to the first question's.
            **options: ``mode``, ``n_perm``, ``seed`` (see :meth:`decide_batch`).

        Returns:
            ``{question_id: {"answer": option, "confidence": p, "probs": {option: p}}}``.
        """
        qs = [questions] if isinstance(questions, (Question, dict)) else list(questions)
        qs = [q.for_audio(None) if isinstance(q, Question) else q for q in qs]
        item = {"audio": audio, "questions": qs, "context": context, "lang": lang}
        return self._decide_items([item], **options)[0]

    def decide_batch(
        self,
        audios: dict | Sequence[Any],
        questions: Sequence[Question | dict],
        *,
        context: dict | Sequence[str | None] | None = None,
        lang: str | None = None,
        mode: str = "packed",
        n_perm: int = 1,
        seed: int = 0,
        max_items: int | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Answer many option groups about many clips in one batched pass.

        Every option group names the clip(s) it is about with ``Question(..., audio=<id>)``, where ``<id>`` is a key
        of ``audios`` (or an index if ``audios`` is a list), a list of ids, or ``"*"`` for every clip.

        Args:
            audios: ``{id: audio}`` or a list of audio (ids are then 0, 1, ...). Audio as in :meth:`decide`.
            questions: option groups, each with ``audio=`` set.
            context: optional context per clip (``{id: str}`` or a list aligned with ``audios``).
            mode: ``"packed"`` (prefix sharing, default) or ``"batch"`` (one row per question; reference).
            n_perm: average over this many option orders (reduces position bias; costs more tokens).
            seed: selects the option orders.
            max_items / max_tokens: split into several forward passes of bounded size.

        Returns:
            ``{audio_id: {question_id: {"answer", "confidence", "probs"}}}`` for every clip that was asked something.
        """
        amap = dict(audios) if isinstance(audios, dict) else dict(enumerate(audios))
        if not amap:
            return {}
        if context is None:
            cmap = {}
        elif isinstance(context, dict):
            cmap = context
        else:
            cmap = dict(zip(amap, context))
        per = {k: [] for k in amap}
        for q in as_questions(questions, unique=False):
            if q.audio is None:
                raise ValueError(f"question {q.id!r}: set audio=<clip id> (or '*' for all clips) in decide_batch")
            targets = list(amap) if q.audio == "*" else (q.audio if isinstance(q.audio, (list, tuple)) else [q.audio])
            for t in targets:
                if t not in amap:
                    raise ValueError(f"question {q.id!r} refers to unknown audio {t!r}; known: {list(amap)}")
                per[t].append(q)
        items, ids = [], []
        for k, qs in per.items():
            if qs:
                seen = [q.id for q in qs]
                if len(set(seen)) != len(seen):
                    raise ValueError(f"audio {k!r}: duplicate question ids {seen}")
                items.append({"audio": amap[k], "questions": qs, "context": cmap.get(k), "lang": lang})
                ids.append(k)
        res = self._decide_items(items, mode=mode, n_perm=n_perm, seed=seed, max_items=max_items, max_tokens=max_tokens)
        return dict(zip(ids, res))

    @torch.no_grad()
    def _decide_items(self, items, *, mode="packed", n_perm=1, seed=0, max_items=None, max_tokens=None) -> list[dict]:
        if not items:
            return []
        t0 = time.perf_counter()
        plans = []
        for it in items:
            qs = as_questions(it.get("questions") or [])
            if not qs:
                raise ValueError("no questions given")
            if it.get("audio") is None:
                raise ValueError("every item needs audio")
            lg = it.get("lang") or qs[0].lang
            pre = self._encode_prefix(it, lg)
            plans.append({"qs": qs, "pre": pre, "sufs": self._suffixes(qs, pre["tail"], n_perm, seed)})
        t_enc = time.perf_counter()
        if mode not in ("packed", "batch"):
            raise ValueError("mode must be 'packed' or 'batch'")
        run = self._run_packed if mode == "packed" else self._run_batch
        n_pass = 0
        for chunk in _chunks(plans, max_items, max_tokens, packed=mode == "packed"):
            run(chunk)
            n_pass += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_fwd = time.perf_counter()
        results = [self._collect(p) for p in plans]
        self.last_stats = {
            "mode": mode,
            "clips": len(items),
            "questions": sum(len(p["qs"]) for p in plans),
            "prefix_tokens": sum(len(p["pre"]["ids"]) for p in plans),
            "suffix_tokens": sum(len(s["ids"]) for p in plans for s in p["sufs"]),
            "prepare_ms": round((t_enc - t0) * 1000, 2),
            "forward_ms": round((t_fwd - t_enc) * 1000, 2),
            "batches": n_pass,
            "decode_steps": 0,
        }
        return results

    def _run_batch(self, plans):
        rows, where = [], []
        for pi, p in enumerate(plans):
            for si, s in enumerate(p["sufs"]):
                rows.append((p["pre"]["ids"] + s["ids"], p["pre"]["audio"]))
                where.append((pi, si))
        batch = self._collate(rows)
        # Ultravox merges the audio inside its own forward; read the last-position logits
        last = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        for r, (pi, si) in enumerate(where):
            s = plans[pi]["sufs"][si]
            s["z"] = last[r, torch.tensor(s["letters"], device=last.device)]

    def _run_packed(self, plans):
        # 1) prefill all prefixes together (left-padded); keep the KV cache
        batch = self._collate([(p["pre"]["ids"], p["pre"]["audio"]) for p in plans])
        P = batch["input_ids"].shape[1]
        out = self.model(**batch, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        # 2) all suffixes of an item in one row; block-diagonal causal mask over the suffix part
        B = len(plans)
        lens = [[len(s["ids"]) for s in p["sufs"]] for p in plans]
        T = max(sum(l) for l in lens)
        dt = self.model.dtype
        neg = torch.finfo(dt).min
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        pos = torch.full((B, T), P, dtype=torch.long)
        mask = torch.full((B, 1, T, P + T), neg, dtype=dt)
        mask[:, 0, :, :P] = torch.where(batch["attention_mask"].cpu()[:, None, :].bool(), 0.0, neg).to(dt)
        last = []
        for b, p in enumerate(plans):
            o, lb = 0, []
            for s in p["sufs"]:
                n = len(s["ids"])
                ids[b, o:o + n] = torch.tensor(s["ids"])
                pos[b, o:o + n] = P + torch.arange(n)
                mask[b, 0, o:o + n, P + o:P + o + n] = torch.triu(torch.full((n, n), neg, dtype=dt), 1)
                lb.append(o + n - 1)
                o += n
            for t in range(o, T):  # padding rows: attend to themselves only (never read)
                mask[b, 0, t, P + t] = 0
            last.append(lb)
        out2 = self.base(
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            position_ids=pos.to(self.device),
            past_key_values=cache,
            use_cache=False,
        )
        h = out2.last_hidden_state
        for b, p in enumerate(plans):
            hb = h[b, torch.tensor(last[b], device=h.device)]
            for s, hv in zip(p["sufs"], hb):
                s["z"] = self._letter_logits(hv, s["letters"])

    def _collect(self, plan) -> dict:
        res = {}
        for qi, q in enumerate(plan["qs"]):
            acc = np.zeros(len(q.options))
            sufs = [s for s in plan["sufs"] if s["q"] == qi]
            for s in sufs:
                z = s["z"]
                p = torch.softmax(z.float(), -1).cpu().numpy()
                for i, j in enumerate(s["perm"]):
                    acc[j] += p[i]
            acc /= len(sufs)
            k = int(acc.argmax())
            res[q.id] = {
                "answer": q.options[k],
                "confidence": round(float(acc[k]), 6),
                "probs": {o: round(float(v), 6) for o, v in zip(q.options, acc)},
            }
        return res


def _chunks(plans, max_items, max_tokens, packed=True):
    """Split plans into passes bounded by item count and by prompt tokens."""
    def cost(p):
        pre, suf = len(p["pre"]["ids"]), [len(s["ids"]) for s in p["sufs"]]
        return pre + sum(suf) if packed else len(suf) * pre + sum(suf)

    cur, tok = [], 0
    for p in plans:
        c = cost(p)
        if cur and ((max_items and len(cur) >= max_items) or (max_tokens and tok + c > max_tokens)):
            yield cur
            cur, tok = [], 0
        cur.append(p)
        tok += c
    if cur:
        yield cur
