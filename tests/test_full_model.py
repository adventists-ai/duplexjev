"""Complete models (encoder + connector + LLM in one repo, like DuplexJev-4B-Para) load with the vendored code and
give the same answers as the connector checkpoint they were built from."""
import json

import pytest

from duplexjev import Decider, Question, default_table
from duplexjev.decider import FULL_ARCH, _is_full_model

QS = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
      Question("sound", "Besides speech, which sound can be heard in this audio?", ["laughter", "a sigh", "None of these"])]


def test_default_table_has_para_rows():
    for lang in ("en", "zh"):
        t = default_table(lang)
        assert len(t) == 10 and [q.id for q in t][-2:] == ["sound", "style"]


def test_full_model_detection(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [FULL_ARCH], "model_type": "ultravox"}))
    assert _is_full_model(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UltravoxModel"], "model_type": "ultravox"}))
    assert not _is_full_model(str(tmp_path))


def test_full_model_matches_connector(decider, clips, tmp_path):
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    m = decider.model
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}  # clone: tied embeddings
    save_file(sd, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    c = m.config.to_dict()
    for k in ("text_model_id", "audio_model_id", "auto_map", "_name_or_path"):
        c.pop(k, None)
    c["architectures"] = [FULL_ARCH]
    c["text_config"] = m.language_model.config.to_dict()
    ac = transformers.WhisperConfig.from_pretrained("openai/whisper-tiny").to_dict()
    ac["_name_or_path"] = "openai/whisper-tiny"  # selects the Whisper encoder class, as in a whisper-based full model
    c["audio_config"] = ac
    (tmp_path / "config.json").write_text(json.dumps(c))
    decider.tok.save_pretrained(tmp_path)
    transformers.WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny").save_pretrained(tmp_path)
    (tmp_path / "processor_config.json").write_text(json.dumps(
        {"encoder_ds_factor": int(getattr(decider.processor, "encoder_ds_factor", 2)), "stack_factor": c["stack_factor"]}))
    full = Decider.from_pretrained(str(tmp_path), device="cpu", dtype=m.dtype)
    for clip in clips:
        a, b = decider.decide(clip, QS), full.decide(clip, QS)
        diff = max(abs(a[q.id]["probs"][o] - b[q.id]["probs"][o]) for q in QS for o in q.options)
        assert diff < 1e-3, diff
