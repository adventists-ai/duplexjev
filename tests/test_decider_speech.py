"""Speech path: answers must not depend on which other clips share the batch.

Needs network and a few hundred MB (Qwen3-0.6B, whisper-tiny, Ultravox code). Run with DUPLEXJEV_SPEECH_TESTS=1.
"""
import json
import os
import shutil

import numpy as np
import pytest

if not os.environ.get("DUPLEXJEV_SPEECH_TESTS"):
    pytest.skip("set DUPLEXJEV_SPEECH_TESTS=1 to run", allow_module_level=True)

from duplexjev import Decider, Question  # noqa: E402


@pytest.fixture(scope="module")
def speech_decider(tmp_path_factory):
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    src, dst = tmp_path_factory.mktemp("uvx_src"), tmp_path_factory.mktemp("uvx_tiny")
    for f in ["ultravox_config.py", "ultravox_model.py", "ultravox_processing.py", "ultravox_pipeline.py",
              "ultravox_tokenizer.py", "config.json"]:
        shutil.copy(hf_hub_download("fixie-ai/ultravox-v0_6-qwen-3-32b", f), src / f)
    c = json.load(open(src / "config.json"))
    c.pop("audio_config", None)
    c.update(audio_model_id="openai/whisper-tiny", text_model_id="Qwen/Qwen3-0.6B", hidden_size=512, torch_dtype="float32")
    json.dump(c, open(src / "config.json", "w"))
    transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(src)
    cfg = transformers.AutoConfig.from_pretrained(src, trust_remote_code=True)
    torch.manual_seed(0)
    transformers.AutoModel.from_config(cfg, trust_remote_code=True).save_pretrained(dst)
    for p in src.iterdir():
        if p.suffix != ".safetensors":
            shutil.copy(p, dst / p.name)
    return Decider.from_pretrained(str(dst), device="cpu")


def test_speech_batch_invariance(speech_decider):
    rng = np.random.default_rng(0)
    clips = [(rng.standard_normal(int(16000 * s)) * 0.1).astype(np.float32) for s in (1.3, 2.7, 4.1)]
    Q = [Question("turn", "Has the user finished the turn?", ["finished", "not finished"]),
         Question("gender", "Speaker gender", ["female", "male"])]
    together = speech_decider.decide(clips, Q)
    batch = speech_decider.decide(clips, Q, mode="batch")
    alone = [speech_decider.decide([c], Q)[0] for c in clips]
    for other in (batch, alone):
        for i in range(3):
            for q in ("turn", "gender"):
                for o, p in together[i][q]["probs"].items():
                    assert abs(p - other[i][q]["probs"][o]) < 1e-3
