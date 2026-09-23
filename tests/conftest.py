"""A tiny speech checkpoint in Ultravox format, built locally: whisper-tiny encoder + random projector + a random
2-layer Qwen3 LLM with the real Qwen3 tokenizer. Fast on CPU and exercises the real code paths.
Needs network once (Qwen3 tokenizer, whisper-tiny, Ultravox model code)."""
import json
import shutil

import pytest
import torch


@pytest.fixture(scope="session")
def decider(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    from duplexjev import Decider

    root = tmp_path_factory.mktemp("tiny")
    llm, src, ckpt = root / "llm", root / "src", root / "ckpt"
    for d in (llm, src, ckpt):
        d.mkdir()
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        for f in ["ultravox_config.py", "ultravox_model.py", "ultravox_processing.py", "ultravox_pipeline.py",
                  "ultravox_tokenizer.py", "config.json"]:
            shutil.copy(hub.hf_hub_download("fixie-ai/ultravox-v0_6-qwen-3-32b", f), src / f)
    except Exception as e:  # offline
        pytest.skip(f"download failed: {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(llm)
    tok.save_pretrained(llm)
    c = json.load(open(src / "config.json"))
    c.pop("audio_config", None)
    c.update(audio_model_id="openai/whisper-tiny", text_model_id=str(llm), hidden_size=128, torch_dtype="float32")
    json.dump(c, open(src / "config.json", "w"))
    tok.save_pretrained(src)
    ucfg = transformers.AutoConfig.from_pretrained(src, trust_remote_code=True)
    transformers.AutoModel.from_config(ucfg, trust_remote_code=True).save_pretrained(ckpt)
    for p in src.iterdir():
        if p.suffix != ".safetensors":
            shutil.copy(p, ckpt / p.name)
    return Decider.from_pretrained(str(ckpt), device="cpu")


@pytest.fixture(scope="session")
def clips():
    import numpy as np

    rng = np.random.default_rng(0)
    return [(rng.standard_normal(int(16000 * s)) * 0.1).astype(np.float32) for s in (1.3, 2.7, 4.1)]
