import pytest, torch


@pytest.fixture(scope="session")
def tiny_decider():
    """A randomly initialised 2-layer Qwen3 with the real Qwen3 tokenizer: fast, and exercises the real code paths."""
    transformers = pytest.importorskip("transformers")
    from duplexjev import Decider

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer download failed: {e}")
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).float()
    return Decider(model, tok, device="cpu")
