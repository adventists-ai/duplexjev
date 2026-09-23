#!/usr/bin/env python3
"""把 stack_factor 接到 TrainConfig → UltravoxConfig。只改 yaml 不会生效。"""
from pathlib import Path

BASE = Path("/data/exp01/exp03_train/ultravox/ultravox/training/config_base.py")
TYPES = Path("/data/exp01/exp03_train/ultravox/ultravox/training/model_types.py")


def patch_base() -> None:
    s = BASE.read_text()
    if "stack_factor:" in s:
        print("config_base already has stack_factor")
        return
    old = "    projector_ln_mid: bool = True\n"
    new = (
        "    projector_ln_mid: bool = True\n"
        "    # Whisper 50 Hz 默认拼 8 帧 → 6.25 Hz。Qwen3-ASR 是 12.5 Hz，拼 2 帧才回到 6.25 Hz。\n"
        "    stack_factor: int = 8\n"
    )
    if old not in s:
        raise SystemExit("projector_ln_mid not found")
    BASE.write_text(s.replace(old, new, 1))
    print("patched config_base.py")


def patch_types() -> None:
    s = TYPES.read_text()
    if "stack_factor=args.stack_factor" in s:
        print("model_types already passes stack_factor")
        return
    old = """            projector_ln_mid=args.projector_ln_mid,
            audio_token_index=ultravox_tokenizer.get_audio_token_id(
                self.text_tokenizer
            ),
        )"""
    new = """            projector_ln_mid=args.projector_ln_mid,
            stack_factor=args.stack_factor,
            audio_token_index=ultravox_tokenizer.get_audio_token_id(
                self.text_tokenizer
            ),
        )"""
    if old not in s:
        raise SystemExit("UltravoxConfig kwargs not found")
    TYPES.write_text(s.replace(old, new, 1))
    print("patched model_types.py")


if __name__ == "__main__":
    patch_base()
    patch_types()
