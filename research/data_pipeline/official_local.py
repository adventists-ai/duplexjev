"""把官方 v0.6 任务名指到本地 official/。

Wenet 走 Lhotse tar+jsonl。续写不读 Llama 列，等 Qwen3-32B 旁路。
"""
from __future__ import annotations

import os

from ultravox.data.configs import commonvoice
from ultravox.data.configs import covost2
from ultravox.data.configs import gigaspeech
from ultravox.data.configs import librispeech
from ultravox.data.configs import multilingual_librispeech
from ultravox.data.configs import musan
from ultravox.data.configs import peoplespeech
from ultravox.data.configs import wenetspeech

OFFICIAL = os.environ.get(
    "ML_ASR_OFFICIAL_ROOT",
    "/data/exp01/exp03_train/data/ml_asr/official",
)

PATH_MAP = {
    "fixie-ai/librispeech_asr": os.path.join(OFFICIAL, "librispeech_asr"),
    "fixie-ai/gigaspeech": os.path.join(OFFICIAL, "gigaspeech"),
    "fixie-ai/peoples_speech": os.path.join(OFFICIAL, "peoples_speech"),
    "fixie-ai/common_voice_17_0": os.path.join(OFFICIAL, "common_voice_17_0"),
    "fixie-ai/covost2": os.path.join(OFFICIAL, "covost2"),
    "fixie-ai/multilingual_librispeech": os.path.join(
        OFFICIAL, "multilingual_librispeech"
    ),
    "facebook/multilingual_librispeech": os.path.join(
        OFFICIAL, "multilingual_librispeech"
    ),
    "fixie-ai/wenetspeech": os.path.join(OFFICIAL, "wenetspeech"),
    "fixie-ai/musan-segments-v2": os.path.join(OFFICIAL, "musan"),
    "huggymissaggy/musan-segments-v2": os.path.join(OFFICIAL, "musan"),
    "fixie-ai/common_voice_17_0-musan": os.path.join(
        OFFICIAL, "common_voice_17_0-musan"
    ),
}

MODULES = (
    librispeech,
    gigaspeech,
    peoplespeech,
    commonvoice,
    covost2,
    multilingual_librispeech,
    musan,
    wenetspeech,
)


def apply(dataset_map: dict) -> None:
    from ultravox.data.registry import register_datasets

    for mod in MODULES:
        keep = []
        for cfg in mod.configs:
            dataset_map.pop(cfg.name, None)
            if cfg.path in PATH_MAP:
                cfg.path = PATH_MAP[cfg.path]
            keep.append(cfg)
        register_datasets(keep)
