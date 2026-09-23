#!/usr/bin/env python3
"""VoiceDataset：10 包切分 + 丢掉 C1 Wenet + 续写改用 32B 旁路。"""
from pathlib import Path

p = Path("/data/exp01/exp03_train/ultravox/ultravox/data/datasets.py")
s = p.read_text()
if "_ml_asr_pack_keep" in s and "ML_ASR_N_PACKS" in s and "n_packs = 10" not in s:
    s = s.replace(
        'n_packs = int(os.environ.get("ML_ASR_N_PACKS", "4"))',
        'n_packs = int(os.environ.get("ML_ASR_N_PACKS", "10"))',
    )
    p.write_text(s)
    print("updated default N_PACKS=10")
if "_ml_asr_pack_keep" in s:
    print("pack filter already present")
    raise SystemExit(0)

hook = '''    def _ml_asr_pack_keep(self, row) -> bool:
        pack = os.environ.get("ML_ASR_PACK")
        if pack is None or pack == "":
            return True
        n_packs = int(os.environ.get("ML_ASR_N_PACKS", "10"))
        pid = int(pack)
        key = ""
        if isinstance(row, dict):
            audio = row.get(getattr(self, "_config", None) and getattr(self._config, "audio_field", None) or "audio")
            if isinstance(audio, dict):
                key = audio.get("path") or ""
            if not key:
                key = str(row.get("id") or row.get("text") or "")
        if not key:
            return True
        digest = hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=8).hexdigest()
        return int(digest, 16) % n_packs == pid

'''
s = s.replace(
    "    def __len__(self):\n        return self._length\n",
    "    def __len__(self):\n        return self._length\n\n" + hook,
    1,
)
old = """        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
"""
new = """        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            if not self._ml_asr_pack_keep(row):
                continue
            actual_length += 1
            sample = self._get_sample(row)
"""
if old not in s:
    raise SystemExit("iter loop pattern missing")
s = s.replace(old, new, 1)
p.write_text(s)
print("patched pack filter")
