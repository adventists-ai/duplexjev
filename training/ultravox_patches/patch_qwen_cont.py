#!/usr/bin/env python3
"""continuation 任务：只用 Qwen3-32B 旁路；没有就跳过，绝不回落到 Llama-8B 列。"""
from pathlib import Path

p = Path("/data/exp01/exp03_train/ultravox/ultravox/data/datasets.py")
s = p.read_text()
if "_ml_asr_qwen_cont" in s:
    print("qwen cont override already present")
    raise SystemExit(0)

helper = '''    def _ml_asr_qwen_cont(self, row):
        """覆盖/删除 parquet 里的 Llama continuation。"""
        if not isinstance(row, dict):
            return row
        name = getattr(getattr(self, "_config", None), "name", "") or ""
        if not name.endswith("-continuation"):
            return row
        audio = row.get(getattr(self._config, "audio_field", None) or "audio")
        path = audio.get("path") if isinstance(audio, dict) else ""
        # 与 split100 的键一致：audio.path 为空（LibriSpeech）时依次用 path / file / id 列
        path = path or row.get("path") or row.get("file") or row.get("id") or ""
        try:
            from ultravox.data.configs.continuation_store import get as _get
        except Exception:
            from continuation_store import get as _get
        qwen = _get(path or "")
        row = dict(row)
        if qwen:
            row["continuation"] = qwen
        else:
            row.pop("continuation", None)
        return row

'''
s = s.replace(
    "    def __len__(self):\n        return self._length\n",
    "    def __len__(self):\n        return self._length\n\n" + helper,
    1,
)
# apply before _get_sample
old = "            sample = self._get_sample(row)"
new = "            row = self._ml_asr_qwen_cont(row)\n            if getattr(getattr(self, \"_config\", None), \"name\", \"\").endswith(\"-continuation\") and not (isinstance(row, dict) and row.get(\"continuation\")):\n                continue\n            sample = self._get_sample(row)"
if old not in s:
    raise SystemExit("no _get_sample call")
s = s.replace(old, new, 1)
p.write_text(s)
print("patched qwen continuation override")
