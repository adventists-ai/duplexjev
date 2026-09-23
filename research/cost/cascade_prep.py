# 用主 venv（有 soundfile/librosa）把级联基线要用的 30 段音频预先解码成 16k npy，venv_vllm 里直接读
import json, random, os, numpy as np, soundfile as sf
ROOT = "/data/exp01/exp03_train"; D = f"{ROOT}/artifacts/paper/cascade/audio_npy"; os.makedirs(D, exist_ok=True)
rows = [json.loads(l) for l in open(f"{ROOT}/artifacts/gender_real/test_manifest.jsonl")]
bins = {"2-3s": (2, 3), "4-6s": (4, 6), "8-12s": (8, 12)}; rng = random.Random(0); sel = []
for b, (lo, hi) in bins.items():
    c = [r for r in rows if lo <= r["duration_s"] <= hi]; rng.shuffle(c); sel += [(b, r) for r in c[:10]]
meta = []
for b, r in sel:
    a, sr = sf.read(r["audio"], dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:
        import librosa; a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    p = f"{D}/{r['id']}.npy"; np.save(p, a); meta.append({"bin": b, "id": r["id"], "lang": r["lang"], "duration_s": r["duration_s"], "npy": p})
json.dump(meta, open(f"{D}/meta.json", "w"), indent=1); print(len(meta))
