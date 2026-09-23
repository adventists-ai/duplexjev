#!/usr/bin/env python3
"""检查某个包的续写覆盖率：本包 _cont + 共用小子集 _shared_cont 里的条目，有多少在 32B 续写库里。
用法：python cont_coverage.py <pack> [min_ratio]   不达标时退出码 1。"""
import glob, json, os, sys
sys.path.insert(0, "/data/exp01/exp03_train/scripts_ml_asr")
os.environ.setdefault("ML_ASR_CONT_DIR", "/data/exp01/exp03_train/artifacts/ml_asr/cont_qwen32b")
import continuation_store as CS
OUT = "/data/exp01/exp03_train/data/ml_asr/packs100"
k = int(sys.argv[1]); need = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
files = glob.glob(os.path.join(OUT, "p%03d" % k, "_cont", "*.jsonl")) + glob.glob(os.path.join(OUT, "_shared_cont", "*.jsonl"))
tot = hit = 0
for f in files:
    for line in open(f, encoding="utf-8"):
        p = json.loads(line)["path"]
        tot += 1
        if CS.get(p):
            hit += 1
r = hit / max(tot, 1)
print("pack %03d continuation coverage %d/%d = %.4f (need %.2f)" % (k, hit, tot, r, need))
sys.exit(0 if r >= need else 1)
