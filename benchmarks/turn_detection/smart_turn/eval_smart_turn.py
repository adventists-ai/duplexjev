"""DuplexJev on Pipecat's Smart Turn test set (pipecat-ai/smart-turn-data-v3.2-test, 31,527 clips, cc-by-4.0).

Each clip is one utterance labelled complete (endpoint_bool=True) or incomplete. Smart Turn v3.2 scores 93.71%
accuracy overall (94.71% English, 90.53% Chinese) at a 0.5 threshold. We report the same: accuracy, precision,
recall, F1 at threshold 0.5 on P(finished), per language, plus AUC (threshold-free) and accuracy at the best
threshold (tuned on the same data -- label it as such if you quote it).

    export DUPLEXJEV_GATEWAY=http://localhost:8420
    PYTHONPATH=/path/to/duplexjev/benchmarks/turn_detection python -m smart_turn.eval_smart_turn --langs eng,zho --out st_results
    # --langs all  for every language (DuplexJev is trained on English and Chinese only)
    # --limit N    for a quick check

The dataset is streamed (4.8 GB in full); per-clip scores go to <out>/scores.csv so a run can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import p_finished, resample

LANG_WORDING = {"zho": "zh", "cmn": "zh", "zh": "zh"}  # everything else is asked in English


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = p >= thr
    tp = int(np.sum(pred & y)); tn = int(np.sum(~pred & ~y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    n = len(y); prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    out = {"n": n, "accuracy": (tp + tn) / max(n, 1), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / max(prec + rec, 1e-9),
           # Smart Turn's benchmark reports FP and FN as shares of *all* samples
           "fp_share": fp / max(n, 1), "fn_share": fn / max(n, 1)}
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < n else None
    except ImportError:
        out["auc"] = None
    ths = np.linspace(0.02, 0.98, 49)
    accs = [float(np.mean((p >= t) == y)) for t in ths]
    out["best_threshold"] = float(ths[int(np.argmax(accs))]); out["accuracy_at_best_threshold"] = max(accs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pipecat-ai/smart-turn-data-v3.2-test")
    ap.add_argument("--split", default="train")
    ap.add_argument("--langs", default="eng,zho", help="comma-separated dataset language codes, or 'all'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="st_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from datasets import load_dataset, Audio
    ds = load_dataset(a.dataset, split=a.split, streaming=True).cast_column("audio", Audio(decode=False))  # decode with soundfile: no torchcodec needed
    langs = None if a.langs == "all" else set(a.langs.split(","))
    path = os.path.join(a.out, "scores.csv"); done = {}
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                done[r["id"]] = r
    fh = open(path, "a", newline=""); w = csv.DictWriter(fh, fieldnames=["id", "language", "dataset", "synthetic", "label", "p_finished"])
    if not done:
        w.writeheader()

    def job(row):
        import io as _io, soundfile as sf
        arr, sr = sf.read(_io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        x = resample(arr, int(sr))
        return row, p_finished(x, LANG_WORDING.get(str(row["language"]).lower(), "en"))

    pool = ThreadPoolExecutor(max_workers=a.workers); pending = []; n = 0; t0 = time.time()

    def drain(block: bool) -> None:
        while pending and (block or pending[0].done()):
            row, p = pending.pop(0).result()
            w.writerow({"id": row["id"], "language": row["language"], "dataset": row.get("dataset", ""),
                        "synthetic": row.get("synthetic", ""), "label": int(bool(row["endpoint_bool"])), "p_finished": f"{p:.6f}"})

    for row in ds:
        if langs and str(row["language"]) not in langs:
            continue
        if str(row["id"]) in done:
            continue
        pending.append(pool.submit(job, row)); n += 1
        if len(pending) >= a.workers * 4:
            drain(False)
            if len(pending) >= a.workers * 4:
                drain(True)
        if n % 500 == 0:
            fh.flush(); print(f"{n} clips  {n / (time.time() - t0):.1f}/s", flush=True)
        if a.limit and n >= a.limit:
            break
    drain(True); fh.close()

    rows = list(csv.DictReader(open(path)))
    report = {}
    for key in ["all"] + sorted({r["language"] for r in rows}):
        sel = rows if key == "all" else [r for r in rows if r["language"] == key]
        y = np.array([r["label"] == "1" for r in sel]); p = np.array([float(r["p_finished"]) for r in sel])
        report[key] = metrics(y, p)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"{'lang':6} {'n':>6} {'acc@0.5':>8} {'F1':>6} {'AUC':>6} {'acc@best':>9} {'thr':>5}")
    for k, m in report.items():
        auc = f"{m['auc']:.3f}" if m["auc"] is not None else "  -  "
        print(f"{k:6} {m['n']:6d} {m['accuracy']*100:7.2f}% {m['f1']:6.3f} {auc:>6} {m['accuracy_at_best_threshold']*100:8.2f}% {m['best_threshold']:5.2f}")
    print("Smart Turn v3.2 (GPU) on the same set: 93.71% overall, 94.71% English, 90.53% Chinese (threshold 0.5).")


if __name__ == "__main__":
    main()
    os._exit(0)  # the streaming dataset leaves threads behind; skip their noisy teardown
