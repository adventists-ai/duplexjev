"""Score-point run of eot-bench for DuplexJev (official eot-bench scoring uses only the prediction 0.2 s into each
pause -- `score_point` -- so this asks the model once per pause instead of every 0.1 s).

Writes a predictions.parquet with the same rows as a full harness run (the template run's grid); every grid point of a
span carries the score-point probability, which is the only value `compute-metrics` reads in score-point mode.

    PYTHONPATH=benchmarks/turn_detection:/path/to/eot-bench python -m eot_bench.run_scorepoints --lang en \
        --template output/.../en/smart_turn_audio_adapter__cb4f57229a --out output/.../en/duplexjev_4b_para
"""
from __future__ import annotations

import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import p_finished, resample

SCORE_POINT = 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--template", required=True, help="a harness run dir for the same language (for the row grid + manifest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="DuplexJev-4B-Para")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=2.1, help="seconds between request starts (rate limit)")
    a = ap.parse_args()
    from eot_harness.io import iter_transformed_row, load_hf_dataset
    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, "scorepoints.jsonl"); cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            d = json.loads(line); cache[(d["id"], d["span_index"])] = d["p"]
    ds = load_hf_dataset("livekit/eot-bench-data", a.lang, "validation")
    targets = []
    for row in ds:
        seen = set()
        for ex in iter_transformed_row(row, max_audio_sec=8.0):
            key = (ex["id"], int(ex["span_index"]))
            if key in seen or ex["silence_dur"] < SCORE_POINT - 1e-6:
                continue
            seen.add(key)
            if key not in cache:
                targets.append((key, ex["audio"]))
    print(f"{len(cache)} cached, {len(targets)} to score", flush=True)
    fh = open(cache_path, "a"); last = [0.0]; t0 = time.time()

    def job(item):
        key, audio = item
        wait = last[0] + a.min_interval - time.time()
        last[0] = max(time.time(), last[0] + a.min_interval)
        if wait > 0:
            time.sleep(wait)
        x = resample(np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"]))
        for attempt in range(6):
            try:
                return key, p_finished(x, "zh" if a.lang == "zh" else "en", retries=1)
            except Exception as e:  # 429 / network: back off
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"failed {key}")

    with ThreadPoolExecutor(a.workers) as pool:
        for n, (key, p) in enumerate(pool.map(job, targets), 1):
            cache[key] = p
            fh.write(json.dumps({"id": key[0], "span_index": key[1], "p": p}) + "\n"); fh.flush()
            if n % 50 == 0:
                print(f"{n}/{len(targets)}  {n / (time.time() - t0) * 60:.1f}/min", flush=True)
    fh.close()
    grid = pd.read_parquet(os.path.join(a.template, "predictions.parquet"))[["id", "language", "span_index", "timestamp", "silence_dur", "label"]].copy()
    grid["p_eot"] = [cache.get((i, int(s)), 0.0) if d >= SCORE_POINT - 1e-6 else 0.0
                     for i, s, d in zip(grid["id"], grid["span_index"], grid["silence_dur"])]
    grid = grid[["id", "language", "span_index", "timestamp", "silence_dur", "p_eot", "label"]]
    grid.to_parquet(os.path.join(a.out, "predictions.parquet"), index=False)
    man = json.load(open(os.path.join(a.template, "manifest.json")))
    run_id = os.path.basename(os.path.normpath(a.out))
    man.update({"model_run_id": run_id, "adapter": "eot_bench.duplexjev_adapter:DuplexJevAdapter", "adapter_id": a.label.lower(),
                "display_name": a.label, "score_point": SCORE_POINT, "note": "score-point run: one model call per pause at 0.2 s"})
    man["model"].update({"adapter": man["adapter"], "adapter_id": man["adapter_id"], "display_name": a.label, "score_point": SCORE_POINT})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("wrote", a.out, "spans scored:", len(cache))


if __name__ == "__main__":
    main()
