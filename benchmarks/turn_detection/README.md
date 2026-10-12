# Turn-detection benchmarks: LiveKit eot-bench and Pipecat Smart Turn

Two public benchmarks, run against a local `duplexjev gateway` (the same HTTP API the LiveKit and Pipecat plugins use).
P(finished) is the probability of the first option of the turn question ("finished, the assistant can reply" /
"话说完了，可以接话"), exactly as in `livekit-plugins-duplexjev` and `pipecat-duplexjev`. Nothing here needs the
transcript.

## 0. Serve the model (one GPU)

```bash
pip install "vllm[audio]>=0.29" duplexjev-vllm "duplexjev[server]>=0.4.1"
vllm serve adventists-ai/DuplexJev-4B-Para --max-model-len 4096 --port 8000      # also try -32B-Turn / -32B-Para
duplexjev gateway --vllm http://localhost:8000/v1 --port 8420 --rate 1000000     # lift the default 30 req/min limit
export DUPLEXJEV_GATEWAY=http://localhost:8420
```

## 1. LiveKit eot-bench (public leaderboard: https://livekit.com/benchmarks/eot-bench)

5,454 real human-to-agent turns in 14 languages (we report `en`: 400 turns and `zh`: 344 turns). Every pause of
≥100 ms is a decision point, scored every 0.1 s with the audio cut at that moment; the last pause of a turn is the
true end of turn. Headline metrics: false-cutoff rate when the agent may wait at most 300 ms / 600 ms, and mean wait
when it may cut the user off at most 5% / 10% of the time.

```bash
git clone https://github.com/livekit/eot-bench && cd eot-bench && uv sync && uv pip install requests scipy
export PYTHONPATH=/path/to/duplexjev/benchmarks/turn_detection DUPLEXJEV_MODEL_LABEL="DuplexJev-4B-Para"
for L in en zh; do
  .venv/bin/eot-harness predict --path livekit/eot-bench-data --name $L --split validation \
    --adapter eot_bench.duplexjev_adapter:DuplexJevAdapter --batch-size 64 --output-dir output
done
R=output/livekit__eot-bench-data__validation__min_silence_100ms
for P in $R/*/duplex*/predictions.parquet; do .venv/bin/eot-harness compute-metrics --predictions $P --output-dir $(dirname $P)/metrics; done
.venv/bin/eot-harness compare-models $R/en      # vs LiveKit Turn Detector v1, SmartTurn v3.2, Baton, VAD (results in the repo)
```

English is 11,191 decision points (tested end to end with a mock gateway: ~3 min at 67 points/s on CPU; a GPU
gateway with `DUPLEXJEV_CONCURRENCY=32` should be faster). Current English board (README of eot-bench):
LiveKit Turn Detector v1 9.9% / 4.5% false cutoffs at 300 / 600 ms; JoinIn Baton 12.3% / 4.8%; SmartTurn v3.2
35.2% / 14.8%; VAD 55.6% / 21.7%.

Leaderboard: third-party models appear as an adapter file + committed results under `output/` + an entry in
`scripts/export_web_data.py`. No written submission process -- open an issue first.

## 2. Pipecat Smart Turn test set

`pipecat-ai/smart-turn-data-v3.2-test`: 31,527 single-utterance clips, complete vs incomplete (balanced), 23
languages, 81% synthetic. Smart Turn v3.2 (GPU): 93.71% accuracy overall, 94.71% English, 90.53% Chinese, threshold 0.5.

```bash
pip install datasets soundfile scipy scikit-learn requests
PYTHONPATH=/path/to/duplexjev/benchmarks/turn_detection \
  python -m smart_turn.eval_smart_turn --langs eng,zho --out st_results          # --limit 200 for a quick check
```

Streams the dataset (no 4.8 GB download), writes per-clip scores to `st_results/scores.csv` (resumable) and
`report.json`: accuracy / precision / recall / F1 at 0.5 (comparable to Smart Turn), AUC, and accuracy at the best
threshold (tuned on the test set itself -- say so if quoted).

## What to send back

`output/.../en|zh/duplex*/metrics/summary.json` and `st_results/report.json`, for 4B-Para and 32B-Turn.
