"""前缀共享读出（EXP_prefix_sharing.md 实现 A：两段式 KV cache）+ M1 一致性 + M2 耗时曲线。
用法：python prefix_share.py A|B [--smoke]
- 同一段音频上的 N 道题：前缀 = 所有题 input_ids 的最长公共前缀（自动包含模板+音频+State），
  只算一次；KV 复制 N 份；N 个后缀右 pad 一次前向读出。题与题互不可见（各自一行）。
- 对照：逐题单跑（batch=1）、现有 batch_repeat（整段复制 N 行）。
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
ROOT = "/data/exp01/exp03_train"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "jev_qwen"))
from jev_qwen.readout import encode_one, forward_last_logits, load_audio, option_token_ids, pad_batch, protocol_s_pack, _to_device  # noqa
from jev_qwen.render_letter import gate_single_tokens  # noqa
from jev_qwen.run_decision import load_pack  # noqa
import run_bench as RB  # noqa

LINE = sys.argv[1]; SMOKE = "--smoke" in sys.argv
CKPT = {"A": f"{ROOT}/ultravox/runs/ml-asr-v2-xattn-2026-09-23/checkpoint-32000",
        "B": f"{ROOT}/ultravox/runs/ml-asr-v1-h18-2026-09-23/checkpoint-32000"}[LINE]
OUTD = f"{ROOT}/artifacts/paper/prefix_share"; os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/{LINE}{'_smoke' if SMOKE else ''}.json"
DEV = "cuda:0"; SEED = 0
pack = load_pack(f"{ROOT}/artifacts/jev_qwen/holdout_cg.json"); items = pack["items"]
model, processor = RB.load_model(CKPT, DEV); tok = processor.tokenizer
lm = model.language_model

def sync(): torch.cuda.synchronize()

def prep(host_audio, qitems):
    encs, tids = [], []
    for it in qitems:
        text, slots, _ = protocol_s_pack(it, SEED, condition="audio")
        labels = list(slots); gate_single_tokens(tok, labels, it["id"])
        enc, _ = encode_one(processor, tok, text, host_audio)
        encs.append(enc); tids.append(option_token_ids(tok, labels, it["id"]))
    return encs, tids

def lcp(encs):
    ids = [e["input_ids"][0].tolist() for e in encs]; P = min(len(x) for x in ids)
    for j in range(P):
        c = ids[0][j]
        if any(x[j] != c for x in ids[1:]): return j
    return P - 1  # 全部相同时留最后一个 token 给后缀

def probs_from(last, tids):
    return [F.softmax(last[i].float()[torch.tensor(t, device=last.device)], -1).cpu().numpy() for i, t in enumerate(tids)]

@torch.no_grad()
def run_single(encs, tids):
    out = []
    for e, t in zip(encs, tids):
        last = forward_last_logits(model, _to_device(dict(e), model, DEV)); out += probs_from(last, [t])
    return out

@torch.no_grad()
def run_repeat(encs, tids):
    last = forward_last_logits(model, pad_batch(encs, DEV, model)); return probs_from(last, tids)

@torch.no_grad()
def run_shared(encs, tids, P=None):
    P = lcp(encs) if P is None else P
    base = {k: v for k, v in encs[0].items() if hasattr(v, "shape")}
    pre = dict(base); pre["input_ids"] = base["input_ids"][:, :P]; pre["attention_mask"] = base["attention_mask"][:, :P]
    pre = _to_device(pre, model, DEV)
    out = model(**pre, use_cache=True); cache = out.past_key_values
    N = len(encs); cache.batch_repeat_interleave(N)
    sufs = [e["input_ids"][0, P:] for e in encs]; L = max(len(s) for s in sufs)
    ids = torch.zeros(N, L, dtype=torch.long); m = torch.zeros(N, L, dtype=torch.long)
    for i, s in enumerate(sufs): ids[i, :len(s)] = s; m[i, :len(s)] = 1
    ids, m = ids.to(DEV), m.to(DEV)
    attn = torch.cat([torch.ones(N, P, dtype=torch.long, device=DEV), m], 1)
    pos = (P + torch.arange(L, device=DEV)).unsqueeze(0).expand(N, L)
    o2 = lm(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=False)
    lastidx = m.sum(1) - 1
    last = o2.logits[torch.arange(N, device=DEV), lastidx]
    return probs_from(last, tids), P, float(np.mean([len(s) for s in sufs]))

def timed(fn, reps=3):
    fn(); sync(); ws = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ws.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ws)), 1)

rep = {"ckpt": CKPT, "line": LINE, "impl": "A: prefix KV once + batch_repeat_interleave + one suffix forward", "m1": {}, "m2": []}
# ---------- M1 一致性 ----------
qpool = items[: (10 if SMOKE else 40)]
hosts = [it for it in items if it["family"] == "content"][:(2 if SMOKE else 12)] + [it for it in items if it["family"] == "gender"][:(1 if SMOKE else 8)]
dmax, dall, agree, flips, n = 0.0, [], 0, [], 0
for h in hosts:
    a = load_audio(h["audio"], h["t_ms"])
    qs = qpool[:10]
    encs, tids = prep(a, qs)
    ps = run_single(encs, tids); pr = run_repeat(encs, tids); sh, P, sl = run_shared(encs, tids)
    for q, x, y, z in zip(qs, ps, pr, sh):
        d = float(np.abs(x - z).max()); dall.append(d); dmax = max(dmax, d); n += 1
        top2 = np.sort(x)[-2:]; margin = float(top2[1] - top2[0])
        if int(x.argmax()) == int(z.argmax()): agree += 1
        else: flips.append({"host": h["id"], "q": q["id"], "margin": round(margin, 4), "dp": round(d, 4)})
    torch.cuda.empty_cache()
rep["m1"] = {"n": n, "argmax_agree": agree, "agree_rate": round(agree / n, 4), "max_abs_dp": round(dmax, 5),
             "median_abs_dp": round(float(np.median(dall)), 6), "flips": flips}
print("M1", json.dumps(rep["m1"])[:400], flush=True)
json.dump(rep, open(OUT, "w"), indent=1)
# ---------- M2 耗时随 N ----------
bydur = sorted([it for it in items if it["t_ms"]], key=lambda it: int(it["t_ms"]))
pick = {"short": bydur[len(bydur) // 10], "mid": bydur[len(bydur) // 2], "long": bydur[-3]}
NS = [1, 2, 5, 10] if SMOKE else [1, 2, 5, 10, 20, 50, 100]
for tag, h in pick.items():
    a = load_audio(h["audio"], h["t_ms"])
    for N in NS:
        qs = [items[i % len(items)] for i in range(N)]
        encs, tids = prep(a, qs)
        try:
            P = lcp(encs) if N > 1 else encs[0]["input_ids"].shape[1] - 1
            row = {"dur_bin": tag, "t_ms": h["t_ms"], "n": N, "prefix_len": P,
                   "suffix_mean_len": round(float(np.mean([e["input_ids"].shape[1] - P for e in encs])), 1),
                   "single_ms": timed(lambda: run_single(encs[:1], tids[:1])),
                   "repeat_ms": timed(lambda: run_repeat(encs, tids)),
                   "shared_ms": timed(lambda: run_shared(encs, tids, P))}
            row["sequential_ms_est"] = round(row["single_ms"] * N, 1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); row = {"dur_bin": tag, "n": N, "oom": True}
        rep["m2"].append(row); print("M2", row, flush=True); json.dump(rep, open(OUT, "w"), indent=1)
        torch.cuda.empty_cache()
print("PREFIX_DONE", flush=True)
