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


# ===================== M3：长共享上下文 + 打包读出（实现 B：单行 + 块对角 4D mask） =====================
# 上下文长度取 已上线车载助手的三类判断调用的中位输入 token：intent 530 / function_check 1471 / quick_response 5020。
# 上下文是合成的工具目录文本（不含任何线上数据），只用于测耗时与一致性，不评准确率。
OUT3 = f"{OUTD}/{LINE}_m3{'_smoke' if SMOKE else ''}.json"

def make_ctx(T):
    lines = ["You are an in-car voice assistant. Available tools and policies:"]
    k = 0
    while True:
        lines.append(f"- tool_{k:03d}(zone, level, duration): controls vehicle function group {k % 37}; "
                     f"call it only when the driver explicitly asks, confirm safety-critical actions, "
                     f"and never act while the vehicle is reversing. 车辆功能 {k}，仅在用户明确要求时调用。")
        k += 1
        ids = tok("\n".join(lines))["input_ids"]
        if len(ids) >= T:
            return tok.decode(ids[:T])

def prep_ctx(host_audio, qitems, ctx):
    encs, tids = [], []
    for it in qitems:
        text, slots, _ = protocol_s_pack(it, SEED, condition="audio")
        labels = list(slots); gate_single_tokens(tok, labels, it["id"])
        enc, _ = encode_one(processor, tok, (ctx + "\n\n" + text) if ctx else text, host_audio)
        encs.append(enc); tids.append(option_token_ids(tok, labels, it["id"]))
    return encs, tids

@torch.no_grad()
def run_packed(encs, tids, P):
    """前缀（模板+上下文+音频等公共部分）算一次；N 个后缀拼成一行，块对角 mask：
    每个后缀只看前缀和自己（因果），位置编码各自从 P 开始。KV 只有 P + sum(L_i)。"""
    base = {k: v for k, v in encs[0].items() if hasattr(v, "shape")}
    pre = dict(base); pre["input_ids"] = base["input_ids"][:, :P]; pre["attention_mask"] = base["attention_mask"][:, :P]
    pre = _to_device(pre, model, DEV)
    cache = model(**pre, use_cache=True).past_key_values
    sufs = [e["input_ids"][0, P:] for e in encs]; lens = [len(s) for s in sufs]; T = sum(lens)
    ids = torch.cat(sufs).unsqueeze(0).to(DEV)
    pos = torch.cat([P + torch.arange(l) for l in lens]).unsqueeze(0).to(DEV)
    neg = torch.finfo(model.dtype).min
    mask = torch.full((T, P + T), neg, dtype=model.dtype, device=DEV)
    mask[:, :P] = 0
    o = 0; lastpos = []
    for l in lens:
        blk = torch.triu(torch.full((l, l), neg, dtype=model.dtype, device=DEV), 1)
        mask[o:o + l, P + o:P + o + l] = blk
        o += l; lastpos.append(o - 1)
    out = lm(input_ids=ids, attention_mask=mask[None, None], position_ids=pos, past_key_values=cache, use_cache=False)
    last = out.logits[0, torch.tensor(lastpos, device=DEV)]
    return probs_from(last, tids)

rep3 = {"ckpt": CKPT, "line": LINE, "impl": "B: prefix KV once + all suffixes packed in one row, block-diagonal 4D mask",
        "ctx_note": "synthetic tool catalogue; lengths = median input tokens (deployed in-car assistant) of intent / function_check / quick_response",
        "m3_equiv": [], "m3": []}
bydur = sorted([it for it in items if it["t_ms"]], key=lambda it: int(it["t_ms"]))
host = bydur[len(bydur) // 2]; a = load_audio(host["audio"], host["t_ms"])
CTXS = [0, 530, 1471, 5020]
# 一致性：每个上下文长度，10 题，逐题单跑 vs 打包
for C in CTXS:
    ctx = make_ctx(C) if C else ""
    encs, tids = prep_ctx(a, items[:10], ctx); P = lcp(encs)
    ps = run_single(encs, tids); pk = run_packed(encs, tids, P)
    d = [float(np.abs(x - z).max()) for x, z in zip(ps, pk)]
    ag = sum(int(x.argmax() == z.argmax()) for x, z in zip(ps, pk))
    r = {"ctx": C, "prefix_len": P, "n": 10, "argmax_agree": ag, "max_abs_dp": round(max(d), 5), "median_abs_dp": round(float(np.median(d)), 5)}
    rep3["m3_equiv"].append(r); print("M3EQ", r, flush=True); json.dump(rep3, open(OUT3, "w"), indent=1)
    torch.cuda.empty_cache()
NS = [1, 10] if SMOKE else [1, 5, 10, 20, 50, 100, 200]
for C in (CTXS[:2] if SMOKE else CTXS):
    ctx = make_ctx(C) if C else ""
    for N in NS:
        qs = [items[i % len(items)] for i in range(N)]
        encs, tids = prep_ctx(a, qs, ctx)
        P = lcp(encs) if N > 1 else encs[0]["input_ids"].shape[1] - 1
        tot_rep = sum(e["input_ids"].shape[1] for e in encs)
        row = {"ctx": C, "n": N, "prefix_len": P, "suffix_mean_len": round(float(np.mean([e["input_ids"].shape[1] - P for e in encs])), 1)}
        try:
            row["single_ms"] = timed(lambda: run_single(encs[:1], tids[:1]))
            row["packed_ms"] = timed(lambda: run_packed(encs, tids, P))
            row["repeat_ms"] = timed(lambda: run_repeat(encs, tids)) if tot_rep <= 60000 else None
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); row["oom"] = True
        row["sequential_ms_est"] = round(row.get("single_ms", 0) * N, 1)
        row["tokens_repeat"] = tot_rep; row["tokens_packed"] = P + sum(e["input_ids"].shape[1] - P for e in encs)
        rep3["m3"].append(row); print("M3", row, flush=True); json.dump(rep3, open(OUT3, "w"), indent=1)
        torch.cuda.empty_cache()
print("M3_DONE", flush=True)
