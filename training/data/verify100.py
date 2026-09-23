#!/usr/bin/env python3
"""复查 packs100：行数守恒、每包文件与统计一致、哈希归属正确、子集齐全、Wenet 守恒、与旧 10 包嵌套、训练加载器视角、续写清单。
结果写 packs100/VERIFY.json，并打印摘要。"""
import glob, json, os, random, sys
import multiprocessing as mp
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, "/data/exp01/exp03_train/scripts_ml_asr")
import split100 as S  # noqa

OUT = S.OUT
N = S.N
P = json.load(open(os.path.join(OUT, "plan.json")))
SHARED = set(P["shared_subsets"])
R = {"errors": []}


def err(msg):
    R["errors"].append(msg)


def check_file(f):
    out = {"rel": f["rel"], "errs": []}
    sp = os.path.join(OUT, "_stats", "parquet", S.safe(f["rel"]) + ".json")
    if not os.path.exists(sp):
        out["errs"].append("missing stats")
        return out
    st = json.load(open(sp))
    if st["rows"] != f["rows"]:
        out["errs"].append("rows %s != meta %s" % (st["rows"], f["rows"]))
    if S.subset_of(f["orel"]) in SHARED:
        for k in range(N):
            op = os.path.join(OUT, "p%03d" % k, f["orel"])
            if not (os.path.islink(op) and os.path.realpath(op) == os.path.realpath(f["src"])):
                out["errs"].append("shared link bad p%d" % k)
                break
        out["per_pack"] = None
        return out
    if sum(st["per_pack"]) != f["rows"]:
        out["errs"].append("sum per_pack %s != %s" % (sum(st["per_pack"]), f["rows"]))
    for k in range(N):
        op = os.path.join(OUT, "p%03d" % k, f["orel"])
        want = st["per_pack"][k]
        if want == 0:
            if os.path.exists(op):
                out["errs"].append("unexpected file p%d" % k)
            continue
        if not os.path.isfile(op):
            out["errs"].append("missing out p%d" % k)
            continue
        n = pq.ParquetFile(op).metadata.num_rows
        if n != want:
            out["errs"].append("p%d rows %s != %s" % (k, n, want))
    rng = random.Random(f["rel"])
    ks = [k for k in range(N) if st["per_pack"][k] > 0]
    for k in rng.sample(ks, min(2, len(ks))):
        op = os.path.join(OUT, "p%03d" % k, f["orel"])
        pf = pq.ParquetFile(op)
        b = next(pf.iter_batches(batch_size=64))
        keys, _ = S.row_keys(b, pf.schema_arrow.names, f["rel"], 0)
        bad = [x for x in keys if "#" not in x and S.pack_of(x) != k]
        if bad:
            out["errs"].append("hash mismatch p%d: %d" % (k, len(bad)))
    out["per_pack"] = st["per_pack"]
    return out


def main():
    with mp.Pool(64) as pool:
        res = pool.map(check_file, P["files"], chunksize=8)
    sub_pack = defaultdict(lambda: [0] * N)
    sub_total = Counter()
    for f, r in zip(P["files"], res):
        for e in r["errs"]:
            err("%s: %s" % (f["rel"], e))
        s = S.subset_of(f["orel"])
        sub_total[s] += f["rows"]
        if r.get("per_pack"):
            for k, v in enumerate(r["per_pack"]):
                sub_pack[s][k] += v
    fb = 0
    for sp in glob.glob(os.path.join(OUT, "_stats", "parquet", "*.json")):
        fb += json.load(open(sp)).get("fallback_keys", 0)
    R["parquet_files"] = len(P["files"])
    R["fallback_keys"] = fb
    empty = {s: [k for k in range(N) if v[k] == 0] for s, v in sub_pack.items()}
    empty = {s: ks for s, ks in empty.items() if ks}
    if empty:
        err("empty subset-in-pack: %s" % empty)
    R["subsets"] = {}
    for s in sorted(sub_total):
        if s in SHARED:
            R["subsets"][s] = {"total": sub_total[s], "shared": True}
        else:
            v = sub_pack[s]
            R["subsets"][s] = {"total": sub_total[s], "per_pack_min": min(v), "per_pack_max": max(v),
                               "expected": round(sub_total[s] / N, 1)}
    ws = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "_stats", "wenet", "*.json")))]
    from wenet_lhotse import _find_pairs
    pairs = _find_pairs(os.path.join(S.SRC, "wenetspeech"))
    covered = sum(len(w["pairs"]) for w in ws)
    wj = [0] * N
    wt = [0] * N
    for w in ws:
        for k in range(N):
            wj[k] += w["per_pack_jsonl"][k]
            wt[k] += w["per_pack_tar"][k]
    miss = sum(w["tar_members_without_text"] for w in ws)
    if covered != len(pairs):
        err("wenet groups cover %d/%d pairs" % (covered, len(pairs)))
    R["wenet"] = {"pairs": len(pairs), "covered": covered, "jsonl_total": sum(wj), "tar_total": sum(wt),
                  "tar_without_text": miss, "per_pack_min": min(wj), "per_pack_max": max(wj),
                  "tar_per_pack_min": min(wt), "tar_per_pack_max": max(wt)}
    for k in range(N):
        d = os.path.join(OUT, "p%03d" % k, "wenetspeech", "data")
        nj = len(glob.glob(os.path.join(d, "cuts_L_fixed.*.jsonl.gz")))
        nt = len(glob.glob(os.path.join(d, "cuts_L_fixed.*.tar.gz")))
        if nj != nt or nj == 0:
            err("wenet p%d: jsonl %d tar %d" % (k, nj, nt))
    old = "/data/exp01/exp03_train/artifacts/ml_asr/b2_packs"
    bad = tot = 0
    for i in range(1, 11):
        with open(os.path.join(old, "p%02d" % i, "audio_text.jsonl"), encoding="utf-8") as fh:
            for j, line in enumerate(fh):
                if j >= 20000:
                    break
                tot += 1
                if S.pack_of(json.loads(line)["path"]) % 10 != i - 1:
                    bad += 1
    R["nesting_old10"] = {"checked": tot, "mismatch": bad}
    if bad:
        err("nesting mismatch %d/%d" % (bad, tot))
    import data_audit_lib as A
    for k in (0, 1, 57, 99):
        rows = A.audit(os.path.join(OUT, "p%03d" % k))
        R["loader_p%03d" % k] = rows
        for row in rows:
            if row.get("flag"):
                err("loader p%03d %s: %s" % (k, row["name"], row["flag"]))
    cont = []
    for k in range(N):
        n = 0
        for p in glob.glob(os.path.join(OUT, "p%03d" % k, "_cont", "*.jsonl")):
            with open(p, "rb") as fh:
                n += sum(1 for _ in fh)
        cont.append(n)
    sh = 0
    for p in glob.glob(os.path.join(OUT, "_shared_cont", "*.jsonl")):
        with open(p, "rb") as fh:
            sh += sum(1 for _ in fh)
    R["cont"] = {"per_pack_min": min(cont), "per_pack_max": max(cont), "total": sum(cont), "shared": sh}
    json.dump(R, open(os.path.join(OUT, "VERIFY.json"), "w"), indent=1, ensure_ascii=False)
    print("ERRORS", len(R["errors"]))
    for e in R["errors"][:40]:
        print("  ", e)
    print(json.dumps({k: v for k, v in R.items() if k not in ("subsets", "errors") and not k.startswith("loader")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
