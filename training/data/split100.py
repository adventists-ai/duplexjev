#!/usr/bin/env python3
"""把官方数据拆成 100 个包，目录结构与 official/ 完全一致（ML_ASR_OFFICIAL_ROOT 指向 packs100/pNNN 即可训练）。

分包键 = blake2b(key) % 100，key 与 export_b2_pack_manifests.py 相同（audio.path → path → file → id），
Wenet 用 jsonl 的 recording source。% 100 嵌套在旧的 % 10 里（新包号 % 10 = 旧包号 - 1）。
- peoples_speech：用 data/（全量 804 文件）写成各包的 peoples_speech/clean/（加载器读 clean）。
- 小子集（总行数 < 6400，即每包 < 64 行）：各包共用全量，用符号链接指向原文件，避免某包该子集为空
  （子集目录缺失时加载器会退回读整个根目录；空数据集会让交错采样提前终止）。
- 不做 C1 排除。
输出：packs100/pNNN/<同 official 结构>、packs100/pNNN/_cont/*.jsonl（本包续写清单）、
      packs100/_shared_cont/*.jsonl（共用小子集的续写清单）、packs100/_stats/**（逐文件统计）。
用法：python split100.py parquet|wenet|plan
"""
import gzip, hashlib, json, os, sys, tarfile, time
import multiprocessing as mp
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = "/data/exp01/exp03_train/data/ml_asr/official"
OUT = os.environ.get("PACKS_OUT", "/data/exp01/exp03_train/data/ml_asr/packs100")
N = 100
SHARED_MAX = 6400
DIRS = ["librispeech_asr", "peoples_speech/data", "gigaspeech", "common_voice_17_0",
        "common_voice_17_0-musan", "covost2", "multilingual_librispeech", "musan"]
TEXT = {"librispeech_asr": ("text", "librispeech"), "peoples_speech": ("text", "peoplespeech"),
        "gigaspeech": ("text", "gigaspeech"), "common_voice_17_0": ("sentence", "commonvoice"),
        "common_voice_17_0-musan": ("sentence", "commonvoice-musan-en"),
        "multilingual_librispeech": ("transcript", "mls")}
MLS_LANG = {"dutch": "nl", "portuguese": "pt"}


def pack_of(key: str) -> int:
    return int(hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=8).hexdigest(), 16) % N


def out_rel(rel: str) -> str:
    if rel.startswith("peoples_speech/data/"):
        return "peoples_speech/clean/" + rel[len("peoples_speech/data/"):]
    return rel


def subset_of(orel: str) -> str:
    return "/".join(orel.split("/")[:2])


def safe(rel: str) -> str:
    return rel.replace("/", "__")


def list_parquet():
    items = []
    for d in DIRS:
        for dp, _, fns in os.walk(os.path.join(SRC, d)):
            for fn in fns:
                fp = os.path.join(dp, fn)
                if fn.endswith(".parquet") and os.path.isfile(fp):
                    rel = os.path.relpath(fp, SRC)
                    items.append((fp, rel, out_rel(rel)))
    return sorted(items)


def plan():
    items = list_parquet()
    tot = Counter()
    rows = {}
    for fp, rel, orel in items:
        n = pq.ParquetFile(fp).metadata.num_rows
        rows[fp] = n
        tot[subset_of(orel)] += n
    shared = sorted(s for s, n in tot.items() if n < SHARED_MAX)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_packs": N, "shared_max": SHARED_MAX, "subset_rows": dict(tot), "shared_subsets": shared,
               "files": [{"src": fp, "rel": rel, "orel": orel, "rows": rows[fp]} for fp, rel, orel in items]},
              open(os.path.join(OUT, "plan.json"), "w"), indent=0)
    print("files", len(items), "subsets", len(tot), "shared", len(shared), shared)


def row_keys(batch, names, rel, start):
    n = batch.num_rows
    keys = [None] * n
    if "audio" in names:
        a = batch.column(names.index("audio"))
        if "path" in [f.name for f in a.type]:
            keys = pc.struct_field(a, "path").to_pylist()
    fb = 0
    for col in ("path", "file", "id"):
        if col in names and any(not k for k in keys):
            vals = batch.column(names.index(col)).to_pylist()
            keys = [k if k else v for k, v in zip(keys, vals)]
    out = []
    for i, k in enumerate(keys):
        if not k:
            k = f"{rel}#{start + i}"
            fb += 1
        out.append(str(k))
    return out, fb


def cont_rec(names, batch, i, key, top, orel):
    col, src = TEXT[top]
    text = batch.column(names.index(col))[i].as_py() if col in names else None
    text = (text or "").strip()
    if not text:
        return None
    if top == "common_voice_17_0":
        lang = orel.split("/")[1]
    elif top == "multilingual_librispeech":
        lang = MLS_LANG.get(orel.split("/")[1], orel.split("/")[1])
    else:
        lang = "en"
    return {"path": key, "text": text, "lang": lang, "source": src, "need_continuation": True}


def do_parquet(task):
    fp, rel, orel, shared = task
    st_path = os.path.join(OUT, "_stats", "parquet", safe(rel) + ".json")
    if os.path.exists(st_path):
        return json.load(open(st_path))
    t0 = time.time()
    top = orel.split("/")[0]
    pf = pq.ParquetFile(fp)
    names = pf.schema_arrow.names
    per_pack = [0] * N
    fb_total = 0
    nrows = 0
    cont_buf = defaultdict(list)
    writers = {}
    try:
        start = 0
        for batch in pf.iter_batches(batch_size=256):
            keys, fb = row_keys(batch, names, rel, start)
            fb_total += fb
            packs = [pack_of(k) for k in keys]
            if not shared:
                idx = defaultdict(list)
                for i, k in enumerate(packs):
                    idx[k].append(i)
                tbl = pa.Table.from_batches([batch])
                for k, ii in idx.items():
                    if k not in writers:
                        op = os.path.join(OUT, "p%03d" % k, orel)
                        os.makedirs(os.path.dirname(op), exist_ok=True)
                        writers[k] = pq.ParquetWriter(op + ".part", pf.schema_arrow)
                    writers[k].write_table(tbl.take(pa.array(ii, type=pa.int64())))
                    per_pack[k] += len(ii)
            if top in TEXT:
                for i, key in enumerate(keys):
                    r = cont_rec(names, batch, i, key, top, orel)
                    if r:
                        cont_buf["shared" if shared else packs[i]].append(r)
            nrows += batch.num_rows
            start += batch.num_rows
    finally:
        for k, w in writers.items():
            w.close()
            op = os.path.join(OUT, "p%03d" % k, orel)
            os.replace(op + ".part", op)
    if shared:
        for k in range(N):
            op = os.path.join(OUT, "p%03d" % k, orel)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            if not os.path.lexists(op):
                os.symlink(fp, op)
    for k, recs in cont_buf.items():
        d = os.path.join(OUT, "_shared_cont") if k == "shared" else os.path.join(OUT, "p%03d" % k, "_cont")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, safe(orel) + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = {"src": fp, "rel": rel, "orel": orel, "shared": shared, "rows": nrows, "meta_rows": pf.metadata.num_rows,
          "per_pack": per_pack, "fallback_keys": fb_total, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(st_path), exist_ok=True)
    json.dump(st, open(st_path + ".tmp", "w"))
    os.replace(st_path + ".tmp", st_path)
    return st


def run_parquet(procs):
    P = json.load(open(os.path.join(OUT, "plan.json")))
    shared = set(P["shared_subsets"])
    tasks = [(f["src"], f["rel"], f["orel"], subset_of(f["orel"]) in shared) for f in P["files"]]
    tasks.sort(key=lambda t: -os.path.getsize(t[0]))
    done = 0
    with mp.Pool(procs) as pool:
        for st in pool.imap_unordered(do_parquet, tasks):
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(time.strftime("%H:%M:%S"), "parquet", done, "/", len(tasks), flush=True)
    print("PARQUET_DONE", flush=True)


# ---------------- Wenet ----------------
sys.path.insert(0, "/data/exp01/exp03_train/scripts_ml_asr")


def _texts(js):
    from wenet_lhotse import WenetLhotseIterable
    return WenetLhotseIterable._load_texts(js)


def do_wenet(task):
    gid, pairs = task
    st_path = os.path.join(OUT, "_stats", "wenet", "g%05d.json" % gid)
    if os.path.exists(st_path):
        return json.load(open(st_path))
    from wenet_lhotse import WenetLhotseIterable
    t0 = time.time()
    stem = "cuts_L_fixed.g%05d" % gid
    jw, tw = {}, {}
    per_pack_js = [0] * N
    per_pack_tar = [0] * N
    miss = 0
    cont = defaultdict(list)

    def jwriter(k):
        if k not in jw:
            d = os.path.join(OUT, "p%03d" % k, "wenetspeech", "data")
            os.makedirs(d, exist_ok=True)
            jw[k] = gzip.open(os.path.join(d, stem + ".jsonl.gz.part"), "wt", encoding="utf-8", compresslevel=1)
        return jw[k]

    def twriter(k):
        if k not in tw:
            d = os.path.join(OUT, "p%03d" % k, "wenetspeech", "data")
            os.makedirs(d, exist_ok=True)
            tw[k] = tarfile.open(os.path.join(d, stem + ".tar.gz.part"), "w:gz", compresslevel=1)
        return tw[k]

    for js, tar in pairs:
        with gzip.open(js, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                sources = (rec.get("recording") or {}).get("sources") or []
                src = (sources[0].get("source") if sources else "") or ""
                if not src:
                    continue
                k = pack_of(src)
                jwriter(k).write(line if line.endswith("\n") else line + "\n")
                per_pack_js[k] += 1
                text = ""
                for sup in rec.get("supervisions") or []:
                    text = (sup.get("text") or "").strip()
                    if text:
                        break
                if text:
                    cont[k].append({"path": src, "text": text, "lang": "zh", "source": "wenetspeech", "need_continuation": True})
        by_full, by_base = WenetLhotseIterable._load_texts(js)
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                text, src = WenetLhotseIterable._lookup(m.name, by_full, by_base)
                if not text:
                    miss += 1
                    continue
                k = pack_of(src)
                twriter(k).addfile(m, tf.extractfile(m))
                per_pack_tar[k] += 1
    for k, w in jw.items():
        w.close()
    for k, w in tw.items():
        w.close()
    for k in set(jw) | set(tw):
        d = os.path.join(OUT, "p%03d" % k, "wenetspeech", "data")
        for ext in (".jsonl.gz", ".tar.gz"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p + ".part"):
                os.replace(p + ".part", p)
    for k, recs in cont.items():
        d = os.path.join(OUT, "p%03d" % k, "_cont")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "wenet_%s.jsonl" % stem), "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = {"gid": gid, "pairs": [str(p[0]) for p in pairs], "per_pack_jsonl": per_pack_js, "per_pack_tar": per_pack_tar,
          "tar_members_without_text": miss, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(st_path), exist_ok=True)
    json.dump(st, open(st_path + ".tmp", "w"))
    os.replace(st_path + ".tmp", st_path)
    return st


def run_wenet(procs, group=8):
    from wenet_lhotse import _find_pairs
    pairs = _find_pairs(os.path.join(SRC, "wenetspeech"))
    groups = [(i // group, pairs[i:i + group]) for i in range(0, len(pairs), group)]
    print("wenet pairs", len(pairs), "groups", len(groups), flush=True)
    done = 0
    with mp.Pool(procs) as pool:
        for st in pool.imap_unordered(do_wenet, groups):
            done += 1
            if done % 10 == 0 or done == len(groups):
                print(time.strftime("%H:%M:%S"), "wenet", done, "/", len(groups), flush=True)
    print("WENET_DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "plan":
        plan()
    elif mode == "parquet":
        run_parquet(int(sys.argv[2]) if len(sys.argv) > 2 else 72)
    elif mode == "wenet":
        run_wenet(int(sys.argv[2]) if len(sys.argv) > 2 else 48)
