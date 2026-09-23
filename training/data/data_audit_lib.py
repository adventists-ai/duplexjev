"""按训练加载器的查找规则（与 ultravox datasets._iter_local_parquet / wenet_lhotse 一致），统计某个数据根目录下
sf2 yaml 每个训练集能读到的行数，并与期望（全量/100，或共用小子集=全量）比较。"""
import glob, gzip, importlib, json, os, pkgutil, sys

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, "/data/exp01/exp03_train/ultravox")
sys.path.insert(0, "/data/exp01/exp03_train/scripts_ml_asr")
YAML = "/data/exp01/exp03_train/scripts_ml_asr/v1_h18_sf2.yaml"
FULL = "/data/exp01/exp03_train/paper_runs/data_audit.json"
OFFICIAL = "/data/exp01/exp03_train/data/ml_asr/official"


def _configs():
    from ultravox.data import registry as R, types as T
    import ultravox.data.configs as CP
    allc = {}
    for m in pkgutil.iter_modules(CP.__path__):
        try:
            mod = importlib.import_module("ultravox.data.configs." + m.name)
        except Exception:
            continue
        for v in vars(mod).values():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, T.DatasetConfig):
                    allc.setdefault(x.name, x)

    def resolve(name):
        cs = [allc[name]]
        while cs[-1].base:
            cs.append(allc[cs[-1].base])
        return R._merge_configs(cs[::-1])
    return resolve


def _find(path, name, split):
    if not os.path.isdir(path):
        return [], path
    search = path
    if name and os.path.isdir(os.path.join(path, name)):
        search = os.path.join(path, name)
    out = []
    for dp, _, fns in os.walk(search):
        for fn in fns:
            if not fn.endswith(".parquet"):
                continue
            fp = os.path.join(dp, fn)
            if not os.path.isfile(fp):
                continue
            rel = os.path.relpath(fp, search)
            if split and not (fn.startswith(split + "-") or fn.startswith(split + ".") or fn.startswith(split + "_")
                              or rel.startswith(split + os.sep) or (os.sep + split + os.sep) in (os.sep + rel)):
                continue
            out.append(fp)
    return sorted(out), search


def audit(root):
    import split100 as S
    plan = json.load(open(os.path.join(S.OUT, "plan.json")))
    shared = set(plan["shared_subsets"])
    full = {r["name"]: r for r in json.load(open(FULL))}
    resolve = _configs()
    import official_local as OL
    cfg = yaml.safe_load(open(YAML))
    rows = []
    cache = {}
    for ts in cfg["train_sets"]:
        name = ts["name"]
        c = resolve(name)
        mapped = OL.PATH_MAP.get(c.path, c.path)
        if isinstance(mapped, str) and mapped.startswith(OFFICIAL):
            mapped = root + mapped[len(OFFICIAL):]
        tr = [sp for sp in (c.splits or []) if "TRAIN" in str(sp.split).upper()]
        r = {"name": name, "w": ts.get("weight", 1), "official_n": sum(sp.num_samples for sp in tr)}
        if "wenet" in name:
            n = 0
            for js in glob.glob(os.path.join(root, "wenetspeech", "data", "cuts_L_fixed.*.jsonl.gz")):
                with gzip.open(js, "rb") as fh:
                    n += sum(1 for _ in fh)
            r["found"] = n
            r["expected"] = round(14621415 / 100)
            r["subset"] = "wenetspeech"
        else:
            files = []
            search = None
            for sp in tr:
                fs, search = _find(mapped, c.subset, sp.name)
                files += fs
            n = 0
            for f in files:
                if f not in cache:
                    cache[f] = pq.ParquetFile(f).metadata.num_rows
                n += cache[f]
            sub = os.path.relpath(search, root) if search else None
            r["subset"] = sub
            r["found"] = n
            base_full = full.get(name, {}).get("found") or 0
            if sub in shared:
                r["expected"] = base_full
            elif name.startswith("peoplespeech"):
                r["expected"] = round(1501271 / 100)
            else:
                r["expected"] = round(base_full / 100)
        e = r["expected"]
        if r["found"] == 0:
            r["flag"] = "EMPTY"
        elif r["subset"] in (".", None) or "/" not in str(r["subset"]) and r["subset"] != "wenetspeech" or (e and r["found"] > 3 * e + 50):
            r["flag"] = "FALLBACK? found %s expected %s search %s" % (r["found"], e, r["subset"])
        elif e and r["found"] < 0.5 * e - 20:
            r["flag"] = "TOO FEW found %s expected %s" % (r["found"], e)
        rows.append(r)
    return rows
