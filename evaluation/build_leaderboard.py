"""Leaderboard for docs/connectors.md (connector checkpoints).
  main language   = mean(qa100, ZJU-ML, Easy-Turn)      (%)
  paralinguistics = mean(gender, emotion)               (%)
  total           = (main language + paralinguistics) / 2
One row per encoder x LLM x connector (A cross-attention / B last layer): the released checkpoint with the best total,
sorted by total, followed by a table with every benchmark. board("main"/"para", ...) prints the two partial boards.
usage: python evaluation/build_leaderboard.py {en|zh} evaluation/leaderboard/*.json > leaderboard.md"""
import json, sys

ORG = "adventists-ai"
ENC = {"qwen3asr": "Qwen3-ASR-0.6B", "whisper": "Whisper-small", "sensevoice": "SenseVoice-Small",
       "moss": "MOSS-Transcribe-Diarize", "turbo": "Whisper-large-v3-turbo"}
LANG = sys.argv[1]
files = sys.argv[2:]

rows = []
for f in files:
    for v in json.load(open(f))["models"]:
        p = v["paper"]
        r = dict(sec=v.get("conn", "B"), enc=ENC.get(v["enc"], v["enc"]), llm=v.get("llm_short", v["llm"]), repo=v["repo"],
                 nc=v["stage"] in ("para", "emotion"), total_M=v.get("params_M"), cpu=(v.get("latency") or {}).get("cpu_ms"),
                 q=p.get("qa100"), z=p.get("zjuml"), t=p.get("easyturn"), g=p.get("gender"), e=p.get("emotion"))
        r["main"] = round((r["q"] + r["z"] + r["t"]) / 3, 1) if None not in (r["q"], r["z"], r["t"]) else None
        r["para"] = round((r["g"] + r["e"]) / 2, 1) if None not in (r["g"], r["e"]) else None
        rows.append(r)

TXT = {
    "en": dict(main="Main language: mean of qa100, ZJU-ML and Easy-Turn", para="Paralinguistics: mean of gender and emotion",
               A="Cross-attention connectors (A)", B="Native last-layer connectors (B)",
               hm=("rank", "encoder", "LLM", "checkpoint", "score", "qa100", "ZJU-ML", "Easy-Turn", "params", "CPU", "licence"),
               hp=("rank", "encoder", "LLM", "checkpoint", "score", "gender", "emotion", "params", "CPU", "licence")),
    "zh": dict(main="主语言：qa100、ZJU-ML、Easy-Turn 三项平均", para="副语言：性别、情绪两项平均",
               A="交叉注意力连接器（A）", B="原生末层连接器（B）",
               hm=("排名", "编码器", "LLM", "模型", "总分", "qa100", "ZJU-ML", "Easy-Turn", "总参数", "CPU", "许可"),
               hp=("排名", "编码器", "LLM", "模型", "总分", "性别", "情绪", "总参数", "CPU", "许可")),
}[LANG]


def board(key, cols, hdr):
    best = {}
    for r in rows:
        if r[key] is None:
            continue
        k = (r["sec"], r["enc"], r["llm"])
        if k not in best or r[key] > best[k][key]:
            best[k] = r
    for sec in ("A", "B"):
        rs = sorted((r for r in best.values() if r["sec"] == sec), key=lambda r: (-r[key], r["total_M"] or 0))
        if not rs:
            continue
        print(f"#### {TXT[sec]}\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---:"] + ["---"] * 3 + ["---:"] * (len(hdr) - 5) + ["---"]) + "|")
        for i, r in enumerate(rs, 1):
            tot = f"{r['total_M'] / 1000:.1f} B" if r["total_M"] else "–"
            cpu = f"{r['cpu'] / 1000:.1f} s" if r["cpu"] else "–"
            lic = "CC BY-NC 4.0" if r["nc"] else "Apache-2.0"
            vals = " | ".join(f"{r[c]:g}" for c in cols)
            print(f"| {i} | {r['enc']} | {r['llm']} | 🤗 [{r['repo']}](https://huggingface.co/{ORG}/{r['repo']}) | "
                  f"**{r[key]:.1f}** | {vals} | {tot} | {cpu} | {lic} |")
        print()



def overall():
    best = {}
    for r in rows:
        if r["main"] is None or r["para"] is None:
            continue
        r["total"] = round((r["main"] + r["para"]) / 2, 1)
        k = (r["sec"], r["enc"], r["llm"])
        if k not in best or r["total"] > best[k]["total"]:
            best[k] = r
    rs = sorted(best.values(), key=lambda r: (-r["total"], r["total_M"] if "total_M" in r else 0))
    H = {"en": ("rank", "encoder", "LLM", "connector", "checkpoint", "total", "main language", "paralinguistics"),
         "zh": ("排名", "编码器", "LLM", "连接器", "模型", "总分", "主语言", "副语言")}[LANG]
    D = {"en": ("checkpoint", "qa100", "ZJU-ML", "Easy-Turn", "gender", "emotion", "params", "CPU (8 threads)", "licence"),
         "zh": ("模型", "qa100", "ZJU-ML", "Easy-Turn", "性别", "情绪", "总参数", "CPU（8 线程）", "许可")}[LANG]
    CN = {"A": {"en": "A · cross-attention", "zh": "A · 交叉注意力"}, "B": {"en": "B · native", "zh": "B · 原生"}}
    print({"en": "### Overall ranking\n\nTotal = (main language + paralinguistics) / 2. Main language = mean of qa100, ZJU-ML and Easy-Turn; paralinguistics = mean of gender and emotion (all %). One checkpoint per encoder × LLM × connector: the one with the best total.\n",
           "zh": "### 总排名\n\n总分 =（主语言 + 副语言）/ 2。主语言 = qa100、ZJU-ML、Easy-Turn 三项平均；副语言 = 性别、情绪两项平均（均为百分制）。每个 编码器 × LLM × 连接器 组合只列总分最高的一个模型。\n"}[LANG])
    print("| " + " | ".join(H) + " |"); print("|---:|---|---|---|---|---:|---:|---:|")
    for i, r in enumerate(rs, 1):
        print(f"| {i} | {r['enc']} | {r['llm']} | {CN[r['sec']][LANG]} | 🤗 [{r['repo']}](https://huggingface.co/{ORG}/{r['repo']}) | **{r['total']:.1f}** | {r['main']:.1f} | {r['para']:.1f} |")
    print({"en": "\n### Details\n", "zh": "\n### 分项成绩\n"}[LANG])
    print("| " + " | ".join(D) + " |"); print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rs:
        tot = f"{r['total_M'] / 1000:.1f} B" if r.get("total_M") else "–"
        cpu = f"{r['cpu'] / 1000:.1f} s" if r["cpu"] else "–"
        lic = "CC BY-NC 4.0" if r["nc"] else "Apache-2.0"
        print(f"| {r['repo']} | {r['q']:g} | {r['z']:g} | {r['t']:g} | {r['g']:g} | {r['e']:g} | {tot} | {cpu} | {lic} |")

overall()
