"""fdx-bench-v1 统一验收入口。三条件 × 分项，输出一张带 bench_sha 的表。

三个条件与它们各自回答的问题：

    audio              给音频 + 提问          -> 主语言分（内容保真度）
    text_only          给转写文字，不给音频     -> 语言先验能拿到多少
    audio_mismatched   给提问 + 别人的音频     -> 音频依赖度的下界

    副语言分 = audio − text_only      （音频比文字多带来的信息）
    音频依赖 = audio − audio_mismatched（模型是否真的在听这一条）

为什么必须有 text_only：上一轮 fdx-p1 的 emotion 看着有 audio_gain，
但没法排除「情绪词本来就写在转写里」。不减掉语言先验，
副语言的分就说不清是听出来的还是读出来的。

**ASR 任务没有 text_only 条件**：把转写给它再让它转写，答案在输入里，
恒等满分，这个数没有意义。TEXT_ONLY 表里因此没有 asr-*。

指标上避开两个上一轮踩到的坑（见 runs/fdx-p1.md §7）：

- WER 饱和：英文 WER 两个模型都是 1.00，因为 max_new_tokens 截断
  制造了大量删除错误，把「v5 明显更贴音频内容」这个真实差异抹平了。
  对策：max_new 按参考长度自适应，另加 M3 关键词召回（有界 [0,1]，
  不会因为截断直接顶到 1）与前缀 CER（只比生成了的那一段）。
- 语义反转：fdx-p1 出现过把「缺乏」说成「过多」，CER 很低但意思相反。
  对策：M4 极性反转筛查。它是启发式筛子，输出的是待人看的条目，
  不是可以直接当分报的指标。
"""

import argparse
import collections
import json
import math
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_questions as BQ  # noqa: E402

BENCH = "/data/exp01/exp03_train/data/fdx_bench_v1"
SAMPLE_RATE = 16000

# --------------------------------------------------------------------------
# 文本归一化与错误率
# --------------------------------------------------------------------------

ZH_PUNCT = "。．，、；：？！…—～·「」『』（）《》〈〉“”‘’\"'()[]{}<>,.;:?!~`"


def norm(s, lang):
    s = (s or "").strip()
    s = re.sub(rf"[{re.escape(ZH_PUNCT)}]", "", s)
    if lang == "zh":
        return re.sub(r"\s+", "", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def units(s, lang):
    return list(s) if lang == "zh" else s.split()


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def err_rate(ref, hyp, lang):
    r, h = units(norm(ref, lang), lang), units(norm(hyp, lang), lang)
    if not r:
        return None
    return edit_distance(r, h) / len(r)


def prefix_err_rate(ref, hyp, lang):
    """只在生成了的那一段长度上比。

    截断时整句 CER 会被删除错误顶满，看不出模型到底跟上音频没有。
    截成同长再比，量的是「已经说出来的那部分对不对」。
    """
    r, h = units(norm(ref, lang), lang), units(norm(hyp, lang), lang)
    if not r or not h:
        return None
    n = min(len(r), len(h))
    return edit_distance(r[:n], h[:n]) / n


# --------------------------------------------------------------------------
# M3 关键词召回
# --------------------------------------------------------------------------

EN_STOP = set(
    """a an the and or but if of to in on at by for with from as is are was were be been
being do does did have has had he she it they we you i his her their our your this that
these those there here not no so than then when while which who whom what how very just
will would can could should may might must shall about into over under again once""".split()
)
# 中文虚词。不用分词器：jieba 不一定装，而且分词粒度变化会让跨轮次的数对不上。
# 直接按字过滤虚词，粒度稳定、可复现。
ZH_STOP = set("的了是在我你他她它们和与及也都就要会有个这那不没吧呢啊呀吗嘛之而其以于对被把给从向让使很更最还又再只才已经过着地得所为")


def keywords(s, lang):
    if lang == "zh":
        return {c for c in norm(s, lang) if c not in ZH_STOP}
    return {w for w in norm(s, lang).split() if w not in EN_STOP and len(w) > 1}


def keyword_recall(ref, hyp, lang):
    kr = keywords(ref, lang)
    if not kr:
        return None
    return len(kr & keywords(hyp, lang)) / len(kr)


# --------------------------------------------------------------------------
# M4 极性反转筛查
# --------------------------------------------------------------------------

ANTONYMS = [
    ("过多", "缺乏"), ("过多", "不足"), ("增加", "减少"), ("上升", "下降"),
    ("提高", "降低"), ("变多", "变少"), ("更多", "更少"), ("充足", "不足"),
    ("可以", "不能"), ("应该", "不该"), ("同意", "反对"), ("支持", "反对"),
    ("成功", "失败"), ("开始", "结束"), ("打开", "关闭"), ("进入", "退出"),
    ("more", "less"), ("increase", "decrease"), ("rise", "fall"),
    ("excess", "lack"), ("agree", "disagree"), ("success", "failure"),
    ("open", "close"), ("enter", "exit"), ("high", "low"),
]
NEG_ZH = re.compile(r"不|没|无|未|别|莫|非")
NEG_EN = re.compile(r"\b(not|no|never|none|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't)\b")


def polarity_flip(ref, hyp, lang):
    """返回反转证据，没有则 None。

    只是筛子：报出来的条目需要人看一眼才能定性。
    它宁可多报（比如原文本来就有否定、模型也照抄了否定的情况会被
    否定计数相等过滤掉，但同义改写仍可能误报），也不要漏报——
    「意思相反但 CER 很低」是比 CER 高得多的危险，值得人工过目。
    """
    r, h = norm(ref, lang), norm(hyp, lang)
    if not h:
        return None
    for a, b in ANTONYMS:
        if (a in r and b in h and a not in h) or (b in r and a in h and b not in h):
            return f"反义词 {a}/{b}"
    rx = NEG_ZH if lang == "zh" else NEG_EN
    nr, nh = len(rx.findall(r)), len(rx.findall(h))
    # 只在参考里有否定而生成里一个都没有（或反过来）时报，
    # 且要求生成长度接近参考，排除截断造成的假阳性。
    if min(len(r), len(h)) / max(len(r), len(h)) > 0.6 and (nr > 0) != (nh > 0):
        return f"否定词数 参考{nr} vs 生成{nh}"
    return None


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------

AUDIO_PH = BQ.AUDIO_PH


QWEN32B = "/data/exp01/models/Qwen3-32B"
ENCODER = "/data/exp01/exp03_train/models/qwen3_asr_0.6b_encoder_v2_crossattn"


def resolve_component(stored, fallback, what):
    """存在就用 checkpoint 自己记的，不存在才回落。

    **不能无条件覆盖成某个固定路径。** 历史 run 分两代 encoder：
    早期是 qwen3_asr_0.6b_encoder（无 cross-attention），
    之后才是 qwen3_asr_0.6b_encoder_v2_crossattn。
    无条件钉到后者，会给旧 checkpoint 装上一个它从未训练过的 encoder，
    权重形状恰好兼容所以**不报错**，跑出来的数全是垃圾——
    而对照组的全部意义就是比较这两代，覆盖掉等于把要测的变量抹掉了。
    """
    if stored and os.path.exists(stored):
        return stored
    print(f"  {what}: checkpoint 记的 {stored!r} 不存在，回落到 {fallback}")
    return fallback


def load_model(ckpt, device):
    config = transformers.AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    # checkpoint 里存的是训练时的绝对路径；路径还在就照用，失效才钉到本地权重
    config.text_model_id = resolve_component(
        getattr(config, "text_model_id", None), QWEN32B, "text_model"
    )
    config.audio_model_id = resolve_component(
        getattr(config, "audio_model_id", None), ENCODER, "audio_model"
    )
    print(f"  encoder = {config.audio_model_id}")
    model = transformers.AutoModel.from_pretrained(
        ckpt,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    processor = transformers.AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    # checkpoint 存的是裸 WhisperFeatureExtractor，而 ultravox_processing 里
    # 按 processor.audio_processor.feature_extractor 取 hop_length，
    # 不裹一层会 AttributeError。与 gate_generalization.py 保持同一处理。
    if isinstance(
        processor.audio_processor, transformers.WhisperFeatureExtractor
    ) and not hasattr(processor.audio_processor, "feature_extractor"):
        processor.audio_processor = transformers.WhisperProcessor(
            feature_extractor=processor.audio_processor,
            tokenizer=transformers.WhisperTokenizer.from_pretrained("openai/whisper-tiny"),
        )
    return model, processor


def build_prompt(tok, question):
    """enable_thinking=False 不能省，原因与直觉相反，所以写清楚。

    训练侧（ultravox_data_proc.py）是拼完整对话、不加 generation prompt、
    也不传 enable_thinking，Qwen3 模板对此渲染出的 assistant 段是：

        <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n答案<|im_end|>\\n

    注意训练时**有**一个空的 <think>\\n\\n</think>\\n\\n。
    而评测侧 add_generation_prompt=True 时：

        enable_thinking=False  -> ...assistant\\n<think>\\n\\n</think>\\n\\n   ← 与训练一致
        不传 enable_thinking   -> ...assistant\\n                            ← 少了那 6 个 token

    所以坑不是「默认开了思考、插进来一段思考把位置挤歪了」，
    而是反过来：**不传时评测侧不插，训练时插了**，少掉的这一段
    让 gold 被接到了模型正准备输出 <think> 的位置上。
    量出来的就是 NLL 25 量级与瞎猜级准确率，试过一次，是这么错的。

    一致性由 assert_template_alignment() 在启动时实测断言，不靠这段注释。
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def assert_template_alignment(tok):
    """启动自检：评测 prompt 必须正好是训练串里「答案之前」的那一段。

    判据不能用「训练串以评测 prompt 开头」——传与不传 enable_thinking
    的两种拼法都满足这个条件（短的那个也是前缀）。
    真正的判据是**减掉 prompt 之后剩下的必须正好是目标答案**：
        一致   -> 剩 "男性<|im_end|>\\n"
        不一致 -> 剩 "<think>\\n\\n</think>\\n\\n男性<|im_end|>\\n"
                  即把 think 标记也当成了要学的答案。
    """
    q, gold = "这是一个对齐自检问题。", "对齐自检答案"
    train = tok.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": gold}],
        tokenize=False,
    )
    prompt = build_prompt(tok, q)
    if not train.startswith(prompt):
        raise RuntimeError(
            "模板口径不一致：评测 prompt 不是训练串的前缀。\n"
            f"  训练串: {train!r}\n  评测prompt: {prompt!r}"
        )
    rest = train[len(prompt) :]
    if not rest.startswith(gold):
        raise RuntimeError(
            "模板口径不一致：减掉评测 prompt 后，剩下的不是以答案开头。\n"
            f"  剩下: {rest!r}\n"
            "  常见原因：训练侧与评测侧的 enable_thinking 口径不同。\n"
            "  若训练侧改过（例如显式关掉了 think 段），此处需同步。"
        )
    return prompt, rest


def make_inputs(processor, tok, question, audio, gold, model, device):
    """返回 (inputs, n_prompt)。audio 为 None 时走纯文本路径（text_only 条件）。"""
    prompt = build_prompt(tok, question)
    kw = {} if audio is None else {"audio": audio, "sampling_rate": SAMPLE_RATE}
    p = processor(text=prompt, **kw)
    f = processor(text=prompt + gold, **kw)
    n_prompt = p["input_ids"].shape[1]
    # tokenizer 可能把 prompt 末字符与 gold 首字符并成一个 token，
    # 那样切点就错了，算的是别的位置的概率。显式挡住，宁可丢这一条。
    if not torch.equal(f["input_ids"][0, :n_prompt], p["input_ids"][0]):
        return None, None
    out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in f.items()}
    if "audio_values" in out:
        out["audio_values"] = out["audio_values"].to(model.dtype)
    return out, int(n_prompt)


@torch.no_grad()
def gold_nll(model, processor, tok, question, audio, gold, device):
    inputs, n_prompt = make_inputs(processor, tok, question, audio, gold, model, device)
    if inputs is None:
        return None, None, None
    logits = model(**inputs).logits[0].float()
    gold_ids = inputs["input_ids"][0, n_prompt:]
    if gold_ids.numel() == 0:
        return None, None, None
    lp = F.log_softmax(logits[n_prompt - 1 : -1], dim=-1)
    nll = -lp[torch.arange(len(gold_ids), device=device), gold_ids].mean().item()
    return nll, logits, n_prompt


@torch.no_grad()
def generate(model, processor, tok, question, audio, device, max_new):
    prompt = build_prompt(tok, question)
    kw = {} if audio is None else {"audio": audio, "sampling_rate": SAMPLE_RATE}
    inputs = processor(text=prompt, **kw)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "audio_values" in inputs:
        inputs["audio_values"] = inputs["audio_values"].to(model.dtype)
    # temperature/top_p/top_k 显式置空：checkpoint 的 generation_config 里带着
    # 采样参数，贪心解码下用不到但会一路刷 UserWarning，把真正的报错埋掉。
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    n = inputs["input_ids"].shape[-1]
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# 错配配对
# --------------------------------------------------------------------------


def build_mismatch_index(golds):
    """为每条找一条 gold 不同的做错配来源。

    不能用「错开一半」这种偷懒配对：小而规整的集合（比如二分类的 gender）
    错开一半有相当概率配到同 gold 的条目，那样错配条件下模型
    照样能答对，音频依赖度就被低估了。上一轮就是这么错过一次。
    """
    by_gold = collections.defaultdict(list)
    for i, g in enumerate(golds):
        by_gold[g].append(i)
    out = []
    for i, g in enumerate(golds):
        others = [j for gg, idxs in by_gold.items() if gg != g for j in idxs]
        out.append(others[i % len(others)] if others else None)
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

OPEN_TASKS = {"asr-zh", "asr-en"}
LANG_OF = {"asr-zh": "zh", "asr-en": "en"}


def lang_of(task, item):
    if task in LANG_OF:
        return LANG_OF[task]
    return "en" if task == "emotion" else "zh"


def run_condition(model, processor, tok, items, task, condition, device, wording, limit):
    qs = BQ.questions_for(task, condition)
    if qs is None:
        return None  # 该条件对该任务不适用
    q_template = qs[0] if wording == "canonical" else qs[1 + (0 if wording == "v1" else 1)]

    rows = items if limit is None else items[:limit]
    golds = [r["gold"] for r in rows]
    mm = build_mismatch_index(golds) if condition == "audio_mismatched" else None

    lang = lang_of(task, rows[0])
    nlls, recs, cers, pcers, exacts, flips = [], [], [], [], [], []
    pred_hist = collections.Counter()
    skipped = 0

    for i, r in enumerate(rows):
        if condition == "text_only":
            audio = None
            question = q_template.format(text=r["text"])
        else:
            if condition == "audio":
                src = r
            elif mm[i] is None:
                # 该任务在当前子集里只有一个 gold 取值，找不到可错配的对象。
                # --limit 冒烟时容易撞上（分项是按类别排序的，前 N 条同类）。
                # 全量跑不会出现；出现就说明这一项的类别只剩一个，指标无意义。
                skipped += 1
                continue
            else:
                src = rows[mm[i]]
            audio, _ = sf.read(f"{BENCH}/{src['audio']}", dtype="float32")
            question = q_template

        gold = r["gold"]
        nll, logits, n_prompt = gold_nll(model, processor, tok, question, audio, gold, device)
        if nll is None:
            # 边界 token 合并，切点不可靠。丢这一条，并计数——
            # 丢太多就说明措辞末尾与答案首字符总在合并，该改措辞。
            skipped += 1
            continue
        nlls.append(nll)

        if task in OPEN_TASKS:
            # max_new 按参考长度自适应，别让截断成为瓶颈。
            # 中文一字约一 token，英文一词约 1.3 token，各留 1.6 倍余量。
            n_ref = len(units(norm(gold, lang), lang))
            max_new = max(24, int(n_ref * (1.6 if lang == "zh" else 2.1)) + 8)
            gen = generate(model, processor, tok, question, audio, device, max_new)
            c = err_rate(gold, gen, lang)
            pc = prefix_err_rate(gold, gen, lang)
            kr = keyword_recall(gold, gen, lang)
            if c is not None:
                cers.append(c)
            if pc is not None:
                pcers.append(pc)
            if kr is not None:
                recs.append(kr)
            fl = polarity_flip(gold, gen, lang)
            if fl:
                flips.append({"id": r["id"], "why": fl, "gold": gold, "gen": gen})
        else:
            opts = r["answer_space"]
            # 取每个选项的首 token，不是首字符。
            # 上一轮这里用了 o[0]，对「男性」这种多字词取到的是「男」，
            # 而模型实际生成的首 token 是「男性」，比的是它几乎不会生成的 token，
            # 准确率被系统性低报。
            opt_ids = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
            if len(set(opt_ids)) != len(opt_ids):
                # 两个选项的首 token 相同，受限 softmax 分不开它们，这一项的
                # 准确率无意义。宁可炸掉也不要报一个看似正常的数。
                raise RuntimeError(
                    f"{task} 的选项首 token 有重复：{list(zip(opts, opt_ids))}"
                )
            probs = F.softmax(logits[n_prompt - 1, opt_ids], dim=-1)
            pred = opts[int(probs.argmax())]
            exacts.append(1.0 if pred == gold else 0.0)
            # 记预测分布：恒答同一类会得到「贴着瞎猜值」的准确率，
            # 与「判得不准」数字相同但含义完全不同，光看准确率分不出来。
            pred_hist[pred] += 1

    def ms(xs):
        if not xs:
            return None
        a = np.array(xs, dtype=float)
        return {
            "mean": round(float(a.mean()), 4),
            "se": round(float(a.std(ddof=1) / math.sqrt(len(a))), 4) if len(a) > 1 else 0.0,
            "n": len(a),
        }

    return {
        "nll": ms(nlls),
        "cer": ms(cers),
        "prefix_cer": ms(pcers),
        "keyword_recall": ms(recs),
        "accuracy": ms(exacts),
        "pred_hist": dict(pred_hist.most_common()),
        "majority_pred_rate": (
            round(pred_hist.most_common(1)[0][1] / sum(pred_hist.values()), 4)
            if pred_hist
            else None
        ),
        "skipped_boundary_merge": skipped,
        "polarity_flips": flips[:20],
        "polarity_flip_rate": round(len(flips) / max(1, len(rows)), 4) if task in OPEN_TASKS else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--manifest", default=f"{BENCH}/bench_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="", help="逗号分隔，空为全部")
    ap.add_argument(
        "--conditions", default="audio,text_only,audio_mismatched", help="逗号分隔"
    )
    ap.add_argument(
        "--loader",
        choices=["fdx", "official"],
        default="fdx",
        help="official=C 路官方 Ultravox v0.6（whisper-turbo 塔，不回落本线 Qwen3-ASR encoder）",
    )
    ap.add_argument("--wording", default="canonical", choices=["canonical", "v1", "v2"])
    ap.add_argument("--limit", type=int, default=None, help="每任务只跑前 N 条，冒烟用")
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="跳过输出文件里已有结果的 任务×措辞×条件 组合。"
        "用于改任务顺序后续跑，不必重算已完成的部分",
    )
    ap.add_argument(
        "--schema-pool",
        default="/data/exp01/exp03_train/data/fdx_p1/schema_pool.json",
        help="用于核验基准措辞与训练措辞零重合",
    )
    args = ap.parse_args()

    if os.path.exists(args.schema_pool):
        n = BQ.verify_questions(args.schema_pool)
        print(f"措辞核验通过：与训练池 {n} 条措辞零重合")
    else:
        print(f"!! 找不到 {args.schema_pool}，跳过措辞重合核验。结果可信度下降。")

    doc = json.load(open(args.manifest, encoding="utf-8"))
    print(f"基准 {doc['name']}  bench_sha={doc['bench_sha'][:16]}…  条目 {len(doc['items'])}")

    by_task = collections.defaultdict(list)
    for it in doc["items"]:
        by_task[it["task"]].append(it)

    want_tasks = [t for t in args.tasks.split(",") if t] or sorted(by_task)
    want_conds = [c for c in args.conditions.split(",") if c]

    if args.loader == "official":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, "/data/exp01/exp03_train")
        from jev_qwen.load_official import load_official

        model, processor = load_official(args.ckpt, args.device)
    else:
        model, processor = load_model(args.ckpt, args.device)
    tok = processor.tokenizer
    prompt, rest = assert_template_alignment(tok)
    print(f"模板对齐自检通过：prompt 尾部 {prompt[-24:]!r}  答案段起始 {rest[:12]!r}")

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding="utf-8"))
        print(f"已有结果，将合并进 {args.out}")
    results.setdefault("meta", {})
    # encoder 与步数记进 meta：对照组横跨两代 encoder，
    # 报表要能机械地按 encoder 分组，不能靠人记得哪个 run 是哪代。
    m = re.search(r"checkpoint-(\d+)", args.ckpt)
    results["meta"].update(
        {
            "ckpt": args.ckpt,
            "run": os.path.basename(os.path.dirname(args.ckpt.rstrip("/"))),
            "step": int(m.group(1)) if m else None,
            "audio_model_id": model.config.audio_model_id
            or getattr(model.config.audio_config, "_name_or_path", None),
            "loader": args.loader,
            "bench_sha": doc["bench_sha"],
            "bench_name": doc["name"],
            "wording": args.wording,
            "limit": args.limit,
        }
    )
    results.setdefault("tasks", {})

    for task in want_tasks:
        items = by_task.get(task)
        if not items:
            print(f"跳过 {task}：基准里没有这个任务")
            continue
        results["tasks"].setdefault(task, {})
        for cond in want_conds:
            if args.skip_done:
                prev = results["tasks"][task].get(args.wording, {}).get(cond)
                na = results["tasks"][task].get(cond)
                if isinstance(prev, dict) or (isinstance(na, dict) and na.get("not_applicable")):
                    print(f"  跳过 {task} / {cond}：已有结果")
                    continue
            print(f"  跑 {task} / {cond} / {args.wording} …", flush=True)
            r = run_condition(
                model, processor, tok, items, task, cond, args.device, args.wording, args.limit
            )
            if r is None:
                print(f"    {cond} 对 {task} 不适用，跳过")
                results["tasks"][task][cond] = {"not_applicable": True}
                continue
            results["tasks"][task].setdefault(args.wording, {})[cond] = r
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            head = r["accuracy"] or r["cer"]
            print(f"    nll={r['nll']} 主指标={head}")

    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
