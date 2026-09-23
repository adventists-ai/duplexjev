"""M5 级联基线（论文表 2）：Qwen3-ASR-0.6B 逐字转写 → Qwen3-32B（vLLM）生成 10 字段 JSON。
同一张卡、bf16、greedy。每个"事件"= 一段音频 + 10 个闭集问题。
报告：ASR 耗时与解码步数；LLM 耗时与输出 token 数；端到端中位数/P95（按音频时长三档）；
以及把全部事件合批时的吞吐（事件/秒）。只写数字，不写任何文本内容到结果里。
"""
import json, os, sys, time, random, statistics as st
import numpy as np, torch
ROOT = "/data/exp01/exp03_train"
OUT = f"{ROOT}/artifacts/paper/cascade/cascade_bench.json"
ASR_ID = "/data/exp01/models/Qwen3-ASR-0.6B-hf"
LLM_ID = "/data/exp01/models/Qwen3-32B"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

QUESTIONS = [
 ("gender", "说话人性别", ["男", "女"]),
 ("language", "说话语言", ["中文", "英文", "其他"]),
 ("duration", "这段话大约多长", ["3秒以内", "3到6秒", "6秒以上"]),
 ("filler", "最合适先播放哪句垫话", ["好的", "稍等，我查一下", "明白了", "嗯嗯", "没问题", "我来帮您看看", "收到", "请再说一遍"]),
 ("route", "应该转到哪个流程", ["导航", "音乐", "车控", "闲聊", "查询信息", "其他"]),
 ("urgency", "紧急程度", ["1", "2", "3", "4", "5"]),
 ("sentiment", "情绪倾向", ["正面", "中性", "负面"]),
 ("handoff", "是否需要转人工", ["是", "否"]),
 ("complete", "这句话说完了吗", ["说完了", "没说完"]),
 ("question", "这是不是一个提问", ["是", "否"]),
]
def llm_prompt(transcript):
    qs = "\n".join(f'- "{k}": {d}，只能从 {opts} 中选一个' for k, d, opts in QUESTIONS)
    return (f"用户刚才说：「{transcript}」\n请回答下列问题，只输出一个 JSON 对象，键如下：\n{qs}\n只输出 JSON，不要解释。")

def pick_audio():
    meta = json.load(open(f"{ROOT}/artifacts/paper/cascade/audio_npy/meta.json"))
    return [(m["bin"], {"audio": m["npy"], "lang": m["lang"], "duration_s": m["duration_s"]}) for m in meta]

def load16(p):
    return np.load(p)

def main():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from vllm import LLM, SamplingParams
    sel = pick_audio()
    proc = AutoProcessor.from_pretrained(ASR_ID)
    asr = AutoModelForMultimodalLM.from_pretrained(ASR_ID, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    llm = LLM(model=LLM_ID, dtype="bfloat16", gpu_memory_utilization=0.75, max_model_len=4096, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    def chat(t):
        return tok.apply_chat_template([{"role": "user", "content": llm_prompt(t)}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    def run_asr(a, lang):
        inp = proc.apply_transcription_request(audio=a, language="Chinese" if lang == "zh" else "English").to(asr.device, asr.dtype)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): out = asr.generate(**inp, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        gen = out[:, inp["input_ids"].shape[1]:]
        return proc.decode(gen, return_format="transcription_only")[0], dt, int(gen.shape[1])
    # warmup
    a0 = load16(sel[0][1]["audio"]); run_asr(a0, sel[0][1]["lang"]); llm.generate([chat("你好")], sp)
    rows = []; transcripts = []
    for b, r in sel:
        a = load16(r["audio"])
        tr, asr_s, asr_steps = run_asr(a, r["lang"])
        t0 = time.perf_counter(); o = llm.generate([chat(tr)], sp, use_tqdm=False)[0]; llm_s = time.perf_counter() - t0
        out_tok = len(o.outputs[0].token_ids)
        try: json.loads(o.outputs[0].text.strip().strip("`").replace("json", "", 1)); ok = True
        except Exception: ok = False
        rows.append({"bin": b, "dur_s": r["duration_s"], "asr_ms": round(asr_s * 1000, 1), "asr_decode_steps": asr_steps,
                     "llm_ms": round(llm_s * 1000, 1), "llm_out_tokens": out_tok, "total_ms": round((asr_s + llm_s) * 1000, 1), "json_ok": ok})
        transcripts.append(tr)
    # 吞吐：所有事件的 LLM 部分一次合批（ASR 部分逐条，上面已计）
    t0 = time.perf_counter(); llm.generate([chat(t) for t in transcripts], sp, use_tqdm=False); batch_s = time.perf_counter() - t0
    def summ(xs):
        return {"n": len(xs), "median": round(st.median(xs), 1), "p95": round(float(np.percentile(xs, 95)), 1)}
    rep = {"asr": ASR_ID, "llm": LLM_ID, "framework": "transformers generate (ASR) + vLLM (LLM), bf16, greedy", "n_questions": len(QUESTIONS),
           "by_bin": {}, "all": {}, "llm_batch_all_events": {"n_events": len(transcripts), "wall_ms": round(batch_s * 1000, 1), "events_per_s": round(len(transcripts) / batch_s, 2)},
           "rows": rows, "json_ok_rate": round(sum(r["json_ok"] for r in rows) / len(rows), 3)}
    for key in ("asr_ms", "asr_decode_steps", "llm_ms", "llm_out_tokens", "total_ms"):
        rep["all"][key] = summ([r[key] for r in rows])
        for b in sorted(set(r["bin"] for r in rows)):
            rep["by_bin"].setdefault(b, {})[key] = summ([r[key] for r in rows if r["bin"] == b])
    json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep["all"], ensure_ascii=False)); print(rep["llm_batch_all_events"])

if __name__ == "__main__":
    main()
