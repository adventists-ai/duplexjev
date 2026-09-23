"""同一推理引擎（vLLM, Qwen3-32B, 1×H200）下三种判断方式的对照：
  gen_json : ASR 转写 → LLM 逐字生成 10 字段 JSON（= cascade_bench 的做法）
  readout  : ASR 转写 → 10 道题各一行，Jev 式单 token 读出（max_tokens=1 + logprobs），共享前缀靠 vLLM prefix caching
             这是 DuplexJev 在优化引擎上的成本上界：真实系统用音频 embedding（~1.6 token/s）替代转写，且不需要 ASR 解码。
每种都测 ctx=0 和 ctx=1471（已上线车载助手 function_check 的中位输入 token；合成工具目录，不含线上数据）。
输出 artifacts/paper/cascade/engine_bench.json
"""
import os, sys, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from cascade_bench import QUESTIONS, llm_prompt, pick_audio, load16, ROOT, ASR_ID, LLM_ID  # noqa

OUT = f"{ROOT}/artifacts/paper/cascade/engine_bench.json"
LET = "ABCDEFGHIJ"

def main():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from vllm import LLM, SamplingParams
    sel = pick_audio()
    proc = AutoProcessor.from_pretrained(ASR_ID)
    asr = AutoModelForMultimodalLM.from_pretrained(ASR_ID, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    llm = LLM(model=LLM_ID, dtype="bfloat16", gpu_memory_utilization=0.75, max_model_len=8192, enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    def make_ctx(T):
        lines = ["You are an in-car voice assistant. Available tools and policies:"]; k = 0
        while True:
            lines.append(f"- tool_{k:03d}(zone, level, duration): controls vehicle function group {k % 37}; "
                         f"call it only when the driver explicitly asks, confirm safety-critical actions, "
                         f"and never act while the vehicle is reversing. 车辆功能 {k}，仅在用户明确要求时调用。")
            k += 1
            ids = tok("\n".join(lines))["input_ids"]
            if len(ids) >= T: return tok.decode(ids[:T])

    def chat(user):
        return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def json_prompt(ctx, tr):
        return chat((ctx + "\n\n" if ctx else "") + llm_prompt(tr))

    rng = random.Random(0)
    def readout_prompts(ctx, tr):
        ps, letter_sets = [], []
        for k, d, opts in QUESTIONS:
            perm = list(range(len(opts))); rng.shuffle(perm)
            lines = "\n".join(f"{LET[i]}. {opts[j]}" for i, j in enumerate(perm))
            user = (ctx + "\n\n" if ctx else "") + f"用户刚才说：「{tr}」\n\n问题：{d}\n{lines}\n只回答一个选项字母。"
            ps.append(chat(user)); letter_sets.append(LET[:len(opts)])
        return ps, letter_sets

    def run_asr(a, lang):
        inp = proc.apply_transcription_request(audio=a, language="Chinese" if lang == "zh" else "English").to(asr.device, asr.dtype)
        with torch.no_grad(): out = asr.generate(**inp, max_new_tokens=128, do_sample=False)
        return proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0]

    trs = [run_asr(load16(r["audio"]), r["lang"]) for _, r in sel]
    sp_json = SamplingParams(temperature=0.0, max_tokens=256)
    sp_one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    OUTS = f"{ROOT}/artifacts/paper/cascade/engine_slo.json"
    rep = {"llm": LLM_ID, "engine": "vLLM", "note": "k 个事件同时到达，一次 generate；wall = 整批完成时间（最慢事件的延迟）", "rows": []}
    KS = [1, 2, 4, 8, 16, 32, 64]
    for C in [0, 1471]:
        ctx = make_ctx(C) if C else ""
        llm.generate([json_prompt(ctx, "你好")] + readout_prompts(ctx, "你好")[0], sp_one, use_tqdm=False)
        for k in KS:
            evs = [trs[i % len(trs)] for i in range(k)]
            for mode in ["readout", "gen_json"]:
                ws = []
                for rep_i in range(3):
                    ps = sum([readout_prompts(ctx, t)[0] for t in evs], []) if mode == "readout" else [json_prompt(ctx, t) for t in evs]
                    t0 = time.perf_counter(); llm.generate(ps, sp_one if mode == "readout" else sp_json, use_tqdm=False); ws.append((time.perf_counter() - t0) * 1000)
                w = float(np.median(ws))
                row = {"ctx": C, "mode": mode, "k_events": k, "wall_ms": round(w, 1), "events_per_s": round(k / w * 1000, 2), "decisions_per_s": round(10 * k / w * 1000, 1)}
                rep["rows"].append(row); print(json.dumps(row), flush=True); json.dump(rep, open(OUTS, "w"), indent=1)
    print("SLO_DONE", flush=True)

if __name__ == "__main__":
    main()
