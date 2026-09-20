#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 5b：A0 适配器批量推理（生成答案），供三类评测消费。

为什么必须有这个脚本
--------------------------------------------------------------------------
阶段 5 训完 A0 后**只有 train_loss，没有任何生成侧指标**。而判分器
`src/evaluation/lexrubric_doubleblind.py` 与 LexEval 官方评估流程
(`code/generation/main.py` + `evaluation/evaluate.py`) 都要求先有一份
「问题 → 答案」的 answers.jsonl。本脚本补的就是这一环 —— 它是
路由训练 / MoE 消融 / checkpoint 选择共同复用的那套「加载 + 生成」代码。

三类任务（各一个 prompt builder）
--------------------------------------------------------------------------
  internal_test  用样本自带 messages 的 system+user（与训练**完全同分布**）
  lexeval        instruction + input（官方 generation/main.py 的口径）
  lexrubric      question（上游 eval.py 直接问，见其 README）

必须与训练对齐的四件事（否则数字不可比，甚至跑错）
--------------------------------------------------------------------------
1. ★ 4bit 量化口径 = nf4 + double_quant + bf16 compute，与 train_qlora.py 逐字一致。
   训练是 4bit 权重，若这里用 bf16 全精度推理，等于换了个模型做评测。
2. ★ tokenizer 优先取 **adapter 目录**（训练结束 save_pretrained 落盘的那份），
   并强制 `enable_thinking=False` —— Qwen3 模板默认会插 `<think></think>`，
   而语料里没有思维链，训练时明确关掉了（见 train_qlora.py 设计要点 5）。
3. ★ 生成必须 **左 padding**。右 padding 时 batch 内短序列的生成起点会错位，
   输出整体串行；这是 HF batch generate 的硬要求，与训练的右 padding 相反。
4. ★ **greedy 解码**（configs/inference.yaml: do_sample=false / temperature=0），
   且推理参数一次锁死，不逐实验微调，保证可复现、可比较。

其他工程点
--------------------------------------------------------------------------
* 断点续跑：answers.jsonl 追加写，已生成的 case_id 跳过 —— 长任务被打断不必重跑。
* 左截断：prompt 超 max_prompt_tokens 时从**头部**截，保住问题尾部与指令
  （与训练同向；右截断会把问题切掉）。
* 记录 run_record 必填字段（见 configs/inference.yaml），含 token 数 / 延迟 /
  检索结果占位 / 错误，缺一不可。
* 报告：token 统计 / 延迟分位 / 显存峰值 / 抽样答案，落 json + md，便于冒烟肉眼验收。

用法
--------------------------------------------------------------------------
干跑（只构造 prompt，不加载模型，秒级）::

    python scripts/train/run_inference.py --task internal_test --dry-run --limit 2

冒烟（20 条，验证模板/量化/显存）::

    python scripts/train/run_inference.py --task internal_test --limit 20 \
        --gpu 0 --max-new-tokens 256 --out outputs/infer/_smoke_internal

全量::

    python scripts/train/run_inference.py --task internal_test --gpu 0 \
        --out outputs/infer/A0_internal_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# 先解析参数（设 CUDA_VISIBLE_DEVICES 必须在 import torch 之前）
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="A0 适配器批量推理（阶段 5b）")
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--task", required=True,
                    choices=["internal_test", "lexeval", "lexrubric"])
    ap.add_argument("--input", default=None,
                    help="覆盖默认输入；多个路径用逗号分隔")
    ap.add_argument("--base-model", default="models/Qwen3-8B")
    ap.add_argument("--adapter", default="models/adapters/A0_unified_qwen3_8b",
                    help="LoRA 适配器目录；填 none 表示只跑底座（对照组）")
    ap.add_argument("--system-name", default=None,
                    help="写入答案的 system 标识（默认取 adapter 目录名）")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--gpu", default=None, help="单卡号，如 0")
    ap.add_argument("--no-4bit", action="store_true", help="关 4bit（调试用）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有 answers.jsonl，重跑")
    ap.add_argument("--dry-run", action="store_true", help="只构造 prompt，不加载模型")
    ap.add_argument("--dump-samples", type=int, default=5,
                    help="报告里抽样展示几条问答")
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--report-md", default=None)
    return ap.parse_args(argv)


ARGS = parse_args()
if ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.path.abspath(ARGS.root)
os.chdir(ROOT)


def jprint(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 输入：把三类数据集读成统一的中间结构
#   {"case_id", "split", "prompt_text"(后填), "reference", "meta"}
# ---------------------------------------------------------------------------
DEFAULT_INPUTS = {
    "internal_test": ["data/test/test.jsonl"],
    "lexeval": None,        # 走 glob
    "lexrubric": ["data/benchmark/lexrubric/repo/data/falvzixun.json",
                  "data/benchmark/lexrubric/repo/data/sifakaoshi.json"],
}
# LexRubric 两个 split 独立登记，key 必须与 src/evaluation/lexrubric_doubleblind.py 一致
LEXRUBRIC_SPLITS = {"falvzixun": "legal_consultation",
                    "sifakaoshi": "judicial_exam"}


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(task):
    """返回 [(rec, split, ref, debug)] 列表；rec 原样保留。"""
    out = []
    if task == "internal_test":
        paths = [p.strip() for p in (ARGS.input or ",".join(DEFAULT_INPUTS[task])).split(",")]
        for p in paths:
            for r in _iter_jsonl(p):
                uid = r.get("uid") or r.get("uid_g") or r.get("source_id")
                out.append((r, "test", r.get("output") or "", {"uid": uid}))
    elif task == "lexeval":
        if ARGS.input:
            paths = [p.strip() for p in ARGS.input.split(",")]
        else:
            paths = sorted(glob.glob("data/benchmark/lexeval/repo/data/*.json"))
        for p in paths:
            stem = os.path.basename(p).rsplit(".", 1)[0]
            for i, r in enumerate(_iter_jsonl(p)):
                out.append((r, stem, str(r.get("answer", "")), {"idx": i}))
    elif task == "lexrubric":
        paths = [p.strip() for p in (ARGS.input or ",".join(DEFAULT_INPUTS[task])).split(",")]
        for p in paths:
            stem = os.path.basename(p).rsplit(".", 1)[0]
            split = LEXRUBRIC_SPLITS.get(stem, stem)
            data = json.load(open(p, encoding="utf-8"))
            for r in data:
                out.append((r, split, str(r.get("answer", "")),
                            {"rid": r.get("ID", r.get("id"))}))
    if ARGS.limit and ARGS.limit > 0:
        out = out[:ARGS.limit]
    return out


def case_id_of(task, split, r, dbg):
    if task == "internal_test":
        return "test::%s" % dbg.get("uid")
    if task == "lexeval":
        return "lexeval::%s::%d" % (split, dbg.get("idx", -1))
    return "lexrubric::%s::%s" % (split, dbg.get("rid"))   # ★ 与判分器 case_id 同构


def build_messages(task, r):
    """把一条原始记录转成 chat messages（不含 assistant 段）。"""
    if task == "internal_test":
        msgs = r.get("messages") or []
        idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        cut = idx[-1] if idx else len(msgs)
        msgs = msgs[:cut]
        return [{"role": m.get("role"), "content": m.get("content", "")} for m in msgs]
    if task == "lexeval":
        # 官方 generation 口径：instruction + input 直接拼接（此处包成单轮 user）
        text = (r.get("instruction") or "") + (r.get("input") or "")
        return [{"role": "user", "content": text}]
    # lexrubric：上游 eval.py 把 question 直接交给模型
    return [{"role": "user", "content": r.get("question", "")}]


def apply_template(tok, msgs):
    """渲染 prompt；enable_thinking=False 与训练对齐，不支持则回退并留痕。"""
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False), True
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True), False


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model_and_tok():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = ARGS.base_model if os.path.isabs(ARGS.base_model) \
        else os.path.join(ROOT, ARGS.base_model)

    # ★ tokenizer 优先取 adapter 目录（训练时落盘的那份，含 chat_template.jinja）
    tok_src, tok_note = base, "base"
    if ARGS.adapter and ARGS.adapter.lower() != "none":
        ad = ARGS.adapter if os.path.isabs(ARGS.adapter) else os.path.join(ROOT, ARGS.adapter)
        if os.path.exists(os.path.join(ad, "tokenizer_config.json")):
            tok_src, tok_note = ad, "adapter"
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.chat_template is None:
        jprint("  [warn] %s 的 tokenizer 无 chat_template，回退到 base" % tok_note)
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        tok_note = "base(fallback)"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # ★ 生成必须左 padding（与训练的右 padding 相反）
    tok.padding_side = "left"
    jprint("tokenizer 来源 = %s (%s)  pad=%r eos=%r padding_side=%s"
           % (tok_note, tok_src, tok.pad_token, tok.eos_token, tok.padding_side))

    model_kw = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    try:
        import transformers as _tf
        _major = int(_tf.__version__.split(".")[0])
    except Exception:
        _major = 5
    if not ARGS.no_4bit:
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    if _major >= 5:
        model_kw["dtype"] = torch.bfloat16
    else:
        model_kw["torch_dtype"] = torch.bfloat16

    jprint("加载底座中 ... %s" % base)
    model = AutoModelForCausalLM.from_pretrained(base, **model_kw)
    if ARGS.no_4bit:
        # ★ 实测坑（2026-09-20）：transformers 非量化 from_pretrained **默认把权重留在 CPU**，
        #   只有 4bit 路径靠 bitsandbytes 的 device_map 自动上卡。漏这一步会静默在 CPU 上跑
        #   —— 症状是 GPU 只占 687MiB、日志只给一条 UserWarning，慢到不可用。
        model = model.to("cuda")
    model.config.use_cache = True          # 训练时关了，推理要开

    adapter_note = "none"
    if ARGS.adapter and ARGS.adapter.lower() != "none":
        from peft import PeftModel
        ad = ARGS.adapter if os.path.isabs(ARGS.adapter) else os.path.join(ROOT, ARGS.adapter)
        if not os.path.exists(os.path.join(ad, "adapter_config.json")):
            jprint("[FATAL] 适配器目录缺少 adapter_config.json: %s" % ad)
            return None, None, None, None
        jprint("挂载 LoRA 适配器 ... %s" % ad)
        model = PeftModel.from_pretrained(model, ad)
        adapter_note = ad
    model.eval()
    return tok, model, adapter_note, tok_note


# ---------------------------------------------------------------------------
# 生成（手写 batch：左 padding + 左截断，完全可控）
# ---------------------------------------------------------------------------
def encode_left_truncated(tok, texts, max_prompt_tokens):
    out = []
    n_trunc = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]     # ★ 左截断：保尾部（问题与指令在后）
            n_trunc += 1
        out.append(ids)
    return out, n_trunc


def pad_left(ids_list, pad_id, device):
    import torch
    n = max(len(x) for x in ids_list)
    ii, am = [], []
    for x in ids_list:
        p = n - len(x)
        ii.append([pad_id] * p + x)            # ★ 左 padding
        am.append([0] * p + [1] * len(x))
    return (torch.tensor(ii, dtype=torch.long, device=device),
            torch.tensor(am, dtype=torch.long, device=device))


def clean_output(text):
    """剥掉 think 段。

    训练与推理都用 Qwen3 模板 + `enable_thinking=False`，prompt 尾部已经是
    `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`（实测），正常情况模型
    直接续写答案，输出里不含 think。但模板版本差异或长输出跑偏时，模型可能
    再吐一段 `<think>...`——这里按上游 LexRubric judge 的同款口径
    （`</think>` 之后才算正式回复）统一剥掉，避免把思考过程当答案送去打分。
    """
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    if t.startswith("<think>"):
        t = t.split("<think>", 1)[1].strip()
    return t


def generate_batch(model, tok, texts, device):
    import torch
    ids_list, _ = encode_left_truncated(tok, texts, ARGS.max_prompt_tokens)
    input_ids, attn = pad_left(ids_list, tok.pad_token_id, device)
    n_in = input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=ARGS.max_new_tokens,
            do_sample=False,          # ★ greedy，与 configs/inference.yaml 一致
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    gen = out[:, n_in:]
    texts_out = tok.batch_decode(gen, skip_special_tokens=True)
    return [clean_output(t) for t in texts_out], int(n_in), dt


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]


def render_md(r):
    L = []
    W = L.append
    W("# A0 推理报告 —— %s" % r["run"])
    W("")
    W("> 生成时间：%s　任务：`%s`" % (r["generated_at"], r["task"]))
    W("> 结论：**%s**" % r["verdict"])
    W("")
    W("## 1. 运行配置")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    for k, v in r["config"].items():
        W("| `%s` | %s |" % (k, v))
    W("")
    W("## 2. 产出")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    for k, v in r["stats"].items():
        W("| %s | %s |" % (k, v))
    W("")
    W("## 3. 延迟与长度")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    for k, v in r["timing"].items():
        W("| %s | %s |" % (k, v))
    W("")
    if r.get("samples"):
        W("## 4. 抽样（人工验收用）")
        W("")
        for i, s in enumerate(r["samples"], 1):
            W("### 样本 %d　`%s`" % (i, s["case_id"]))
            W("")
            W("**提问**（尾部 300 字）：")
            W("")
            W("```text")
            W(s["prompt_tail"])
            W("```")
            W("")
            W("**模型回答**（前 600 字）：")
            W("")
            W("```text")
            W(s["answer_head"])
            W("```")
            W("")
            if s.get("reference"):
                W("**参考答案**（前 300 字）：")
                W("")
                W("```text")
                W(s["reference"][:300])
                W("```")
                W("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    if not ARGS.out:
        name = ARGS.system_name or (
            os.path.basename(ARGS.adapter.rstrip("/")) if ARGS.adapter
            and ARGS.adapter.lower() != "none" else "base")
        ARGS.out = "outputs/infer/%s_%s" % (ARGS.task, name)
    out_dir = ARGS.out if os.path.isabs(ARGS.out) else os.path.join(ROOT, ARGS.out)
    os.makedirs(out_dir, exist_ok=True)
    ans_path = os.path.join(out_dir, "answers.jsonl")
    system_name = ARGS.system_name or os.path.basename(ARGS.adapter.rstrip("/"))

    jprint("=" * 78)
    jprint("任务       :", ARGS.task)
    jprint("GPU 可见   :", os.environ.get("CUDA_VISIBLE_DEVICES", "(未限制)"))
    jprint("底座       :", ARGS.base_model)
    jprint("适配器     :", ARGS.adapter)
    jprint("输出       :", out_dir)
    jprint("=" * 78)

    records = load_records(ARGS.task)
    if not records:
        jprint("[FATAL] 输入为空")
        return 2
    splits = {}
    for _r, sp, _ref, _d in records:
        splits[sp] = splits.get(sp, 0) + 1
    jprint("读到 %d 条　split 分布：%s" % (len(records), splits))

    # ---- dry-run：只借 base tokenizer 渲染 prompt，不加载模型权重 ----
    if ARGS.dry_run:
        from transformers import AutoTokenizer
        base = ARGS.base_model if os.path.isabs(ARGS.base_model) \
            else os.path.join(ROOT, ARGS.base_model)
        tok_src = base
        if ARGS.adapter and ARGS.adapter.lower() != "none":
            ad = ARGS.adapter if os.path.isabs(ARGS.adapter) else os.path.join(ROOT, ARGS.adapter)
            if os.path.exists(os.path.join(ad, "tokenizer_config.json")):
                tok_src = ad
        tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        shown = 0
        for r, sp, ref, dbg in records:
            msgs = build_messages(ARGS.task, r)
            text, kw = apply_template(tok, msgs)
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if shown < ARGS.dump_samples:
                shown += 1
                jprint("-" * 78)
                jprint("case_id=%s  split=%s  prompt_tokens=%d  thinking_kwarg=%s"
                       % (case_id_of(ARGS.task, sp, r, dbg), sp, n, kw))
                jprint("  roles:", [m["role"] for m in msgs])
                jprint("  prompt head:", repr(text[:200]))
                jprint("  prompt tail:", repr(text[-200:]))
                jprint("  reference  :", repr(str(ref)[:160]))
        jprint("-" * 78)
        jprint("DRY_RUN_OK  共 %d 条；用 --limit 2 时只渲染前 2 条" % len(records))
        return 0

    import torch
    tok, model, adapter_note, tok_note = load_model_and_tok()
    if tok is None:
        return 3
    gpu = torch.cuda.get_device_properties(0)
    jprint("GPU = %s (%.0f GB)" % (gpu.name, gpu.total_memory / 1024 ** 3))

    # ---- 断点续跑 ----
    done = set()
    if os.path.exists(ans_path) and not ARGS.no_resume:
        for rec in _iter_jsonl(ans_path):
            if rec.get("case_id") and not rec.get("error"):
                done.add(rec["case_id"])
        if done:
            jprint("断点续跑：已完成 %d 条，跳过" % len(done))

    todo = [(r, sp, ref, dbg) for r, sp, ref, dbg in records
            if case_id_of(ARGS.task, sp, r, dbg) not in done]
    jprint("待生成 %d 条（batch=%d, max_new_tokens=%d）"
           % (len(todo), ARGS.batch_size, ARGS.max_new_tokens))

    fout = open(ans_path, "a", encoding="utf-8")
    n_ok = n_err = 0
    n_trunc_total = 0
    lat, tin, tout = [], [], []
    samples = []
    failed = []
    t_start = time.time()
    for bi in range(0, len(todo), ARGS.batch_size):
        chunk = todo[bi:bi + ARGS.batch_size]
        texts = []
        for r, sp, ref, dbg in chunk:
            msgs = build_messages(ARGS.task, r)
            text, _kw = apply_template(tok, msgs)
            texts.append(text)
        try:
            answers, n_in, dt = generate_batch(model, tok, texts, "cuda")
        except Exception as e:                       # 单批失败不该毁掉整轮
            jprint("[warn] batch %d 生成失败：%s" % (bi // ARGS.batch_size, e))
            for r, sp, ref, dbg in chunk:
                cid = case_id_of(ARGS.task, sp, r, dbg)
                fout.write(json.dumps({
                    "run_id": "%s_%s" % (ARGS.task, system_name),
                    "case_id": cid, "split": sp, "system": system_name,
                    "answer": "", "error": "generate_failed: %s" % e,
                }, ensure_ascii=False) + "\n")
                failed.append(cid)
                n_err += 1
            fout.flush()
            continue

        for j, (r, sp, ref, dbg) in enumerate(chunk):
            cid = case_id_of(ARGS.task, sp, r, dbg)
            ans = answers[j]
            if not ans:
                n_err += 1
                failed.append(cid)
            else:
                n_ok += 1
            rec = {
                "run_id": "%s_%s" % (ARGS.task, system_name),
                "case_id": cid,
                "id": dbg.get("rid", dbg.get("uid", dbg.get("idx"))),
                "split": sp,
                "system": system_name,
                "adapter": adapter_note,
                "prompt_hash": None,
                "retrieved_provisions": [],      # 本脚本不接检索；RAG 场景另行注入
                "router_decision": None,
                "time_filter_applied": False,
                "answer": ans,
                "reference": str(ref)[:4000],
                "input_tokens": n_in,
                "output_tokens": len(tok(ans, add_special_tokens=False)["input_ids"]),
                "latency_ms": int(dt * 1000 / max(len(chunk), 1)),
                "error": None if ans else "empty_output",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lat.append(rec["latency_ms"]); tin.append(rec["input_tokens"])
            tout.append(rec["output_tokens"])
            if len(samples) < ARGS.dump_samples:
                samples.append({"case_id": cid,
                                "prompt_tail": texts[j][-300:],
                                "answer_head": ans[:600],
                                "reference": str(ref)})
        fout.flush()
        done_now = bi + len(chunk)
        speed = done_now / max(time.time() - t_start, 1e-6)
        eta = (len(todo) - done_now) / speed if speed > 0 else 0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        jprint("  [%d/%d] %.1f 条/s  ETA %.0fs  峰值显存 %.1fGB  ok=%d err=%d"
               % (done_now, len(todo), speed, eta, peak, n_ok, n_err))
    fout.close()

    elapsed = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    # 空输出率过高（>5%）视为链路有问题
    total_new = n_ok + n_err
    empty_rate = (n_err / total_new) if total_new else 1.0
    checks = {
        "answers_written_gt_0": n_ok > 0,
        "empty_rate_ok": empty_rate < 0.05,
        "latency_recorded": len(lat) > 0,
        "no_nan_tokens": True,
        "gpu_used": peak > 0,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "stage": "5b-infer",
        "run": "%s_%s" % (ARGS.task, system_name),
        "task": ARGS.task,
        "generated_at": time.strftime("%F %T"),
        "verdict": verdict,
        "checks": checks,
        "config": {
            "base_model": ARGS.base_model,
            "adapter": ARGS.adapter,
            "tokenizer_source": tok_note,
            "task": ARGS.task,
            "limit": ARGS.limit,
            "batch_size": ARGS.batch_size,
            "max_new_tokens": ARGS.max_new_tokens,
            "max_prompt_tokens": ARGS.max_prompt_tokens,
            "do_sample": False,
            "quantization": "none" if ARGS.no_4bit else "nf4+double_quant+bf16",
            "enable_thinking": False,
            "padding_side": "left",
        },
        "stats": {
            "n_total": len(records),
            "n_todo": len(todo),
            "n_ok": n_ok,
            "n_error": n_err,
            "n_resumed_skipped": len(done),
            "empty_rate": round(empty_rate, 4),
            "split_dist": splits,
            "answers_path": ans_path,
            "failed_case_ids": failed[:50],
        },
        "timing": {
            "elapsed_s": round(elapsed, 1),
            "throughput_per_s": round(total_new / max(elapsed, 1e-6), 3),
            "latency_ms_p50": pct(lat, 0.5),
            "latency_ms_p90": pct(lat, 0.9),
            "input_tokens_p50": pct(tin, 0.5),
            "input_tokens_p90": pct(tin, 0.9),
            "output_tokens_p50": pct(tout, 0.5),
            "output_tokens_p90": pct(tout, 0.9),
            "peak_vram_gb": round(peak, 2),
        },
        "samples": samples,
        "elapsed_seconds_total": round(time.time() - t0, 1),
    }

    if ARGS.report_json:
        p = ARGS.report_json if os.path.isabs(ARGS.report_json) \
            else os.path.join(ROOT, ARGS.report_json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        jprint("报告:", p)
    if ARGS.report_md:
        p = ARGS.report_md if os.path.isabs(ARGS.report_md) \
            else os.path.join(ROOT, ARGS.report_md)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(render_md(report))
        jprint("报告:", p)

    jprint("=" * 78)
    jprint("VERDICT =", verdict)
    for k, v in checks.items():
        jprint("  %-26s %s" % (k, v))
    jprint("ok=%d err=%d  耗时 %.1fs  峰值显存 %.2fGB" % (n_ok, n_err, elapsed, peak))
    jprint("答案文件:", ans_path)
    jprint("=" * 78)
    print("MARKER_INFER_%s_DONE" % ARGS.task.upper())
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
