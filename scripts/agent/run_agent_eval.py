# -*- coding: utf-8 -*-
"""智能体批量评测：内部 test 抽样 → 逐题 trace(JSONL) + 汇总报告(MD)。

对照设计（论文消融）：--max-steps 1 即"单轮 RAG"基线，--max-steps 2 为
智能体主链路；两者共用同一检索/生成口径，只差「核验 + 补检」。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from agent.law_agent import LawAgent, build_agent  # noqa: E402


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--adapter", default="models/adapters/A0_unified_qwen3_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--final-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--tag", default=None, help="输出文件标签，默认 A0_ms<steps>")
    ap.add_argument("--report-md", default=None)
    args = ap.parse_args()

    root = args.root.rstrip("/")
    tag = args.tag or ("A0_ms%d" % args.max_steps)
    out_dir = os.path.join(root, "outputs", "agent")
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "agent_traces_%s.jsonl" % tag)
    report_md = args.report_md or os.path.join(root, "docs", "agent",
                                               "AGENT_EVAL_%s.md" % tag)

    # 断点续跑：已完成的 question 跳过
    done = set()
    if os.path.exists(trace_path):
        for r in iter_jsonl(trace_path):
            done.add(r["question"])
    fout = open(trace_path, "a", encoding="utf-8")

    agent = build_agent(root, args.adapter, args.device,
                        max_steps=args.max_steps, final_k=args.final_k)

    n = 0
    t_start = time.time()
    for rec in iter_jsonl(os.path.join(root, "data", "test", "test.jsonl")):
        if n >= args.limit:
            break
        q = (rec.get("input") or "").strip() if "input" in rec else \
            (rec["messages"][-2]["content"] if len(rec.get("messages", [])) >= 2 else "")
        if not q or q in done:
            continue
        tr = agent.answer(q)
        tr.domain = rec.get("domain", "")
        fout.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
        fout.flush()
        n += 1
        print("[%d/%d] %.1fs requery=%s citations=%d"
              % (n, args.limit, tr.latency_sec, tr.requery_triggered,
                 len(tr.citations_final)), flush=True)
    fout.close()

    # ---- 汇总 ----
    recs = list(iter_jsonl(trace_path))
    n_total = len(recs)
    n_requery = sum(1 for r in recs if r.get("requery_triggered"))
    cit_first = sum(len(r.get("citations_first") or []) for r in recs)
    cit_final = sum(len(r.get("citations_final") or []) for r in recs)
    ver = [r.get("verified_final") or {} for r in recs]
    n_found = sum(len(v.get("found") or []) for v in ver)
    n_missing = sum(len(v.get("missing") or []) for v in ver)
    n_ooc = sum(len(v.get("out_of_ctx") or []) for v in ver)
    n_unres = sum(len(v.get("law_unresolved") or []) for v in ver)
    n_mentions = sum(r.get("n_mentions") or 0 for r in recs) or \
        sum(v.get("n_mentions") or 0 for v in ver)
    n_verdictable = n_found + n_missing + n_ooc
    lat = [r.get("latency_sec") or 0 for r in recs]
    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0

    md = f"""# 智能体评测：{tag}

> 生成时间：{time.strftime('%F %T')}　adapter：`{args.adapter}`　max_steps={args.max_steps}

| 指标 | 值 |
|---|---:|
| 题数 | {n_total} |
| 补检触发率 | {n_requery / n_total:.4f}（{n_requery}/{n_total}） |
| 引用条数（首轮 → 终答，**去重**） | {cit_first} → {cit_final} |
| 终答引用三态 | 落实 {n_found} / 疑似编造 {n_missing} / 出上下文 {n_ooc} |
| 法名无法解析（单列，**不计入编造**） | {n_unres} |
| 引用落实率（终答，去重后） | {n_found / max(1, n_verdictable):.4f} |
| 引用提及次数（含重复，原始） | {n_mentions} |
| 延迟 p50 / 总耗时 | {p50:.1f}s / {time.time() - t_start:.0f}s |

> 口径说明：同一《法名》第X条在答案里重复出现只计 **1 条**（去重）；
> 法名无法解析到 `law_id` 的引用单列 `law_unresolved`，**不并入「疑似编造」**，
> 避免把「解析器认不出」误报成模型在编造。

逐题 trace：`outputs/agent/agent_traces_{tag}.jsonl`
"""
    os.makedirs(os.path.dirname(report_md), exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
