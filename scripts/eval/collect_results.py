#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总终评结果 → 一张可比矩阵（json + markdown + README 数据块）。

读什么（全部只读、容错）
--------------------------------------------------------------------------
* `outputs/score/lexeval_<sys>.json`   ← `scripts/eval/score_answers.py --task lexeval --metrics all`
* `outputs/score/internal_<sys>.json`  ← 同上，`--task internal_test --metrics all`
* `outputs/judge/lexrubric/<sys>/SUMMARY.json` ← `src.evaluation.lexrubric_doubleblind`

写什么
--------------------------------------------------------------------------
* `docs/eval/RESULTS_MATRIX.json`  —— 机读，供 `make_figures.py` 画热力图
* `docs/eval/RESULTS_MATRIX.md`    —— 人读表格
* `docs/eval/README_BLOCK.md`      —— 可直接粘进仓库 README 的一段（带 AUTO 标记）

用法
--------------------------------------------------------------------------
    envs/main/bin/python scripts/eval/collect_results.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 系统 → (显示名, score 文件名前缀, judge 目录名)
SYSTEMS = [
    ("base",   "base (Qwen3-8B, 无微调)", "base",   "base"),
    ("A0",     "A0 (统一适配器)",          "A0",     "A0"),
    ("moe_L2", "MoE-L2 (L2 门控混合)",     "moe_L2", "moe_L2"),
    ("criminal", "专家: criminal",         "criminal", "criminal"),
    ("civil",    "专家: civil",            "civil",    "civil"),
    ("procedure", "专家: procedure",       "procedure", "procedure"),
]


# 内部验证集 answers 目录特例（历史命名，见 collect_one 注释）
INTERNAL_ANS = {
    "A0": "outputs/infer/A0_internal_test/answers.jsonl",
}


def jprint(*a):
    print(*a, flush=True)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                            # noqa: BLE001
        return None


def n_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:                                            # noqa: BLE001
        return 0


def pick(d, *path, default=None):
    """安全取嵌套键。"""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_one(key, display, score_prefix, judge_name):
    row = {"key": key, "display": display}
    warns = []

    # ---- answers 规模（"跑没跑" 的第一手证据）----
    # ★ A0 的内部集目录名是历史遗留的 `A0_internal_test`（不是 `internal_A0`），
    #   而打分产物统一叫 `outputs/score/internal_A0.json` —— 两边命名不一致，
    #   这里显式映射，避免"有分数但 answers 显示 0 行"的假象。
    internal_ans = INTERNAL_ANS.get(key, "outputs/infer/internal_%s/answers.jsonl" % key)
    row["n_answers"] = {
        "lexrubric": n_lines("outputs/infer/lexrubric_%s/answers.jsonl" % score_prefix),
        "lexeval":   n_lines("outputs/infer/lexeval_%s/answers.jsonl" % score_prefix),
        "internal":  n_lines(internal_ans),
    }

    # ---- LexEval 客观 + 生成（同一份 answers.jsonl，score_answers 自动分流）----
    s = load_json("outputs/score/lexeval_%s.json" % score_prefix)
    if s:
        acc = pick(s, "metrics", "accuracy", default={}) or {}
        rg = pick(s, "metrics", "rouge_l", default={}) or {}
        row["lexeval_acc"] = acc.get("exact_match")
        row["lexeval_acc_subset"] = acc.get("subset_match")
        row["lexeval_n_choice"] = acc.get("n_choice_questions")
        row["lexeval_unparsed"] = acc.get("unparsed")
        row["lexeval_rouge"] = rg.get("f_mean")
        row["lexeval_rouge_n"] = rg.get("n")
        row["lexeval_by_task"] = pick(acc, "by_task", default=None)
    else:
        warns.append("缺 outputs/score/lexeval_%s.json" % score_prefix)

    # ---- 内部验证集 ----
    si = load_json("outputs/score/internal_%s.json" % score_prefix)
    if si:
        row["internal_rouge"] = pick(si, "metrics", "rouge_l", "f_mean")
        row["internal_statute"] = pick(si, "metrics", "statute_hit", "hit_rate_mean")
        row["internal_acc"] = pick(si, "metrics", "accuracy", "exact_match")
        row["internal_n"] = si.get("n_valid")
    else:
        warns.append("缺 outputs/score/internal_%s.json" % score_prefix)

    # ---- LexRubric（MiniMax-M3 判分）----
    js = load_json("outputs/judge/lexrubric/%s/SUMMARY.json" % judge_name)
    if js:
        ov = js.get("overall", {}) or {}
        row["lexrubric_mean"] = pick(ov, "judge_mean", "mean")
        row["lexrubric_pct"] = pick(ov, "normalized_pct", "mean")
        row["lexrubric_ci95"] = pick(ov, "normalized_pct", "ci95")
        row["lexrubric_n"] = ov.get("n")
        row["lexrubric_by_split"] = {
            k: {"n": v.get("n"),
                "mean": pick(v, "judge_mean", "mean"),
                "pct": pick(v, "normalized_pct", "mean")}
            for k, v in (js.get("by_split") or {}).items()
        }
        row["lexrubric_dimensions"] = pick(ov, "by_dimension", default=None)
    else:
        warns.append("缺 outputs/judge/lexrubric/%s/SUMMARY.json" % judge_name)

    row["warnings"] = warns
    return row


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/eval")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for key, display, sp, jn in SYSTEMS:
        r = collect_one(key, display, sp, jn)
        # 完全没数据的系统（专家行在没跑的时候）不进矩阵，避免误导
        has_any = any(r.get(k) is not None for k in
                      ("lexrubric_pct", "lexeval_acc", "internal_rouge", "internal_statute"))
        if not has_any and sum(r["n_answers"].values()) == 0:
            jprint("[skip] %-10s 无任何产出" % key)
            continue
        rows.append(r)
        jprint("[ok]   %-10s rubric_pct=%s acc=%s rouge=%s int_rouge=%s int_stat=%s  警告=%d"
               % (key, fmt(r.get("lexrubric_pct"), 2), fmt(r.get("lexeval_acc")),
                  fmt(r.get("lexeval_rouge")), fmt(r.get("internal_rouge")),
                  fmt(r.get("internal_statute")), len(r["warnings"])))

    doc = {"n_systems": len(rows), "systems": {r["key"]: r for r in rows}, "rows": rows}
    with open(os.path.join(a.out_dir, "RESULTS_MATRIX.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    # ---- markdown 表 ----
    hdr = ("| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L "
           "| 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |")
    sep = "|---|---:|---:|---:|---:|---:|---|"
    lines = [hdr, sep]
    for r in rows:
        na = r["n_answers"]
        lines.append("| %s | %s | %s | %s | %s | %s | %d / %d / %d |" % (
            r["display"], fmt(r.get("lexrubric_pct"), 2), fmt(r.get("lexeval_acc")),
            fmt(r.get("lexeval_rouge")), fmt(r.get("internal_rouge")),
            fmt(r.get("internal_statute")), na["lexrubric"], na["lexeval"], na["internal"]))
    md = "\n".join(lines) + "\n"
    with open(os.path.join(a.out_dir, "RESULTS_MATRIX.md"), "w", encoding="utf-8") as f:
        f.write("# law-agent 终评结果矩阵\n\n" + md)
        f.write("\n> LexRubric 归一化 % = mean_total / max_score × 100（逐 case 平均）；"
                "LexEval 客观 Acc 为 exact_match；ROUGE-L 为字符级 LCS f，cap=800。\n")
        warn = [(r["key"], w) for r in rows for w in r["warnings"]]
        if warn:
            f.write("\n## 缺失项\n\n")
            for k, w in warn:
                f.write("- `%s`：%s\n" % (k, w))

    # ---- README 块（供自动 patch 进仓库 README）----
    rb = ["<!-- AUTO_RESULTS_BEGIN 由 scripts/eval/collect_results.py 自动生成，勿手改 -->",
          "### 终评结果（自动汇总）", "", md.strip(), "",
          "> 生成口径：LexEval 客观题 cap=256 / 生成题 cap=1536 / LexRubric cap=1536；",
          "> 判分 MiniMax-M3（`configs/judge.yaml`）；完整机读数据见 "
          "`docs/eval/RESULTS_MATRIX.json`，图见 `docs/figures/`。",
          "<!-- AUTO_RESULTS_END -->", ""]
    with open(os.path.join(a.out_dir, "README_BLOCK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rb))
    jprint("写出 %s/{RESULTS_MATRIX.json,RESULTS_MATRIX.md,README_BLOCK.md}" % a.out_dir)
    print("MARKER_COLLECT_DONE n=%d" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
