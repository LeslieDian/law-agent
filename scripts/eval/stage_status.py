#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描过夜链的产物/日志/标志，生成 `docs/eval/STAGE_STATUS.{md,json}`。

为什么要这个脚本
--------------------------------------------------------------------------
用户要求"**每完成一个阶段就更新 README 并推送 GitHub**"。手写进度表必然与
事实漂移；这里把"当前跑到哪"变成**确定性扫描**的产物 —— 只看四类客观证据：
  1. 答案文件行数（outputs/infer/<sys>/answers.jsonl）
  2. 日志里的 `rc=` 与 `[n/N]` 进度
  3. 完成标志（MARKER_* / JUDGE_*）
  4. 报告文件是否存在（docs/moe/L2_GATE.json 等）
任何一项没到位就如实写"待跑/进行中"，**不做推测、不美化**。

用法（在仓库根目录，服务器侧）
--------------------------------------------------------------------------
    envs/main/bin/python scripts/eval/stage_status.py
    # 只打印不落盘
    envs/main/bin/python scripts/eval/stage_status.py --dry-run

输出
--------------------------------------------------------------------------
    docs/eval/STAGE_STATUS.md   ← 由 patch_readme.py --tag AUTO_STAGE 贴进 README
    docs/eval/STAGE_STATUS.json ← 机读版（供本地同步脚本判断"有没有新阶段完成"）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(path: str) -> str:
    p = os.path.join(ROOT, path)
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _nlines(path: str) -> int:
    """数 answers.jsonl 的有效行数（坏行不计）。"""
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return 0
    n = 0
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


def _flat(text: str) -> str:
    """tqdm 用 \\r 刷新，先拍平。"""
    return text.replace("\r", "\n").replace("\x1b", "")


def _progress(log: str, total: int) -> str:
    """从日志里取最后一个 `[n/total]` 或 `n/total [..]` 进度。"""
    t = _flat(_read(log))
    pats = [r"\[(\d+)/%d\]" % total, r"(\d+)/%d \[" % total]
    for pat in pats:
        m = re.findall(pat, t)
        if m:
            return "%s/%d" % (m[-1], total)
    return ""


def _last_rc(log: str) -> str:
    m = re.findall(r"rc=(-?\d+)", _flat(_read(log)))
    return m[-1] if m else ""


def _has(path: str) -> bool:
    return os.path.exists(os.path.join(ROOT, path))


def _marker_in(path: str, marker: str) -> bool:
    return marker in _read(path)


# ---- 阶段定义： (阶段, 系统, 证据函数, 总量) ----------------------------------
def build_rows():
    rows = []

    def add(stage, system, status, progress, note):
        rows.append({"stage": stage, "system": system, "status": status,
                     "progress": progress, "note": note})

    # ---- GPU0：A0 三阶段评测链 ----
    a0_rub = _nlines("outputs/infer/lexrubric_A0_unified_qwen3_8b/answers.jsonl")
    a0_eva = _nlines("outputs/infer/lexeval_A0_unified_qwen3_8b/answers.jsonl")
    a0_chain = "logs/eval_A0_chain.log"
    a0_done = _marker_in(a0_chain, "MARKER_FULL_INFER_A0_DONE")
    add("A0 评测链", "LexRubric 649", "已完成" if a0_rub >= 649 else
        ("进行中" if a0_rub else "待跑"), "%d/649" % a0_rub,
        "cap=1536；AI 裁判判分")
    add("A0 评测链", "LexEval 客观 11,400 + 生成 2,750", "已完成" if a0_eva >= 14150 else
        ("进行中" if a0_eva else "待跑"), "%d/14150" % a0_eva,
        "客观 cap=256 / 生成 cap=1536")
    if a0_done:
        # 三阶段完打标志；用于 gpu0_lane 的等待条件
        add("A0 评测链", "整链标志", "已完成", "", "MARKER_FULL_INFER_A0_DONE")

    # ---- GPU1：L2 门控训练 ----
    gate_pt = _has("models/moe/L2_gate/gate_weights.pt")
    gate_rep = _has("docs/moe/L2_GATE.json")
    gp = _progress("logs/train/L2_gate.log", 1125)
    if gate_pt:
        st = "已完成"
    elif gp:
        st = "进行中"
    else:
        st = "待跑"
    add("阶段 6 门控", "L2 门控训练", st, gp or ("—" if not gp else gp),
        "72 gates / 2.36M 参数；→ %s" % ("gate_weights.pt 已出" if gate_pt else "等 gate_weights.pt"))

    # ---- GPU1：MoE 评测队列 Q1–Q6 ----
    qs = [
        ("MoE 队列", "Q1 MoE 冒烟", "logs/eval/Q1_moe_smoke.log", "outputs/infer/_smoke_moe_L2/answers.jsonl", 2),
        ("MoE 队列", "Q2 MoE 内部集 1k", "logs/eval/Q2_moe_internal.log", "outputs/infer/internal_moe_L2/answers.jsonl", 1000),
        ("MoE 队列", "Q3 MoE LexRubric 649", "logs/eval/Q3_moe_lexrubric.log", "outputs/infer/lexrubric_moe_L2/answers.jsonl", 649),
        ("MoE 队列", "Q4 MoE 客观 11,400", "logs/eval/Q4_moe_lexeval_obj.log", "outputs/infer/lexeval_moe_L2/answers.jsonl", 11400),
        ("MoE 队列", "Q5 MoE 生成 2,750", "logs/eval/Q5_moe_lexeval_gen.log", "outputs/infer/lexeval_moe_L2/answers.jsonl", 14150),
        ("MoE 队列", "Q6 A0 内部集 cap=1024", "logs/eval/Q6_a0_internal_c1024.log", "outputs/infer/A0_internal_test_c1024/answers.jsonl", 1000),
    ]
    for stage, label, log, ans, total in qs:
        n = _nlines(ans)
        rc = _last_rc(log)
        if total and n >= total:
            st = "已完成"
        elif rc and rc != "0":
            st = "失败"
        elif n or _has(log):
            st = "进行中"
        else:
            st = "待跑"
        note = ("rc=%s" % rc) if rc else ""
        if stage == "MoE 队列" and label.startswith("Q1") and not _has("models/moe/L2_gate/gate_weights.pt"):
            st, note = "待跑", "等门控权重"
        add(stage, label, st, ("%d/%d" % (n, total)) if total else str(n), note)
    if _marker_in("logs/gpu1_after_gate_chain.log", "MARKER_GATE_QUEUE_DONE"):
        add("MoE 队列", "整队列标志", "已完成", "", "MARKER_GATE_QUEUE_DONE")

    # ---- GPU0：base 三阶段 + 三专家内部集 ----
    base_rub = _nlines("outputs/infer/lexrubric_base/answers.jsonl")
    base_eva = _nlines("outputs/infer/lexeval_base/answers.jsonl")
    add("base 评测", "LexRubric 649", "已完成" if base_rub >= 649 else
        ("进行中" if base_rub else "待跑"), "%d/649" % base_rub, "cap=1536")
    add("base 评测", "LexEval 客观+生成 14,150", "已完成" if base_eva >= 14150 else
        ("进行中" if base_eva else "待跑"), "%d/14150" % base_eva, "")
    for dom in ("criminal", "civil", "procedure"):
        n = _nlines("outputs/infer/internal_%s/answers.jsonl" % dom)
        add("三专家内部集", "internal_%s 1k" % dom,
            "已完成" if n >= 1000 else ("进行中" if n else "待跑"),
            "%d/1000" % n, "cap=1024；域专业化分析")

    # ---- API：判分 ----
    jr = _read("logs/overnight/judge_results.txt")
    lanelog = _read("logs/overnight/judge_lane.log")
    # 判分通道是串行的 A0 → moe_L2 → base：靠 lane 日志里的 READY/冒烟目录判断轮到谁了
    for sysname, smoke_name in (("A0_unified_qwen3_8b", "_smoke_A0"),
                                ("moe_L2", "_smoke_moe_L2"),
                                ("base", "_smoke_base")):
        ok = ("JUDGE_OK_%s" % sysname) in jr
        fail = ("JUDGE_FAIL_%s" % sysname) in jr
        smoke_fail = ("JUDGE_SMOKE_FAIL_%s" % sysname) in jr
        smoke_here = os.path.isdir(os.path.join(ROOT, "outputs", "judge", "lexrubric", smoke_name))
        if ok:
            st = "已完成"
        elif smoke_fail:
            st = "失败(冒烟)"
        elif fail:
            st = "失败"
        elif smoke_here:
            st = "进行中"
        else:
            # 通道是否已经轮到这个系统？（lane 日志里有 "READY <sys>" 就是已就绪）
            st = "已就绪" if ("READY %s" % sysname) in lanelog else "待跑"
        add("判分(API)", "MiniMax-M3 × %s" % sysname, st, "",
            "649 题 / 22 维度" + ("；冒烟 8 条中" if st == "进行中" else ""))

    # ---- 收尾 ----
    mx = os.path.join(ROOT, "docs", "eval", "RESULTS_MATRIX.json")
    have_mx = os.path.exists(mx)
    mx_note = "collect_results.py + make_figures.py"
    if have_mx:
        mx_note += "；最近一次 %s" % datetime.fromtimestamp(
            os.path.getmtime(mx)).strftime("%m-%d %H:%M")
    add("收尾", "汇总 + 出图", "已完成" if have_mx else "待跑", "", mx_note)
    add("收尾", "过夜链整链", "已完成" if _marker_in("logs/overnight_chain.log", "MARKER_OVERNIGHT_DONE")
        else "进行中", "", "MARKER_OVERNIGHT_DONE")

    return rows


ICON = {"已完成": "✅", "进行中": "🔄", "已就绪": "🟡", "待跑": "⬜", "失败": "❌", "失败(冒烟)": "❌"}


def render_md(rows) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = sum(1 for r in rows if r["status"] == "已完成")
    total = len(rows)
    out = []
    # ★ 块必须自带标记：patch_readme.py 是"连标记一起替换"，
    #   块内不带标记 → 第一次替换后标记就永久消失了（实测踩过）。参见 collect_results.py。
    out.append("<!-- AUTO_STAGE_BEGIN 由 scripts/eval/stage_status.py 自动生成，勿手改 -->")
    out.append("")
    out.append("### 阶段进度（自动汇总）")
    out.append("")
    out.append("> **本段由自动化回填**（`scripts/eval/stage_status.py` 扫描产物/日志/标志后生成，"
               "经 `patch_readme.py --tag AUTO_STAGE` 贴入）。每完成一个阶段刷新一次，"
               "并自动提交推送。机读版：`docs/eval/STAGE_STATUS.json`。")
    out.append(">")
    out.append("> 生成时间 **%s** ｜ 已完成 **%d/%d** 项" % (ts, done, total))
    out.append("")
    out.append("| 阶段 | 项 | 状态 | 进度 | 备注 |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        out.append("| %s | %s | %s %s | %s | %s |" % (
            r["stage"], r["system"], ICON.get(r["status"], ""), r["status"],
            r["progress"] or "—", r["note"] or "—"))
    out.append("")
    out.append("<sub>状态来源：答案文件行数 / 日志 `rc=` 与 `[n/N]` 进度 / 完成标志 / 报告文件。"
               "未到位一律如实标注，不做推测。</sub>")
    out.append("")
    out.append("<!-- AUTO_STAGE_END -->")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = build_rows()
    md = render_md(rows)
    print(md)

    if a.dry_run:
        print("[dry-run] 未落盘")
        return 0

    d = os.path.join(ROOT, "docs", "eval")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "STAGE_STATUS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "done": sum(1 for r in rows if r["status"] == "已完成"),
               "total": len(rows), "rows": rows}
    with open(os.path.join(d, "STAGE_STATUS.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已写 docs/eval/STAGE_STATUS.md + .json")
    print("MARKER_STAGE_STATUS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
