#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线出图：把训练 / 门控 / 等价性 / 终评结果画成论文与汇报用图。

设计原则
--------------------------------------------------------------------------
* **不占 GPU、不联网、只读**：纯 CPU 从已落盘的 json 读数据画图。
* **容错优先**：每张图独立 try/except —— 缺数据的图跳过并打印原因，
  绝不因为一张图缺输入而让整批失败（过夜链里任何一个环节都不能拖垮后续）。
* **英文标注**：服务器无中文字体，写中文会出豆腐块 ▯。图内一律英文。
* 输出：`docs/figures/*.png`（150 dpi，够论文用）。

用法
--------------------------------------------------------------------------
    envs/main/bin/python scripts/eval/make_figures.py
    envs/main/bin/python scripts/eval/make_figures.py --out-dir docs/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # 无显示环境必须
import matplotlib.pyplot as plt            # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

# --------------------------------------------------------------------------
# 训练运行 → (图例名, 报告路径候选)
# ---------------------------------------------------------------------------
TRAIN_RUNS = [
    ("A0-unified (general)", ["docs/train/A0_unified_report.json",
                              "docs/train/A0_unified_qwen3_8b_report.json"]),
    ("Criminal expert",      ["docs/train/A0_criminal_report.json"]),
    ("Civil expert",         ["docs/train/A0_civil_report.json"]),
    ("Procedure expert",     ["docs/train/A0_procedure_report.json"]),
]

# 终评矩阵：行 = 系统（**必须英文** —— 服务器无中文字体，写中文会出豆腐块 ▯）
# 键 → 图上的英文行标签；collect_results.py 的 systems 键与之一致
MATRIX_ROWS = [
    ("base",      "base (Qwen3-8B)"),
    ("A0",        "A0 (unified adapter)"),
    ("moe_L2",    "MoE-L2 (gated mixture)"),
    ("criminal",  "Expert: criminal"),
    ("civil",     "Expert: civil"),
    ("procedure", "Expert: procedure"),
]
MATRIX_COLS = [
    ("lexrubric_pct", "LexRubric\n(norm. %)", "%.1f"),
    ("lexeval_acc",   "LexEval obj\nAccuracy", "%.4f"),
    ("lexeval_rouge", "LexEval gen\nROUGE-L", "%.4f"),
    ("internal_rouge", "Internal\nROUGE-L", "%.4f"),
    ("internal_statute", "Internal\nStatute hit", "%.4f"),
]


def jprint(*a):
    print(*a, flush=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------------------
# 图 1：四条训练 loss 曲线
# ---------------------------------------------------------------------------
def fig_training_loss(out_dir):
    series = []
    for name, cands in TRAIN_RUNS:
        p = first_existing(cands)
        if not p:
            jprint("  [skip] %s：报告不存在（试过 %s）" % (name, cands))
            continue
        d = load_json(p)
        lc = d.get("loss_curve") or []
        if len(lc) < 2:
            jprint("  [skip] %s：loss_curve 长度 %d 不足以画图" % (name, len(lc)))
            continue
        xs = [float(a) for a, _ in lc]
        ys = [float(b) for _, b in lc]
        series.append((name, xs, ys, d.get("verdict"), d.get("train_seconds")))
    if not series:
        raise RuntimeError("没有任何可用的 loss_curve")

    f, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, xs, ys, verdict, sec in series:
        lab = "%s%s" % (name, "" if verdict is None else "  [%s]" % verdict)
        ax.plot(xs, ys, linewidth=1.3, label=lab)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.set_title("QLoRA training loss of L2 expert bank (Qwen3-8B, 4-bit nf4, LoRA r=16)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7.5)
    f.tight_layout()
    p = os.path.join(out_dir, "fig_training_loss_curves.png")
    f.savefig(p)
    plt.close(f)
    return p, len(series)


# --------------------------------------------------------------------------
# 图 2：门控专家使用率（L2_GATE.json: expert_utilization）
# --------------------------------------------------------------------------
def fig_expert_utilization(out_dir):
    cands = ["docs/moe/L2_GATE.json"]
    p = first_existing(cands)
    if not p:
        raise RuntimeError("docs/moe/L2_GATE.json 不存在（门控未训完）")
    d = load_json(p)
    util = d.get("expert_utilization") or d.get("expert_usage")
    if not util:
        raise RuntimeError("报告里没有 expert_utilization 字段")
    if isinstance(util, dict):
        names = list(util.keys())
        vals = [float(util[k]) for k in names]
    else:
        names = [str(i) for i in range(len(util))]
        vals = [float(v) for v in util]

    f, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = ax.bar(names, vals, color="#4878b0", width=0.6)
    ax.set_ylabel("Mean gate weight")
    ax.set_title("L2 gate expert utilization (eval split)")
    if vals and max(vals) > 0:
        ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.3f" % v,
                ha="center", va="bottom", fontsize=8)
    if vals and min(vals) < 0.02:
        ax.set_xlabel("WARNING: an expert is near 0 -> gate collapse, raise --balance-alpha",
                      fontsize=7, color="#b03030")
    f.tight_layout()
    q = os.path.join(out_dir, "fig_expert_utilization.png")
    f.savefig(q)
    plt.close(f)
    return q, dict(zip(names, vals))


# --------------------------------------------------------------------------
# 图 3：门控训练 loss（L2_GATE.json loss_curve，若有）
# ---------------------------------------------------------------------------
def fig_gate_loss(out_dir):
    p = first_existing(["docs/moe/L2_GATE.json"])
    if not p:
        raise RuntimeError("docs/moe/L2_GATE.json 不存在")
    d = load_json(p)
    lc = d.get("loss_curve") or []
    if len(lc) < 2:
        raise RuntimeError("L2_GATE.json 无 loss_curve（长度 %d）" % len(lc))
    xs = [float(a) for a, _ in lc]
    ys = [float(b) for _, b in lc]
    f, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs, ys, color="#b06a30", linewidth=1.3)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Gate training loss")
    ax.set_title("L2 gating training loss (experts frozen, 2.36M trainable)")
    ax.grid(alpha=0.3, linewidth=0.5)
    f.tight_layout()
    q = os.path.join(out_dir, "fig_l2_gate_loss.png")
    f.savefig(q)
    plt.close(f)
    return q, len(lc)


# --------------------------------------------------------------------------
# 图 4：MoE 装配等价性 —— fp32 逐位相等 vs 4bit 量化偏差
# ---------------------------------------------------------------------------
def fig_equivalence(out_dir):
    fp = first_existing(["docs/moe/VERIFY_MIXTURE_GPU_FP32_36L.json"])
    q4 = first_existing(["docs/moe/VERIFY_MIXTURE_GPU_4BIT.json"])
    if not fp:
        raise RuntimeError("缺 fp32 全 36 层等价性报告")
    a = load_json(fp)
    names = [e["expert"] for e in a["per_expert"]]
    fp_abs = [float(e["max_abs_diff"]) for e in a["per_expert"]]
    q_abs = None
    if q4:
        b = load_json(q4)
        q_abs = [float(e.get("relative", 0.0)) for e in b["per_expert"]]

    f, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.bar(names, fp_abs, color="#3f8f5f", width=0.6)
    ax.set_title("fp32, all 36 layers:  max|delta| = %.1e\n(bit-exact -> assembly & math correct)"
                 % max(fp_abs or [0.0]))
    ax.set_ylabel("max |delta|  (abs)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    ax = axes[1]
    if q_abs:
        tol = float(load_json(q4).get("tolerance_relative", 0.01))
        ax.bar(names, q_abs, color="#b05050", width=0.6)
        ax.axhline(tol, color="#333333", linestyle="--", linewidth=1,
                   label="nominal tol = %.2f" % tol)
        ax.legend(fontsize=7.5)
        ax.set_title("nf4 4-bit: relative diff = quantization noise\n(recorded, NOT a correctness gate)")
        ax.set_ylabel("relative diff")
    else:
        ax.text(0.5, 0.5, "4-bit report missing", ha="center", va="center")
        ax.set_axis_off()
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    f.tight_layout()
    r = os.path.join(out_dir, "fig_moe_equivalence.png")
    f.savefig(r)
    plt.close(f)
    return r, {"fp32_max": max(fp_abs or [0.0]), "4bit_rel": q_abs}


# --------------------------------------------------------------------------
# 图 5：终评矩阵热力图
# --------------------------------------------------------------------------
def fig_results_matrix(out_dir):
    p = "docs/eval/RESULTS_MATRIX.json"
    if not os.path.isfile(p):
        raise RuntimeError("docs/eval/RESULTS_MATRIX.json 不存在（先跑 collect_results.py）")
    d = load_json(p)
    sysmap = d.get("systems", {})
    rows, labels = [], []
    for key, label in MATRIX_ROWS:
        s = sysmap.get(key)
        if s is None:
            continue
        vals = [s.get(k) for k, _, _ in MATRIX_COLS]
        if all(v is None for v in vals):
            continue                 # 整行无数据：画出来只有 n/a，纯噪声
        labels.append(label)
        rows.append(vals)
    if not rows:
        raise RuntimeError("矩阵里没有可画的系统行（所有系统都还没有分数）")

    # 归一化后画色块（每列独立 min-max），数值仍按原样标注
    import math
    norm = []
    for c in range(len(MATRIX_COLS)):
        col = [r[c] for r in rows]
        val = [v for v in col if isinstance(v, (int, float)) and not math.isnan(v)]
        lo, hi = (min(val), max(val)) if val else (0.0, 1.0)
        norm.append([(0.5 if v is None or (isinstance(v, float) and math.isnan(v))
                      else ((v - lo) / (hi - lo) if hi > lo else 1.0)) for v in col])

    f, ax = plt.subplots(figsize=(1.9 * len(MATRIX_COLS) + 2.4, 0.75 * len(labels) + 2.0))
    grid = [[norm[c][r] for c in range(len(MATRIX_COLS))] for r in range(len(labels))]
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MATRIX_COLS)))
    ax.set_xticklabels([t for _, t, _ in MATRIX_COLS], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for r in range(len(labels)):
        for c, (_, _, fmt) in enumerate(MATRIX_COLS):
            v = rows[r][c]
            txt = "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt % v
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.5,
                    color="#111111" if grid[r][c] < 0.6 else "#ffffff")
    ax.set_title("Law-agent evaluation matrix (col normalized; higher = better per column)", fontsize=9.5)
    f.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    f.tight_layout()
    q = os.path.join(out_dir, "fig_results_matrix.png")
    f.savefig(q)
    plt.close(f)
    return q, rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="离线出图（不占 GPU）")
    ap.add_argument("--out-dir", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    todo = [
        ("fig_training_loss", fig_training_loss),
        ("fig_l2_gate_loss", fig_gate_loss),
        ("fig_expert_utilization", fig_expert_utilization),
        ("fig_moe_equivalence", fig_equivalence),
        ("fig_results_matrix", fig_results_matrix),
    ]
    ok, fail = [], []
    for name, fn in todo:
        try:
            p, extra = fn(a.out_dir)
            ok.append((name, p))
            jprint("[OK]   %-24s -> %s  %s" % (name, p, extra))
        except Exception as e:                                  # noqa: BLE001
            fail.append((name, str(e)))
            jprint("[FAIL] %-24s %s" % (name, e))
    jprint("-" * 70)
    jprint("出图成功 %d 张，跳过 %d 张" % (len(ok), len(fail)))
    for n, why in fail:
        jprint("   跳过 %s：%s" % (n, why))
    # 哪怕只出一张也算成功（过夜链不该被一张图卡住）
    print("MARKER_FIGURES_DONE ok=%d fail=%d" % (len(ok), len(fail)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
