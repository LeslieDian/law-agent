#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 4b-2b：**热度权重**（用户 2026-09-19 拍板）。

为什么要这一步
--------------
检索排序如果只看向量相似度，会出现「同样相关、但公认更重要/更常被引用的条文排在后面」。
热度权重给这类节点一个**先验加分**，落在 Neo4j 节点属性 + 检索排序公式里。

两个热度信号（都**只从检索库自身**算，不看评测集 → 零泄漏）
--------------------------------------------------------------------------
1. `Provision.hotness_cites`
   CITES 边的**入度**：该条文被其他条文引用多少次（`本法第十二条` / `《刑法》第二百九十四条`）。
   实测语料里 CITES 覆盖稀疏（约 7 千条边），所以它只对**少数被反复引用**的条文生效 ——
   这正是我们要的「热点条文」。

2. `Case.hotness_same_case`
   同一 `case_sha1` 在**检索库内**被多少个样本代表（同案多视角：判决预测 / 文书摘要 / 要素抽取）。
   代表数越多 = 语料里被反复讨论的典型案件。
   ★ 只用检索库内的样本计数，**绝不把 train/val/test/router 的样本算进来** ——
     否则评测题会通过「计数」给被评测案件加权，构成隐形泄漏。

合成公式（写进论文实现细节，不要改了口径不说）
--------------------------------------------------------------------------
    norm_log1p(x) = log1p(x) / log1p(max_x)            # 0..1
    rank_norm     = (LAW_RANK[t] - RANK_MIN) / (RANK_MAX - RANK_MIN)   # 0..1，法源位阶

    Provision.hotness = 0.85 * norm_log1p(in_cites) + 0.15 * rank_norm
    Case.hotness      = 1.00 * norm_log1p(n_same_case)

检索侧用法（`W_HOT` 默认 0.05）::

    final_score = rrf_score * (1 + W_HOT * hotness)

即**热度只做微调先验**（最大 +5%），不允许盖过相关性本身。

输出
----
    indexes/retrieval/hotness.json          # {provision:{pid:{...}}, case:{uid:{...}}, meta}
    docs/retrieval/HOTNESS_REPORT.json/.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter

# 法源位阶（效力等级）—— 只用于 rank_norm，权重差距刻意压小
LAW_RANK = {
    "法律": 1.00,
    "行政法规": 0.85,
    "司法解释": 0.80,
    "法律解释": 0.78,
}
RANK_DEFAULT = 0.70
RANK_MIN = min(list(LAW_RANK.values()) + [RANK_DEFAULT])
RANK_MAX = max(list(LAW_RANK.values()) + [RANK_DEFAULT])

CASE_COLLECTION_PREFIX = "case_embedding_"
PROVISION_COLLECTION = "provision_embedding"


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_log1p(v, vmax):
    if not vmax or vmax <= 0:
        return 0.0
    return math.log1p(max(0, v)) / math.log1p(vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--edges-dir", default="indexes/retrieval")
    ap.add_argument("--emb-root", default="indexes/retrieval/embeddings")
    ap.add_argument("--out-json", default="indexes/retrieval/hotness.json")
    ap.add_argument("--report-json", default="docs/retrieval/HOTNESS_REPORT.json")
    ap.add_argument("--report-md", default="docs/retrieval/HOTNESS_REPORT.md")
    ap.add_argument("--metric", default="cosine")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    os.chdir(root)
    t0 = time.time()
    checks = {}
    notes = []

    edges_dir = args.edges_dir if os.path.isabs(args.edges_dir) else os.path.join(root, args.edges_dir)
    emb_root = args.emb_root if os.path.isabs(args.emb_root) else os.path.join(root, args.emb_root)

    def R(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    # ---------------------------------------------------------------- 法源位阶
    law_type = {}
    lp = os.path.join(edges_dir, "law_nodes.jsonl")
    if os.path.exists(lp):
        for r in iter_jsonl(lp):
            law_type[r["law_id"]] = r.get("law_type_std") or "未标注"
    print("[1/4] law_nodes =%d" % len(law_type), flush=True)

    # ---------------------------------------------------------------- CITES 入度
    in_cites = Counter()
    out_cites = Counter()
    ep = os.path.join(edges_dir, "edges_cites.jsonl")
    n_cites = 0
    if os.path.exists(ep):
        for e in iter_jsonl(ep):
            in_cites[e["end"]] += 1
            out_cites[e["start"]] += 1
            n_cites += 1
    print("[2/4] CITES 边 =%d  被引条文(入度>0) =%d" % (n_cites, len(in_cites)), flush=True)
    checks["cites_edges_read"] = n_cites > 0

    # ---------------------------------------------------------------- Provision 热度
    prov_hot = {}
    max_in = max(in_cites.values()) if in_cites else 0
    by_level = Counter()
    by_domain = Counter()
    n_prov_rows = 0
    prov_meta_glob = os.path.join(emb_root, PROVISION_COLLECTION, "shard_*.jsonl")
    for p in sorted(glob.glob(prov_meta_glob)):
        for m in iter_jsonl(p):
            pid = m.get("id")
            if not pid:
                continue
            n_prov_rows += 1
            law_id = m.get("law_id") or ""
            lt = law_type.get(law_id, "未标注")
            rank = LAW_RANK.get(lt, RANK_DEFAULT)
            rank_norm = (rank - RANK_MIN) / (RANK_MAX - RANK_MIN) if RANK_MAX > RANK_MIN else 0.0
            ic = in_cites.get(pid, 0)
            hot = 0.85 * norm_log1p(ic, max_in) + 0.15 * rank_norm
            prov_hot[pid] = {
                "hotness": round(hot, 6),
                "in_cites": ic,
                "out_cites": out_cites.get(pid, 0),
                "law_id": law_id,
                "law_type_std": lt,
                "law_rank": rank,
                "level": m.get("level"),
                "domain4": m.get("domain4"),
            }
            by_level[m.get("level") or "?"] += 1
            by_domain[m.get("domain4") or "?"] += 1
    print("[3/4] provision 热度 =%d  按 level=%s  按域=%s"
          % (len(prov_hot), dict(by_level), dict(by_domain)), flush=True)
    checks["provision_hotness_gt_0"] = len(prov_hot) > 0
    # ★ 真校验（原来是写死的 True，是「假断言」）：
    #   向量库 meta 行数 == 热度表条目数 —— provision_id 有唯一约束，
    #   一对一不漏才说明「每个可检索条文都有热度」（含 level=doc 的 doc 级行）。
    checks["provision_hotness_covers_all"] = (
        n_prov_rows > 0 and len(prov_hot) == n_prov_rows)
    if n_prov_rows != len(prov_hot):
        notes.append("provision 热度覆盖不全：meta 行 %d vs 热度条目 %d（有 provision_id 重复？）"
                     % (n_prov_rows, len(prov_hot)))

    # ---------------------------------------------------------------- Case 热度
    case_hot = {}
    case_multi = Counter()
    case_sha1_of = {}
    n_case_rows = 0
    for p in sorted(glob.glob(os.path.join(emb_root, CASE_COLLECTION_PREFIX + "*", "shard_*.jsonl"))):
        for m in iter_jsonl(p):
            uid = m.get("uid")
            if not uid:
                continue
            n_case_rows += 1
            cs = m.get("case_sha1") or ""
            case_sha1_of[uid] = cs
            if cs:
                case_multi[cs] += 1
    max_multi = max(case_multi.values()) if case_multi else 0
    for uid, cs in case_sha1_of.items():
        n = case_multi.get(cs, 0)
        case_hot[uid] = {
            "hotness": round(norm_log1p(n, max_multi), 6),
            "n_same_case": n,
            "case_sha1": cs,
        }
    print("[4/4] case 热度 =%d  唯一 case_sha1 =%d  同案最大复现 =%d"
          % (len(case_hot), len(case_multi), max_multi), flush=True)
    checks["case_hotness_gt_0"] = len(case_hot) > 0
    checks["case_multiplicity_gt_0"] = max_multi > 0
    # ★ 真校验：4 个 case 库的 meta 总行数 == 唯一 uid 数（uid 必须跨库不撞）
    checks["case_hotness_covers_all"] = (
        n_case_rows > 0 and len(case_sha1_of) == n_case_rows)
    if n_case_rows != len(case_sha1_of):
        notes.append("case uid 在 4 个库间有重复：行 %d vs 唯一 uid %d"
                     % (n_case_rows, len(case_sha1_of)))

    # ---------------------------------------------------------------- 断言/说明
    if not checks["cites_edges_read"]:
        notes.append("CITES 边为空 → 所有 provision 热度只剩法源位阶项（0.15 权重），请核查 4b-2 产物")
    if not prov_hot:
        notes.append("provision 热度为空 → 多半是 embeddings 未生成或 --emb-root 不对")

    verdict = "PASS" if all(checks.values()) else "FAIL"

    top_prov = sorted(prov_hot.items(), key=lambda kv: -kv[1]["hotness"])[:20]
    top_case = sorted(case_hot.items(), key=lambda kv: -kv[1]["hotness"])[:20]
    hist = Counter()
    for v in prov_hot.values():
        hist["in_cites==0"] += (v["in_cites"] == 0)
        hist["in_cites>=1"] += (v["in_cites"] >= 1)
        hist["in_cites>=5"] += (v["in_cites"] >= 5)
        hist["in_cites>=20"] += (v["in_cites"] >= 20)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provision": prov_hot,
        "case": case_hot,
        "meta": {
            "formula": {
                "norm_log1p": "log1p(x)/log1p(max_x)",
                "provision_hotness": "0.85*norm_log1p(in_cites) + 0.15*rank_norm",
                "case_hotness": "norm_log1p(n_same_case)",
                "ranking": "final = rrf_score * (1 + W_HOT * hotness)   [W_HOT 默认 0.05]",
            },
            "law_rank": LAW_RANK, "rank_default": RANK_DEFAULT,
            "max_in_cites": max_in, "max_same_case": max_multi,
            "n_provision": len(prov_hot), "n_case": len(case_hot),
            "leakage_note": "case 复现次数只在**检索库内**统计，不含 train/val/test/router",
        },
    }
    op = R(args.out_json)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    rep = {
        "generated_at": out["generated_at"], "verdict": verdict,
        "checks": checks, "notes": notes,
        "formula": out["meta"]["formula"], "law_rank": LAW_RANK,
        "counts": {"provision": len(prov_hot), "case": len(case_hot),
                   "provision_meta_rows": n_prov_rows, "case_meta_rows": n_case_rows,
                   "cites_edges": n_cites, "unique_case_sha1": len(case_multi)},
        "provision_in_cites_hist": dict(hist),
        "provision_by_level": dict(by_level), "provision_by_domain": dict(by_domain),
        "top_provision": [{"provision_id": k, **v} for k, v in top_prov],
        "top_case": [{"uid": k, **v} for k, v in top_case],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rj = R(args.report_json)
    os.makedirs(os.path.dirname(rj), exist_ok=True)
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    md = []
    W = md.append
    W("# 阶段 4b-2b 热度权重报告\n")
    W("> 生成时间：%s　脚本：`scripts/retrieval/build_hotness.py`　结论：**%s**\n"
      % (rep["generated_at"], verdict))
    W("## 1. 公式（论文实现细节，口径不得改动而不声明）\n")
    W("```")
    for k, v in rep["formula"].items():
        W("%-22s %s" % (k, v))
    W("```\n")
    W("## 2. 规模\n")
    W("| 项 | 值 |")
    W("|---|---:|")
    for k, v in rep["counts"].items():
        W("| %s | %d |" % (k, v))
    W("")
    W("| 入度分档 | 条文数 |")
    W("|---|---:|")
    for k in ("in_cites==0", "in_cites>=1", "in_cites>=5", "in_cites>=20"):
        W("| %s | %d |" % (k, rep["provision_in_cites_hist"].get(k, 0)))
    W("")
    W("| 法源 | 条文数 |")
    W("|---|---:|")
    for k, v in sorted(rep["provision_by_level"].items()):
        W("| %s | %d |" % (k, v))
    W("")
    W("| 域 | 条文数 |")
    W("|---|---:|")
    for k, v in sorted(rep["provision_by_domain"].items()):
        W("| %s | %d |" % (k, v))
    W("")
    W("## 3. 热点条文 Top20（按热度）\n")
    W("| provision_id | 热度 | 被引 | 法律 | 法源 |")
    W("|---|---:|---:|---|---|")
    for x in rep["top_provision"]:
        W("| `%s` | %.4f | %d | %s | %s |"
          % (x["provision_id"][:52], x["hotness"], x["in_cites"],
             x["law_id"][:26], x["law_type_std"]))
    W("")
    W("## 4. 热点案件 Top20（按同案复现次数）\n")
    W("| uid | 热度 | 同案复现 |")
    W("|---|---:|---:|")
    for x in rep["top_case"]:
        W("| `%s` | %.4f | %d |" % (str(x["uid"])[:52], x["hotness"], x["n_same_case"]))
    W("")
    W("## 5. 泄漏说明\n")
    W("案件复现次数**只在检索库内**统计，不含 `train` / `dev` / `test` / `router` 的任何样本；")
    W("provision 热度只用 CITES 边与法源位阶，与评测集无关。**本条须写进论文。**\n")
    rm = R(args.report_md)
    os.makedirs(os.path.dirname(rm), exist_ok=True)
    with open(rm, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("=" * 62)
    print("verdict =", verdict)
    for k, v in checks.items():
        print("  %-30s %s" % (k, v))
    print("热度表 ->", op)
    print("报告   ->", rj)
    print("HOTNESS_DONE")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
