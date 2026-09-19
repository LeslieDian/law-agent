#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-0：检索库范围探查 + 泄漏门禁（只读，不产出任何语料）

背景
----
论文 3.3 / 3.4 把「裁判文书」作为向量化对象（入口 = Case 节点，`Case.embedding` 存 Neo4j），
而阶段 4 的 train / dev / test / router 四份 split **也都来自同一批案件语料**。
若评测样本进了检索库，模型等于「对考题开卷检索」→ 全部 RAG 指标作废。

本脚本只读，回答三件事
--------------------
1. 四份 split 是否两两 disjoint（复核阶段 4 的结论，用 uid 口径独立重算）
2. 清洗语料的 `task_kind` 分布 → 给出「案件正文类」的**可操作口径**
3. 三种检索库候选范围的实得量 + 与评测集的交集断言

用法
----
    python scripts/retrieval/probe_pool_scope.py \
        --root /mnt/data/lidian/law-agent \
        --out-json docs/corpus/RETRIEVAL_POOL_PROBE.json

设计原则
--------
- **只读**：不写任何语料文件，只写一份 JSON 报告
- **口径显式**：所有集合运算都用 `uid`（阶段 2 修复后与 `uid_g` 同口径）
- **汇总由明细重算**：不手写任何总数
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys
import time


def iter_jsonl(path):
    """流式读 jsonl，跳过空行与坏行（坏行计数返回给调用方）。"""
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                bad += 1
    if bad:
        sys.stderr.write("[WARN] %s: %d 行解析失败\n" % (path, bad))


def uid_of(rec):
    """统一取 uid：优先 uid_g（阶段 4 产物），退化到 uid。"""
    return rec.get("uid_g") or rec.get("uid") or ""


def scan_uids(paths, keep_fields=()):
    """扫一遍 jsonl，返回 (uid 列表, 重复统计, 附加字段桶)。"""
    uids = []
    dup = collections.Counter()
    seen = set()
    extra = collections.defaultdict(list)
    n = 0
    for p in paths:
        for rec in iter_jsonl(p):
            n += 1
            u = uid_of(rec)
            if not u:
                dup["__missing_uid__"] += 1
                continue
            uids.append(u)
            if u in seen:
                dup[u] += 1
            else:
                seen.add(u)
            for f in keep_fields:
                extra[f].append(rec.get(f))
    return uids, dup, extra, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--case-kinds", default="case_analysis",
        help="判定「案件正文类」的 task_kind 白名单（逗号分隔）；"
             "实测分布会一并输出，便于核对口径")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    case_kinds = set(x.strip() for x in args.case_kinds.split(",") if x.strip())
    t0 = time.time()

    qa_files = sorted(glob.glob(root + "/data/corpus/decontaminated/qa/*.jsonl"))
    item_files = sorted(glob.glob(root + "/data/corpus/statute_items/*.jsonl"))
    derived_files = sorted(glob.glob(root + "/data/corpus/derived/*.jsonl"))
    split_files = {
        "train": root + "/data/train/train.jsonl",
        "dev": root + "/data/dev/dev.jsonl",
        "test": root + "/data/test/test.jsonl",
        "router": root + "/data/router/router_train.jsonl",
    }

    R = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "host": os.uname().nodename if hasattr(os, "uname") else "",
         "root": root, "case_kinds_used": sorted(case_kinds)}

    # ---------- 1. 四份 split ----------
    split_uids = {}
    for name, p in split_files.items():
        if not os.path.exists(p):
            R.setdefault("errors", []).append("split 不存在: %s" % p)
            continue
        uids = [uid_of(r) for r in iter_jsonl(p)]
        split_uids[name] = set(uids)
        R.setdefault("splits", {})[name] = {
            "file": p.replace(root + "/", ""),
            "rows": len(uids),
            "unique_uids": len(set(uids)),
        }

    # 两两 disjoint（阶段 4 结论的独立复核）
    names = sorted(split_uids)
    pair_overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_overlap["%s∩%s" % (a, b)] = len(split_uids[a] & split_uids[b])
    R["split_pair_overlap"] = pair_overlap
    R["split_pair_overlap_max"] = max(pair_overlap.values()) if pair_overlap else 0

    eval_uids = set()
    for n in ("dev", "test", "router"):
        eval_uids |= split_uids.get(n, set())
    all_split_uids = set()
    for n in split_uids:
        all_split_uids |= split_uids[n]
    R["eval_set"] = {"members": ["dev", "test", "router"],
                     "unique_uids": len(eval_uids)}
    R["all_four_splits"] = {"unique_uids": len(all_split_uids)}

    # ---------- 2. 清洗后 qa：task_kind 分布 ----------
    kind_cnt = collections.Counter()
    kind_uids = collections.defaultdict(set)
    task_cnt = collections.Counter()
    qa_uids = set()
    qa_rows = 0
    for p in qa_files:
        for rec in iter_jsonl(p):
            qa_rows += 1
            u = uid_of(rec)
            qa_uids.add(u)
            k = rec.get("task_kind") or "(none)"
            kind_cnt[k] += 1
            kind_uids[k].add(u)
            task_cnt[rec.get("task") or "(none)"] += 1

    R["qa"] = {
        "files": [p.replace(root + "/", "") for p in qa_files],
        "rows": qa_rows,
        "unique_uids": len(qa_uids),
        "task_kind_counts": dict(sorted(kind_cnt.items(),
                                        key=lambda kv: -kv[1])),
        "task_counts_top30": dict(task_cnt.most_common(30)),
    }

    case_uids = set()
    for k in case_kinds:
        case_uids |= kind_uids.get(k, set())
    R["case_scope"] = {
        "task_kinds": sorted(case_kinds),
        "records": sum(kind_cnt.get(k, 0) for k in case_kinds),
        "unique_uids": len(case_uids),
        "distribution_by_kind": [
            {"task_kind": k,
             "records": kind_cnt.get(k, 0),
             "unique_uids": len(kind_uids.get(k, set())),
             "overlap_with_eval": len(kind_uids.get(k, set()) & eval_uids),
             "overlap_with_all_splits": len(kind_uids.get(k, set()) & all_split_uids)}
            for k in sorted(kind_cnt, key=lambda x: -kind_cnt[x])
        ],
    }

    # ---------- 3. statute_items / derived ----------
    item_uids = set()
    item_rows = 0
    for p in item_files:
        for rec in iter_jsonl(p):
            item_rows += 1
            item_uids.add(uid_of(rec))
    derived_uids = set()
    derived_rows = 0
    for p in derived_files:
        for rec in iter_jsonl(p):
            derived_rows += 1
            derived_uids.add(uid_of(rec))

    R["statute_items"] = {
        "files": [p.replace(root + "/", "") for p in item_files],
        "rows": item_rows, "unique_uids": len(item_uids),
        "overlap_with_eval": len(item_uids & eval_uids),
        "overlap_with_all_splits": len(item_uids & all_split_uids),
    }
    R["derived"] = {
        "rows": derived_rows, "unique_uids": len(derived_uids),
        "overlap_with_eval": len(derived_uids & eval_uids),
    }

    # ---------- 4. 检索库三方案 ----------
    # 源池 = 清洗后 qa ∪ statute_items（法条切条）
    source_pool = qa_uids | item_uids
    # A：与四份 split 全不相交（最保守，任何指标都无可指摘）
    pool_A = source_pool - all_split_uids
    # B：只与评测集（dev∪test∪router）不相交，train 可进检索库
    pool_B = source_pool - eval_uids
    # C：仅案件类，与四份全不相交（论文 3.3 口径：向量化对象 = 裁判文书）
    pool_C = case_uids - all_split_uids

    def _breakdown(s):
        return {"total": len(s),
                "from_qa": len(s & qa_uids),
                "from_items": len(s & item_uids),
                "case_like": len(s & case_uids)}

    R["pool_options"] = {
        "A_source_minus_all_splits": _breakdown(pool_A),
        "B_source_minus_eval_only": _breakdown(pool_B),
        "C_case_only_minus_all_splits": _breakdown(pool_C),
    }

    # ---------- 5. 硬断言 ----------
    checks = {
        "split_pair_overlap_zero": max(pair_overlap.values() or [0]) == 0,
        "qa_uid_unique": len(qa_uids) == qa_rows,
        "items_uid_unique": len(item_uids) == item_rows,
        "items_not_in_eval": len(item_uids & eval_uids) == 0,
        "pool_A_not_in_eval": len(pool_A & eval_uids) == 0,
        "pool_A_not_in_any_split": len(pool_A & all_split_uids) == 0,
        "pool_B_not_in_eval": len(pool_B & eval_uids) == 0,
        "pool_C_not_in_any_split": len(pool_C & all_split_uids) == 0,
        "derived_excluded_from_pool": len(derived_uids & pool_A) == 0,
    }
    R["checks"] = checks
    R["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    R["elapsed_sec"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    # ---------- 6. 控制台摘要 ----------
    print("=" * 66)
    print("阶段 4b-0 检索库范围探查（只读）")
    print("=" * 66)
    for n in names:
        s = R["splits"][n]
        print("  split %-7s rows=%7d unique=%7d" % (n, s["rows"], s["unique_uids"]))
    print("  两两交集最大值 = %d" % R["split_pair_overlap_max"])
    print("  评测集(dev∪test∪router) unique = %d" % len(eval_uids))
    print()
    print("  qa rows=%d unique=%d" % (qa_rows, len(qa_uids)))
    print("  task_kind 分布：")
    for k, c in sorted(kind_cnt.items(), key=lambda kv: -kv[1]):
        print("     %-28s %7d  (eval 交叠 %d)"
              % (k, c, len(kind_uids[k] & eval_uids)))
    print()
    print("  statute_items rows=%d unique=%d (与评测集交叠 %d)"
          % (item_rows, len(item_uids), len(item_uids & eval_uids)))
    print()
    print("  检索库候选范围：")
    for k, v in R["pool_options"].items():
        print("     %-34s total=%7d  qa=%7d  items=%7d  case=%7d"
              % (k, v["total"], v["from_qa"], v["from_items"], v["case_like"]))
    print()
    bad = [k for k, v in checks.items() if not v]
    print("  断言：%s" % ("全部通过" if not bad else ("失败 " + ", ".join(bad))))
    print("  verdict = %s   耗时 %.1fs" % (R["verdict"], R["elapsed_sec"]))
    print("  报告 -> %s" % args.out_json)
    print("PROBE_POOL_SCOPE_DONE")
    return 0 if R["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
