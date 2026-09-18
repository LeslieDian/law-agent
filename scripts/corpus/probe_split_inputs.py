#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4 前置探查：统计去污后 QA 池与法条条目的分层结构（只读）。

为「分层下采样 + train/val/test/路由集切分」提供真实分层基数。

硬约定：
  * 只读 `data/corpus/decontaminated/qa/*.jsonl`，**显式排除 `_all.jsonl`**
    （合并副本，与逐文件流重复，读它必然双计）。
  * 不改动任何输入文件。

用法：
  python scripts/corpus/probe_split_inputs.py \
      --root /mnt/data/lidian/law-agent \
      --out-json /tmp/probe_split.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

# ---------------------------------------------------------------- 工具

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


# ---------------------------------------------------------------- 主流程

def probe_qa(qa_dir, stats, samples):
    files = sorted(
        f for f in os.listdir(qa_dir)
        if f.endswith(".jsonl") and not f.startswith("_")
    )
    seen_uids = collections.Counter()
    stats["qa_files"] = files
    stats["qa_skipped_files"] = sorted(
        f for f in os.listdir(qa_dir) if f.startswith("_")
    )

    per_file = {}
    for fn in files:
        n = 0
        n_bad = 0
        for rec in iter_jsonl(os.path.join(qa_dir, fn)):
            if rec is None:
                n_bad += 1
                continue
            n += 1
            stats["total"] += 1

            uid = rec.get("uid") or ""
            seen_uids[uid] += 1

            dom = rec.get("domain") or "<empty>"
            task = rec.get("task") or "<empty>"
            src = rec.get("source_dataset") or "<empty>"
            kind = rec.get("task_kind") or "<empty>"
            dsrc = rec.get("domain_source") or "<empty>"

            stats["by_domain"][dom] += 1
            stats["by_domain_task"]["%s|%s" % (dom, task)] += 1
            stats["by_domain_source"]["%s|%s" % (dom, src)] += 1
            stats["by_task"][task] += 1
            stats["by_source"][src] += 1
            stats["by_task_kind"][kind] += 1
            stats["by_domain_source_signal"][dsrc] += 1

            # 标记位
            if rec.get("synthetic"):
                stats["flags_synthetic"] += 1
            if rec.get("replay"):
                stats["flags_replay"] += 1
            if rec.get("source_grade") and rec["source_grade"] != "A":
                stats["flags_nonA"] += 1

            # 字段完整性（供下采样时选参）
            for k in ("messages", "instruction", "output", "content_sha1"):
                if not rec.get(k):
                    stats["missing_%s" % k] += 1
            msgs = rec.get("messages") or []
            if len(msgs) != 2:
                stats["messages_len_ne_2"] += 1
            else:
                ulen = len(msgs[0].get("content") or "")
                alen = len(msgs[1].get("content") or "")
                if ulen == 0 or alen == 0:
                    stats["messages_empty_side"] += 1
                # 路由集需要 query 文本 → 记录 user 侧长度分布
                stats["user_len_buckets"][_bucket(ulen)] += 1
                stats["out_len_buckets"][_bucket(alen)] += 1
        per_file[fn] = {"records": n, "bad_json_lines": n_bad}

    stats["qa_per_file"] = per_file
    stats["qa_dup_uid"] = sum(c - 1 for c in seen_uids.values() if c > 1)
    stats["qa_unique_uid"] = len(seen_uids)

    # 抽 2 条样例看字段
    for fn in files[:1]:
        for i, rec in enumerate(iter_jsonl(os.path.join(qa_dir, fn))):
            if i >= 2:
                break
            samples.append({"file": fn, "keys": sorted(rec.keys()),
                            "sample": _trim(rec)})
    return files


def _bucket(n: int) -> str:
    for lo, hi in ((0, 32), (33, 128), (129, 512), (513, 2048), (2049, 10 ** 9)):
        if lo <= n <= hi:
            return "%d-%d" % (lo, hi if hi < 10 ** 9 else 99999)
    return "?"


def _trim(rec, limit=700):
    out = {}
    for k, v in rec.items():
        if isinstance(v, str):
            out[k] = v[:limit]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = v[:3]
        elif isinstance(v, dict):
            out[k] = {kk: (vv[:200] if isinstance(vv, str) else vv)
                      for kk, vv in list(v.items())[:6]}
        else:
            out[k] = str(v)[:200]
    return out


def probe_statutes(st_dir, stats, samples):
    """法条条目流：为「法条任务派生 / 向量库主料」两路提供基数。"""
    files = sorted(
        f for f in os.listdir(st_dir)
        if f.endswith(".jsonl") and not f.startswith("_")
    ) if os.path.isdir(st_dir) else []
    stats["st_files"] = files
    if not files:
        return
    uids = collections.Counter()
    for fn in files:
        n = 0
        for rec in iter_jsonl(os.path.join(st_dir, fn)):
            if rec is None:
                continue
            n += 1
            stats["st_total"] += 1
            stats["st_by_domain"][rec.get("domain") or "<empty>"] += 1
            stats["st_by_level"][rec.get("level") or "<empty>"] += 1
            stats["st_by_source"][rec.get("source_dataset") or "<empty>"] += 1
            stats["st_by_status"][rec.get("status") or "<empty>"] += 1
            stats["st_by_law_type"][rec.get("law_type") or "<empty>"] += 1
            uids[rec.get("uid") or ""] += 1
            if not rec.get("text_full"):
                stats["st_missing_text_full"] += 1
        stats["st_per_file"][fn] = n
    stats["st_dup_uid"] = sum(c - 1 for c in uids.values() if c > 1)
    stats["st_unique_uid"] = len(uids)
    for i, rec in enumerate(iter_jsonl(os.path.join(st_dir, files[0]))):
        if i >= 2:
            break
        samples.append({"file": files[0], "kind": "statute",
                        "keys": sorted(rec.keys()), "sample": _trim(rec)})


# ---------------------------------------------------------------- 输出

def dump_md(stats, path):
    W = lambda s="": path_fh.write(s + "\n")
    with open(path, "w", encoding="utf-8") as fh:
        path_fh = fh
        W("# 阶段 4 输入探查（只读）")
        W()
        W("- QA 池总记录：**%d**（唯一 uid %d，重复 %d）"
          % (stats["total"], stats["qa_unique_uid"], stats["qa_dup_uid"]))
        W("- 读取文件：%s" % ", ".join(stats["qa_files"]))
        W("- 已排除（合并副本，防双计）：%s"
          % (", ".join(stats["qa_skipped_files"]) or "无"))
        W()
        W("## QA 逐域")
        W()
        W("| 域 | 条数 | 占比 |")
        W("|---|---|---|")
        for k, v in stats["by_domain"].most_common():
            W("| %s | %d | %.2f%% |" % (k, v, pct(v, stats["total"])))
        W()
        W("## QA 逐任务")
        W()
        W("| task | 条数 | 占比 |")
        W("|---|---|---|")
        for k, v in stats["by_task"].most_common():
            W("| %s | %d | %.2f%% |" % (k, v, pct(v, stats["total"])))
        W()
        W("## QA 逐来源")
        W()
        W("| source_dataset | 条数 | 占比 |")
        W("|---|---|---|")
        for k, v in stats["by_source"].most_common():
            W("| %s | %d | %.2f%% |" % (k, v, pct(v, stats["total"])))
        W()
        W("## QA 域 x 任务（分层基数）")
        W()
        W("| 域\\|task | 条数 |")
        W("|---|---|")
        for k, v in sorted(stats["by_domain_task"].items(), key=lambda x: -x[1]):
            W("| %s | %d |" % (k, v))
        W()
        W("## 标记位与完整性")
        W()
        W("| 项 | 值 |")
        W("|---|---|")
        for k in ("flags_synthetic", "flags_replay", "flags_nonA",
                  "missing_messages", "missing_instruction", "missing_output",
                  "missing_content_sha1", "messages_len_ne_2",
                  "messages_empty_side"):
            W("| %s | %d |" % (k, stats[k]))
        W()
        W("user 侧长度分桶：%s" % dict(stats["user_len_buckets"]))
        W()
        W("答案侧长度分桶：%s" % dict(stats["out_len_buckets"]))
        W()
        if stats["st_files"]:
            W("## 法条条目流")
            W()
            W("- 总条目：**%d**（唯一 uid %d，重复 %d）"
              % (stats["st_total"], stats["st_unique_uid"], stats["st_dup_uid"]))
            W("- 逐域：%s" % dict(stats["st_by_domain"]))
            W("- 逐 level：%s" % dict(stats["st_by_level"]))
            W("- 逐来源：%s" % dict(stats["st_by_source"]))
            W("- 逐 status：%s" % dict(stats["st_by_status"]))
            W("- 缺 text_full：%d" % stats["st_missing_text_full"])
        W()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    qa_dir = os.path.join(args.root, "data/corpus/decontaminated/qa")
    st_dir = os.path.join(args.root, "data/corpus/statute_items")
    if not os.path.isdir(qa_dir):
        sys.exit("QA 目录不存在：%s" % qa_dir)

    stats = collections.defaultdict(int)
    for key in ("by_domain", "by_domain_task", "by_domain_source", "by_task",
                "by_source", "by_task_kind", "by_domain_source_signal",
                "user_len_buckets", "out_len_buckets",
                "st_by_domain", "st_by_level", "st_by_source", "st_by_status",
                "st_by_law_type", "st_per_file"):
        stats[key] = collections.Counter()
    stats["qa_per_file"] = {}
    stats["total"] = 0
    samples = []

    probe_qa(qa_dir, stats, samples)
    probe_statutes(st_dir, stats, samples)

    payload = {
        "generated_by": "scripts/corpus/probe_split_inputs.py",
        "qa_dir": qa_dir,
        "statutes_dir": st_dir,
        "stats": _to_jsonable(stats),
        "samples": samples,
    }
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    if args.out_md:
        dump_md(stats, args.out_md)

    print("QA_TOTAL=%d UNIQUE_UID=%d DUP=%d"
          % (stats["total"], stats["qa_unique_uid"], stats["qa_dup_uid"]))
    print("ST_TOTAL=%d" % stats["st_total"])
    print("DOMAIN=%s" % dict(stats["by_domain"]))
    print("FLAGS synthetic=%d replay=%d nonA=%d"
          % (stats["flags_synthetic"], stats["flags_replay"], stats["flags_nonA"]))
    print("OK -> %s" % args.out_json)


def _to_jsonable(rec):
    if isinstance(rec, collections.Counter):
        return dict(rec)
    if isinstance(rec, dict):
        return {k: _to_jsonable(v) for k, v in rec.items()}
    return rec


if __name__ == "__main__":
    main()
