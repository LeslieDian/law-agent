#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长文本隐患只读探查：item / doc 级极端长度样本的来源与性质。

回答两个问题（见 docs/retrieval_design.md 的 H5 / H6）：

    H6  item 最长 26,980 字 —— 是「切条遗漏（整篇未切开）」还是「某条本身就很长」？
    H5  doc 级 337 条最长 61,387 字 —— 是什么内容？能不能与 item 同池做向量检索？

判定思路：
    - 若 item 正文里**行首**出现 >= 2 个条号 → 说明该条里塞进了多条 → 切条遗漏；
    - 实测该计数 **= 0**（质检 C3 已保证行首无残留条号），故重点转为「长条是什么内容」；
    - 取样最长样本的正文开头，看是否为**附表 / 清单 / ASCII 表格**。

用法（在服务器上）：
    python scripts/corpus/probe_long_text.py [--items-dir data/corpus/statute_items]

只读：不写任何文件、不修改语料。输出到 stdout。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

# 行首条号（「第十二条」「第 12 条」）。切条正确的产物里，正文**不应**出现行首条号。
ART_LINE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百千零〇]+\s*条", re.M)
# 表格 / 清单的可疑标记
TABLE_MARKS = ("+---", "|---", "｜", "序号", "附表", "附件", "目录")


def load(items_dir: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    skipped: list[str] = []
    for root, _dirs, files in os.walk(items_dir):
        for fn in sorted(files):
            if not fn.endswith(".jsonl"):
                continue
            if fn.startswith("_"):          # 合并副本 / 调试残留一律不参与
                skipped.append(fn)
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    return rows, skipped


def qstats(xs: list[int]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return {}
    at = lambda p: xs[min(n - 1, int(p * n))]
    return {"n": n, "p50": at(.5), "p90": at(.9), "p99": at(.99),
            "max": xs[-1], "mean": round(sum(xs) / n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="data/corpus/statute_items")
    ap.add_argument("--sample-chars", type=int, default=400)
    args = ap.parse_args()

    rows, skipped = load(args.items_dir)
    print("跳过(下划线前缀):", skipped)
    print("载入行数:", len(rows))
    if not rows:
        print("[ERROR] 没有载入任何记录，检查 --items-dir")
        return 1

    print("level 分布:", dict(collections.Counter(r.get("level") for r in rows)))
    item = [r for r in rows if r.get("level") == "item"]
    doc = [r for r in rows if r.get("level") == "doc"]

    print("\n--- item 级 (%d) ---" % len(item))
    print("字数:", qstats([len(r.get("text", "") or "") for r in item]))
    print("\n--- doc 级 (%d) ---" % len(doc))
    if doc:
        print("字数:", qstats([len(r.get("text", "") or "") for r in doc]))
        print("域分布:", dict(collections.Counter(r.get("domain") for r in doc)))
        print("来源:", dict(collections.Counter(r.get("source_dataset") for r in doc)))
        print("law_type 分布:", dict(collections.Counter(r.get("law_type") for r in doc)))
        print("distinct law_title:", len({r.get("law_title") for r in doc}))

    # ---------- H6 判据：行首出现 >= 2 个条号 = 疑似未切开 ----------
    print("\n=== H6 疑似『未切开的整篇』（正文行首出现 >= 2 个条号）===")
    susp = [(len(ART_LINE.findall(r.get("text", "") or "")), len(r.get("text", "") or ""),
             r.get("law_title", "")) for r in item]
    susp = [s for s in susp if s[0] >= 2]
    susp.sort(reverse=True)
    print("命中条数:", len(susp), "（0 = 无切条遗漏）")
    for s in susp[:15]:
        print("  %3d 条号 | %7d 字 | %s" % s)

    # ---------- 长度分档 ----------
    print("\n=== item 长度分档 ===")
    b = collections.Counter()
    for r in item:
        n = len(r.get("text", "") or "")
        b["<=500" if n <= 500 else "501-1000" if n <= 1000 else
          "1001-3000" if n <= 3000 else "3001-10000" if n <= 10000 else ">10000"] += 1
    for k in ("<=500", "501-1000", "1001-3000", "3001-10000", ">10000"):
        print("  %-12s %6d  (%.3f%%)" % (k, b[k], 100.0 * b[k] / len(item)))
    over = b["1001-3000"] + b["3001-10000"] + b[">10000"]
    print("  >1000 字合计: %d (%.3f%%)" % (over, 100.0 * over / len(item)))

    # ---------- 最长样本内容性质 ----------
    n = args.sample_chars
    print("\n=== 最长 item 的内容性质 ===")
    top = max(item, key=lambda x: len(x.get("text", "") or ""))
    t = top.get("text", "") or ""
    print("  law_title:", top.get("law_title"), "| article_label:", top.get("article_label"))
    print("  len(text):", len(t), "| 换行数:", t.count("\n"))
    print("  表格/清单标记命中:", [m for m in TABLE_MARKS if m in t])
    print("  ---- 开头 %d 字 ----" % n)
    print("  " + t[:n].replace("\n", " / "))

    if doc:
        print("\n=== 最长 doc 的内容性质 ===")
        td = max(doc, key=lambda x: len(x.get("text", "") or ""))
        t2 = td.get("text", "") or ""
        print("  law_title:", td.get("law_title"), "| 换行数:", t2.count("\n"))
        print("  表格/清单标记命中:", [m for m in TABLE_MARKS if m in t2])
        print("  ---- 开头 %d 字 ----" % n)
        print("  " + t2[:n].replace("\n", " / "))

    # ---------- 超长样本是否集中在特定 law_type ----------
    for th in (1000, 3000):
        c = collections.Counter(r.get("law_type") for r in item
                                if len(r.get("text", "") or "") > th)
        print("\n  item > %d 字 (%d 条): %s" % (th, sum(c.values()), dict(c)))

    # ---------- char_len 字段自洽 ----------
    bad = [r.get("uid") for r in rows
           if r.get("char_len") is not None and r.get("char_len") != len(r.get("text", "") or "")]
    print("\n=== char_len 字段 == len(text) 校验 ===")
    print("  不一致条数:", len(bad), bad[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
