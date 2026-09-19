#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-1：法名规范化 + 影子节点合并（只读探查，不产出语料）

背景（H4 更正）
--------------
`:Law`（法名层）节点的自然键是法名，但语料里同一个法有**两种**写法差异，
必须**两级都规范化**，否则图谱会出现「同一实体多个节点」：

  L0 原始法名          —— `《中华人民共和国刑法》` / `中华人民共和国刑法`
  L1 剥书名号 + 去空白 —— 跨源差异（pandalla 全部带《》，twang2218 不带）。
                          实测两源 raw 交集 **0**，不处理会让图谱被撕成两半。
  L2 L1 + 剥版本后缀   —— 同源多版本差异（`…法(2019修正)` vs `…法`）。
                          ⚠️ 此前 H4 把这一级误记为「书名号造成的 228 个重复」，
                          实为**两个独立动作**，本脚本分别统计。

规范化规则（保守：只统一**形式**，不动**语义**）
1. 剥离任意层书名号 `《》`
2. 去除全部空白（含全角空格 U+3000 / NBSP）
3. 全角括号统一为半角
4. 只剥离**版本标记**后缀（`(2019修正)` / `（2020年修订）` / `[2018修改]`），
   **不剥离**「（试行）」等实质后缀 —— 那是不同的法

用法
----
    python scripts/retrieval/normalize_law_title.py \
        --root /mnt/data/lidian/law-agent \
        --out-json docs/corpus/LAW_TITLE_NORM.json
"""
import argparse
import collections
import glob
import itertools
import json
import os
import re
import sys
import time

WS_RE = re.compile(r"[\s\u3000\u00a0]+")
# 与 scripts/corpus/probe_vector_db.py 的 VER_SUFFIX 保持同一口径
VER_SUFFIX = re.compile(
    r"[（(\[][^（）()\[\]]{0,30}?(?:修正|修订|修改|废止)[^（）()\[\]]{0,30}?[)）\]]\s*$")


def strip_wrappers(t):
    """L1：剥任意层书名号 + 去空白 + 统一括号形式。"""
    if not t:
        return ""
    s = str(t).strip()
    guard = 0
    while len(s) >= 2 and s[0] == "《" and s[-1] == "》" and guard < 5:
        s = s[1:-1].strip()
        guard += 1
    s = WS_RE.sub("", s)
    s = s.replace("（", "(").replace("）", ")")
    return s


def strip_version(t):
    """剥版本标记后缀，循环到稳定（可能叠加多层，如 `(2019修正)(2020修订)`）。"""
    prev = None
    s = t
    while prev != s:
        prev = s
        s = VER_SUFFIX.sub("", s).strip()
    return s


def law_title_of(rec):
    """统一取法名：items 用顶层 `law_title`；statutes 用嵌套 `statute.title`。"""
    t = rec.get("law_title") or rec.get("law_name")
    if t:
        return str(t)
    st = rec.get("statute")
    if isinstance(st, dict):
        return str(st.get("title") or st.get("name") or "")
    return ""


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def scan_file(path):
    rows = 0
    missing = 0
    raw_cnt = collections.Counter()
    l1_cnt = collections.Counter()
    l2_cnt = collections.Counter()
    raw2canon = collections.defaultdict(set)
    canon2raw = collections.defaultdict(set)
    for rec in iter_jsonl(path):
        rows += 1
        t = law_title_of(rec)
        if not t:
            missing += 1
            continue
        raw = t.strip()
        l1 = strip_wrappers(raw)
        l2 = strip_version(l1)
        raw_cnt[raw] += 1
        l1_cnt[l1] += 1
        l2_cnt[l2] += 1
        raw2canon[raw].add(l2)
        canon2raw[l2].add(raw)
    dup_groups = {c: sorted(rs) for c, rs in canon2raw.items() if len(rs) > 1}
    over_split = {r: sorted(cs) for r, cs in raw2canon.items() if len(cs) > 1}
    return {
        "file": os.path.basename(path),
        "rows": rows,
        "rows_missing_title": missing,
        "raw_unique": len(raw_cnt),
        "l1_unique": len(l1_cnt),
        "l2_unique": len(l2_cnt),
        "merged_l1": len(raw_cnt) - len(l1_cnt),
        "merged_l2": len(l1_cnt) - len(l2_cnt),
        "merged_total": len(raw_cnt) - len(l2_cnt),
        "merged_groups": len(dup_groups),
        "over_split_raw": len(over_split),
        "top_merged_groups": [
            {"canonical": c, "variants": [v[:70] for v in rs[:5]],
             "variants_total": len(rs)}
            for c, rs in sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:20]
        ],
        "_raw": set(raw_cnt), "_l1": set(l1_cnt), "_l2": set(l2_cnt),
        "_l2_cnt": l2_cnt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    root = args.root.rstrip("/")
    t0 = time.time()

    groups = {
        "statute_items": sorted(glob.glob(root + "/data/corpus/statute_items/*.jsonl")),
        "statutes": sorted(glob.glob(root + "/data/corpus/decontaminated/statutes/*.jsonl")),
    }

    R = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": root,
         "normalization": {
             "L1": "剥书名号 + 去空白 + 统一括号",
             "L2": "L1 + 剥版本标记后缀（与 probe_vector_db.py 的 VER_SUFFIX 同口径）",
         }}
    detail = {}
    for label, files in groups.items():
        detail[label] = [scan_file(p) for p in files]

    cross = {}
    for label, items in detail.items():
        for a, b in itertools.combinations(items, 2):
            cross["%s:: %s | %s" % (label, a["file"], b["file"])] = {
                "raw_overlap": len(a["_raw"] & b["_raw"]),
                "l1_overlap": len(a["_l1"] & b["_l1"]),
                "l2_overlap": len(a["_l2"] & b["_l2"]),
                "a_raw": a["raw_unique"], "b_raw": b["raw_unique"],
                "a_l1": a["l1_unique"], "b_l1": b["l1_unique"],
                "a_l2": a["l2_unique"], "b_l2": b["l2_unique"],
            }
    R["cross_file"] = cross

    streams = {}
    for label, items in detail.items():
        raw = set(); l1 = set(); l2 = set()
        l2_cnt = collections.Counter()
        rows = missing = 0
        for it in items:
            raw |= it["_raw"]; l1 |= it["_l1"]; l2 |= it["_l2"]
            l2_cnt.update(it["_l2_cnt"])
            rows += it["rows"]; missing += it["rows_missing_title"]
        owner = collections.defaultdict(set)
        for it in items:
            for c in it["_l2"]:
                owner[c].add(it["file"])
        streams[label] = {
            "rows": rows,
            "rows_missing_title": missing,
            "raw_unique": len(raw),
            "l1_unique": len(l1),
            "l2_unique": len(l2),
            "merged_l1": len(raw) - len(l1),
            "merged_l2": len(l1) - len(l2),
            "merged_total": len(raw) - len(l2),
            "canonical_shared_by_files": sum(1 for c, fs in owner.items() if len(fs) > 1),
            "top_canonical_by_items": [
                {"law": c, "items": n} for c, n in l2_cnt.most_common(15)],
        }

    # L2 法名被多个源文件共享的明细（建图时会被合并的节点）
    shared_detail = []
    for label, items in detail.items():
        owner = {}
        for it in items:
            for c in it["_l2"]:
                owner.setdefault(c, set()).add(it["file"])
        shared_detail += [
            {"stream": label, "canonical": c, "files": sorted(fs)}
            for c, fs in owner.items() if len(fs) > 1]
    R["shared_across_files"] = {
        "count": len(shared_detail),
        "sample": shared_detail[:15],
    }

    for label in detail:
        for it in detail[label]:
            for k in ("_raw", "_l1", "_l2", "_l2_cnt"):
                it.pop(k, None)
    R["by_file"] = detail
    R["by_stream"] = streams

    checks = {
        "no_missing_title": all(
            it["rows_missing_title"] == 0 for label in detail for it in detail[label]),
        "no_over_split": all(
            it["over_split_raw"] == 0 for label in detail for it in detail[label]),
        "l2_le_l1_le_raw": all(
            it["l2_unique"] <= it["l1_unique"] <= it["raw_unique"]
            for label in detail for it in detail[label]),
        "statutes_law_positive": streams.get("statutes", {}).get("l2_unique", 0) > 0,
    }
    key = [k for k in cross if k.startswith("statute_items::")]
    if key:
        c0 = cross[key[0]]
        checks["norm_connects_two_sources"] = (
            c0["raw_overlap"] == 0 and c0["l1_overlap"] > 0)
    R["checks"] = checks
    R["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    R["elapsed_sec"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("阶段 4b-1 法名规范化探查（只读）")
    print("=" * 76)
    for label in detail:
        for it in detail[label]:
            print("  [%s] %s" % (label, it["file"]))
            print("       rows=%d miss=%d | raw=%d -> L1=%d -> L2=%d | merged L1=%d L2=%d"
                  " | over_split=%d"
                  % (it["rows"], it["rows_missing_title"], it["raw_unique"],
                     it["l1_unique"], it["l2_unique"], it["merged_l1"],
                     it["merged_l2"], it["over_split_raw"]))
    print()
    print("  跨文件交集：")
    for k, v in cross.items():
        print("     %s" % k)
        print("        raw %d∩%d=**%d**   L1 %d∩%d=**%d**   L2 %d∩%d=**%d**"
              % (v["a_raw"], v["b_raw"], v["raw_overlap"],
                 v["a_l1"], v["b_l1"], v["l1_overlap"],
                 v["a_l2"], v["b_l2"], v["l2_overlap"]))
    print()
    for label, s in streams.items():
        print("  [%s] rows=%d | raw=%d -> L1=%d -> L2=%d  (merged 合计 %d)"
              % (label, s["rows"], s["raw_unique"], s["l1_unique"], s["l2_unique"],
                 s["merged_total"]))
        print("       被多个源文件共享的 L2 法名 = %d" % s["canonical_shared_by_files"])
    print()
    print("  最高频法名（items，L2 口径）：")
    for x in streams["statute_items"]["top_canonical_by_items"][:8]:
        print("     %6d  %s" % (x["items"], x["law"][:56]))
    print()
    print("  合并变体最多的 6 组（items）：")
    for e in detail["statute_items"][0]["top_merged_groups"][:6]:
        print("     %-40s <- %d 个变体" % (e["canonical"][:40], e["variants_total"]))
    bad = [k for k, v in checks.items() if not v]
    print()
    print("  断言：%s" % ("全部通过" if not bad else ("失败 " + ", ".join(bad))))
    print("  verdict = %s   耗时 %.1fs" % (R["verdict"], R["elapsed_sec"]))
    print("PROBE_LAW_TITLE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
