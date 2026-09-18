#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2b 前置探查（只读）：法条切条前的「字段 + 正文结构」画像。

目的：在动手写 split_statutes.py 之前，用实测数据回答四个问题
  1. type / status / office_category 的真实取值分布 —— 才知道「只留法律/司法解释/行政法规」怎么过滤、
     status="7" 这类脏值到底有多少、还有什么别的脏值
  2. 正文（output）的 char 长度分布 —— 判断「整部法规」的量级，评估切条后条数规模
  3. 条文标记（第X条）的形态 —— 切条的锚点，以及是否存在条款号重复（需消歧）
  4. 章节结构（第X编/章/节）与目录 —— 切条时应剥离的噪声

不修改任何输入文件。输出 JSON 到 stdout（重定向到文件后下载，避免终端编码问题）。

用法：
    python3 scripts/corpus/probe_statutes.py > /tmp/probe_statutes.json
    SAMPLE=500 python3 scripts/corpus/probe_statutes.py     # 抽样快跑
"""
import collections
import json
import os
import re
import sys

SRC = os.environ.get(
    "STATUTES_SRC", "data/corpus/decontaminated/statutes")
SAMPLE = int(os.environ.get("SAMPLE", "0"))

FILES = [
    "twang2218__chinese-law-and-regulations.jsonl",
    "pandalla__chinese_law_examples.jsonl",
]

# 条文锚点：第X条 / 第X条之一（修正案常用「之一」）
RE_ARTICLE = re.compile(r"第[〇零一二三四五六七八九十百千0-9]+条(?:之[一二三四五六七八九十]+)?")
RE_CHAPTER = re.compile(r"^[\s\u3000]*第[〇零一二三四五六七八九十百千0-9]+章")
RE_SECTION = re.compile(r"^[\s\u3000]*第[〇零一二三四五六七八九十百千0-9]+节")
RE_PART = re.compile(r"^[\s\u3000]*第[〇零一二三四五六七八九十百千0-9]+编")


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


def profile_law_text(text):
    """单部法规正文的结构画像"""
    lines = text.split("\n")
    arts = RE_ARTICLE.findall(text)
    return {
        "chars": len(text),
        "lines": len(lines),
        "articles": len(arts),
        "unique_articles": len(set(arts)),
        "chapters": sum(1 for ln in lines if RE_CHAPTER.match(ln)),
        "sections": sum(1 for ln in lines if RE_SECTION.match(ln)),
        "parts": sum(1 for ln in lines if RE_PART.match(ln)),
        "has_toc": bool(re.search(r"目\s*录", text[:2000])),
        "has_appendix": "附则" in text,
        "indent_ideographic": text.count("\u3000"),
        "indent_space": len(re.findall(r"\n ", text)),
        "blank_lines": sum(1 for ln in lines if not ln.strip()),
    }


def main():
    report = {
        "src": SRC,
        "sample": SAMPLE,
        "files": {},
    }

    for fname in FILES:
        path = os.path.join(SRC, fname)
        if not os.path.exists(path):
            report["files"][fname] = {"error": "NOT_FOUND"}
            continue

        f = {
            "records": 0,
            "top_level_keys": collections.Counter(),
            "statute_keys": collections.Counter(),
            "type": collections.Counter(),
            "status": collections.Counter(),
            "office_category": collections.Counter(),
            "office_level": collections.Counter(),
            "task": collections.Counter(),
            "domain": collections.Counter(),
            "char_lens": [],
            "struct": collections.Counter(),
            "status_x_type": collections.Counter(),
            "samples": [],
        }
        struct_sum = collections.Counter()

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                f["records"] += 1
                if SAMPLE and f["records"] > SAMPLE:
                    break

                for k in r:
                    f["top_level_keys"][k] += 1
                st = r.get("statute") or {}
                for k in st:
                    f["statute_keys"][k] += 1
                f["type"][st.get("type", "<none>")] += 1
                f["status"][st.get("status", "<none>")] += 1
                f["office_category"][st.get("office_category", "<none>")] += 1
                f["office_level"][st.get("office_level", "<none>")] += 1
                f["task"][r.get("task", "<none>")] += 1
                f["domain"][r.get("domain", "<none>")] += 1
                f["status_x_type"]["%s|%s" % (st.get("status", "<none>"), st.get("type", "<none>"))] += 1

                text = r.get("output", "") or ""
                f["char_lens"].append(len(text))

                if r.get("task") == "statute_doc":
                    p = profile_law_text(text)
                    for k, v in p.items():
                        if k in ("chars", "lines"):
                            continue
                        struct_sum[k] += v
                    if len(f["samples"]) < 3:
                        f["samples"].append({
                            "source_id": r.get("source_id"),
                            "title": st.get("title"),
                            "type": st.get("type"),
                            "status": st.get("status"),
                            "head": text[:700],
                        })

        cl = sorted(f.pop("char_lens"))
        f["char_stats"] = {
            "min": cl[0] if cl else None,
            "p10": pct(cl, 0.10), "p50": pct(cl, 0.50), "p90": pct(cl, 0.90),
            "max": cl[-1] if cl else None,
        }
        f["struct_totals"] = dict(struct_sum)
        for k in ("top_level_keys", "statute_keys", "type", "status",
                  "office_category", "office_level", "task", "domain",
                  "struct", "status_x_type"):
            f[k] = dict(f[k].most_common(30))
        report["files"][fname] = f

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
