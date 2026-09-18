#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2b 质检：法条切条产物（statute_items）的硬检查，verdict 必须 PASS 才能进向量库。

检查项（每项都给出数字与样例，不靠"看起来没问题"）：
  C1 schema 完整性   —— 必需字段存在且非空；uid 全局唯一
  C2 内部一致性      —— text_full == 「法规名 条号 正文」；content_sha1 可复算
  C3 切条正确性      —— 正文里**不应再出现行首条号锚点**（说明没切干净）
                       正文不应含「目 录」目录块
  C4 碎片检测        —— 过短条文（< 12 字）比例 + 样例（交叉引用误切会表现为碎片）
  C5 章节归属        —— chapter 非空的占比；同法内 chapter 不应为空串与「第X章」混杂异常
  C6 域分布          —— 三法域（civil/criminal/procedural）必须有量，且各域抽样可读
  C7 版本链          —— 同一 source_id 下条号重复（多版本）情况登记

用法：
    python3 scripts/corpus/verify_statute_items.py --root /mnt/data/lidian/law-agent \
        --out-json docs/corpus/statute_items_VERIFY.json \
        --out-md   docs/corpus/statute_items_VERIFY.md
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

RE_HEAD_ARTICLE = re.compile(r"(^|\n)[ \t\u3000]*第[〇零一二三四五六七八九十百千两0-9]+条")
REQUIRED = ["uid", "law_title", "text", "text_full", "content_sha1",
            "domain", "status", "source_dataset"]
REQUIRED_FOR_ITEM_LEVEL = ["article_label"]   # 仅 level="item" 才要求有条款号
MAX_NEAR = 200        # 报告里最多列出的样例数（计数不受影响）
SHORT_CHARS = 12
VERDICT_MAX_SHORT_SHARE = 0.15
VERDICT_MAX_RESIDUAL_SHARE = 0.02


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--items-dir", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()

    t0 = time.time()
    items_dir = a.items_dir or os.path.join(a.root, "data/corpus/statute_items")
    files = sorted(fn for fn in os.listdir(items_dir) if fn.endswith(".items.jsonl"))

    res = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items_dir": items_dir,
        "files": files,
        "counts": {"total": 0, "by_file": {}},
        "C1_schema": {"missing": collections.Counter(), "dup_uids": 0,
                      "dup_uid_samples": []},
        "C2_consistency": {"text_full_mismatch": 0, "sha_mismatch": 0,
                           "samples": []},
        "C3_residual": {"head_anchor_in_text": 0, "has_toc": 0, "samples": []},
        "C4_short": {"count": 0, "share": None, "samples": []},
        "C5_chapter": {"with_chapter": 0, "without_chapter": 0, "share": None},
        "C6_domain": {"by_domain": collections.Counter(), "samples": {}},
        "C7_versions": {"same_law_dup_article_label": 0, "samples": []},
        "per_law_item_count": {},
        "blockers": [],
    }

    seen_uids = set()
    short_n = 0
    chap_y = chap_n = 0
    dom_samples = collections.defaultdict(list)
    law_lab = collections.defaultdict(collections.Counter)

    for fn in files:
        path = os.path.join(items_dir, fn)
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                n += 1
                res["counts"]["total"] += 1

                for k in REQUIRED:
                    if it.get(k) in (None, ""):
                        res["C1_schema"]["missing"][k] += 1
                if it.get("level", "item") == "item":
                    for k in REQUIRED_FOR_ITEM_LEVEL:
                        if it.get(k) in (None, ""):
                            res["C1_schema"]["missing"]["item." + k] += 1
                else:
                    res["C1_schema"]["doc_level_items"] = \
                        res["C1_schema"].get("doc_level_items", 0) + 1

                uid = it.get("uid", "")
                if uid in seen_uids:
                    res["C1_schema"]["dup_uids"] += 1
                    if len(res["C1_schema"]["dup_uid_samples"]) < MAX_NEAR:
                        res["C1_schema"]["dup_uid_samples"].append(uid)
                else:
                    seen_uids.add(uid)

                exp_full = re.sub(r"[ \t\u3000]+", " ",
                                  "%s %s %s" % (it.get("law_title", ""),
                                                it.get("article_label", ""),
                                                it.get("text", ""))).strip()
                if it.get("text_full") != exp_full:
                    res["C2_consistency"]["text_full_mismatch"] += 1
                    if len(res["C2_consistency"]["samples"]) < 5:
                        res["C2_consistency"]["samples"].append(
                            {"uid": uid, "got": it.get("text_full", "")[:80],
                             "exp": exp_full[:80]})
                if it.get("content_sha1") and sha1(it.get("text_full", "")) != it["content_sha1"]:
                    res["C2_consistency"]["sha_mismatch"] += 1

                txt = it.get("text", "") or ""
                if it.get("level", "item") == "item" and RE_HEAD_ARTICLE.search(txt):
                    res["C3_residual"]["head_anchor_in_text"] += 1
                    if len(res["C3_residual"]["samples"]) < MAX_NEAR:
                        res["C3_residual"]["samples"].append(
                            {"uid": uid, "text": txt[:160].replace("\n", " ")})
                if "目 录" in txt or "目录" == txt.strip():
                    res["C3_residual"]["has_toc"] += 1
                res["split_mode"] = res.get("split_mode", collections.Counter())
                res["split_mode"][it.get("split_mode", "item_split")] += 1
                res["level"] = res.get("level", collections.Counter())
                res["level"][it.get("level", "item")] += 1

                if len(txt) < SHORT_CHARS:
                    short_n += 1
                    if len(res["C4_short"]["samples"]) < 15:
                        res["C4_short"]["samples"].append(
                            {"uid": uid, "len": len(txt),
                             "text": txt[:80].replace("\n", " ")})

                if it.get("chapter"):
                    chap_y += 1
                else:
                    chap_n += 1

                dom = it.get("domain", "")
                res["C6_domain"]["by_domain"][dom] += 1
                if len(dom_samples[dom]) < 3:
                    dom_samples[dom].append({"law": it.get("law_title", "")[:30],
                                             "label": it.get("article_label", ""),
                                             "text": txt[:70].replace("\n", " ")})

                law_lab[it.get("source_id", "")][it.get("article_label", "")] += 1

        res["counts"]["by_file"][fn] = n

    total = res["counts"]["total"] or 1
    res["C4_short"]["count"] = short_n
    res["C4_short"]["share"] = round(short_n / total, 4)
    res["C5_chapter"]["with_chapter"] = chap_y
    res["C5_chapter"]["without_chapter"] = chap_n
    res["C5_chapter"]["share"] = round(chap_y / total, 4)
    res["C6_domain"]["samples"] = {k: v for k, v in dom_samples.items()}

    dup_laws = 0
    for sid, cnt in law_lab.items():
        d = sum(c - 1 for c in cnt.values() if c > 1)
        if d:
            dup_laws += d
            if len(res["C7_versions"]["samples"]) < 5:
                res["C7_versions"]["samples"].append(
                    {"source_id": sid, "dup_article_labels": d,
                     "total_items": sum(cnt.values())})
    res["C7_versions"]["same_law_dup_article_label"] = dup_laws

    res["per_law_item_count"] = {
        "laws": len(law_lab),
        "items_per_law_p50": sorted(sum(c.values()) for c in law_lab.values())[len(law_lab) // 2] if law_lab else 0,
        "laws_with_few_items": sum(1 for c in law_lab.values() if sum(c.values()) <= 3),
    }

    # --- verdict ---
    if res["C1_schema"]["missing"]:
        res["blockers"].append("C1 必需字段缺失：%s" % dict(res["C1_schema"]["missing"]))
    if res["C1_schema"]["dup_uids"]:
        res["blockers"].append("C1 uid 重复 %d 条" % res["C1_schema"]["dup_uids"])
    if res["C2_consistency"]["text_full_mismatch"] or res["C2_consistency"]["sha_mismatch"]:
        res["blockers"].append("C2 text_full/sha1 不一致")
    if res["C3_residual"]["head_anchor_in_text"] / total > VERDICT_MAX_RESIDUAL_SHARE:
        res["blockers"].append(
            "C3 残留行首条号 %.2f%% > %.0f%%（切条不干净）"
            % (100 * res["C3_residual"]["head_anchor_in_text"] / total,
               100 * VERDICT_MAX_RESIDUAL_SHARE))
    if res["C4_short"]["share"] > VERDICT_MAX_SHORT_SHARE:
        res["blockers"].append("C4 过短碎片占比 %.2f%% > %.0f%%"
                               % (100 * res["C4_short"]["share"],
                                  100 * VERDICT_MAX_SHORT_SHARE))
    for dom in ("civil", "criminal", "procedural"):
        if res["C6_domain"]["by_domain"].get(dom, 0) == 0:
            res["blockers"].append("C6 域 %s 条数为 0" % dom)

    res["verdict"] = "FAIL" if res["blockers"] else "PASS"

    def jsonable(o):
        if isinstance(o, collections.Counter):
            return dict(o.most_common())
        return str(o)

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=jsonable)

    # --- Markdown ---
    L = []
    W = L.append
    W("# 阶段 2b 质检报告（STATUTE ITEMS VERIFY）\n")
    W("> 生成时间：%s　耗时 %.1fs" % (res["generated_at"], time.time() - t0))
    W("> **verdict = `%s`**\n" % res["verdict"])
    if res["blockers"]:
        for b in res["blockers"]:
            W("- ⛔ %s" % b)
        W("")
    W("## 总量\n")
    W("| 文件 | 条数 |")
    W("|---|---|")
    for k, v in res["counts"]["by_file"].items():
        W("| `%s` | %d |" % (k, v))
    W("| **合计** | **%d** |" % res["counts"]["total"])
    W("")
    W("## C1 schema / C2 一致性\n")
    W("| 项 | 值 |")
    W("|---|---|")
    W("| 必需字段缺失 | %s |" % (dict(res["C1_schema"]["missing"]) or "无"))
    W("| uid 重复 | %d |" % res["C1_schema"]["dup_uids"])
    W("| text_full 不一致 | %d |" % res["C2_consistency"]["text_full_mismatch"])
    W("| sha1 不可复算 | %d |" % res["C2_consistency"]["sha_mismatch"])
    W("")
    W("## C3 切条残留\n")
    W("| 项 | 条数 | 占比 |")
    W("|---|---|---|")
    W("| 正文里仍有行首条号（仅 level=item） | %d | %.3f%% |"
      % (res["C3_residual"]["head_anchor_in_text"],
         100.0 * res["C3_residual"]["head_anchor_in_text"] / total))
    W("| 正文含目录块 | %d | %.3f%% |"
      % (res["C3_residual"]["has_toc"], 100.0 * res["C3_residual"]["has_toc"] / total))
    W("")
    W("条目层级与切分方式：")
    W("")
    W("| level | 条数 |")
    W("|---|---|")
    for k, v in (res.get("level") or collections.Counter()).most_common():
        W("| %s | %d |" % (k, v))
    W("")
    W("| split_mode | 条数 |")
    W("|---|---|")
    for k, v in (res.get("split_mode") or collections.Counter()).most_common():
        W("| `%s` | %d |" % (k, v))
    W("")
    for s in res["C3_residual"]["samples"][:5]:
        W("- `%s`：%s" % (s["uid"][-30:], s["text"]))
    W("")
    W("## C4 过短碎片（< %d 字）\n" % SHORT_CHARS)
    W("合计 **%d** 条（%.2f%%）" % (short_n, 100.0 * short_n / total))
    W("")
    for s in res["C4_short"]["samples"]:
        W("- (%d 字) %s" % (s["len"], s["text"]))
    W("")
    W("## C5 章节归属\n")
    W("带章标记 %d（%.1f%%）／无章标记 %d" % (chap_y, 100.0 * chap_y / total, chap_n))
    W("")
    W("## C6 域分布与抽样\n")
    W("| 域 | 条数 | 占比 |")
    W("|---|---|---|")
    for k, v in res["C6_domain"]["by_domain"].most_common():
        W("| %s | %d | %.1f%% |" % (k, v, 100.0 * v / total))
    W("")
    for dom, ss in res["C6_domain"]["samples"].items():
        W("**%s**：" % dom)
        for s in ss:
            W("  - 《%s》%s　%s" % (s["law"].strip("《》"), s["label"], s["text"]))
        W("")
    W("## C7 版本链\n")
    W("同法内条号重复（多版本/修正）合计 %d 处；涉及法 %d 部"
      % (res["C7_versions"]["same_law_dup_article_label"],
         len(res["C7_versions"]["samples"])))
    W("")
    W("## 每法条数\n")
    W("| 法数 | 条数中位数 | ≤3 条的法 |")
    W("|---|---|---|")
    W("| %d | %d | %d |" % (res["per_law_item_count"]["laws"],
                             res["per_law_item_count"]["items_per_law_p50"],
                             res["per_law_item_count"]["laws_with_few_items"]))
    W("")
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("VERDICT = %s" % res["verdict"])
    for b in res["blockers"]:
        print("BLOCKER:", b)
    print("items=%d files=%d short_share=%.4f residual_head=%d chapter_share=%.3f"
          % (res["counts"]["total"], len(files), res["C4_short"]["share"] or 0,
             res["C3_residual"]["head_anchor_in_text"], res["C5_chapter"]["share"] or 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
