#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 4b 前置探查：向量库 / 知识图谱设计的**可行性实测**（只读，不写语料）

设计文档（configs/retrieval.yaml + docs/neo4j_setup.cypher）假设了一批字段与关系。
本脚本在**真实语料**上逐条核对，回答四件事：

  1) Law / LawVersion / Provision 三类节点各能派生多少？
     - Law      = law_title 剥掉 "(2019修正)" 之类的版本后缀后的规范法名
     - LawVersion = 规范法名 + (版本标记, publish_date)；若每个规范法名只有 1 个
                    law_title，说明语料里**没有多版本**，LawVersion 层就是空壳
     - Provision = 逐条 item（65,037 里 level=item 的部分）
  2) retrieval.yaml 的 time_filter 有没有数据支撑？
     - 规则依赖 effective_from + effective_to；effective_to 字段是否存在？
     - status 的真实取值分布是什么？
  3) 图谱的边能自动建多少？重点：Provision-[:CITES]->Provision
     - 条文正文里的交叉引用（《X法》第Y条 / 本法第Y条）能否解析到**目标法条**
  4) 检索粒度：item vs doc、text 长度分布；评测表里的「条+款/条+款+项」有没有款、项数据

用法：
  PYTHONUTF8=1 python scripts/corpus/probe_vector_db.py \
      --items-dir data/corpus/statute_items \
      --out-json docs/corpus/VECTOR_DB_PROBE.json
"""

import argparse
import collections
import glob
import json
import os
import re
import time

# ---------- 正则 ----------
# 版本后缀：(2019修正) （2020年修订） [2018修改] 等，出现在标题末尾
VER_SUFFIX = re.compile(r"[（(\[][^（）()\[\]]{0,30}?(?:修正|修订|修改|废止)[^（）()\[\]]{0,30}?[)）\]]\s*$")
# 任意位置的「第X条」（含交叉引用）
ART_ANY = re.compile(r"第[〇零一二三四五六七八九十百千两0-9]+条")
# 行首锚点：「第X条」在行首（正文残留检测用，C3 已要求为 0）
ART_LINE = re.compile(r"(?m)^[ \t\u3000]*第[〇零一二三四五六七八九十百千两0-9]+条")
# 可解析的具名引用：《中华人民共和国刑法》第十二条
CITE_NAMED = re.compile(r"《([^》]{2,60})》\s*第([〇零一二三四五六七八九十百千两0-9]+)条")
# 可解析的自身引用：本法第十二条 / 本条例第三条
CITE_SELF = re.compile(r"本(?:法|条例|规定|办法|规则|解释|决定)\s*第([〇零一二三四五六七八九十百千两0-9]+)条")

CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def cn2int(s):
    """中文数字转 int（支持 十/百/千 与阿拉伯数字混排），失败返回 None。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, section, number = 0, 0, 0
    units = {"十": 10, "百": 100, "千": 1000}
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in units:
            u = units[ch]
            if number == 0:
                number = 1
            section += number * u
            number = 0
        else:
            return None
    total = section + number
    return total if total > 0 else None


def canon_title(t):
    """剥掉版本后缀，得到规范法名（用于 Law 节点聚合）。"""
    t = (t or "").strip()
    prev = None
    while prev != t:
        prev = t
        t = VER_SUFFIX.sub("", t).strip()
    return t


def resolve_law(name, canon_set, canon_suffix_index):
    """
    把引用里的法名解析到语料中的规范法名。
    依次尝试：精确 → 去掉空格 → 语料规范名以它结尾（《民法典》→《中华人民共和国民法典》）。
    """
    n = name.strip()
    for cand in (n, n.replace(" ", "")):
        if cand in canon_set:
            return cand
    for cand in (n, n.replace(" ", "")):
        hit = canon_suffix_index.get(cand)
        if hit:
            return hit[0]
    return None


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--items-dir", default="data/corpus/statute_items")
    ap.add_argument("--out-json", default="docs/corpus/VECTOR_DB_PROBE.json")
    ap.add_argument("--sample-cite", type=int, default=15)
    a = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.root, a.items_dir, "*.items.jsonl")))
    if not files:
        files = sorted(glob.glob(os.path.join(a.root, a.items_dir, "*.jsonl")))
    if not files:
        raise SystemExit("no items file found under %s" % a.items_dir)

    keys_union = collections.Counter()
    by_level = collections.Counter()
    by_split_mode = collections.Counter()
    by_dataset = collections.Counter()
    by_law_type = collections.Counter()
    by_domain = collections.Counter()
    status_dist = collections.Counter()

    title_counter = collections.Counter()          # 原始 law_title
    canon_to_titles = collections.defaultdict(set)  # 规范名 -> {law_title}
    law_article_range = {}                          # 规范名 -> [min_no, max_no]
    law_article_seqs = collections.defaultdict(set)  # 规范名 -> {article_no}

    len_by_level = collections.defaultdict(list)

    eff_from_missing = 0
    eff_period_present = 0
    eff_period_samples = []
    publish_missing = 0
    office_present = 0
    has_effective_to = 0
    article_no_null = 0
    article_label_empty = 0
    para_or_sub_present = 0
    line_anchor_hits = 0
    chap_marked = 0
    sha1_key = collections.Counter()

    cite_any_total = 0
    cite_self_events = 0
    cite_named_events = 0
    cite_named_laws = collections.Counter()
    cite_named_resolved = 0
    cite_named_sample = []
    cite_self_resolved = 0
    cite_self_unresolved_sample = []

    total = 0
    docs_scanned = 0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                keys_union.update(rec.keys())

                lvl = rec.get("level", "?")
                by_level[lvl] += 1
                by_split_mode[rec.get("split_mode", "")] += 1
                by_dataset[rec.get("source_dataset", "")] += 1
                by_law_type[rec.get("law_type", "")] += 1
                if lvl == "item":
                    by_domain[rec.get("domain", "")] += 1

                title = rec.get("law_title", "")
                title_counter[title] += 1
                cn = canon_title(title)
                canon_to_titles[cn].add(title)

                if not rec.get("effective_from"):
                    eff_from_missing += 1
                if not rec.get("publish_date"):
                    publish_missing += 1
                if rec.get("effective_period"):
                    eff_period_present += 1
                    if len(eff_period_samples) < 5:
                        eff_period_samples.append(str(rec["effective_period"])[:120])
                if rec.get("office"):
                    office_present += 1
                if "effective_to" in rec:
                    has_effective_to += 1
                status_dist[rec.get("status") or "(empty)"] += 1

                if rec.get("article_no") is None:
                    article_no_null += 1
                if not rec.get("article_label"):
                    article_label_empty += 1
                # 款/项粒度：语料是否真有这两个层级的数据
                if rec.get("article_sub") is not None or rec.get("paragraph"):
                    para_or_sub_present += 1
                if rec.get("chapter"):
                    chap_marked += 1

                txt = rec.get("text") or ""
                len_by_level[lvl].append(len(txt))
                docs_scanned += 1

                for k in ("content_sha1", "sha1", "text_sha1"):
                    if k in rec:
                        sha1_key[k] += 1

                if lvl == "item":
                    line_anchor_hits += len(ART_LINE.findall(txt))
                    hits = ART_ANY.findall(txt)
                    cite_any_total += len(hits)

                    named = CITE_NAMED.findall(txt)
                    cite_named_events += len(named)
                    for nm, num in named:
                        cite_named_laws[nm] += 1
                        if len(cite_named_sample) < a.sample_cite:
                            cite_named_sample.append({"law": nm, "article": num})

                    selfc = CITE_SELF.findall(txt)
                    cite_self_events += len(selfc)

                    no = rec.get("article_no")
                    if cn not in law_article_range and no is not None:
                        law_article_range[cn] = [no, no]
                    elif cn in law_article_range and no is not None:
                        r = law_article_range[cn]
                        r[0] = min(r[0], no)
                        r[1] = max(r[1], no)
                    if no is not None:
                        law_article_seqs[cn].add(no)

    # ---- 引用可解析率（需要全量 law 集合，故放循环后） ----
    canon_set = set(canon_to_titles.keys())
    # 后缀索引：规范名 -> [规范名]（用于《民法典》→《中华人民共和国民法典》）
    suffix_index = collections.defaultdict(list)
    for cn in canon_set:
        for cut in range(0, min(6, len(cn))):
            suffix_index[cn[cut:]].append(cn)

    # 重扫一遍自身引用（避免存明细占内存）：只算可解析率
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("level") != "item":
                    continue
                cn = canon_title(rec.get("law_title", ""))
                txt = rec.get("text") or ""
                seqs = law_article_seqs.get(cn, set())
                for num in CITE_SELF.findall(txt):
                    v = cn2int(num)
                    if v is not None and v in seqs:
                        cite_self_resolved += 1
                    elif v is not None and len(cite_self_unresolved_sample) < a.sample_cite:
                        cite_self_unresolved_sample.append(
                            {"law": cn, "cited": num, "parsed": v,
                             "law_articles": len(seqs)})

    for nm, cnt in cite_named_laws.items():
        if resolve_law(nm, canon_set, suffix_index):
            cite_named_resolved += cnt

    multi_version = {k: sorted(v) for k, v in canon_to_titles.items() if len(v) > 1}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [os.path.basename(f) for f in files],
        "total_items": total,

        "schema": {
            "keys_union": sorted(keys_union.keys()),
            "key_presence": dict(keys_union),
            "content_sha1_like_keys": dict(sha1_key),
        },

        "node_feasibility": {
            "Law": {"from": "canon_title(law_title)", "distinct_canonical": len(canon_to_titles)},
            "LawVersion": {
                "distinct_raw_law_title": len(title_counter),
                "canonical_with_multi_versions": len(multi_version),
                "multi_version_share": (len(multi_version) / len(canon_to_titles)) if canon_to_titles else 0,
                "samples": dict(list(multi_version.items())[:5]),
            },
            "Provision": {
                "item_level": by_level.get("item", 0),
                "doc_level": by_level.get("doc", 0),
            },
            "note": "Case / LegalConcept / Cause / Court 四类节点：语料内无对应字段（见 report 结论）",
        },

        "by_level": dict(by_level),
        "by_split_mode": dict(by_split_mode),
        "by_dataset": dict(by_dataset),
        "by_law_type": dict(by_law_type),
        "by_domain_item": dict(by_domain),
        "status_dist": dict(status_dist.most_common()),

        "time_filter_feasibility": {
            "has_effective_to_field_rows": has_effective_to,
            "effective_from_missing_rows": eff_from_missing,
            "effective_period_present_rows": eff_period_present,
            "effective_period_samples": eff_period_samples,
            "publish_date_missing_rows": publish_missing,
            "office_present_rows": office_present,
            "verdict": "has_effective_to = 0 → retrieval.yaml 的 time_filter 规则无字段可依"
                       if has_effective_to == 0 else "field present, check value coverage",
        },

        "grain": {
            "article_no_null_rows": article_no_null,
            "article_label_empty_rows": article_label_empty,
            "paragraph_or_sub_rows": para_or_sub_present,
            "chapter_marked_rows": chap_marked,
            "text_len_chars": {
                lvl: {"n": len(v), "p10": pct(sorted(v), 0.10), "p50": pct(sorted(v), 0.50),
                      "p90": pct(sorted(v), 0.90), "p99": pct(sorted(v), 0.99),
                      "max": max(v) if v else None}
                for lvl, v in len_by_level.items()
            },
            "line_anchor_remaining": line_anchor_hits,
        },

        "graph_edges": {
            "HAS_PROVISION": {"buildable": True, "rows": by_level.get("item", 0)},
            "NEXT": {"buildable": True, "from": "article_seq / article_no 排序"},
            "CITES_named": {
                "events": cite_named_events,
                "distinct_target_names": len(cite_named_laws),
                "resolved_events": cite_named_resolved,
                "resolved_share": (cite_named_resolved / cite_named_events) if cite_named_events else 0,
                "top_targets": dict(cite_named_laws.most_common(15)),
                "samples": cite_named_sample,
            },
            "CITES_self": {
                "events": cite_self_events,
                "resolved_events": cite_self_resolved,
                "resolved_share": (cite_self_resolved / cite_self_events) if cite_self_events else 0,
                "unresolved_samples": cite_self_unresolved_sample,
            },
            "cite_any_total": cite_any_total,
            "unparsable_share": ((cite_any_total - cite_named_events - cite_self_events) / cite_any_total)
                                 if cite_any_total else 0,
        },

        "elapsed_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.join(a.root, a.out_json)), exist_ok=True)
    with open(os.path.join(a.root, a.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
