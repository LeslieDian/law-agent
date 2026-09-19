#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-2：知识图谱「边抽取」（只读语料 → 边表落 indexes/retrieval/）

产出（每条边都可回溯到语料 uid）
--------------------------------
  law_nodes.jsonl            :Law 节点（L2 法名层）
  provision_nodes.jsonl      :Provision 节点（= `statute_items` 行数）
  edges_has_provision.jsonl  (Law)-[:HAS_PROVISION]->(Provision)   = Provision 数
  edges_next.jsonl           (Provision)-[:NEXT]->(Provision)      ★ 图谱扩展主力
  edges_cites.jsonl          (Provision)-[:CITES]->(Provision)     ← 稀疏，仅作辅助

> 节点/边数量**随上游语料变化**，报告里的断言一律与输入行数自洽（不写死具体数字）。

设计依据
--------
`docs/retrieval_design.md` 第 1.4 / 3 节；接口见 `docs/corpus/RETRIEVAL_SCOPE.md` 第 3 节。

关键口径（实测确定，不可想当然）
--------------------------------
1. **L2 法名**（剥书名号 + 剥版本前缀后缀）作 `:Law` 主键 —— 不归一化时两个来源的法名
   字符串交集为 **0**，图谱会被撕成两半（H4）。
2. `article_seq` 的语义**按来源不同**：
   - `twang2218`：`split_doc()` 赋的**文档内条序**（每文档从 1 起）→ 可用于连 NEXT；
   - `pandalla`：`law_item` 路径赋的**文件内递增序号**，每条记录自成一"文档" → 不产生 NEXT。
   因此 NEXT 的分组键取**【来源文档】**（`<dataset>__<file_stem>:<source_index>`），
   **不是** `law_id` —— 同一 L2 法名可能对应多个源文档（228 组跨源合并），
   按 `law_id` 连会把不同版本的第 2 条接在一起，产生错误边。
3. `provision_id` 按设计取 `{law_id}-a{article_no}`；同一 L2 法名下多文档时条号会撞，
   撞号者追加 `-v{k}` 消歧并**单独统计**（这是 L2 合并的必然代价，必须留痕）。
4. doc 级条目（337 条，无条号）也建节点、也挂 HAS_PROVISION，但**不作为 NEXT / CITES 的端点**。
5. `CITES` 只接受**可解析**的引用：`本法第X条`（同法）与 `《法名》第X条`（具名，精确 L2 匹配）。
   无前缀的「第X条」**不抽**（无法定位目标，硬猜等于造边）。

用法
----
    python scripts/retrieval/extract_edges.py \
        --root /mnt/data/lidian/law-agent \
        --out-dir indexes/retrieval \
        --report-json docs/retrieval/EDGE_EXTRACT.json \
        --report-md   docs/retrieval/EDGE_EXTRACT.md
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_law_title import iter_jsonl, strip_version, strip_wrappers  # noqa: E402

# ---------------------------------------------------------------- 中文数字
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}
NUM_PAT = r"[0-9〇零一二三四五六七八九十百千两]+"


def cn2int(s):
    """中文/阿拉伯条号 → int。无法解析返回 None。

    「一百一十四」→114　「十」→10　「二十」→20　「十二」→12
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, cur, seen = 0, 0, False
    for ch in s:
        if ch in CN_DIGITS:
            cur = CN_DIGITS[ch]
            seen = True
        elif ch in CN_UNITS:
            u = CN_UNITS[ch]
            if cur == 0:
                cur = 1
            total += cur * u
            cur = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return total + cur


# 自指引用：正文里的「本法 / 本条例 / 本解释 …第X条」都指向**当前条文所属的法**
# （实测语料里 twang2218 的行政法规/司法解释普遍用「本条例」「本解释」自指，
#  只匹配「本法」会漏掉一半以上 —— 这正是 H3「CITES 太稀疏」的成因之一）
SELF_NOUN = r"(?:法|条例|规定|解释|规则|办法|细则|准则|意见|批复|通知|决定|命令)"
RE_SELF = re.compile(r"本\s*" + SELF_NOUN + r"\s*第\s*(" + NUM_PAT + r")\s*条")
# 具名引用：书名号也可能是 〈〉（外层已用《》时的嵌套写法）
RE_NAMED = re.compile(r"[《〈]\s*([^《》〈〉]{2,80}?)\s*[》〉]\s*第\s*(" + NUM_PAT + r")\s*条")

# 正文指纹的宽松口径：去空白与标点后再哈希（跨源同一法条常只差标点全半角）
PUNCT_RE = re.compile(r"[\s\u3000,，。、;；:：!！?？\"'“”‘’()（）《》〈〉\[\]【】—\-－…·]+")


def hard_hash(t):
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()


def loose_hash(t):
    return hashlib.sha1(PUNCT_RE.sub("", t or "").encode("utf-8")).hexdigest()


# twang2218 的 law_type 口径；pandalla 的是「类别 : XXX」（语义不同，H7），全部并入 raw
TYPE_STD = ("法律", "行政法规", "司法解释", "法律解释")


def file_stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def doc_key_of(rec):
    """来源文档全局键 —— NEXT 的连线单位（与阶段 2b 的 doc_uid 同构）。"""
    return "%s__%s:%s" % (
        (rec.get("source_dataset") or "?").replace("/", "__"),
        file_stem(rec.get("source_file") or "?"),
        rec.get("source_index"))


def official_files(pattern):
    """只取**正式产出文件**：排除 `_` 开头的合并副本（`_all.jsonl`）与 `*.sample.jsonl` 调试残留。

    ★ 与 normalize_corpus / apply_decontam / downsample_split 的硬约定 1 同口径。
    读 `_all.jsonl` 必然**双计**（它是分文件的拼接副本，不是独立样本）；
    读 `*.sample.jsonl` 会把调试残留当正式语料 → 图谱规模虚高。
    """
    return [p for p in sorted(glob.glob(pattern))
            if not os.path.basename(p).startswith("_")
            and ".sample." not in os.path.basename(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-dir", default="indexes/retrieval")
    ap.add_argument("--report-json", default="docs/retrieval/EDGE_EXTRACT.json")
    ap.add_argument("--report-md", default="docs/retrieval/EDGE_EXTRACT.md")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用，0=全量）")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    item_files = official_files(root + "/data/corpus/statute_items/*.jsonl")
    if not item_files:
        print("[FATAL] 找不到 statute_items", file=sys.stderr)
        return 2

    # ---------------------------------------------------------- 1. 载入 + 建 provision 键
    charges = []          # 每条条文的工作记录
    law_type_votes = collections.defaultdict(set)
    law_domains = collections.defaultdict(set)
    law_sources = collections.defaultdict(set)
    law_docs = collections.defaultdict(set)
    law_raw_names = collections.defaultdict(set)
    by_law_article = collections.defaultdict(list)  # (law_id, article_no) -> [provision_id]
    n_rows = 0
    n_no_title = 0
    n_bad_seq = 0

    stop = False
    for p in item_files:
        if stop:
            break
        for rec in iter_jsonl(p):
            if args.limit and len(charges) >= args.limit:
                stop = True
                break
            n_rows += 1
            raw_title = (rec.get("law_title") or rec.get("law_name") or "").strip()
            if not raw_title:
                n_no_title += 1
                continue
            law_id = strip_version(strip_wrappers(raw_title))
            if not law_id:
                n_no_title += 1
                continue
            level = rec.get("level") or "item"
            article_no = rec.get("article_no")
            seq = rec.get("article_seq")
            if not isinstance(seq, int) or seq <= 0:
                seq = 1
                if level != "doc":
                    n_bad_seq += 1
            ltype = (rec.get("law_type") or "").strip()
            law_type_votes[law_id].add(ltype)
            law_domains[law_id].add(rec.get("domain") or "unknown")
            law_sources[law_id].add(rec.get("source_dataset") or "")
            law_raw_names[law_id].add(raw_title)
            dkey = doc_key_of(rec)
            law_docs[law_id].add(dkey)
            charges.append({
                "law_id": law_id, "law_name": law_id, "raw_title": raw_title,
                "level": level, "article_no": article_no,
                "article_label": rec.get("article_label") or "",
                "article_seq": seq, "doc_key": dkey,
                "law_type": ltype,
                "status": rec.get("status") or "",
                "effective_from": rec.get("effective_from") or "",
                "effective_period": rec.get("effective_period") or "",
                "publish_date": rec.get("publish_date") or "",
                "chapter": rec.get("chapter") or "",
                "domain": rec.get("domain") or "",
                "source_dataset": rec.get("source_dataset") or "",
                "uid": rec.get("uid") or "",
                "text": rec.get("text") or "",
                "text_hash": rec.get("content_sha1") or "",
                "char_len": rec.get("char_len") or 0,
            })

    # ---------------------------------------------------------- 2. provision_id 分配
    # 先按 (law_id, article_no) 分组，组内按 uid 排序保证确定性
    grouped = collections.defaultdict(list)
    doc_level_recs = []
    for c in charges:
        if c["level"] == "doc" or c["article_no"] in (None, ""):
            doc_level_recs.append(c)
        else:
            grouped[(c["law_id"], c["article_no"])].append(c)

    n_disambiguated = 0
    disambig_groups = []
    for key in sorted(grouped):
        lst = sorted(grouped[key], key=lambda x: x["uid"])
        for i, c in enumerate(lst):
            base = "%s-a%s" % (c["law_id"], c["article_no"])
            c["provision_id"] = base if i == 0 else "%s-v%d" % (base, i + 1)
            if i > 0:
                n_disambiguated += 1
        if len(lst) > 1:
            disambig_groups.append({"law_id": key[0], "article_no": key[1],
                                    "count": len(lst),
                                    "provision_ids": [x["provision_id"] for x in lst[:6]]})

    # doc 级：provision_id = {law_id}-doc；同一法名下多文档时加 -v{k}
    used_ids = set(c["provision_id"] for c in charges
                   if c.get("provision_id") and c["level"] != "doc")
    doc_counter = collections.Counter()
    for c in sorted(doc_level_recs, key=lambda x: x["uid"]):
        base = "%s-doc" % c["law_id"]
        doc_counter[base] += 1
        k = doc_counter[base]
        pid = base if k == 1 else "%s-v%d" % (base, k)
        while pid in used_ids:
            k += 1
            pid = "%s-v%d" % (base, k)
        used_ids.add(pid)
        c["provision_id"] = pid

    # 索引：provision_id -> 记录； (law_id, article_no) -> [pid...]
    prov_by_id = {}
    for c in charges:
        prov_by_id[c["provision_id"]] = c
        if c["level"] != "doc" and c["article_no"] not in (None, ""):
            by_law_article[(c["law_id"], c["article_no"])].append(c["provision_id"])

    pid_unique = len(prov_by_id) == len(charges)

    # ---------------------------------------------------------- 3. NEXT 边（按来源文档）
    next_edges = []
    chain_heads = 0
    doc_sizes = collections.Counter()
    doc_groups = collections.defaultdict(list)
    for c in charges:
        if c["level"] != "doc":
            doc_groups[c["doc_key"]].append(c)
    for dkey in sorted(doc_groups):
        lst = doc_groups[dkey]
        doc_sizes[len(lst)] += 1
        lst.sort(key=lambda x: (x["article_seq"], x["article_no"] or 0, x["uid"]))
        chain_heads += 1
        for a, b in zip(lst, lst[1:]):
            next_edges.append({"start": a["provision_id"], "end": b["provision_id"],
                               "doc": dkey})

    # ---------------------------------------------------------- 4. CITES 边
    law_id_set = set(c["law_id"] for c in charges)

    def resolve_law(name_l2):
        """精确 L2 命中优先；否则做**唯一后缀**兜底（简称→全称），有歧义则放弃。

        实测：具名引用里大量是简称（`《土地管理法》` 而法名是 `中华人民共和国土地管理法`），
        只做精确匹配会让一半具名引用解析失败。兜底只在**唯一匹配**时生效，
        避免把 `《民事诉讼法》` 错连到 `最高人民法院关于适用…的解释` 之类的多个候选。
        """
        if not name_l2:
            return [], "miss"
        if name_l2 in law_id_set:
            return [name_l2], "exact"
        if len(name_l2) >= 3:
            cands = [lid for lid in law_id_set if lid.endswith(name_l2)]
            if len(cands) == 1:
                return cands, "alias"
        return [], "miss"

    cites = set()
    cites_stats = collections.Counter()
    unparsed_named = collections.Counter()
    for c in charges:
        if c["level"] == "doc":
            continue
        pid = c["provision_id"]
        text = c["text"]
        # (a) 自指：「本法 / 本条例 / 本解释 …第X条」→ 同法内条文
        for m in RE_SELF.finditer(text):
            tgt = cn2int(m.group(1))
            if tgt is None:
                continue
            cites_stats["self_raw"] += 1
            hits = by_law_article.get((c["law_id"], tgt)) or []
            if not hits:
                cites_stats["self_unresolved"] += 1
                continue
            for h in hits:
                if h != pid:
                    cites.add((pid, h, "self", m.group(0)))
            cites_stats["self_resolved"] += 1
        # (b) 具名：「《法名》第X条」→ 指定法的条文
        for m in RE_NAMED.finditer(text):
            name_raw, num_raw = m.group(1), m.group(2)
            cites_stats["named_raw"] += 1
            tgt = cn2int(num_raw)
            if tgt is None:
                continue
            laws, how = resolve_law(strip_version(strip_wrappers(name_raw)))
            cites_stats["named_%s" % how] += 1
            if not laws:
                unparsed_named[name_raw] += 1
                continue
            hits = []
            for L in laws:
                hits += by_law_article.get((L, tgt)) or []
            if not hits:
                cites_stats["named_unresolved"] += 1
                unparsed_named[name_raw] += 1
                continue
            for h in hits:
                if h != pid:
                    cites.add((pid, h, "named", m.group(0)))
            cites_stats["named_resolved"] += 1

    cites_edges = [{"start": s, "end": e, "kind": k, "evidence": ev}
                   for (s, e, k, ev) in sorted(cites)]

    # ---------------------------------------------------------- 5. 落盘
    def write_jsonl(name, rows):
        fp = os.path.join(out_dir, name)
        with open(fp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return fp

    law_rows = []
    for law_id in sorted(law_domains):
        types_raw = sorted(t for t in law_type_votes[law_id] if t)
        types_std = sorted(t for t in types_raw if t in TYPE_STD)
        law_rows.append({
            "law_id": law_id, "law_name": law_id,
            "law_type_std": (types_std[0] if len(types_std) == 1
                             else ("多类型" if len(types_std) > 1 else "未标注")),
            "law_type_raw": types_raw,
            "domains": sorted(law_domains[law_id]),
            "source_datasets": sorted(law_sources[law_id]),
            "n_docs": len(law_docs[law_id]),
            "raw_name_variants": sorted(law_raw_names[law_id]),
        })
    law_item_cnt = collections.Counter()
    for c in charges:
        law_item_cnt[c["law_id"]] += 1
    for r in law_rows:
        r["n_provisions"] = law_item_cnt[r["law_id"]]

    prov_rows = [{
        "provision_id": c["provision_id"], "uid": c["uid"],
        "law_id": c["law_id"], "law_name": c["law_name"],
        "level": c["level"], "article_no": c["article_no"],
        "article_label": c["article_label"], "article_seq": c["article_seq"],
        "doc_key": c["doc_key"], "chapter": c["chapter"],
        "status": c["status"], "effective_from": c["effective_from"],
        "effective_period": c["effective_period"], "publish_date": c["publish_date"],
        "domain": c["domain"], "law_type": c["law_type"],
        "source_dataset": c["source_dataset"],
        "text_hash": c["text_hash"], "char_len": c["char_len"],
    } for c in charges]

    write_jsonl("law_nodes.jsonl", law_rows)
    write_jsonl("provision_nodes.jsonl", prov_rows)
    write_jsonl("edges_has_provision.jsonl",
                [{"start": c["law_id"], "end": c["provision_id"]} for c in charges])
    write_jsonl("edges_next.jsonl", next_edges)
    write_jsonl("edges_cites.jsonl", cites_edges)

    # ---------------------------------------------------------- 6. 断言 + 报告
    # 重复条文登记（**只登记、不改口径**：Provision 节点仍按语料条数建）
    # ⚠️ 不能用语料的 `content_sha1`：它是对 `text_full`（含法名 + 条号前缀）算的，
    #    跨源同一条文因前缀写法不同必然不同 → 用它判跨源重复是**假阴性**（实测 0 组）。
    #    这里改用**正文**的两级指纹：精确（原文）+ 宽松（去空白标点）。
    def dup_stats(keyfn):
        cnt = collections.Counter()
        src = collections.defaultdict(set)
        law = {}
        for c in charges:
            if c["level"] == "doc":
                continue
            k = keyfn(c)
            cnt[k] += 1
            src[k].add(c["source_dataset"])
            if k not in law:
                law[k] = c["law_id"]
        groups = [k for k, v in cnt.items() if v > 1]
        return {
            "unique_texts": len(cnt),
            "dup_groups": len(groups),
            "dup_rows": sum(cnt[k] - 1 for k in groups),
            "cross_source_groups": sum(1 for k in groups if len(src[k]) > 1),
            "same_source_groups": sum(1 for k in groups if len(src[k]) == 1),
        }

    dup_exact = dup_stats(lambda c: hard_hash(c["text"]))
    dup_loose = dup_stats(lambda c: loose_hash(c["text"]))
    multi_doc_laws = sum(1 for r in law_rows if r["n_docs"] > 1)

    n_law = len(law_rows)
    n_prov = len(prov_rows)
    n_item = sum(1 for c in charges if c["level"] != "doc")
    n_doc = n_prov - n_item
    checks = {
        # ★ 2026-09-19：不要写死 1,579 / 65,037 —— 上游语料一变（如阶段 3 去污修复），
        #   写死的断言会把一次**正确**的重跑判成 FAIL。改为「与输入行数自洽 + 合理区间」。
        "provision_nodes_matches_input": (n_prov == n_rows) if not args.limit else True,
        "law_nodes_plausible_500_5000": (500 <= n_law <= 5000) if not args.limit else True,
        "provision_id_unique": pid_unique,
        "has_provision_covers_all": len(charges) == n_prov,
        "next_edges_plausible": len(next_edges) >= 50000 if not args.limit else True,
        "cites_edges_plausible": 1000 <= len(cites_edges) <= 60000 if not args.limit else True,
        "no_empty_provision_id": all(c.get("provision_id") for c in charges),
        # ★ 输入里不得混入合并副本 / 调试残留（否则节点数与法条数双计）
        "inputs_are_official_only": all(
            not os.path.basename(p).startswith("_")
            and ".sample." not in os.path.basename(p) for p in item_files),
    }
    R = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": root, "limit": args.limit,
        "input": {"files": [os.path.basename(p) for p in item_files], "rows": n_rows},
        "nodes": {"Law": n_law, "Provision": n_prov,
                  "Provision_item": n_item, "Provision_doc": n_doc},
        "edges": {"HAS_PROVISION": n_prov, "NEXT": len(next_edges), "CITES": len(cites_edges)},
        "next_detail": {"chain_heads": chain_heads,
                        "avg_chain_len": round(n_item / max(chain_heads, 1), 2),
                        "doc_size_hist_top": dict(doc_sizes.most_common(10))},
        "cites_detail": {
            "self_raw": cites_stats["self_raw"],
            "self_resolved": cites_stats["self_resolved"],
            "named_raw": cites_stats["named_raw"],
            "named_exact": cites_stats["named_exact"],
            "named_alias": cites_stats["named_alias"],
            "named_resolved": cites_stats["named_resolved"],
            "raw_total": cites_stats["self_raw"] + cites_stats["named_raw"],
            "resolved_total": cites_stats["self_resolved"] + cites_stats["named_resolved"],
            "resolve_rate": round(
                (cites_stats["self_resolved"] + cites_stats["named_resolved"]) /
                max(cites_stats["self_raw"] + cites_stats["named_raw"], 1), 4),
            "top_unresolved_named": [{"name": k, "n": v}
                                     for k, v in unparsed_named.most_common(20)],
        },
        "provision_id_disambiguation": {
            "count": n_disambiguated,
            "groups": len(disambig_groups),
            "laws_with_multi_docs": multi_doc_laws,
            "note": "同一 L2 法名下存在多个源文档（版本变体 / 跨源重号）时条号会撞",
            "top_groups": disambig_groups[:15],
        },
        "duplicate_provision_text": {
            "exact": dup_exact,
            "loose": dup_loose,
            "note": ("只登记不改口径：Provision 节点仍按语料条数建。"
                     "exact = 正文原文 sha1；loose = 去空白标点后 sha1"
                     "（跨源同一条文常只差全半角标点）"),
        },
        "law_type_dist": dict(collections.Counter(r["law_type_std"] for r in law_rows)),
        "rows_missing_title": n_no_title,
        "rows_bad_seq": n_bad_seq,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    for path in (args.report_json, args.report_md):
        d = path if os.path.isabs(path) else os.path.join(root, path)
        os.makedirs(os.path.dirname(d), exist_ok=True)
    rj = args.report_json if os.path.isabs(args.report_json) else os.path.join(root, args.report_json)
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    md = []
    W = md.append
    W("# 阶段 4b-2 边抽取报告\n")
    W("> 生成时间：%s　脚本：`scripts/retrieval/extract_edges.py`　"
      "输入：`data/corpus/statute_items/*.jsonl`（%d 行）\n" % (R["generated_at"], n_rows))
    W("## 1. 节点\n")
    W("| 节点 | 数量 |")
    W("|---|---:|")
    W("| `:Law`（L2 法名层） | **%d** |" % n_law)
    W("| `:Provision` 合计 | **%d** |" % n_prov)
    W("| └ `level=item` | %d |" % n_item)
    W("| └ `level=doc`（无条号，兜底） | %d |" % n_doc)
    W("")
    W("`law_type` 分布（标准化口径）：%s\n" % json.dumps(R["law_type_dist"], ensure_ascii=False))
    W("## 2. 边\n")
    W("| 关系 | 数量 | 说明 |")
    W("|---|---:|---|")
    W("| `HAS_PROVISION` | %d | 每条条文都挂在它的 L2 法下 |" % n_prov)
    W("| `NEXT` | **%d** | ★ 图谱扩展主力（覆盖全部条文） |" % len(next_edges))
    W("| `CITES` | %d | 交叉引用，稀疏，仅作辅助 |" % len(cites_edges))
    W("")
    W("`NEXT` 明细：链数 **%d**，平均链长 %.2f。"
      % (chain_heads, R["next_detail"]["avg_chain_len"]))
    W("链数以**来源文档**为单位（不是法名）——`article_seq` 只在同一源文档内连续。\n")
    W("`CITES` 明细：可解析引用 **%d / %d**（解析率 **%.1f%%**）。"
      % (R["cites_detail"]["resolved_total"], R["cites_detail"]["raw_total"],
         100.0 * R["cites_detail"]["resolve_rate"]))
    W("- 自指（`本法`/`本条例`/`本解释`…第X条）：%d → 解析 %d"
      % (R["cites_detail"]["self_raw"], R["cites_detail"]["self_resolved"]))
    W("- 具名（`《法名》第X条`）：%d → 解析 %d（精确 %d + 简称兜底 %d）"
      % (R["cites_detail"]["named_raw"], R["cites_detail"]["named_resolved"],
         R["cites_detail"]["named_exact"], R["cites_detail"]["named_alias"]))
    W("- 无前缀「第X条」**不抽**（无法定位目标，硬猜等于造边）\n")
    W("## 3. `provision_id` 消歧（L2 合并的必然代价）\n")
    W("同一 L2 法名下可能有多个源文档（版本变体），条号会撞 → 撞号者加 `-v{k}`：")
    W("**%d 条**被消歧（%d 个组）。\n"
      % (n_disambiguated, len(disambig_groups)))
    if disambig_groups:
        W("| 法名 | 条号 | 冲突条数 |")
        W("|---|---:|---:|")
        for g in disambig_groups[:10]:
            W("| %s | %s | %d |" % (g["law_id"][:40], g["article_no"], g["count"]))
        W("")
    W("## 4. 重复条文（只登记，不改口径）\n")
    d = R["duplicate_provision_text"]
    W("> ⚠️ 判重用的是**正文**指纹，**不是**语料的 `content_sha1` —— 后者是对 `text_full`")
    W("> （含法名 + 条号前缀）计算的，跨源同一条文因前缀写法不同必然不同，")
    W("> 用它判跨源重复会得到 **0 组**（假阴性，第一版即踩此坑）。\n")
    W("| 口径 | 唯一正文 | 重复组 | 多余副本 | 跨源重复组 | 同源重复组 |")
    W("|---|---:|---:|---:|---:|---:|")
    for label, key in (("精确（正文原文）", "exact"), ("宽松（去空白标点）", "loose")):
        x = d[key]
        W("| %s | %d | %d | %d | **%d** | %d |"
          % (label, x["unique_texts"], x["dup_groups"], x["dup_rows"],
             x["cross_source_groups"], x["same_source_groups"]))
    W("")
    W("多文档 L2 法名 **%d** 个（`provision_id` 消歧与重复条文的共同根因）。\n" % multi_doc_laws)
    W("## 5. 断言\n")
    for k, v in checks.items():
        W("- `%s` = %s" % (k, "true" if v else "**false**"))
    W("")
    W("**verdict = %s**　耗时 %.1fs\n" % (R["verdict"], R["elapsed_sec"]))
    rmd = args.report_md if os.path.isabs(args.report_md) else os.path.join(root, args.report_md)
    with open(rmd, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("=" * 70)
    print("阶段 4b-2 边抽取")
    print("=" * 70)
    print("  输入 %d 行（%s）" % (n_rows, ", ".join(os.path.basename(p) for p in item_files)))
    print("  :Law = %d    :Provision = %d (item %d + doc %d)" % (n_law, n_prov, n_item, n_doc))
    print("  HAS_PROVISION = %d" % n_prov)
    print("  NEXT          = %d  (链 %d，平均链长 %.2f)"
          % (len(next_edges), chain_heads, R["next_detail"]["avg_chain_len"]))
    print("  CITES         = %d  (解析率 %.1f%%，raw %d)"
          % (len(cites_edges), 100.0 * R["cites_detail"]["resolve_rate"],
             R["cites_detail"]["raw_total"]))
    print("  provision_id 消歧 %d 条 / %d 组" % (n_disambiguated, len(disambig_groups)))
    print("  law_type 分布 %s" % json.dumps(R["law_type_dist"], ensure_ascii=False))
    bad = [k for k, v in checks.items() if not v]
    print("  断言：%s" % ("全部通过" if not bad else "失败 " + ", ".join(bad)))
    print("  verdict = %s   耗时 %.1fs" % (R["verdict"], R["elapsed_sec"]))
    print("  边表 -> %s" % out_dir)
    print("EDGE_EXTRACT_DONE")
    return 0 if R["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
