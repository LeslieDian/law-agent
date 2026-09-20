#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-4：导入 Neo4j（约束 → 节点 → 边 → 热度 → 向量索引 → 计数验收）

★ 库布局（2026-09-19 按用户拍板：**case 4 库 + 法条 1 库**）
---------------------------------------------------------------
磁盘（`indexes/retrieval/embeddings/`）
    provision_embedding           法条 1 库（item + doc 两种 level 混库）
    case_embedding_general        案件库：通用
    case_embedding_civil          案件库：民事
    case_embedding_criminal       案件库：刑事
    case_embedding_procedural     案件库：程序

Neo4j 侧一一对应（**每个库一个向量索引，落在各自的属性上**）
    (p:Provision).embedding        ← provision_embedding 的 level=item 行
    (p:Provision).embedding_doc    ← provision_embedding 的 level=doc 行
    (c:Case).embedding_general     ← case_embedding_general
    (c:Case).embedding_civil       ← case_embedding_civil
    (c:Case).embedding_criminal    ← case_embedding_criminal
    (c:Case).embedding_procedural  ← case_embedding_procedural

★ 为什么 case 用「每库一个属性 + 一个索引」，不是「一个属性 + 域过滤」
  ① 与磁盘布局一一对应，「按域检索」是**真的走不同索引**，不是查询时过滤，
     消融实验才有意义（索引更小、域内纯度由构建期保证）；
  ② Neo4j 向量索引对「同一 (label, property) 建多个」没有意义（内容相同）。
  代价：一个 Case 只有一个非空 embedding_* 属性，未建的属性不进该索引 —— 正是我们要的。

★ 为什么法条 item / doc 仍分两个属性
  item 中位 91 字、doc 级最长 61,387 字，混池会污染 top-k。构建期两者虽同库
  （为保持「法条 1 库」的口径），入库时按 `level` 分流到两个索引，检索侧可各取所需。

★ 热度（用户 2026-09-19 拍板，脚本 `build_hotness.py`）
    Provision.hotness / hotness_cites / out_cites
    Case.hotness      / n_same_case
    排序用法：final = rrf_score * (1 + W_HOT * hotness)，W_HOT 默认 0.05
    （热度只做 ≤5% 的先验微调，不允许盖过相关性）

节点
----
  :Law            法名层（L2 规范化后）           ← `law_nodes.jsonl`
  :Provision      条文（item + doc），**带 1024d 向量** ← `provision_nodes.jsonl`
  :Case           案件文书（样本级，与四份 split 强隔离），**带 1024d 向量**
  :Domain 4  :LawType 4  :SourceDataset 2

边
--
  (:Law)-[:HAS_PROVISION]->(:Provision)          ← `provision_nodes.jsonl` 行数
  (:Provision)-[:NEXT]->(:Provision)             ★ 图谱扩展主力
  (:Provision)-[:CITES]->(:Provision)            ← 辅助
  (:Provision)-[:IN_DOMAIN|OF_TYPE|FROM_SOURCE]
  (:Case)-[:SAME_CASE]->(:Case)                  同案多视角星形连接

★ 验收期望值一律**从输入产物现算**（`build_expectations()`），不写死数字 ——
  法条流会随上游修复而变化，写死会让一次正确的重跑被误判为 FAIL。

★ 幂等
------
全程 MERGE + IF NOT EXISTS，可反复重跑；`--step reset` 才会清库。

用法
----
    python scripts/retrieval/import_neo4j.py --step all \
        --root /mnt/data/lidian/law-agent \
        --uri bolt://127.0.0.1:7687 --user neo4j --password *** \
        --report-md docs/retrieval/NEO4J_IMPORT.md
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_law_title import iter_jsonl  # noqa: E402

DOMAINS = ["civil", "criminal", "procedural", "general"]
DOMAIN4 = ("general", "civil", "criminal", "procedural")
LAW_TYPES = ["法律", "行政法规", "司法解释", "法律解释"]

PROVISION_COLLECTION = "provision_embedding"
CASE_COLLECTIONS = {
    "general": "case_embedding_general",
    "civil": "case_embedding_civil",
    "criminal": "case_embedding_criminal",
    "procedural": "case_embedding_procedural",
}
CASE_EMB_PROP = {d: "embedding_" + d for d in DOMAIN4}

CONSTRAINTS = [
    ("Law", "law_id"), ("Provision", "provision_id"), ("Case", "case_id"),
    ("Domain", "name"), ("LawType", "name"), ("SourceDataset", "name"),
]
RANGE_INDEXES = [
    ("provision_law_id_idx", "Provision", ["law_id"]),
    ("provision_article_no_idx", "Provision", ["law_id", "article_no"]),
    ("provision_status_idx", "Provision", ["status"]),
    ("provision_level_idx", "Provision", ["level"]),
    ("case_sha1_idx", "Case", ["case_sha1"]),
    ("case_domain_idx", "Case", ["domain"]),
    ("case_domain4_idx", "Case", ["domain4"]),
    ("case_lib_idx", "Case", ["case_lib"]),
]
VECTOR_INDEXES = [
    ("provision_embedding", "Provision", "embedding", 1024),
    ("provision_doc_embedding", "Provision", "embedding_doc", 1024),
] + [(CASE_COLLECTIONS[d], "Case", CASE_EMB_PROP[d], 1024) for d in DOMAIN4]
FULLTEXT_INDEXES = [
    ("provision_fulltext", "Provision", ["law_name", "text"]),
    ("case_fulltext", "Case", ["facts", "question"]),
]
# 旧实验遗留的向量索引（与本轮口径不符）—— `--drop-legacy` 时清掉
LEGACY_VECTOR_INDEXES = ["provision_embedding_4b", "provision_embedding_bert768",
                         "provision_embedding_bge_m3"]
LEGACY_CONSTRAINTS = ["law_version_id_unique", "cause_name_unique",
                      "court_name_unique", "legal_concept_name_unique"]


def batches(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def count_meta_rows(coll_dir):
    """一个向量 collection 的向量行数（= 分片 meta 的 jsonl 行数）。"""
    n = 0
    for p in sorted(glob.glob(os.path.join(coll_dir, "shard_*.jsonl"))):
        n += count_lines(p)
    return n


def count_meta_by_level(coll_dir):
    """provision 库按 level 分流计数 → {"item": n, "doc": n}"""
    acc = collections.Counter()
    for p in sorted(glob.glob(os.path.join(coll_dir, "shard_*.jsonl"))):
        for m in iter_jsonl(p):
            acc[m.get("level") or "item"] += 1
    return dict(acc)


def load_hotness(path):
    if not os.path.exists(path):
        return None, {}
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    return h, h.get("meta", {})


def build_expectations(edges_root, emb_root):
    """★ 验收期望值**从输入产物现算**，不写死。

    2026-09-19 教训：原来把 65,037 / 62,258 / 6,459 硬编码进断言，
    阶段 3 去污修复后法条流从 22,771 恢复到 23,510，切条数一变这些断言就全部失真
    （而且会以「FAIL」的形式把一次正确重跑判成失败）。改为与输入 JSONL 行数自洽比对。
    """
    from os.path import join as J
    prov_dir = J(emb_root, PROVISION_COLLECTION)
    by_level = count_meta_by_level(prov_dir)
    n_prov_item = by_level.get("item", 0)
    n_prov_doc = by_level.get("doc", 0)
    n_case = {}
    for d in DOMAIN4:
        n_case[d] = count_meta_rows(J(emb_root, CASE_COLLECTIONS[d]))
    n_case_all = sum(n_case.values())
    hot, _meta = load_hotness(J(edges_root, "hotness.json"))
    exp = {
        "Law": count_lines(J(edges_root, "law_nodes.jsonl")),
        "Provision": count_lines(J(edges_root, "provision_nodes.jsonl")),
        "HAS_PROVISION": count_lines(J(edges_root, "provision_nodes.jsonl")),
        "NEXT": count_lines(J(edges_root, "edges_next.jsonl")),
        "CITES": count_lines(J(edges_root, "edges_cites.jsonl")),
        "provision_embedding_rows": n_prov_item,
        "provision_doc_embedding_rows": n_prov_doc,
        "Case": n_case_all,
        "case_embedding_rows": n_case_all,
    }
    exp["_detail"] = {
        "provision_by_level": by_level,
        "case_rows_by_domain": n_case,
        "hotness_available": bool(hot),
    }
    if hot:
        exp["hotness_provision"] = len(hot.get("provision") or {})
        exp["hotness_case"] = len(hot.get("case") or {})
    return exp


def iter_vectors(coll_dir):
    """流式读一个 collection 的 (meta, vector_list)，按 shard 顺序。"""
    import numpy as np
    files = sorted(glob.glob(os.path.join(coll_dir, "shard_*.npy")))
    for npy in files:
        meta = npy[:-4] + ".jsonl"
        if not os.path.exists(meta):
            continue
        vecs = np.load(npy)
        with open(meta, "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line), vecs[i].tolist()
                i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--uri", default="bolt://127.0.0.1:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--step", default="all",
                    choices=["all", "reset", "constraints", "nodes", "edges",
                             "hotness", "indexes", "verify"])
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="每类只导前 N 条（冒烟用）")
    ap.add_argument("--drop-legacy", action="store_true",
                    help="顺带删除旧实验遗留的向量索引与约束（schema 级清理）")
    ap.add_argument("--report-md", default="docs/retrieval/NEO4J_IMPORT.md")
    args = ap.parse_args()

    if not args.password:
        print("[FATAL] 需要 --password 或环境变量 NEO4J_PASSWORD", file=sys.stderr)
        return 2

    from neo4j import GraphDatabase  # noqa: E402
    root = args.root.rstrip("/")
    emb_root = os.path.join(root, "indexes/retrieval/embeddings")
    edges_root = os.path.join(root, "indexes/retrieval")
    t0 = time.time()
    log = []
    counters = {}

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    print("连接 %s ..." % args.uri, flush=True)

    def run(cypher, **kw):
        with driver.session() as s:
            return s.run(cypher, **kw).consume()

    def count(label=None, rel=None, where=""):
        if rel:
            q = "MATCH ()-[r:%s]->() RETURN count(r) AS c" % rel
        elif label:
            q = "MATCH (n:%s) %s RETURN count(n) AS c" % (label, where)
        else:
            q = "MATCH (n) RETURN count(n) AS c"
        with driver.session() as s:
            return s.run(q).single()["c"]

    def step(name):
        log.append("[%6.1fs] %s" % (time.time() - t0, name))
        print("  " + log[-1], flush=True)

    # ------------------------------------------------------------- reset
    if args.step == "reset":
        step("清库 MATCH (n) DETACH DELETE n")
        run("MATCH (n) DETACH DELETE n")
        if args.drop_legacy:
            step("删旧实验向量索引 / 约束（--drop-legacy）")
            for nm in LEGACY_VECTOR_INDEXES:
                try:
                    run("DROP INDEX %s IF EXISTS" % nm)
                    log.append("      dropped index %s" % nm)
                except Exception as e:
                    log.append("      drop index %s 失败：%s" % (nm, e))
            for nm in LEGACY_CONSTRAINTS:
                try:
                    run("DROP CONSTRAINT %s IF EXISTS" % nm)
                    log.append("      dropped constraint %s" % nm)
                except Exception as e:
                    log.append("      drop constraint %s 失败：%s" % (nm, e))
        print("DONE_RESET")
        return 0

    # ------------------------------------------------------- constraints
    if args.step in ("all", "constraints"):
        step("建约束与索引")
        for label, prop in CONSTRAINTS:
            run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:%s) REQUIRE n.%s IS UNIQUE"
                % (label, prop))
        for name, label, props in RANGE_INDEXES:
            run("CREATE INDEX %s IF NOT EXISTS FOR (n:%s) ON (%s)"
                % (name, label, ", ".join("n." + p for p in props)))
        for name, label, props in FULLTEXT_INDEXES:
            run("CREATE FULLTEXT INDEX %s IF NOT EXISTS FOR (n:%s) ON EACH [%s]"
                % (name, label, ", ".join("n." + p for p in props)))

    # ------------------------------------------------------------- nodes
    if args.step in ("all", "nodes"):
        # 1) Law / LawType / Domain / SourceDataset
        step("导 Law / LawType / Domain / SourceDataset")
        law_rows = [json.loads(l) for l in
                    open(edges_root + "/law_nodes.jsonl", encoding="utf-8") if l.strip()]
        if args.limit:
            law_rows = law_rows[:args.limit]
        for b in batches(law_rows, args.batch):
            run("""UNWIND $rows AS r
                   MERGE (l:Law {law_id: r.law_id})
                   SET l.law_name = r.law_name, l.law_type = r.law_type_std,
                       l.n_provisions = r.n_provisions, l.n_docs = r.n_docs,
                       l.domains = r.domains""", rows=b)
        run("UNWIND $rows AS r MERGE (d:Domain {name: r}) SET d.label = r", rows=DOMAINS)
        run("UNWIND $rows AS r MERGE (t:LawType {name: r})", rows=LAW_TYPES)
        run("""UNWIND $rows AS r MERGE (s:SourceDataset {name: r})
               SET s.short = split(r, '/')[1]""",
            rows=["twang2218/chinese-law-and-regulations",
                  "pandalla/chinese_law_examples"])
        counters["Law"] = count("Law")

        # 2) Provision（item / doc 分流到两个属性 → 两个向量索引）
        step("导 Provision（item → p.embedding；doc → p.embedding_doc）")
        text_by_uid = {}
        for p in sorted(glob.glob(root + "/data/corpus/statute_items/*.jsonl")):
            for rec in iter_jsonl(p):
                text_by_uid[rec["uid"]] = rec.get("text") or ""
        prov_meta = [json.loads(l) for l in
                     open(edges_root + "/provision_nodes.jsonl", encoding="utf-8")
                     if l.strip()]
        meta_by_id = {m["provision_id"]: m for m in prov_meta}
        if args.limit:
            meta_by_id = dict(list(meta_by_id.items())[:args.limit])

        n_item = n_doc = 0
        item_buf, doc_buf = [], []
        for m, vec in iter_vectors(os.path.join(emb_root, PROVISION_COLLECTION)):
            src = meta_by_id.get(m["id"])
            if not src:
                continue
            row = dict(src, text=text_by_uid.get(src["uid"], ""))
            row["domain4"] = m.get("domain4") or ""
            if (m.get("level") or "item") == "doc":
                row["embedding_doc"] = vec
                doc_buf.append(row)
                n_doc += 1
                if len(doc_buf) >= 500:
                    run(PROV_DOC_CYPHER, rows=doc_buf)
                    doc_buf = []
            else:
                row["embedding"] = vec
                item_buf.append(row)
                n_item += 1
                if len(item_buf) >= 500:
                    run(PROV_CYPHER, rows=item_buf)
                    item_buf = []
        if item_buf:
            run(PROV_CYPHER, rows=item_buf)
        if doc_buf:
            run(PROV_DOC_CYPHER, rows=doc_buf)
        counters["Provision_item_with_vec"] = n_item
        counters["Provision_doc_with_vec"] = n_doc

        # 无向量的条文（去表后无独立语义 / 被截断的极端条目）也要有节点
        step("补齐无向量的 Provision 节点（只入图谱）")
        have = set()
        with driver.session() as s:
            for rec in s.run("MATCH (p:Provision) RETURN p.provision_id AS pid"):
                have.add(rec["pid"])
        missing = [m for m in meta_by_id.values() if m["provision_id"] not in have]
        for b in batches(missing, args.batch):
            run(PROV_CYPHER_NOVEC, rows=[dict(x, text=text_by_uid.get(x["uid"], ""))
                                         for x in b])
        counters["Provision_no_vector"] = len(missing)
        counters["Provision"] = count("Provision")

        # 3) Case（4 库 → 4 个属性 / 4 个向量索引）
        step("导 Case（4 库 → c.embedding_<domain4>）")
        # ★ 2026-09-20 修复：向量库 scope="eval" 只排除 dev/test（KB 包含 train/router 案例），
        #   但 facts 挂载此前把四份 split 全部排除 → 62,519 条 train/router 案例
        #   有真向量、facts/question/output 全空（命中后送生成器的是空案例）。
        #   现与 build_embeddings 的 scope="eval" 对齐：只排 dev/test。
        split_uids = set()
        for p in (root + "/data/dev/dev.jsonl", root + "/data/test/test.jsonl"):
            for rec in iter_jsonl(p):
                u = rec.get("uid_g") or rec.get("uid")
                if u:
                    split_uids.add(u)
        case_text = {}
        case_extra = {}
        for p in sorted(glob.glob(root + "/data/corpus/decontaminated/qa/*.jsonl")):
            for rec in iter_jsonl(p):
                if rec.get("task_kind") != "case_analysis":
                    continue
                u = rec.get("uid_g") or rec.get("uid") or ""
                if not u or u in split_uids:
                    continue
                case_text[u] = rec.get("input") or ""
                case_extra[u] = {
                    "ask": (rec.get("ask") or "")[:400],
                    "output": (rec.get("output") or "")[:2000],
                }
        n_case_by_dom = collections.Counter()
        n_case_no_text = 0
        for d in DOMAIN4:
            coll = CASE_COLLECTIONS[d]
            buf = []
            for m, vec in iter_vectors(os.path.join(emb_root, coll)):
                uid = m["uid"]
                if uid not in case_text:
                    n_case_no_text += 1
                ex = case_extra.get(uid, {})
                buf.append({
                    "case_id": uid, "uid": uid,
                    "case_sha1": m.get("case_sha1") or "",
                    "domain": m.get("domain") or "",
                    "domain4": m.get("domain4") or d,
                    "case_lib": coll,
                    "task": m.get("task") or "",
                    "source_dataset": m.get("source_dataset") or "",
                    "char_len": m.get("chars") or 0,
                    "facts": case_text.get(uid, ""),
                    "question": ex.get("ask") or "",
                    "output": ex.get("output") or "",
                    "emb_prop": CASE_EMB_PROP[d],
                    "embedding": vec,
                })
                n_case_by_dom[d] += 1
                if len(buf) >= 500:
                    run(CASE_CYPHER, rows=buf)
                    buf = []
            if buf:
                run(CASE_CYPHER, rows=buf)
        counters["Case"] = sum(n_case_by_dom.values())
        counters["case_rows_by_domain"] = dict(n_case_by_dom)
        counters["case_rows_without_text"] = n_case_no_text
        # ★ 门禁：案例向量全部来自非 dev/test 的 qa 流，理论上 100% 可挂载 facts。
        #   2026-09-19 那次 62,519 条空 facts（43%）就是口径错位且无门禁才静默入库的。
        n_total_case = sum(n_case_by_dom.values())
        if n_total_case and n_case_no_text / n_total_case > 0.005:
            print("[FATAL] case facts 挂载失败率 %.2f%%（%d/%d）> 0.5%% —— "
                  "facts 来源与向量库口径不一致，拒绝入库"
                  % (100.0 * n_case_no_text / n_total_case,
                     n_case_no_text, n_total_case), file=sys.stderr)
            return 2

    # ------------------------------------------------------------- edges
    if args.step in ("all", "edges"):
        # ★ 必须用「按索引逐点匹配 + 分批提交」，不能写
        #   `MATCH (l:Law), (p:Provision) WHERE p.law_id = l.law_id`
        #   —— 那是 1,579 × 72,449 ≈ 1.1 亿次笛卡尔积比较，会跑到天亮。
        step("导 HAS_PROVISION")
        run("""MATCH (p:Provision)
               CALL (p) {
                 MATCH (l:Law {law_id: p.law_id})
                 MERGE (l)-[:HAS_PROVISION]->(p)
               } IN TRANSACTIONS OF 5000 ROWS""")
        counters["HAS_PROVISION"] = count(rel="HAS_PROVISION")

        # ★ 回填 Provision.law_type ← Law.law_type_std
        #   源数据里 pandalla 的 law_type 是**分类串**（「类别 : 商标综合规定」），
        #   只有 twang2218 才是标准类型；不回填这些条文就拿不到 OF_TYPE 边。
        step("回填 Provision.law_type ← Law.law_type_std（补 pandalla 的分类串）")
        run("""MATCH (l:Law)-[:HAS_PROVISION]->(p:Provision)
               WHERE l.law_type IN $types
                 AND (p.law_type IS NULL OR NOT p.law_type IN $types)
               SET p.law_type = l.law_type""", types=LAW_TYPES)

        step("导 IN_DOMAIN / OF_TYPE / FROM_SOURCE")
        run("""MATCH (p:Provision)
               CALL (p) {
                 MATCH (d:Domain {name: p.domain})
                 MERGE (p)-[:IN_DOMAIN]->(d)
               } IN TRANSACTIONS OF 5000 ROWS""")
        run("""MATCH (p:Provision) WHERE p.law_type IN $types
               CALL (p) {
                 MATCH (t:LawType {name: p.law_type})
                 MERGE (p)-[:OF_TYPE]->(t)
               } IN TRANSACTIONS OF 5000 ROWS""", types=LAW_TYPES)
        run("""MATCH (p:Provision)
               CALL (p) {
                 MATCH (sd:SourceDataset {name: p.source_dataset})
                 MERGE (p)-[:FROM_SOURCE]->(sd)
               } IN TRANSACTIONS OF 5000 ROWS""")
        counters["IN_DOMAIN"] = count(rel="IN_DOMAIN")
        counters["OF_TYPE"] = count(rel="OF_TYPE")
        counters["FROM_SOURCE"] = count(rel="FROM_SOURCE")

        for rel, fname in (("NEXT", "edges_next.jsonl"),
                           ("CITES", "edges_cites.jsonl")):
            step("导 %s" % rel)
            path = os.path.join(edges_root, fname)
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            if args.limit:
                rows = rows[:args.limit]
            done = 0
            for b in batches(rows, args.batch):
                run("""UNWIND $rows AS r
                       MATCH (a:Provision {provision_id: r.start})
                       MATCH (b:Provision {provision_id: r.end})
                       MERGE (a)-[:%s]->(b)""" % rel, rows=b)
                done += len(b)
                if done % 20000 == 0:
                    print("      %s %d/%d" % (rel, done, len(rows)), flush=True)
            counters[rel] = count(rel=rel)

        step("导 SAME_CASE（同案多视角，星形连接）")
        run("""MATCH (c:Case) WHERE c.case_sha1 <> ''
               WITH c.case_sha1 AS h, collect(c) AS cs
               WHERE size(cs) > 1
               WITH head(cs) AS hub, tail(cs) AS rest
               UNWIND rest AS x
               MERGE (hub)-[:SAME_CASE]->(x)""")
        counters["SAME_CASE"] = count(rel="SAME_CASE")

    # ----------------------------------------------------------- hotness
    if args.step in ("all", "hotness"):
        step("导热度权重（hotness.json → 节点属性）")
        hot, hmeta = load_hotness(os.path.join(edges_root, "hotness.json"))
        if not hot:
            print("⚠️ 缺 indexes/retrieval/hotness.json —— 先跑 build_hotness.py")
            counters["hotness"] = "MISSING"
        else:
            prov = hot.get("provision") or {}
            rows = [{"pid": pid, "hotness": v.get("hotness", 0.0),
                     "in_cites": v.get("in_cites", 0), "out_cites": v.get("out_cites", 0)}
                    for pid, v in prov.items()]
            if args.limit:
                rows = rows[:args.limit]
            for b in batches(rows, args.batch):
                run("""UNWIND $rows AS r
                       MATCH (p:Provision {provision_id: r.pid})
                       SET p.hotness = r.hotness, p.hotness_cites = r.in_cites,
                           p.out_cites = r.out_cites""", rows=b)
            counters["hotness_provision"] = count(
                "Provision", where="WHERE n.hotness IS NOT NULL")

            cases = hot.get("case") or {}
            crows = [{"uid": uid, "hotness": v.get("hotness", 0.0),
                      "n_same_case": v.get("n_same_case", 0)}
                     for uid, v in cases.items()]
            if args.limit:
                crows = crows[:args.limit]
            for b in batches(crows, args.batch):
                run("""UNWIND $rows AS r
                       MATCH (c:Case {case_id: r.uid})
                       SET c.hotness = r.hotness, c.n_same_case = r.n_same_case""",
                    rows=b)
            counters["hotness_case"] = count(
                "Case", where="WHERE n.hotness IS NOT NULL")
            counters["hotness_formula"] = hmeta.get("formula", {})
            counters["W_HOT_default"] = 0.05

    # ----------------------------------------------------------- indexes
    if args.step in ("all", "indexes"):
        step("建向量索引（数据就绪后建，更快）")
        for name, label, prop, dim in VECTOR_INDEXES:
            run("""CREATE VECTOR INDEX %s IF NOT EXISTS
                   FOR (n:%s) ON (n.%s)
                   OPTIONS {indexConfig: {
                     `vector.dimensions`: %d,
                     `vector.similarity_function`: 'cosine'}}"""
                % (name, label, prop, dim))
        step("等待索引 ONLINE")
        states = {}
        for _ in range(180):
            with driver.session() as s:
                rows = list(s.run("""SHOW INDEXES YIELD name, state, type
                                     WHERE type = 'VECTOR' OR type = 'FULLTEXT'
                                     RETURN name, state"""))
            states = {r["name"]: r["state"] for r in rows}
            pending = [k for k, v in states.items() if v != "ONLINE"]
            if not pending:
                break
            time.sleep(2)
        counters["index_states"] = states

    # ------------------------------------------------------------ verify
    if args.step in ("all", "verify"):
        step("验收计数")
        for lb in ("Law", "Provision", "Case", "Domain", "LawType", "SourceDataset"):
            counters[lb] = count(lb)
        for rl in ("HAS_PROVISION", "NEXT", "CITES", "IN_DOMAIN", "OF_TYPE",
                   "FROM_SOURCE", "SAME_CASE"):
            counters[rl] = count(rel=rl)
        with driver.session() as s:
            n_vec = s.run("MATCH (p:Provision) WHERE p.embedding IS NOT NULL "
                          "RETURN count(p) AS c").single()["c"]
            n_dvec = s.run("MATCH (p:Provision) WHERE p.embedding_doc IS NOT NULL "
                           "RETURN count(p) AS c").single()["c"]
            n_cvec = s.run("MATCH (c:Case) WHERE c.hotness IS NOT NULL "
                           "RETURN count(c) AS c").single()["c"]
            per_dom = {d: s.run("MATCH (c:Case) WHERE c.%s IS NOT NULL "
                                "RETURN count(c) AS c" % CASE_EMB_PROP[d]).single()["c"]
                       for d in DOMAIN4}
            idx = [dict(r) for r in s.run(
                "SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' "
                "RETURN name, state ORDER BY name")]
        counters["provision_embedding_rows"] = n_vec
        counters["provision_doc_embedding_rows"] = n_dvec
        counters["case_embedding_rows"] = sum(per_dom.values())
        counters["case_embedding_rows_by_domain"] = per_dom
        counters["case_hotness_rows"] = n_cvec
        counters["vector_indexes"] = idx

    driver.close()

    checks = {}
    expect = {}
    if args.step in ("all", "verify") and not args.limit:
        expect = build_expectations(edges_root, emb_root)
        exp_flat = {k: v for k, v in expect.items() if not k.startswith("_")}
        checks = {"%s_%d" % (k, v): counters.get(k) == v for k, v in exp_flat.items()}
        checks["vector_indexes_online"] = all(
            x["state"] == "ONLINE" for x in counters.get("vector_indexes", []))
        # ★ 4 库各自的行数必须与磁盘分库一致（不是只看总数）
        by_dom_disk = expect.get("_detail", {}).get("case_rows_by_domain", {})
        by_dom_graph = counters.get("case_embedding_rows_by_domain", {})
        checks["case_rows_by_domain_match"] = bool(by_dom_disk) and all(
            by_dom_graph.get(d) == n for d, n in by_dom_disk.items())

    R = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uri": args.uri, "step": args.step, "limit": args.limit,
        "library_layout": {
            "provision_library": PROVISION_COLLECTION,
            "case_libraries": {d: CASE_COLLECTIONS[d] for d in DOMAIN4},
            "neo4j_case_embedding_prop": CASE_EMB_PROP,
        },
        "counters": counters, "checks": checks,
        "expected_from_inputs": expect,
        "verdict": ("PASS" if all(checks.values()) else "FAIL") if checks else "N/A",
        "elapsed_sec": round(time.time() - t0, 1), "log": log,
    }
    rp = os.path.join(root, args.report_md)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    md = ["# 阶段 4b-4 Neo4j 导入报告\n",
          "> 生成时间：%s　脚本：`scripts/retrieval/import_neo4j.py`　"
          "step=`%s`\n" % (R["generated_at"], args.step),
          "## 0. 库布局（case 4 库 + 法条 1 库）\n",
          "| 磁盘 collection | Neo4j 向量索引 | 属性 |", "|---|---|---|",
          "| `%s` (level=item) | `provision_embedding` | `Provision.embedding` |"
          % PROVISION_COLLECTION,
          "| `%s` (level=doc) | `provision_doc_embedding` | `Provision.embedding_doc` |"
          % PROVISION_COLLECTION]
    for d in DOMAIN4:
        md.append("| `%s` | `%s` | `Case.%s` |"
                  % (CASE_COLLECTIONS[d], CASE_COLLECTIONS[d], CASE_EMB_PROP[d]))
    md += ["", "## 1. 计数\n", "| 项 | 数量 |", "|---|---:|"]
    for k, v in counters.items():
        if isinstance(v, list):
            for x in v:
                md.append("| `%s` | %s |" % (x.get("name"), x.get("state")))
            continue
        if isinstance(v, dict):
            md.append("| `%s` | %s |"
                      % (k, ", ".join("%s=%s" % (a, b) for a, b in v.items())))
            continue
        md.append("| `%s` | %s |" % (k, v))
    md += ["", "## 2. 断言（期望值从输入产物现算，不写死）\n"]
    if expect:
        exp_flat = {k: v for k, v in expect.items() if not k.startswith("_")}
        md += ["| 项 | 期望（输入行数） | 实测（Neo4j） | 通过 |", "|---|---:|---:|:--:|"]
        for k, v in exp_flat.items():
            got = counters.get(k)
            md.append("| `%s` | %d | %s | %s |"
                      % (k, v, got, "✅" if got == v else "❌"))
        md.append("| `vector_indexes_online` | — | — | %s |"
                  % ("✅" if checks.get("vector_indexes_online") else "❌"))
        md.append("| `case_rows_by_domain_match` | — | — | %s |"
                  % ("✅" if checks.get("case_rows_by_domain_match") else "❌"))
        if expect.get("_detail"):
            md += ["", "分库明细：`%s`" % json.dumps(expect["_detail"], ensure_ascii=False)]
        md.append("")
    md += ["**verdict = %s**　耗时 %.1fs" % (R["verdict"], R["elapsed_sec"]), "",
           "## 3. 执行轨迹\n", "```"] + log + ["```", ""]
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("=" * 70)
    for k, v in counters.items():
        print("  %-32s %s" % (k, v))
    bad = [k for k, v in checks.items() if not v]
    if checks:
        print("  断言：%s" % ("全部通过" if not bad else "失败 " + ", ".join(bad)))
        print("  verdict = %s" % R["verdict"])
    print("  耗时 %.1fs" % R["elapsed_sec"])
    print("IMPORT_NEO4J_DONE")
    return 0 if R["verdict"] in ("PASS", "N/A") else 3


PROV_CYPHER = """UNWIND $rows AS r
MERGE (p:Provision {provision_id: r.provision_id})
SET p.uid = r.uid, p.law_id = r.law_id, p.law_name = r.law_id,
    p.level = r.level, p.article_no = r.article_no,
    p.article_label = r.article_label, p.article_seq = r.article_seq,
    p.chapter = r.chapter, p.status = r.status,
    p.effective_from = r.effective_from, p.effective_period = r.effective_period,
    p.publish_date = r.publish_date, p.domain = r.domain, p.domain4 = r.domain4,
    p.law_type = r.law_type,
    p.source_dataset = r.source_dataset, p.text_hash = r.text_hash,
    p.char_len = r.char_len, p.text = r.text, p.embedding = r.embedding"""

PROV_CYPHER_NOVEC = """UNWIND $rows AS r
MERGE (p:Provision {provision_id: r.provision_id})
SET p.uid = r.uid, p.law_id = r.law_id, p.law_name = r.law_id,
    p.level = r.level, p.article_no = r.article_no,
    p.article_label = r.article_label, p.article_seq = r.article_seq,
    p.chapter = r.chapter, p.status = r.status,
    p.effective_from = r.effective_from, p.effective_period = r.effective_period,
    p.publish_date = r.publish_date, p.domain = r.domain,
    p.law_type = r.law_type,
    p.source_dataset = r.source_dataset, p.text_hash = r.text_hash,
    p.char_len = r.char_len, p.text = r.text"""

PROV_DOC_CYPHER = """UNWIND $rows AS r
MERGE (p:Provision {provision_id: r.provision_id})
SET p.uid = r.uid, p.law_id = r.law_id, p.law_name = r.law_id,
    p.level = 'doc', p.article_no = r.article_no,
    p.article_label = r.article_label, p.article_seq = r.article_seq,
    p.chapter = r.chapter, p.status = r.status,
    p.effective_from = r.effective_from, p.effective_period = r.effective_period,
    p.publish_date = r.publish_date, p.domain = r.domain, p.domain4 = r.domain4,
    p.law_type = r.law_type,
    p.source_dataset = r.source_dataset, p.text_hash = r.text_hash,
    p.char_len = r.char_len, p.text = r.text, p.embedding_doc = r.embedding_doc"""

# ★ 4 库 → 4 属性：按 r.emb_prop 分流（Cypher 不能动态属性名，
#   所以用 CASE 分支；这里只有 4 种取值，显式写开更安全）
CASE_CYPHER = """UNWIND $rows AS r
MERGE (c:Case {case_id: r.case_id})
SET c.uid = r.uid, c.case_sha1 = r.case_sha1, c.domain = r.domain,
    c.domain4 = r.domain4, c.case_lib = r.case_lib,
    c.task = r.task, c.source_dataset = r.source_dataset, c.char_len = r.char_len,
    c.facts = r.facts, c.question = r.question, c.output = r.output,
    c.embedding_general = CASE WHEN r.emb_prop = 'embedding_general'
                               THEN r.embedding ELSE c.embedding_general END,
    c.embedding_civil = CASE WHEN r.emb_prop = 'embedding_civil'
                             THEN r.embedding ELSE c.embedding_civil END,
    c.embedding_criminal = CASE WHEN r.emb_prop = 'embedding_criminal'
                                THEN r.embedding ELSE c.embedding_criminal END,
    c.embedding_procedural = CASE WHEN r.emb_prop = 'embedding_procedural'
                                  THEN r.embedding ELSE c.embedding_procedural END"""


if __name__ == "__main__":
    sys.exit(main())
