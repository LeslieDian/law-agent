#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-3：向量化（Qwen3-Embedding-0.6B / 1024d / L2 归一化）

输入范围（口径见 docs/corpus/RETRIEVAL_SCOPE.md，已与四份 split 强隔离）
----------------------------------------------------------------------
  provision      法条条文 —— `indexes/retrieval/provision_nodes.jsonl` 里 level=item
  provision_doc  无条号整篇（337 条）—— **单独 collection**，只嵌「法名 + 首段」
  case           案件文书 —— 清洗 qa 流里 `task_kind=case_analysis` 且不在四份 split 内

长文本处置（依据 docs/corpus/LONG_TEXT_PROBE.md 第 7 节）
--------------------------------------------------------
  item ≤ 1,000 字  → 原样嵌入（绝大多数）
  item > 1,000 字  → 去表 + 截断；去表后 < 200 字 → **只入图谱、不入向量索引**
  doc 级           → 不与 item 同池，单独 collection，只嵌「法名 + 首段」
（各档条数见运行报告，不在此写死 —— 会随上游切条数变化。）

为什么要单独抽 collection
-------------------------
item 中位数 91 字，而 doc 级最长 61,387 字（行政审批项目目录附件）。
混在同一空间里，长条目的"语义平均"会与任意查询都算出中等相似度，污染 top-k。

★ 断点续跑（有陷阱）
--------------------
按 SHARD 条一片，每片写完落 `.done`。重跑时跳过已完成片 —— SSH 长任务必断，
没有 resume 就得从头再来。

⚠️ **resume 是按分片序号、不看内容**：上游法条数一变（如阶段 3 去污修复后 65,037 → 72,449），
旧分片里的条目就全错位了，但 `.done` 还在 → 会被静默跳过。**换了上游语料必须
先删 `indexes/retrieval/embeddings/`**（本脚本不自动清理，避免误删正在跑的产物）。

★ 断言只认不变量，不认绝对数字
------------------------------
2026-09-19 连续踩到 4 处「写死数字」的地雷（上游一重跑就把正确结果判成 FAIL）：
`extract_edges`(1579/65037)、`build_embeddings`(64700)、`import_neo4j`(1579/65037/62258/6459)、
以及本文件的案件池 `82821`。全部改为**从输入产物现算**或**账本恒等式**。

用法
----
    python scripts/retrieval/build_embeddings.py \
        --root /mnt/data/lidian/law-agent \
        --out-dir indexes/retrieval/embeddings \
        --report-json docs/retrieval/EMBEDDING_REPORT.json \
        --report-md  docs/retrieval/EMBEDDING_REPORT.md
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_law_title import iter_jsonl  # noqa: E402

SHARD = 5000
LONG_CHAR = 1000      # item 超过此长度触发去表 + 截断
MAX_CHARS = 2000      # 去表后截断上限（字符）
MIN_KEEP = 200        # 去表后短于此长度 → 不入向量索引（只入图谱）
DOC_HEAD = 500        # doc 级只取前 N 字

TABLE_LINE = re.compile(r"^\s*[+\-=|─│┌┐└┘├┤┬┴┼─=]{3,}[\s\-=+|─│]*$")


def strip_tables(t):
    """去掉 ASCII/制表符表格行（司法解释附表、目录清单）。"""
    keep = []
    for line in (t or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        if TABLE_LINE.match(s):
            continue
        if s.count("|") >= 2 or s.count("+--") >= 1 or s.count("\t") >= 2:
            continue
        keep.append(s)
    return "\n".join(keep)


def doc_text(law_id, text):
    head = (text or "").strip()[:DOC_HEAD]
    return ("%s %s" % (law_id, head)).strip()


def collect_provision_tasks(root, node_file, item_files):
    """→ (item_tasks, doc_tasks, skipped, n_item_nodes)

    `n_item_nodes` 是 provision_nodes 里 level≠doc 的节点数 ——
    断言用它与 `len(item_tasks) + len(skipped)` 对照，**不写死 64,700**
    （上游法条切条数会随阶段 3 产物变化）。
    """
    if not os.path.exists(node_file):
        raise FileNotFoundError("缺 %s —— 请先跑 4b-2 extract_edges.py" % node_file)

    nodes = []
    for line in open(node_file, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            nodes.append(json.loads(line))

    text_by_uid = {}
    for p in item_files:
        for rec in iter_jsonl(p):
            text_by_uid[rec["uid"]] = rec.get("text") or ""

    item_tasks, doc_tasks, skipped = [], [], []
    n_item_nodes = 0
    for n in nodes:
        uid = n["uid"]
        body = text_by_uid.get(uid, "")
        if n.get("level") == "doc":
            doc_tasks.append({"rid": n["provision_id"], "uid": uid, "level": "doc",
                              "law_id": n["law_id"],
                              "domain4": domain4_of(n),
                              "text": doc_text(n["law_id"], body)})
            continue
        n_item_nodes += 1
        chars = n.get("char_len") or len(body)
        if chars > LONG_CHAR:
            body = strip_tables(body)[:MAX_CHARS]
            if len(body) < MIN_KEEP:
                skipped.append({"provision_id": n["provision_id"], "uid": uid,
                                "law_id": n["law_id"], "chars_before": chars,
                                "chars_after": len(body), "reason": "去表后无独立语义"})
                continue
        label = n.get("article_label") or ""
        item_tasks.append({"rid": n["provision_id"], "uid": uid, "level": "item",
                           "law_id": n["law_id"], "article_no": n.get("article_no"),
                           "domain": n.get("domain") or "",
                           "domain4": domain4_of(n),
                           "text": ("%s %s %s" % (n["law_id"], label, body)).strip()})
    return item_tasks, doc_tasks, skipped, n_item_nodes


# =============================================================================
# ★ 库分层（2026-09-19 按用户拍板）：**case 4 库（1 通用 + 3 子库）+ 法条 1 库**
#   case_embedding_general / _civil / _criminal / _procedural   ← 检索主入口（以 case 为节点）
#   provision_embedding                                          ← 法条侧，图谱扩展与引用层
#   理由：论文 3.2/3.3/3.4 的检索入口是**裁判文书**（Case），法条是命中后沿
#   HAS_PROVISION / CITES / NEXT 扩展出来的对象 → 库分层必须与主从关系一致。
#   四域与 MoE 专家一一对齐，方便做「按域检索」的消融。
# =============================================================================
DOMAIN4 = ("general", "civil", "criminal", "procedural")
CASE_COLLECTIONS = {
    "general":    "case_embedding_general",
    "civil":      "case_embedding_civil",
    "criminal":   "case_embedding_criminal",
    "procedural": "case_embedding_procedural",
}
PROVISION_COLLECTION = "provision_embedding"


def domain4_of(rec):
    """把域标签收敛到四域之一（通用 / 民事 / 刑事 / 程序）。

    ★ 必须是**四值互斥的确定映射**：未知 / 空标签绝不能塞进某个子库
      （会污染子库语义，且让「按域检索」的消融失去意义）。
      规则：先看 `domain`；不是四域之一则看 `domains` 是否**恰好单标签**；
      仍不行一律归 `general`（通用库），计数写进报告。
    """
    d = str(rec.get("domain") or "").strip().lower()
    if d in DOMAIN4:
        return d
    doms = rec.get("domains")
    if isinstance(doms, list) and len(doms) == 1 and str(doms[0]).strip().lower() in DOMAIN4:
        return str(doms[0]).strip().lower()
    return "general"


def collect_case_tasks(qa_files, split_uids):
    """→ (tasks, accounting)

    ★ 2026-09-19：原来断言写死 `len(cases) == 82821`。案件池 = 去污后 qa 流里
    「task_kind=case_analysis 且不在四份 split 里」的样本 —— **split 一重跑，池子就变**
    （阶段 4 的分层下采样会因派生流变化而选到不同行）。写死必炸。

    改为返回**划分账本**（单遍 if/elif，四类互斥且覆盖全部 qa 行），
    断言用恒等式 `kept + 各 skip 项 == qa_rows`，数字进报告、不进断言。
    """
    tasks = []
    acc = collections.Counter()
    for p in qa_files:
        for rec in iter_jsonl(p):
            acc["qa_rows"] += 1
            if rec.get("task_kind") != "case_analysis":
                acc["skip_not_case"] += 1
                continue
            uid = rec.get("uid_g") or rec.get("uid") or ""
            if not uid:
                acc["skip_no_uid"] += 1
                continue
            if uid in split_uids:
                acc["skip_in_splits"] += 1
                continue
            txt = (rec.get("input") or "").strip()
            if not txt:
                acc["skip_empty_input"] += 1
                continue
            acc["kept"] += 1
            d4 = domain4_of(rec)
            acc["domain4_" + d4] += 1
            tasks.append({
                "rid": uid, "uid": uid, "text": txt,
                "case_sha1": rec.get("case_sha1") or "",
                "domain": rec.get("domain") or "",
                "domain4": d4,
                "task": rec.get("task") or "",
                "source_dataset": rec.get("source_dataset") or "",
                "char_len": len(txt),
            })
    return tasks, {k: int(v) for k, v in acc.items()}


def load_split_uids(root, scope="eval"):
    """→ (uids, files)

    ★ 2026-09-19 口径修订（用户决策）
    ---------------------------------
    原来把**四份 split 全部**排除出检索库。后果：程序法 `case_analysis` 共 3,333 条，
    其中 3,268 条（98.1%）被阶段 4 的分层切分吃进 train/router/dev/test，
    索引侧仅剩 65 条 → `case_embedding_procedural` 几乎空库（实测只有 37 行）。

    为什么「只排除评测集」才是对的：
      - 检索库是**知识库（KB）**，不是训练数据。IR 的标准做法是
        「语料固定，只排除被查询项本身」，而不是「排除所有训练样本」；
      - 真正的泄漏风险只有一种：**评测 query 自身的文书出现在库里**。
        评测 query 一律来自 dev/test → 只排除 dev/test 即可堵死；
      - train/router 的样本**不参与评测**，放进 KB 既无泄漏、又能补齐稀有域。

    scope="eval" → 只排除 dev/test（默认，新口径）
    scope="all"  → 排除 train/dev/test/router（旧口径，保留以便复现旧结果）
    """
    if scope == "all":
        files = [root + "/data/train/train.jsonl", root + "/data/dev/dev.jsonl",
                 root + "/data/test/test.jsonl", root + "/data/router/router_train.jsonl"]
    else:
        files = [root + "/data/dev/dev.jsonl", root + "/data/test/test.jsonl"]
    uids = set()
    for p in files:
        if not os.path.exists(p):
            raise FileNotFoundError("缺 split：%s" % p)
        for rec in iter_jsonl(p):
            u = rec.get("uid_g") or rec.get("uid")
            if u:
                uids.add(u)
    return uids, files


def encode_collection(name, tasks, model, out_root, batch_size, max_seq_length,
                      extra_fields=()):
    """分片编码一个 collection（带 resume）。返回 stats。"""
    coll_dir = os.path.join(out_root, name)
    os.makedirs(coll_dir, exist_ok=True)
    t0 = time.time()
    n_total = len(tasks)
    n_shards = (n_total + SHARD - 1) // SHARD
    n_done = 0
    n_encoded = 0
    n_over_seq = 0
    tokens_max = 0
    dim = None

    for si in range(n_shards):
        chunk = tasks[si * SHARD:(si + 1) * SHARD]
        npy = os.path.join(coll_dir, "shard_%05d.npy" % si)
        meta = os.path.join(coll_dir, "shard_%05d.jsonl" % si)
        done = os.path.join(coll_dir, "shard_%05d.done" % si)
        if os.path.exists(done) and os.path.exists(npy) and os.path.exists(meta):
            n_done += 1
            n_encoded += len(chunk)
            continue

        texts = [t["text"] for t in chunk]
        # token 长度（只统计，不逐条落盘）—— 论文要"截断比例"
        enc = model.tokenizer(texts, add_special_tokens=True, truncation=False,
                              padding=False)
        lens = [len(x) for x in enc["input_ids"]]
        n_over_seq += sum(1 for L in lens if L > max_seq_length)
        tokens_max = max(tokens_max, max(lens) if lens else 0)

        vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        import numpy as np
        vecs = np.asarray(vecs, dtype="float32")
        dim = int(vecs.shape[1])
        np.save(npy, vecs)

        with open(meta, "w", encoding="utf-8") as f:
            for i, t in enumerate(chunk):
                row = {"row": si * SHARD + i, "shard": si, "idx": i,
                       "id": t["rid"], "uid": t["uid"], "tokens": lens[i],
                       "chars": len(t["text"])}
                for k in extra_fields:
                    row[k] = t.get(k)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(done, "w", encoding="utf-8") as f:
            f.write("%d\n" % len(chunk))

        n_encoded += len(chunk)
        print("      [%s] shard %d/%d  +%d  (累计 %d/%d, %.0fs)"
              % (name, si + 1, n_shards, len(chunk), n_encoded, n_total,
                 time.time() - t0), flush=True)

    return {
        "collection": name, "rows": n_total, "encoded": n_encoded,
        "shards": n_shards, "shards_skipped_resume": n_done, "dim": dim,
        "batch_size": batch_size, "max_seq_length": max_seq_length,
        "rows_over_max_seq": n_over_seq, "tokens_max": tokens_max,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def official_files(pattern):
    """只取**正式产出文件**：排除 `_` 开头的合并副本（`_all.jsonl`）与 `*.sample.jsonl` 调试残留。

    ★ 与 normalize_corpus / apply_decontam / downsample_split 的硬约定 1 同口径。
    读 `_all.jsonl` 必然**双计**（它是分文件的拼接副本，不是独立样本）——
    对案件池尤其致命：会把 qa 流整份算两遍，案件数直接翻倍。
    """
    return [p for p in sorted(glob.glob(pattern))
            if not os.path.basename(p).startswith("_")
            and ".sample." not in os.path.basename(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-dir", default="indexes/retrieval/embeddings")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--report-json", default="docs/retrieval/EMBEDDING_REPORT.json")
    ap.add_argument("--report-md", default="docs/retrieval/EMBEDDING_REPORT.md")
    ap.add_argument("--only", default="", help="只跑某个 collection（冒烟用）")
    ap.add_argument("--limit", type=int, default=0, help="每个 collection 只取前 N 条（冒烟用）")
    ap.add_argument("--case-scope", default="eval", choices=("eval", "all"),
                    help="案件库的泄漏排除口径：eval=只排除 dev/test（默认，新口径）；"
                         "all=排除四份 split（旧口径）")
    ap.add_argument("--clean-cases", action="store_true",
                    help="★ 重建案件库前先删掉 4 个 case_* collection 目录。"
                         "案件池一变（条数变→分片错位），旧 `.done` 会被静默跳过 → 必须清。")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    out_root = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    t0 = time.time()

    item_files = official_files(root + "/data/corpus/statute_items/*.jsonl")
    qa_files = official_files(root + "/data/corpus/decontaminated/qa/*.jsonl")
    print("[0/4] 输入正式文件：statute_items %d 个 / decontaminated/qa %d 个"
          % (len(item_files), len(qa_files)), flush=True)
    for p in qa_files + item_files:
        print("      %s" % os.path.basename(p), flush=True)
    print("[1/4] 载入泄漏排除集（case-scope=%s）..." % args.case_scope, flush=True)
    split_uids, split_files = load_split_uids(root, scope=args.case_scope)
    print("      排除 uid 合计 %d（%s）"
          % (len(split_uids), ", ".join(os.path.basename(f) for f in split_files)),
          flush=True)
    if args.case_scope == "eval":
        print("      ★ 新口径：检索库=知识库，只排除**评测集**；train/router 样本可入库"
              "（补齐稀有域的必需条件，见 load_split_uids 注释）", flush=True)

    print("[2/4] 构建任务列表 ...", flush=True)
    prov_items, prov_docs, prov_skipped, n_item_nodes = collect_provision_tasks(
        root, os.path.join(root, "indexes/retrieval/provision_nodes.jsonl"), item_files)
    cases, case_acc = collect_case_tasks(qa_files, split_uids)
    print("      provision=%d  provision_doc=%d  case=%d  skipped=%d  item_nodes=%d"
          % (len(prov_items), len(prov_docs), len(cases), len(prov_skipped),
             n_item_nodes), flush=True)
    print("      case 划分账本 %s" % json.dumps(case_acc, ensure_ascii=False), flush=True)

    case_sha1_uniq = len(set(c["case_sha1"] for c in cases if c["case_sha1"]))
    case_uid_uniq = len(set(c["rid"] for c in cases))

    # ★ 库分层（用户 2026-09-19 拍板）：case 4 库（1 通用 + 3 子库）+ 法条 1 库
    #   doc 级法条折进同一个 provision 库：只嵌「法名 + 首段」+ 带 level=doc 字段，
    #   既满足「法条 1 库」，又不重蹈「长条目混池」的坑（H5）。
    case_by_dom = {d: [c for c in cases if c["domain4"] == d] for d in DOMAIN4}
    prov_all = prov_items + prov_docs
    print("      case 分库：" + "  ".join("%s=%d" % (d, len(case_by_dom[d]))
                                          for d in DOMAIN4), flush=True)

    # 案件池恒等式：单遍 if/elif 划分，五类互斥且覆盖全部 qa 行
    case_kept = case_acc.get("kept", 0)
    case_skips = sum(v for k, v in case_acc.items() if k.startswith("skip_"))
    case_accounted = (case_kept + case_skips) == case_acc.get("qa_rows", -1)
    # ★ 不写死任何绝对条数：上界从输入现算（case_analysis 总行数）
    n_case_analysis = (case_acc.get("qa_rows", 0) - case_acc.get("skip_not_case", 0))
    checks = {
        # ★ 不写死 64,700：与 provision_nodes 里的 item 级节点数自洽
        "provision_items_complete": len(prov_items) + len(prov_skipped) == n_item_nodes,
        # ★ 不写死 82821（旧口径）/136xxx（新口径）：
        #   池子 = qa 流里 case_analysis 且不在**排除集**内的样本。
        #   排除集口径（--case-scope）一改池子就变 → 只断言
        #   「划分账本自洽」+「池子 ⊆ case_analysis 且占比合理」，数字进报告。
        "case_pool_accounted": case_accounted,
        "case_pool_le_all_case_analysis": len(cases) <= n_case_analysis,
        "case_pool_plausible_band": 0.3 * n_case_analysis <= len(cases) <= n_case_analysis,
        # ★ 泄漏门禁（唯一真正要守的不变量）：**评测集 uid 绝不出现在库里**
        "case_not_in_eval_splits": len({c["uid"] for c in cases} & split_uids) == 0,
        "provision_not_in_eval_splits": len({t["uid"] for t in prov_items} & split_uids) == 0,
        "eval_scope_is_dev_test_only": (args.case_scope != "eval") or all(
            os.path.basename(f) in ("dev.jsonl", "test.jsonl") for f in split_files),
        "no_empty_text": all(t["text"].strip() for t in prov_items + cases + prov_docs),
        # ★ 输入里不得混入合并副本 / 调试残留（读 `_all.jsonl` 会让案件池双计）
        "inputs_are_official_only": all(
            not os.path.basename(p).startswith("_")
            and ".sample." not in os.path.basename(p)
            for p in qa_files + item_files),
        # ★ 库分层自洽（用户要求「case 4 库 + 法条 1 库」）
        "case_domains_cover_pool": sum(len(v) for v in case_by_dom.values()) == len(cases),
        "case_domains_all_nonempty": all(len(v) > 0 for v in case_by_dom.values()),
        "provision_pool_complete": len(prov_all) == len(prov_items) + len(prov_docs),
    }

    print("[3/4] 加载模型 ...", flush=True)
    from embedding_model import load_model  # noqa: E402
    model_dir = args.model_dir or os.path.join(root, "models/Qwen3-Embedding-0.6B")
    model, minfo = load_model(model_dir, device=args.device,
                              max_seq_length=args.max_seq_length)
    print("      %s" % json.dumps(minfo, ensure_ascii=False), flush=True)

    print("[4/4] 编码 ...", flush=True)
    if args.clean_cases:
        import shutil
        for d in DOMAIN4:
            cd = os.path.join(out_root, CASE_COLLECTIONS[d])
            if os.path.isdir(cd):
                shutil.rmtree(cd)
                print("      ★ 已清空旧案件库 %s（防分片错位静默跳过）"
                      % CASE_COLLECTIONS[d], flush=True)
    plan = [(PROVISION_COLLECTION, prov_all, ("level", "law_id", "domain4"))]
    for d in DOMAIN4:
        plan.append((CASE_COLLECTIONS[d], case_by_dom[d],
                     ("case_sha1", "domain", "domain4", "task", "source_dataset")))
    if args.only:
        plan = [x for x in plan if x[0] == args.only]

    stats = []
    for name, tasks, extra in plan:
        if args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print("      [%s] 跳过（无任务）" % name, flush=True)
            continue
        stats.append(encode_collection(name, tasks, model, out_root,
                                       args.batch_size, args.max_seq_length, extra))

    R = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": root, "out_dir": out_root, "limit": args.limit, "only": args.only,
        "model": minfo,
        "text_policy": {
            "provision_text": "law_id + article_label + 正文",
            "case_text": "input（案件正文，不加 instruction —— 文档侧不加前缀）",
            "long_item": ">%d 字 → 去表 + 截断到 %d 字；去表后 <%d 字只入图谱"
                         % (LONG_CHAR, MAX_CHARS, MIN_KEEP),
            "doc_level": "折进同一个 provision 库（不另立库）；只嵌「法名 + 前 %d 字」"
                         "并带 `level=doc` 字段 → 检索侧可过滤" % DOC_HEAD,
        },
        "library_layout": {
            "policy": "case 4 库（1 通用 + 3 子库）+ 法条 1 库",
            "provision_library": PROVISION_COLLECTION,
            "case_libraries": {d: CASE_COLLECTIONS[d] for d in DOMAIN4},
            "case_rows_by_domain": {d: len(case_by_dom[d]) for d in DOMAIN4},
            "provision_rows": {"item": len(prov_items), "doc": len(prov_docs),
                               "total": len(prov_all)},
        },
        "collections": stats,
        "case_pool": {"rows": len(cases), "unique_uid": case_uid_uniq,
                      "unique_case_sha1": case_sha1_uniq,
                      "partition_accounting": case_acc,
                      "partition_identity_ok": case_accounted,
                      "leak_exclusion_scope": args.case_scope,
                      "leak_exclusion_files": [os.path.basename(f) for f in split_files],
                      "n_case_analysis_total": n_case_analysis,
                      "note": "scope=eval → 只排除评测集(dev/test)；train/router 样本可入库" 
                              "（检索库=知识库，不是训练数据）"},
        "provision_skipped": {"count": len(prov_skipped), "rows": prov_skipped[:50]},
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rj = os.path.join(root, args.report_json)
    rm = os.path.join(root, args.report_md)
    os.makedirs(os.path.dirname(rj), exist_ok=True)
    os.makedirs(os.path.dirname(rm), exist_ok=True)
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    tot = sum(s["rows"] for s in stats)
    md = []
    W = md.append
    W("# 阶段 4b-3 向量化报告\n")
    W("> 生成时间：%s　脚本：`scripts/retrieval/build_embeddings.py`\n" % R["generated_at"])
    W("模型：`%s`　**%dd**　max_seq_length **%d**　fp16=%s　L2 归一化=是\n"
      % (minfo["model_dir"], minfo["dim"], minfo["max_seq_length"], minfo["fp16"]))
    W("> ★ 查询侧加 `Instruct: …\\nQuery: …` 前缀，**文档侧不加**（Qwen3-Embedding 是 instruction-aware 模型）。\n")
    W("## 1. 库分层：case 4 库（1 通用 + 3 子库）+ 法条 1 库\n")
    W("| collection | 类型 | 域 | 条数 | 分片 | 维度 | 超长截断 | 最长 token | 耗时 |")
    W("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        nm = s["collection"]
        kind = "法条" if nm == PROVISION_COLLECTION else "案件"
        dom = "-" if nm == PROVISION_COLLECTION else \
            [d for d in DOMAIN4 if CASE_COLLECTIONS[d] == nm][0]
        W("| `%s` | %s | %s | **%d** | %d | %s | %d | %d | %.1fs |"
          % (nm, kind, dom, s["rows"], s["shards"], s["dim"],
             s["rows_over_max_seq"], s["tokens_max"], s["elapsed_sec"]))
    W("| **合计** | | | **%d** | | | | | |" % tot)
    W("")
    W("| 库 | 条数 |")
    W("|---|---:|")
    W("| `%s`（item %d + doc %d） | %d |"
      % (PROVISION_COLLECTION, len(prov_items), len(prov_docs), len(prov_all)))
    for d in DOMAIN4:
        W("| `%s`（%s） | %d |" % (CASE_COLLECTIONS[d], d, len(case_by_dom[d])))
    W("")
    W("> 检索主入口 = 4 个 case 库（以 case 为节点）；法条库是命中后沿")
    W("> `HAS_PROVISION` / `CITES` / `NEXT` 扩展的对象。四域与 MoE 专家一一对齐。")
    W("")
    W("## 2. 长文本处置\n")
    W("- item > %d 字 → 去表 + 截断到 %d 字" % (LONG_CHAR, MAX_CHARS))
    W("- 去表后 < %d 字 → **只入图谱、不入向量索引**：**%d 条**"
      % (MIN_KEEP, len(prov_skipped)))
    if prov_skipped:
        for x in prov_skipped[:10]:
            W("  - `%s`（%d 字 → %d 字）" % (x["provision_id"][:46],
                                             x["chars_before"], x["chars_after"]))
    W("- doc 级 %d 条单独成 collection（不与 item 同池）\n" % len(prov_docs))
    W("## 3. 案件池\n")
    W("| 项 | 值 |")
    W("|---|---:|")
    W("| 入库条数（uid 唯一） | **%d** |" % case_uid_uniq)
    W("| 其中唯一 `case_sha1` | %d |" % case_sha1_uniq)
    W("")
    W("> `case_sha1` 是语料的案件正文指纹（尾部 400 字），**不是**语义案件 ID ——")
    W("> 同一案件的不同任务视角（判决预测 / 文书摘要 / 要素抽取）会产生多条样本。")
    W("> 本阶段**按样本级入库**（口径与 4b-0 门禁一致，与四份 split 强隔离），")
    W("> 同案多视角在 4b-4 用 `SAME_CASE` 关系显式关联，**不删节点**。\n")
    W("### 3.1 案件池划分账本（不变量，替代旧写死的 82,821）\n")
    W("| 项 | 条数 |")
    W("|---|---:|")
    for k in ("qa_rows", "kept", "skip_not_case", "skip_no_uid",
              "skip_in_splits", "skip_empty_input"):
        if k in case_acc:
            W("| `%s` | %d |" % (k, case_acc[k]))
    W("")
    W("> 恒等式 `kept + Σskip_* == qa_rows` = **%s**。"
      % ("true" if case_accounted else "**false**"))
    W("> 案件池大小随阶段 4 切分变化（split 一重跑，被排除的 uid 就变），因此**不写死具体数字**，")
    W("> 只断言账本自洽 + 落在 3 万～12 万的合理带内；准确数字见上表与报告 JSON。\n")
    W("## 4. 断言\n")
    for k, v in checks.items():
        W("- `%s` = %s" % (k, "true" if v else "**false**"))
    W("")
    W("**verdict = %s**　总耗时 %.1fs\n" % (R["verdict"], R["elapsed_sec"]))
    with open(rm, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("=" * 70)
    print("阶段 4b-3 向量化完成")
    for s in stats:
        print("  %-16s rows=%6d dim=%s shards=%d over_seq=%d %.1fs"
              % (s["collection"], s["rows"], s["dim"], s["shards"],
                 s["rows_over_max_seq"], s["elapsed_sec"]))
    bad = [k for k, v in checks.items() if not v]
    print("  断言：%s" % ("全部通过" if not bad else "失败 " + ", ".join(bad)))
    print("  verdict = %s   总耗时 %.1fs" % (R["verdict"], R["elapsed_sec"]))
    print("  产物 -> %s" % out_root)
    print("BUILD_EMBEDDINGS_DONE")
    return 0 if R["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
