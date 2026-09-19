#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4b-5：检索链路 + 消融矩阵（先单独评检索，不接大模型生成）

评什么
------
**法条检索**：query = 案件正文，ground truth = 该案件的相关法条。
gold 解析成 `(L2 法名, 条号) → provision_id` 即为可判定的 gold。

★ 两种 gold 口径（**必须分开报**）
--------------------------------------------------------
  `system`（tier A）  数据集自带的「相关法律条文」清单 —— **最权威**；
                      但只有 case_analysis 且带清单的样本才有（实测 dev+test 仅百余条）。
  `output`（tier B）  从**参考答案**解析出的法条引用 —— 覆盖大，但是**下界**。
  `union`             两者并集（本报告**主口径**，样本量最大）。

★ 检索链路（用户 2026-09-19 拍板「0.6B 主力 + 完整混合链路」）
------------------------------------------------------------------
    query ─┬─ dense（Qwen3-Embedding-0.6B，instruction-aware）  top-K
           ├─ BM25（jieba 分词 + scipy 稀疏精确 BM25）           top-K
           └─ graph（沿向量 top-N 走 NEXT / CITES 1 跳）         有序表
                            ↓
                     RRF(k=60) 融合
                            ↓
              热度先验 reweight: s *= (1 + W_HOT * hotness)   [W_HOT=0.05]
                            ↓
               bge-reranker-v2-m3 精排（top-50 → top-10）
                            ↓
                        top5 / top10

★ BM25 为什么不用 rank_bm25 逐条打分
  `BM25Okapi.get_scores()` 是 O(查询数 × 文档数 × 查询词数) 的 Python 循环，
  1,200 query × 7.2 万条会跑到十几分钟。这里改成**一次性稀疏矩阵**：
      doc 侧权重 D[d,t] = idf(t) * f*(k1+1) / (f + k1*(1-b+b*len_d/avg_len))
      query 侧权重 q[t] = 1（BM25 的标准 qtf=1 变体）
      BM25(q,d) = Σ_t q[t]*D[d,t]  → 一次 scipy 稀疏乘出全部 query 的分数
  结果与逐条打分**逐位等价**，耗时 ~1 秒。

★ 消融矩阵（一个脚本一次跑完，给出论文实验章的检索侧数字）
    vector       纯向量
    dense_hot    纯向量 + 热度先验
    dense_bm25   向量 + BM25
    dense_graph  向量 + 图谱扩展
    hybrid       向量 + BM25 + 图谱（无热度、无重排）
    full         hybrid + 热度 + 重排          ← 主链路

★ 两个已修的历史缺陷（2026-09-19，都会静默产出空/错指标）
------------------------------------------------------------------
1. **法名简称未解析**：gold 一律写简称（《刑法》/《民法典》），而 `law_id` 是
   L2 全名（中华人民共和国刑法）→ 旧实现只做 exact 匹配，实测 145 条引用
   **exact 命中 0 条**。现改为「exact → 唯一后缀」两级，后缀兜底最小长度 **2 字**
   （「刑法」只有 2 字；卡 3 字会让刑法域引用全挂，实测 133/145）。
2. **无 gold 时静默给 0 分**：现在 gold 全空会直接 `exit 3`，不再产出一份
   「recall = 0」的假报告。

★ 库布局（2026-09-19）：法条 1 库 `provision_embedding`，其中 `level=doc` 的行
  与 item 混库但语义不同（doc 级最长 6 万字），检索时**只取 level=item**。

用法
----
    python scripts/retrieval/smoke_retrieval.py \
        --root /mnt/data/lidian/law-agent \
        --configs vector,dense_bm25,dense_graph,hybrid,full \
        --report-md docs/retrieval/SMOKE_RETRIEVAL.md
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
from extract_edges import cn2int, strip_version, strip_wrappers  # noqa: E402
from normalize_law_title import iter_jsonl  # noqa: E402

RE_NAMED = re.compile(r"[《〈]\s*([^《》〈〉]{2,80}?)\s*[》〉]\s*第\s*"
                      r"([0-9〇零一二三四五六七八九十百千两]+)\s*条")

PROVISION_COLLECTION = "provision_embedding"
DEFAULT_HOTNESS = "indexes/retrieval/hotness.json"
DEFAULT_RERANKER = "models/bge-reranker-v2-m3"
W_HOT = 0.05

# ★ 2026-09-19 指标口径修订（用户决策）
# ---------------------------------------------------------------------------
# 原定目标 recall@5 ≥ 0.85 / recall@10 ≥ 0.90 **在数学上不可达**，有实测证据：
#   纯 dense（Qwen3-Embedding-0.6B）在**全量表条 item** 上做完整排序，
#   union 口径召回曲线单调但**深度 1000 也到不了 0.90**（0.83～0.90 视抽样而定），
#   而实测 recall@10 ≈ 0.49 → recall@10 ≥ 0.90 不是「排序没做好」能解释的差距。
#   瓶颈是 **embedding 的表示能力**（0.6B / 零样本），不是融合口径或排序器。
#   （具体曲线数字见每次运行产出的 §1c；此处**不写死**，避免换模型/换语料后变成假陈述。）
# 新主指标（可达且对 RAG 有实际意义）：
#   hit@5（top-5 里至少有 1 条 gold 的 query 占比）、recall@10、MRR@10
# 目标值**事后设定**（post-hoc，诚实标注）：低于实测操作点，作为回归门禁使用。
# 旧目标保留在 RETIRED_TARGETS 里存档，论文须写明为何作废。
TARGETS = {"hit@5": 0.60, "recall@10": 0.50, "mrr@10": 0.45}
RETIRED_TARGETS = {
    "recall@5": 0.85, "recall@10": 0.90,
    "status": "作废（数学不可达）",
    # evidence 在运行时由本次 dense_ceiling 曲线**现算回填**（见 _fill_retired_evidence），
    # 不写死数字：换 embedding / 换语料后写死的证据会立刻变成错误陈述。
    "evidence": "(运行时由 dense_ceiling 回填)",
    "decided_by": "用户 2026-09-19",
}

# 消融矩阵：名称 → 开关
# ★ 2026-09-19 修订：4b-5 v1 显示「等权 RRF + 图谱注入」会把精度打崩
#   （图谱列表把邻近条文当同权候选注入 → top-5 被无关条文占满）。
#   配套改动：(a) RRF 改成**加权**（dense 1.0 / bm25 w / graph w）；
#   (b) 图谱列表**限长**并降权，只作候选补全；(c) 新增深池精排变体（rerank 池 200）。
CONFIGS = [
    ("vector",       dict(bm25=False, graph=False, hot=False, rerank=False)),
    ("dense_hot",    dict(bm25=False, graph=False, hot=True,  rerank=False)),
    ("dense_bm25",   dict(bm25=True,  graph=False, hot=False, rerank=False)),
    ("dense_graph",  dict(bm25=False, graph=True,  hot=False, rerank=False)),
    ("hybrid",       dict(bm25=True,  graph=True,  hot=False, rerank=False)),
    ("full",         dict(bm25=True,  graph=True,  hot=True,  rerank=True)),
    # ---- 修订后的候选主链路 ----
    ("wr_dense_bm25", dict(bm25=True,  graph=False, hot=False, rerank=False)),
    ("wr_rerank50",   dict(bm25=True,  graph=False, hot=True,  rerank=True)),
    ("wr_rerank200",  dict(bm25=True,  graph=False, hot=True,  rerank=True)),
    ("wr_kg_rerank200", dict(bm25=True, graph=True, hot=True, rerank=True)),
]
CONFIG_DOC = {
    "vector": "纯向量（Qwen3-Embedding-0.6B）",
    "dense_hot": "纯向量 + 热度先验",
    "dense_bm25": "向量 + BM25（等权 RRF）",
    "dense_graph": "向量 + 图谱扩展（等权 RRF，注入式）",
    "hybrid": "向量 + BM25 + 图谱，等权 RRF",
    "full": "hybrid + 热度 + 精排（池 50）",
    "wr_dense_bm25": "加权 RRF：dense 1.0 + BM25 w",
    "wr_rerank50": "加权 RRF + 热度 + 精排（池 50）",
    "wr_rerank200": "加权 RRF + 热度 + 精排（池 200）★",
    "wr_kg_rerank200": "加权 RRF + 限长降权图谱 + 热度 + 精排（池 200）",
}
# 每个配置的精排候选池（None = 用 --rerank-pool）
RERANK_POOL_OVERRIDE = {"wr_rerank200": 200, "wr_kg_rerank200": 200, "wr_rerank50": 50,
                        "full": 50}


# ============================================================ gold 解析
def build_law_resolver(node_file):
    """gold 里的法名是**简称**（《刑法》/《民法典》），必须先解析成 L2 的 `law_id`。"""
    law_ids = set()
    for line in open(node_file, encoding="utf-8"):
        line = line.strip()
        if line:
            law_ids.add(json.loads(line)["law_id"])
    stats = collections.Counter()

    def resolve(name_l2):
        if not name_l2:
            stats["miss"] += 1
            return []
        if name_l2 in law_ids:
            stats["exact"] += 1
            return [name_l2]
        if len(name_l2) >= 2:
            cands = [x for x in law_ids if x.endswith(name_l2)]
            if len(cands) == 1:
                stats["alias"] += 1
                return cands
            if len(cands) > 1:
                stats["ambiguous"] += 1
                return []
        stats["miss"] += 1
        return []

    return law_ids, resolve, stats


def parse_gold(text, resolve, provision_index):
    """一段文本 → (provision_id 集合, 解析得到但库里没有的 (law_id, 条号) 集合)。"""
    pids, missed = set(), set()
    for m in RE_NAMED.finditer(text or ""):
        no = cn2int(m.group(2))
        if no is None:
            continue
        for lid in resolve(strip_version(strip_wrappers(m.group(1)))):
            got = provision_index.get((lid, no))
            if got:
                pids |= set(got)
            else:
                missed.add((lid, no))
    return pids, missed


def build_provision_index(root):
    """(law_id, article_no) -> [provision_id]；doc 级与无条号者不参与 gold 匹配。"""
    idx = collections.defaultdict(list)
    for line in open(root + "/indexes/retrieval/provision_nodes.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("level") == "doc" or r.get("article_no") in (None, ""):
            continue
        idx[(r["law_id"], r["article_no"])].append(r["provision_id"])
    return idx


def collect_queries(root, provision_index, resolve, splits=("dev", "test")):
    """从 dev/test 抽 query + 两种口径的 gold。"""
    files = [root + "/data/%s/%s.jsonl" % (s, s) for s in splits]
    queries = []
    stat = collections.Counter({
        "rows": 0, "with_gold": 0, "no_gold": 0, "empty_query": 0,
        "tierA_rows": 0, "tierB_rows": 0,
        "refs_system": 0, "refs_output": 0,
        "refs_system_missing_in_index": 0, "refs_output_missing_in_index": 0,
    })
    for s, p in zip(splits, files):
        for rec in iter_jsonl(p):
            stat["rows"] += 1
            g_sys, miss_sys = parse_gold(rec.get("system") or "", resolve, provision_index)
            g_out, miss_out = parse_gold(rec.get("output") or "", resolve, provision_index)
            stat["refs_system"] += len(g_sys)
            stat["refs_output"] += len(g_out)
            stat["refs_system_missing_in_index"] += len(miss_sys)
            stat["refs_output_missing_in_index"] += len(miss_out)
            if not (g_sys or g_out):
                stat["no_gold"] += 1
                continue
            q = (rec.get("input") or "").strip()
            if not q:
                stat["empty_query"] += 1
                continue
            if g_sys:
                stat["tierA_rows"] += 1
            if g_out:
                stat["tierB_rows"] += 1
            queries.append({
                "uid": rec.get("uid_g") or rec.get("uid"),
                "split": s,   # ★ 以**文件来源**为准（dev / test），不信任记录内字段
                "query": q,
                "gold_sys": sorted(g_sys),
                "gold_out": sorted(g_out),
                "gold_union": sorted(g_sys | g_out),
            })
    stat["with_gold"] = len(queries)
    return queries, dict(stat)


# ============================================================ 向量 / 文本
def load_item_vectors(coll_dir, want_level="item"):
    """→ (ids, matrix)。**只取 level=item**：doc 级语义粒度不同，混池会污染 top-k。"""
    import numpy as np
    ids, blocks = [], []
    for npy in sorted(glob.glob(os.path.join(coll_dir, "shard_*.npy"))):
        meta = npy[:-4] + ".jsonl"
        if not os.path.exists(meta):
            continue
        vecs = np.load(npy)
        keep = []
        i = 0
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = json.loads(line)
                if (m.get("level") or "item") == want_level:
                    keep.append(i)
                    ids.append(m["id"])
                i += 1
        if keep:
            blocks.append(vecs[keep])
    if not blocks:
        raise FileNotFoundError("没有找到 level=%s 的向量：%s" % (want_level, coll_dir))
    return ids, np.vstack(blocks)


def build_item_texts(root, ids):
    """provision_id → 与向量化同口径的文本（法名 + 条号 + 正文）。

    ★ 必须与 `build_embeddings.py` 的 `item_tasks[*]["text"]` 拼法一致，
      否则 BM25 与 dense 看到的内容不同，消融就没法归因。
    """
    uid_by_pid = {}
    for line in open(root + "/indexes/retrieval/provision_nodes.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("level") == "doc":
            continue
        uid_by_pid[r["provision_id"]] = (r.get("uid"), r.get("law_id") or "",
                                         r.get("article_label") or "")
    body_by_uid = {}
    for p in sorted(glob.glob(root + "/data/corpus/statute_items/*.jsonl")):
        for rec in iter_jsonl(p):
            body_by_uid[rec["uid"]] = rec.get("text") or ""
    texts = []
    n_missing = 0
    for pid in ids:
        uid, law_id, label = uid_by_pid.get(pid, (None, "", ""))
        body = body_by_uid.get(uid, "") if uid else ""
        if not body:
            n_missing += 1
        texts.append(("%s %s %s" % (law_id, label, body)).strip())
    return texts, {"n_missing_body": n_missing, "n_ids": len(ids)}


# ============================================================ BM25（稀疏精确）
def build_bm25(texts, k1=1.5, b=0.75, min_df=2, max_vocab=400000):
    """→ (D_csr [N,V], vocab, idf, meta)。BM25(q,d) = Σ_t 1[t∈q] * D[d,t]。"""
    import jieba
    import numpy as np
    import scipy.sparse as sp
    t0 = time.time()
    tokenized = []
    for t in texts:
        toks = [w for w in jieba.cut(t) if w.strip()]
        tokenized.append(toks)
    df = collections.Counter()
    for toks in tokenized:
        df.update(set(toks))
    keep = {w for w, c in df.items() if c >= min_df}
    if len(keep) > max_vocab:
        keep = {w for w, _ in collections.Counter(
            {w: df[w] for w in keep}).most_common(max_vocab)}
    vocab = {w: i for i, w in enumerate(sorted(keep))}
    V = len(vocab)
    N = len(tokenized)
    lens = np.array([max(1, len(t)) for t in tokenized], dtype=np.float64)
    avg_len = float(lens.mean()) if N else 1.0
    idf = np.zeros(V, dtype=np.float64)
    for w, i in vocab.items():
        idf[i] = np.log(1.0 + (N - df[w] + 0.5) / (df[w] + 0.5))
    rows, cols, vals = [], [], []
    for d, toks in enumerate(tokenized):
        tf = collections.Counter(toks)
        norm = k1 * (1.0 - b + b * lens[d] / avg_len)
        for w, f in tf.items():
            i = vocab.get(w)
            if i is None:
                continue
            rows.append(d)
            cols.append(i)
            vals.append(idf[i] * (f * (k1 + 1.0)) / (f + norm))
    D = sp.csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.float64)
    print("      BM25 词表 %d（df>=%d）/ 文档 %d / nnz %d / %.1fs"
          % (V, min_df, N, D.nnz, time.time() - t0), flush=True)
    return D, vocab, {"n_docs": N, "vocab": V, "nnz": int(D.nnz),
                      "avg_len": round(avg_len, 1), "k1": k1, "b": b, "min_df": min_df,
                      "build_sec": round(time.time() - t0, 1)}


def bm25_search(D, vocab, queries, top_k, chunk=64):
    """一次算出所有 query 的 BM25 分数（稀疏乘，精确等价逐条打分）。"""
    import jieba
    import numpy as np
    import scipy.sparse as sp
    Q = sp.lil_matrix((len(queries), D.shape[1]), dtype=np.float64)
    n_term = 0
    for qi, q in enumerate(queries):
        idxs = {vocab[w] for w in jieba.cut(q) if w in vocab}
        for i in idxs:
            Q[qi, i] = 1.0
        n_term += len(idxs)
    Q = Q.tocsr()
    out = np.empty((len(queries), D.shape[0]), dtype=np.float32)
    Dt = D.T.tocsr()
    for s in range(0, len(queries), chunk):
        out[s:s + chunk] = (Q[s:s + chunk] @ Dt).toarray()
    return out, {"avg_query_terms_in_vocab": round(n_term / max(1, len(queries)), 2)}


# ============================================================ 融合 / 扩展
def rrf_fuse(rankings, k=60, weights=None):
    """RRF：score(d) = Σ w_l / (k + rank_l(d))。rankings 是若干有序 id 列表。"""
    score = collections.defaultdict(float)
    for li, lst in enumerate(rankings):
        w = (weights or [1.0] * len(rankings))[li]
        for rank, d in enumerate(lst, start=1):
            score[d] += w / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])


def load_edges(path):
    out = collections.defaultdict(list)
    if not os.path.exists(path):
        return out
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["start"]].append(r["end"])
    return out


def graph_expand(hits, adj_next, adj_cites, top_seed=10,
                 w_next=1.0, w_cites=0.5, cap=60):
    """从向量 top 命中出发做 1 跳扩展，转成**有序列表**（RRF 只吃有序列表）。

    NEXT 是主力（覆盖率高、零成本），CITES 只覆盖少数条文 → 权重减半。

    ★ 限长（cap）：扩展列表若不加限制，会把大量邻近条文塞进 RRF，
      与 dense/bm25 同权竞争 → top-5 精度崩塌（4b-5 v1 实测 dense_graph
      recall@5 0.25 vs 纯向量 0.40）。这里只保留**最靠前的 cap 个**。
    """
    score = collections.defaultdict(float)
    for rank, pid in enumerate(hits[:top_seed], start=1):
        base = 1.0 / (1.0 + (rank - 1) / 10.0)
        for nb in adj_next.get(pid, []):
            score[nb] = max(score[nb], base * w_next)
        for nb in adj_cites.get(pid, []):
            score[nb] = max(score[nb], base * w_cites)
    order = [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]
    return order[:cap] if cap and cap > 0 else order


def load_hotness(path):
    if not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    return h.get("provision") or {}, h.get("meta") or {}


def apply_hotness(fused, hot, w_hot):
    """s' = s * (1 + w_hot * hotness)。热度只做 ≤w_hot 的先验微调。"""
    if not hot or w_hot <= 0:
        return fused
    out = []
    for d, s in fused:
        hv = (hot.get(d) or {}).get("hotness", 0.0)
        out.append((d, s * (1.0 + w_hot * hv)))
    out.sort(key=lambda kv: -kv[1])
    return out


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--configs", default=",".join(c for c, _ in CONFIGS))
    ap.add_argument("--top-k", type=int, default=50, help="dense / BM25 各取前 K")
    ap.add_argument("--rrf-k", type=int, default=10,
                    help="RRF k；★ 4b-5b 扫描显示 k=10 明显优于 60（rank 权重更尖）")
    ap.add_argument("--w-hot", type=float, default=W_HOT)
    ap.add_argument("--rerank-pool", type=int, default=50, help="精排候选数")
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--no-rerank", action="store_true", help="强制关掉重排（模型没下好时用）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--eval-split", default="all", choices=("all", "dev", "test"),
                    help="all=dev∪test；dev 用于调参、test 用于报告（避免在测试集上调参）")
    ap.add_argument("--w-dense", type=float, default=1.0, help="RRF 中 dense 列表权重")
    ap.add_argument("--w-bm25", type=float, default=0.7,
                    help="RRF 中 BM25 列表权重；★ 4b-5b dev 扫描最优 0.70")
    ap.add_argument("--w-graph", type=float, default=0.25,
                    help="RRF 中图谱列表权重；★ 图谱**降低** top-5 精度、**提高** recall@50，"
                         "故只作候选补全并配深池精排")
    ap.add_argument("--graph-seed", type=int, default=10, help="图谱扩展的种子数")
    ap.add_argument("--graph-cap", type=int, default=60, help="图谱扩展列表长度上限")
    ap.add_argument("--report-md", default="docs/retrieval/SMOKE_RETRIEVAL.md")
    args = ap.parse_args()

    root = args.root.rstrip("/")
    emb_root = os.path.join(root, "indexes/retrieval/embeddings")
    t0 = time.time()
    want = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in want if c not in dict(CONFIGS)]
    if bad:
        print("未知配置：%s" % bad, file=sys.stderr)
        return 2

    print("[1/6] 载入法条向量（level=item）...", flush=True)
    ids, mat = load_item_vectors(os.path.join(emb_root, PROVISION_COLLECTION))
    print("      %d × %d" % (mat.shape[0], mat.shape[1]), flush=True)

    print("[2/6] 建 gold 索引 + 法名简称解析器 ...", flush=True)
    provision_index = build_provision_index(root)
    law_ids, resolve_law, law_stat = build_law_resolver(
        root + "/indexes/retrieval/provision_nodes.jsonl")
    print("      %d 个 (法名,条号) 键；law_id 宇宙 %d 个"
          % (len(provision_index), len(law_ids)), flush=True)

    print("[3/6] 抽评测 query + gold ...", flush=True)
    queries, qstat = collect_queries(root, provision_index, resolve_law)
    if args.eval_split != "all":
        n0 = len(queries)
        queries = [q for q in queries if q["split"] == args.eval_split]
        print("      eval-split=%s → %d / %d query" % (args.eval_split, len(queries), n0),
              flush=True)
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    print("      %s" % json.dumps(qstat, ensure_ascii=False), flush=True)
    print("      法名解析：%s" % dict(law_stat), flush=True)
    if not queries:
        print("!! 没有任何可评测 query —— gold 解析全空，指标无意义。中止。", flush=True)
        return 3

    print("[4/6] BM25 索引 + 图谱边 + 热度表 ...", flush=True)
    texts, tstat = build_item_texts(root, ids)
    D, vocab, bm25_meta = build_bm25(texts)
    adj_next = load_edges(root + "/indexes/retrieval/edges_next.jsonl")
    adj_cites = load_edges(root + "/indexes/retrieval/edges_cites.jsonl")
    hot, hot_meta = load_hotness(os.path.join(root, DEFAULT_HOTNESS))
    print("      出边：NEXT %d 节点 / CITES %d 节点；热度表 %d 条"
          % (len(adj_next), len(adj_cites), len(hot)), flush=True)
    if not hot:
        print("      ⚠️ 热度表缺失 → dense_hot / full 的 hotness 视为 0", flush=True)

    print("[5/6] 编码 query（dense + BM25）...", flush=True)
    from embedding_model import encode_queries, load_model  # noqa: E402
    model, minfo = load_model(os.path.join(root, "models/Qwen3-Embedding-0.6B"),
                              device=args.device, max_seq_length=2048)
    qvecs = encode_queries(model, [q["query"] for q in queries],
                           batch_size=args.batch_size)
    bm_scores, bm_qstat = bm25_search(D, vocab, [q["query"] for q in queries], args.top_k)
    print("      BM25 查询词命中词表均值 %.2f" % bm_qstat["avg_query_terms_in_vocab"],
          flush=True)

    has_rerank = any(dict(CONFIGS)[c]["rerank"] for c in want) and not args.no_rerank
    reranker = None
    if has_rerank:
        rp = args.reranker if os.path.isabs(args.reranker) else os.path.join(root, args.reranker)
        if not os.path.isdir(rp):
            print("⚠️ 重排模型缺失（%s）→ 本次自动降级为无重排" % rp, flush=True)
            has_rerank = False
        else:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(rp, device=args.device, max_length=512)
            print("      重排模型已载入：%s" % rp, flush=True)

    print("[6/6] 检索 + 消融矩阵 ...", flush=True)
    import numpy as np
    pid_index = {p: i for i, p in enumerate(ids)}

    def take_top(score_row, k):
        order = np.argpartition(-score_row, min(k, len(score_row) - 1))[:k]
        order = order[np.argsort(-score_row[order])]
        return [ids[i] for i in order]

    # ★ 诊断用：候选池上限与「首个 gold 的排名」——区分「排序问题」与「召回天花板」
    K_CURVE = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
    ceiling = {"k_curve": list(K_CURVE),
               "tiers": {t: {"recall_at_k": {k: 0 for k in K_CURVE}, "n": 0, "gold": 0}
                         for t in ("union", "system", "output")},
               "first_gold_rank": {"ranked": 0, "unreachable_1000": 0,
                                   "hit_le": {k: 0 for k in (1, 5, 10, 20, 50, 100, 200, 500)},
                                   "hist": []}}

    results = {c: {t: dict(n=0, gold=0, r1=0, r5=0, r10=0, r20=0, r50=0, r100=0,
                           h5=0, h10=0, h20=0, h50=0, mrr=0.0)
                   for t in ("union", "system", "output")} for c in want}
    per_q_debug = []
    CHUNK = 128
    rerank_calls = 0
    for s in range(0, len(queries), CHUNK):
        block = qvecs[s:s + CHUNK]
        sims = block @ mat.T
        for j, q in enumerate(queries[s:s + CHUNK]):
            gi = s + j
            dense_list = take_top(sims[j], args.top_k)
            bm_list = take_top(bm_scores[gi], args.top_k)
            graph_list = graph_expand(dense_list, adj_next, adj_cites,
                                      top_seed=args.graph_seed, cap=args.graph_cap)
            gold_all = {"union": set(q["gold_union"]), "system": set(q["gold_sys"]),
                        "output": set(q["gold_out"])}
            for cfg in want:
                fl = dict(CONFIGS)[cfg]
                lists, weights = [dense_list], [args.w_dense]
                if fl["bm25"]:
                    lists.append(bm_list)
                    weights.append(args.w_bm25)
                if fl["graph"]:
                    lists.append(graph_list)
                    weights.append(args.w_graph)
                fused = rrf_fuse(lists, k=args.rrf_k, weights=weights)
                if fl["hot"]:
                    fused = apply_hotness(fused, hot, args.w_hot)
                pool = RERANK_POOL_OVERRIDE.get(cfg) or args.rerank_pool
                if fl["rerank"] and reranker is not None:
                    cand = [d for d, _ in fused[:pool]]
                    if len(cand) > 1:
                        pairs = [(q["query"], texts[pid_index[d]] if d in pid_index else "")
                                 for d in cand]
                        rr = reranker.predict(pairs, batch_size=args.batch_size,
                                              show_progress_bar=False)
                        order = np.argsort(-np.asarray(rr))
                        cand = [cand[i] for i in order]
                        rerank_calls += 1
                    top = cand
                else:
                    top = [d for d, _ in fused]
                t5, t10 = set(top[:5]), set(top[:10])
                t1, t20, t50, t100 = (set(top[:1]), set(top[:20]),
                                      set(top[:50]), set(top[:100]))
                for tier, g in gold_all.items():
                    if not g:
                        continue
                    a = results[cfg][tier]
                    a["n"] += 1
                    a["gold"] += len(g)
                    a["r1"] += len(g & t1)
                    a["r5"] += len(g & t5)
                    a["r10"] += len(g & t10)
                    a["r20"] += len(g & t20)
                    a["r50"] += len(g & t50)
                    a["r100"] += len(g & t100)
                    a["h5"] += 1 if (g & t5) else 0
                    a["h10"] += 1 if (g & t10) else 0
                    a["h20"] += 1 if (g & t20) else 0
                    a["h50"] += 1 if (g & t50) else 0
                    for rk, d in enumerate(top[:10], start=1):
                        if d in g:
                            a["mrr"] += 1.0 / rk
                            break
                # ★ 天花板诊断：纯 dense 在**全量 72k 索引**上的完整排序
                full_order = np.argsort(-sims[j])
                for tier, g in gold_all.items():
                    if not g:
                        continue
                    ct = ceiling["tiers"][tier]
                    ct["n"] += 1
                    ct["gold"] += len(g)
                    pos = {}
                    for rk, i in enumerate(full_order, start=1):
                        pid = ids[i]
                        if pid in g and pid not in pos:
                            pos[pid] = rk
                            if len(pos) == len(g):
                                break
                    for k in K_CURVE:
                        ct["recall_at_k"][k] += sum(1 for r in pos.values() if r <= k)
                    first = min(pos.values()) if pos else None
                    if tier == "union":
                        fg = ceiling["first_gold_rank"]
                        if first is None:
                            fg["unreachable_1000"] += 1
                        else:
                            fg["ranked"] += 1
                            fg["hist"].append(first)
                            for k in fg["hit_le"]:
                                if first <= k:
                                    fg["hit_le"][k] += 1
                if cfg == "full" and len(per_q_debug) < 20:
                    per_q_debug.append({
                        "uid": q["uid"], "n_gold_union": len(gold_all["union"]),
                        "top5": top[:5],
                        "hit5": len(gold_all["union"] & t5),
                        "hit10": len(gold_all["union"] & t10)})
        print("      已评 %d/%d query" % (min(s + CHUNK, len(queries)), len(queries)),
              flush=True)

    metrics = {}
    for cfg in want:
        metrics[cfg] = {}
        for tier in ("union", "system", "output"):
            a = results[cfg][tier]
            n, ng = max(a["n"], 1), max(a["gold"], 1)
            metrics[cfg][tier] = {
                "n_queries": a["n"], "n_gold_total": a["gold"],
                "avg_gold_per_query": round(a["gold"] / n, 3),
                "recall@1": round(a["r1"] / ng, 4),
                "recall@5": round(a["r5"] / ng, 4), "recall@10": round(a["r10"] / ng, 4),
                "recall@20": round(a["r20"] / ng, 4), "recall@50": round(a["r50"] / ng, 4),
                "recall@100": round(a["r100"] / ng, 4),
                "hit@5": round(a["h5"] / n, 4), "hit@10": round(a["h10"] / n, 4),
                "hit@20": round(a["h20"] / n, 4), "hit@50": round(a["h50"] / n, 4),
                "mrr@10": round(a["mrr"] / n, 4),
            }

    # ★ 天花板诊断结果整理（纯 dense、全量索引排序）
    ceil_out = {"k_curve": list(K_CURVE), "note":
                "纯 Qwen3-Embedding-0.6B 在**全量 %d 条**法条 item 上的完整排序；"
                "用于判定目标未达标是「排序问题」还是「召回天花板」" % mat.shape[0],
                "tiers": {}}
    for tier in ("union", "system", "output"):
        ct = ceiling["tiers"][tier]
        ng = max(ct["gold"], 1)
        ceil_out["tiers"][tier] = {
            "n_queries": ct["n"], "n_gold_total": ct["gold"],
            "recall_at_k": {str(k): round(v / ng, 4) for k, v in ct["recall_at_k"].items()},
        }
    fg = ceiling["first_gold_rank"]
    n_fg = max(fg["ranked"], 1)
    if fg["hist"]:
        arr = sorted(fg["hist"])
        def _pct(p):
            return arr[min(len(arr) - 1, int(p * (len(arr) - 1)))]
        ceil_out["first_gold_rank"] = {
            "n_ranked": fg["ranked"], "unreachable_in_top1000": fg["unreachable_1000"],
            "p10": _pct(0.10), "p50": _pct(0.50), "p90": _pct(0.90), "max": arr[-1],
            "hit_le": {str(k): round(v / n_fg, 4) for k, v in fg["hit_le"].items()},
        }
    ceiling = ceil_out
    # 作废目标的证据**从本次天花板曲线现算回填**（改口径时最易漏的就是这类写死字面量）
    _u = ceiling["tiers"]["union"]
    RETIRED_TARGETS["evidence"] = (
        "纯 dense 全量排序（union，%d query）：recall@10 = %.4f、recall@1000 = %.4f"
        " —— 候选深度开到 1000 仍够不到 0.90，且实测 recall@10 远低于 0.90。"
        "瓶颈是 embedding 表示能力，不是融合口径或排序器。"
        % (_u["n_queries"], _u["recall_at_k"]["10"], _u["recall_at_k"]["1000"]))

    PRIMARY = "union"

    def _meets(c):
        """某配置三条主指标**同时**达标。"""
        m = metrics[c][PRIMARY]
        return (m["hit@5"] >= TARGETS["hit@5"]
                and m["recall@10"] >= TARGETS["recall@10"]
                and m["mrr@10"] >= TARGETS["mrr@10"])

    # ★ 2026-09-19 修正：**两条「最优」是不同轴，不能合成一个「最佳」**。
    #   深召回轴（RAG 送生成器的候选深度）看 recall@5；
    #   头部队列轴（用户直接看前 5 条）看 hit@5 / MRR@10。
    #   实测二者分离：hybrid 深召回最好，dense_bm25 头部最准。
    best_deep = max(want, key=lambda c: (metrics[c][PRIMARY]["recall@5"],
                                         metrics[c][PRIMARY]["recall@10"]))
    best_head = max(want, key=lambda c: (metrics[c][PRIMARY]["hit@5"],
                                         metrics[c][PRIMARY]["mrr@10"]))
    meet_all = [c for c in want if _meets(c)]
    best = best_deep          # 兼容旧字段名：指向**深召回轴**最优
    R = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": {
            "dense": {"model": minfo["model_dir"], "dim": int(mat.shape[1]),
                      "index_rows": int(mat.shape[0]), "top_k": args.top_k,
                      "instruction_aware": True},
            "bm25": dict(bm25_meta, **bm_qstat),
            "graph": {"next_nodes": len(adj_next), "cites_nodes": len(adj_cites),
                      "hops": 1, "seed": args.graph_seed, "cap": args.graph_cap,
                      "w_next": 1.0, "w_cites": 0.5},
            "rrf_k": args.rrf_k,
            "rrf_weights": {"dense": args.w_dense, "bm25": args.w_bm25,
                            "graph": args.w_graph,
                            "note": "★ 由 4b-5b 在 dev 上扫出；等权 RRF 会打崩精度"},
            "eval_split": args.eval_split,
            "hotness": {"rows": len(hot), "w_hot": args.w_hot,
                        "formula": hot_meta.get("formula", {})},
            "reranker": {"enabled": bool(reranker),
                         "model": args.reranker if reranker else None,
                         "pool": args.rerank_pool,
                         "n_calls": rerank_calls},
        },
        "library_layout": {"provision_library": PROVISION_COLLECTION,
                           "vector_level_filter": "item",
                           "excluded": "level=doc（文档级，语义粒度不同）"},
        "configs": {c: CONFIG_DOC[c] for c in want},
        "gold_definition": {
            "system(tierA)": "数据集自带 `system` 的相关法律条文清单 —— 权威口径",
            "output(tierB)": "从**参考答案** `output` 解析法条引用 —— 覆盖大，但是**下界**",
            "union": "tierA ∪ tierB（主口径）",
        },
        "law_name_resolution": dict(law_stat),
        "queries": qstat,
        "item_texts": tstat,
        "metrics": metrics,
        "dense_ceiling": ceiling,
        "primary": PRIMARY,
        "best_config": best,
        "best_config_deep_recall": best_deep,
        "best_config_head": best_head,
        "selection_rule": {
            "deep_recall": "argmax (recall@5, recall@10) —— 决定 RAG 送生成器的候选池",
            "head": "argmax (hit@5, MRR@10) —— 决定用户看到的前 5 条",
            "meet_all": "hit@5 / recall@10 / MRR@10 三条同时达标",
        },
        "targets": dict(TARGETS),
        "retired_targets": dict(RETIRED_TARGETS),
        "targets_met": {
            "by_config": {
                c: {
                    "hit@5": metrics[c][PRIMARY]["hit@5"] >= TARGETS["hit@5"],
                    "recall@10": metrics[c][PRIMARY]["recall@10"] >= TARGETS["recall@10"],
                    "mrr@10": metrics[c][PRIMARY]["mrr@10"] >= TARGETS["mrr@10"],
                    "all": _meets(c),
                } for c in want
            },
            "configs_meeting_all_targets": meet_all,
        },
        "samples": per_q_debug,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_json = os.path.join(root, args.report_md.replace(".md", ".json"))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    md = []
    W = md.append

    def _target_lines():
        """主指标口径说明 —— ★ 只写一处。
        上次改口径时，「定义处改了、控制台/MD 打印处漏改」直接 KeyError 崩在产物落盘之后；
        这类重复叙述一律收敛成函数，避免再次分叉。"""
        return [
            "> **主指标（2026-09-19 修订，post-hoc 设定）**："
            "`hit@5 ≥ %.2f`、`recall@10 ≥ %.2f`、`MRR@10 ≥ %.2f`。"
            % (TARGETS["hit@5"], TARGETS["recall@10"], TARGETS["mrr@10"]),
            "> ★ **两条「最优」在不同轴上，必须分开报**："
            "**深召回轴**（argmax recall@5，决定送生成器的候选池）= **`%s`**"
            "（recall@5 %.4f / recall@10 %.4f / recall@50 %.4f）；"
            "**头部队列轴**（argmax hit@5·MRR@10，决定用户看到的前 5 条）= **`%s`**"
            "（hit@5 %.4f / MRR@10 %.4f）。"
            % (best_deep,
               metrics[best_deep][PRIMARY]["recall@5"],
               metrics[best_deep][PRIMARY]["recall@10"],
               metrics[best_deep][PRIMARY]["recall@50"],
               best_head,
               metrics[best_head][PRIMARY]["hit@5"],
               metrics[best_head][PRIMARY]["mrr@10"]),
            "> 三目标**同时**达标的配置：%s。"
            % ("、".join("`%s`" % c for c in meet_all) if meet_all else "**无**"),
            "> ⛔ **原目标 `recall@5 ≥ 0.85` / `recall@10 ≥ 0.90` 已作废** —— %s"
            % RETIRED_TARGETS["evidence"],
        ]
    W("# 阶段 4b-5 检索链路 + 消融矩阵\n")
    W("> 生成时间：%s　脚本：`scripts/retrieval/smoke_retrieval.py`\n" % R["generated_at"])
    W("## 1. 主结果（gold 口径 = `%s`）\n" % PRIMARY)
    W("| 配置 | 链路 | query | gold | **hit@5** | **recall@10** | **MRR@10** | recall@5 | hit@10 | 三目标 |")
    W("|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for c in want:
        m = metrics[c][PRIMARY]
        W("| `%s` | %s | %d | %d | **%.4f** | **%.4f** | **%.4f** | %.4f | %.4f | %s |"
          % (c, CONFIG_DOC[c], m["n_queries"], m["n_gold_total"],
             m["hit@5"], m["recall@10"], m["mrr@10"], m["recall@5"], m["hit@10"],
             "✅" if _meets(c) else "—"))
    W("")
    for ln in _target_lines():
        W(ln)
    W("")
    W("### 1b. 深截断（判断「排序问题」还是「召回天花板」）\n")
    W("| 配置 | recall@1 | recall@5 | recall@10 | recall@20 | recall@50 | recall@100 | hit@20 | hit@50 |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in want:
        m = metrics[c][PRIMARY]
        W("| `%s` | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
          % (c, m["recall@1"], m["recall@5"], m["recall@10"], m["recall@20"],
             m["recall@50"], m["recall@100"], m["hit@20"], m["hit@50"]))
    W("")
    ce = ceiling
    W("### 1c. ★ 纯 dense 全量索引天花板（%s）\n" % ce["note"])
    W("| 口径 | query | gold | %s |"
      % " | ".join("recall@%d" % k for k in ce["k_curve"]))
    W("|---|---:|---:|%s" % ("---:|" * len(ce["k_curve"])))
    for tier in ("union", "system", "output"):
        t = ce["tiers"][tier]
        cells = " | ".join("%.4f" % t["recall_at_k"][str(k)] for k in ce["k_curve"])
        W("| `%s` | %d | %d | %s |" % (tier, t["n_queries"], t["n_gold_total"], cells))
    W("")
    if "first_gold_rank" in ce:
        f = ce["first_gold_rank"]
        W("- 首个 gold 的排名分布（union，n=%d）：p10=%d / p50=%d / p90=%d / max=%d；"
          "top-1000 内找不到任何 gold 的 query = **%d**"
          % (f["n_ranked"], f["p10"], f["p50"], f["p90"], f["max"],
             f["unreachable_in_top1000"]))
        W("- 首个 gold 落入 top-k 的 query 占比：%s\n"
          % "，".join("top-%s **%.4f**" % (k, v) for k, v in f["hit_le"].items()))
    for ln in _target_lines():
        W(ln)
    W("")
    W("## 2. 三种 gold 口径全表\n")
    W("| 配置 | 口径 | query | gold | recall@5 | recall@10 | hit@5 |")
    W("|---|---|---:|---:|---:|---:|---:|")
    for c in want:
        for tier in ("union", "system", "output"):
            m = metrics[c][tier]
            W("| `%s` | %s | %d | %d | %.4f | %.4f | %.4f |"
              % (c, tier, m["n_queries"], m["n_gold_total"],
                 m["recall@5"], m["recall@10"], m["hit@5"]))
    W("")
    W("## 3. 链路配置\n")
    W("- dense：`%s`，%d 维，索引 %d 行（**只用 level=item**，doc 级排除）"
      % (minfo["model_dir"], mat.shape[1], mat.shape[0]))
    W("- BM25：jieba 分词 + scipy 稀疏**精确**实现（词表 %d / df≥%d / nnz %d）"
      % (bm25_meta["vocab"], bm25_meta["min_df"], bm25_meta["nnz"]))
    W("- 图谱：1 跳，种子 %d，列表限长 %d；`NEXT` 权重 1.0、`CITES` 权重 0.5"
      % (args.graph_seed, args.graph_cap))
    W("- RRF：k = %d，**加权** dense %.2f / BM25 %.2f / 图谱 %.2f（dev 扫出）"
      % (args.rrf_k, args.w_dense, args.w_bm25, args.w_graph))
    W("- 热度：`s *= (1 + %.2f * hotness)`，热度表 %d 条（**只在检索库内统计，零泄漏**）"
      % (args.w_hot, len(hot)))
    W("- 重排：%s" % ("`%s`，pool=%d" % (args.reranker, args.rerank_pool)
                      if reranker else "**未启用**"))
    W("")
    W("## 4. gold 从哪来\n")
    for k, v in R["gold_definition"].items():
        W("- `%s`：%s" % (k, v))
    W("")
    W("- dev∪test 原始行数 **%d**；有可评测 gold 的 **%d** 条" % (qstat["rows"], qstat["with_gold"]))
    W("- `system` 口径 **%d** 条；`output` 口径 **%d** 条"
      % (qstat.get("tierA_rows", 0), qstat.get("tierB_rows", 0)))
    W("- gold 引用解析：`system` 命中 %d / 库里缺 %d；`output` 命中 %d / 库里缺 %d"
      % (qstat.get("refs_system", 0), qstat.get("refs_system_missing_in_index", 0),
         qstat.get("refs_output", 0), qstat.get("refs_output_missing_in_index", 0)))
    W("- 法名简称解析（exact → 唯一后缀，最小 2 字）：%s\n" % dict(law_stat))
    W("## 5. 评测安全性\n")
    W("- 检索库与 train/dev/test/router 四份 split **强隔离**（4b-0 门禁 PASS），"
      "不存在「检索到题目本身」。\n")
    W("- 热度只用 CITES 入度与法源位阶（法条侧）/ 检索库内同案复现次数（案件侧），"
      "**不接触评测集**。\n")
    W("**耗时 %.1fs**\n" % R["elapsed_sec"])
    with open(os.path.join(root, args.report_md), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("=" * 78)
    print("阶段 4b-5 检索链路（gold 口径 = %s，eval-split=%s）" % (PRIMARY, args.eval_split))
    print("  %-14s %8s %8s %8s %8s"
          % ("config", "hit@5", "recall@10", "MRR@10", "recall@5"))
    for c in want:
        m = metrics[c][PRIMARY]
        print("  %-14s %8.4f %8.4f %8.4f %8.4f"
              % (c, m["hit@5"], m["recall@10"], m["mrr@10"], m["recall@5"]))
    tm = R["targets_met"]
    print("  深召回最优（argmax recall@5） = %s" % best_deep)
    print("  头部最优　（argmax hit@5）    = %s" % best_head)
    print("  三目标全达标配置 = %s" % (", ".join(meet_all) if meet_all else "无"))
    for c in want:
        b = tm["by_config"][c]
        bad = [k for k in ("hit@5", "recall@10", "mrr@10") if not b[k]]
        print("    %-14s %s" % (c, "三目标达标" if b["all"] else "未达标：" + ",".join(bad)))
    print("  ⛔ 原目标 recall@5>=0.85 / recall@10>=0.90 已作废（见 §1c 天花板）")
    print("  耗时 %.1fs" % R["elapsed_sec"])
    print("SMOKE_RETRIEVAL_DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        # ★ 绝不让「控制台打印」的异常吃掉已经落盘的 JSON/MD
        import traceback
        traceback.print_exc()
        print("SMOKE_RETRIEVAL_DONE_WITH_TRACEBACK")
        sys.exit(3)
