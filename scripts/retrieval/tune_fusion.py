# -*- coding: utf-8 -*-
"""阶段 4b-5b：在 **dev** 上扫融合权重（一次缓存、无 GPU 重算）。

动机（2026-09-19 实测）：
  4b-5 v1 的等权 RRF + 图谱注入把精度打崩 —— dense_graph recall@5 0.2545
  vs 纯向量 0.3961；hybrid 0.3215；full 0.3566。所有"加料"都变差，
  说明**融合口径错了**（图谱邻近条文与 dense 同权竞争），不是组件没用。

本脚本把 query 侧的 dense / BM25 / 图谱候选**各算一次**，然后在 dev 上
穷举权重组合，直接读指标 —— 避免"一个配置跑 758s"的暴力搜索。

输出：docs/retrieval/FUSION_SWEEP.{json,md}
  - 权重网格：w_bm25 ∈ {0,0.1,0.2,0.3,0.5,0.7,1.0} × w_graph ∈ {0,0.05,0.1,0.2,0.3,0.5}
  - rrf_k ∈ {10,60}，hotness ∈ {0, 0.05}
  - 指标：recall@{1,5,10,20,50,100} / hit@5 / MRR@10（gold 口径三选）
★ 只在 dev 上调参；test 的数字一律由 smoke_retrieval.py 单独出。
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoke_retrieval import (  # noqa: E402
    DEFAULT_HOTNESS, PROVISION_COLLECTION, apply_hotness, bm25_search,
    build_bm25, build_item_texts, build_law_resolver, build_provision_index,
    collect_queries, graph_expand, load_edges, load_hotness, load_item_vectors,
    rrf_fuse,
)

K_CURVE = (1, 5, 10, 20, 50, 100)
W_BM25_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
W_GRAPH_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
RRF_K_GRID = (10, 60)
HOT_GRID = (0.0, 0.05)


def evaluate(ranked_ids, gold, ks=K_CURVE):
    """→ dict(recall@k, hit@5, mrr@10)"""
    out = {"n": 1, "gold": len(gold)}
    tops = {k: set(ranked_ids[:k]) for k in ks}
    for k in ks:
        out["r%d" % k] = len(gold & tops[k])
    out["h5"] = 1 if (gold & tops[5]) else 0
    out["mrr"] = 0.0
    for rk, d in enumerate(ranked_ids[:10], start=1):
        if d in gold:
            out["mrr"] = 1.0 / rk
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--top-k", type=int, default=500, help="各检索器候选深度")
    ap.add_argument("--graph-seed", type=int, default=10)
    ap.add_argument("--graph-cap", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--report-md", default="docs/retrieval/FUSION_SWEEP.md")
    args = ap.parse_args()

    import numpy as np

    root = args.root.rstrip("/")
    emb_root = os.path.join(root, "indexes/retrieval/embeddings")
    t0 = time.time()

    print("[1/5] dense 向量 ...", flush=True)
    ids, mat = load_item_vectors(os.path.join(emb_root, PROVISION_COLLECTION))
    print("      %d × %d" % mat.shape, flush=True)

    print("[2/5] gold 索引 ...", flush=True)
    provision_index = build_provision_index(root)
    law_ids, resolve_law, law_stat = build_law_resolver(
        root + "/indexes/retrieval/provision_nodes.jsonl")

    print("[3/5] query + gold（split=%s）..." % args.split, flush=True)
    queries, qstat = collect_queries(root, provision_index, resolve_law,
                                     splits=(args.split,))
    print("      %d query 带 gold" % len(queries), flush=True)
    if not queries:
        print("!! 没有可评测 query", flush=True)
        return 3

    print("[4/5] BM25 / 图谱 / 热度 ...", flush=True)
    texts, _ = build_item_texts(root, ids)
    D, vocab, bm25_meta = build_bm25(texts)
    adj_next = load_edges(root + "/indexes/retrieval/edges_next.jsonl")
    adj_cites = load_edges(root + "/indexes/retrieval/edges_cites.jsonl")
    hot, hot_meta = load_hotness(os.path.join(root, DEFAULT_HOTNESS))

    print("[5/5] 编码 query + 缓存候选 ...", flush=True)
    from embedding_model import encode_queries, load_model
    model, minfo = load_model(os.path.join(root, "models/Qwen3-Embedding-0.6B"),
                              device=args.device, max_seq_length=2048)
    qvecs = encode_queries(model, [q["query"] for q in queries],
                           batch_size=args.batch_size)
    bm_scores, _ = bm25_search(D, vocab, [q["query"] for q in queries], args.top_k)

    def take_top(row, k):
        kk = min(k, len(row))
        order = np.argpartition(-row, kk - 1)[:kk]
        return [ids[i] for i in order[np.argsort(-row[order])]]

    # 候选各算一次
    cache = []
    sims_all = qvecs @ mat.T
    for j, q in enumerate(queries):
        dense_list = take_top(sims_all[j], args.top_k)
        bm_list = take_top(bm_scores[j], args.top_k)
        gl = graph_expand(dense_list, adj_next, adj_cites,
                          top_seed=args.graph_seed, cap=args.graph_cap)
        gold = {"union": set(q["gold_union"]), "system": set(q["gold_sys"]),
                "output": set(q["gold_out"])}
        cache.append((q["uid"], dense_list, bm_list, gl, gold))
    print("      候选缓存完毕（%d query，深度 %d）" % (len(cache), args.top_k), flush=True)

    # ---- 扫描
    rows = []
    for rrf_k in RRF_K_GRID:
        for w_bm in W_BM25_GRID:
            for w_g in W_GRAPH_GRID:
                for w_hot in HOT_GRID:
                    acc = {t: collections.Counter() for t in ("union", "system", "output")}
                    for _uid, dl, bl, gl, gold in cache:
                        lists, weights = [dl], [1.0]
                        if w_bm > 0:
                            lists.append(bl)
                            weights.append(w_bm)
                        if w_g > 0 and gl:
                            lists.append(gl)
                            weights.append(w_g)
                        fused = rrf_fuse(lists, k=rrf_k, weights=weights)
                        if w_hot > 0:
                            fused = apply_hotness(fused, hot, w_hot)
                        ranked = [d for d, _ in fused]
                        for t, g in gold.items():
                            if not g:
                                continue
                            acc[t].update(evaluate(ranked, g))
                    for t in ("union", "system", "output"):
                        a = acc[t]
                        if not a["n"]:
                            continue
                        n, ng = a["n"], max(a["gold"], 1)
                        rows.append({
                            "gold": t, "rrf_k": rrf_k, "w_bm25": w_bm,
                            "w_graph": w_g, "w_hot": w_hot, "n": n,
                            **{("recall@%d" % k): round(a["r%d" % k] / ng, 4) for k in K_CURVE},
                            "hit@5": round(a["h5"] / n, 4),
                            "mrr@10": round(a["mrr"] / n, 4),
                        })
    print("      扫描组合数 %d" % len(rows), flush=True)

    def top_of(gold, key, k=15):
        sub = [r for r in rows if r["gold"] == gold]
        return sorted(sub, key=lambda r: -r[key])[:k]

    R = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_used_for_tuning": args.split,
        "note": "只在 dev 上扫权重；test 指标由 smoke_retrieval.py --eval-split test 单独出",
        "grid": {"w_bm25": list(W_BM25_GRID), "w_graph": list(W_GRAPH_GRID),
                 "rrf_k": list(RRF_K_GRID), "w_hot": list(HOT_GRID),
                 "top_k": args.top_k, "graph_seed": args.graph_seed,
                 "graph_cap": args.graph_cap},
        "n_queries": len(queries),
        "dense_model": minfo["model_dir"],
        "baselines": {
            "pure_dense_hit@5": round(sum(
                1 for _u, dl, _b, _g, gold in cache if set(dl[:5]) & gold["union"])
                / max(len(cache), 1), 4),
        },
        "top_by_recall@5": {"union": top_of("union", "recall@5"),
                            "system": top_of("system", "recall@5"),
                            "output": top_of("output", "recall@5")},
        "top_by_mrr@10": {"union": top_of("union", "mrr@10")},
    }
    out_json = os.path.join(root, args.report_md.replace(".md", ".json"))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    md = ["# 阶段 4b-5b 融合权重扫描（调参集 = `%s`）\n" % args.split,
          "> 生成时间：%s　网格：w_bm25 %s × w_graph %s × rrf_k %s × w_hot %s，候选深度 %d\n"
          % (R["generated_at"], list(W_BM25_GRID), list(W_GRAPH_GRID),
             list(RRF_K_GRID), list(HOT_GRID), args.top_k),
          "- query 数 **%d**；dense 模型 `%s`" % (len(queries), minfo["model_dir"]),
          "- 纯 dense（w_bm25=0, w_graph=0）**hit@5** = **%.4f**\n"
          % R["baselines"]["pure_dense_hit@5"]]
    for gold in ("union", "system", "output"):
        md.append("\n## gold = `%s` —— recall@5 Top-12\n" % gold)
        md.append("| w_bm25 | w_graph | rrf_k | w_hot | recall@5 | recall@10 | recall@50 | hit@5 | MRR@10 |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in top_of(gold, "recall@5", 12):
            md.append("| %.2f | %.2f | %d | %.2f | **%.4f** | %.4f | %.4f | %.4f | %.4f |"
                      % (r["w_bm25"], r["w_graph"], r["rrf_k"], r["w_hot"],
                         r["recall@5"], r["recall@10"], r["recall@50"],
                         r["hit@5"], r["mrr@10"]))
    md.append("\n## 参考：w_graph 的作用（union，固定 w_bm25 最佳值、其它取扫描最优）\n")
    best_bm = top_of("union", "recall@5", 1)[0]["w_bm25"]
    md.append("| w_bm25=%.2f | w_graph | recall@5 | recall@20 | recall@50 |" % best_bm)
    md.append("|---:|---:|---:|---:|")
    for r in sorted([x for x in rows if x["gold"] == "union" and x["w_bm25"] == best_bm
                     and x["rrf_k"] == 60 and x["w_hot"] == 0.0],
                    key=lambda x: x["w_graph"]):
        md.append("| %.2f | %.2f | %.4f | %.4f | %.4f |"
                  % (r["w_bm25"], r["w_graph"], r["recall@5"],
                     r["recall@20"], r["recall@50"]))
    md.append("\n_耗时 %.1fs_\n" % (time.time() - t0))
    with open(os.path.join(root, args.report_md), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 控制台摘要
    for gold in ("union", "system", "output"):
        b = top_of(gold, "recall@5", 3)
        print("  best[%s] r@5: %s" % (gold, [
            "%.4f (bm=%.2f g=%.2f k=%d hot=%.2f)" % (
                x["recall@5"], x["w_bm25"], x["w_graph"], x["rrf_k"], x["w_hot"])
            for x in b]), flush=True)
    print("FUSION_SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(3)
