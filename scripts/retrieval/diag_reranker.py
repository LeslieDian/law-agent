# -*- coding: utf-8 -*-
"""诊断：bge-reranker-v2-m3 为什么把 top-5 打崩（4b-5 实测 −0.09 recall@5）。

现象（dev 427 query）：
  wr_dense_bm25（无精排） recall@5 0.4194 / hit@5 0.6651 / MRR 0.5147
  wr_rerank200（同链路+精排）recall@5 0.3295 / hit@5 0.5644 / MRR 0.4424
精排**越深越差**（pool 50 比 200 好）—— 对 CrossEncoder 而言不合常理，怀疑：
  (a) `max_length=512` 把**法条正文**挤没了（query 是长案件文本，passage 被截断）；
  (b) 配对顺序 / 输入模板不对；
  (c) 打分方向反了（score 越小越相关）。

本脚本用**受控实验**定位：
  A. 判别力：gold 法条 vs 随机负例，看 mean score 与 AUC；再按 query 截断长度做敏感性；
  B. 端到端：对 dense top-50 精排，比较  {不精排, ml=512, ml=1024, 截断query@512, 截断query@256}
     的 recall@5 / hit@5 / MRR@10。
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_retrieval import (  # noqa: E402
    PROVISION_COLLECTION, build_item_texts, build_law_resolver,
    build_provision_index, collect_queries, load_item_vectors,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--n-neg", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--reranker", default="models/bge-reranker-v2-m3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report-json", default="docs/retrieval/RERANKER_DIAG.json")
    args = ap.parse_args()

    import numpy as np
    root = args.root.rstrip("/")
    rnd = random.Random(42)

    print("[1/4] 载入向量 + 文本 ...", flush=True)
    ids, mat = load_item_vectors(os.path.join(root, "indexes/retrieval/embeddings",
                                             PROVISION_COLLECTION))
    texts, _ = build_item_texts(root, ids)
    pos = {p: i for i, p in enumerate(ids)}

    print("[2/4] gold ...", flush=True)
    pi = build_provision_index(root)
    _, resolve, _ = build_law_resolver(root + "/indexes/retrieval/provision_nodes.jsonl")
    queries, _ = collect_queries(root, pi, resolve, splits=(args.split,))
    rnd.shuffle(queries)
    queries = queries[:args.n_queries]
    print("      %d query" % len(queries), flush=True)

    print("[3/4] 编码 query ...", flush=True)
    from embedding_model import encode_queries, load_model
    model, _ = load_model(os.path.join(root, "models/Qwen3-Embedding-0.6B"),
                          device=args.device, max_seq_length=2048)
    qvecs = encode_queries(model, [q["query"] for q in queries], batch_size=64)
    del model
    sims = qvecs @ mat.T

    def take_top(row, k):
        kk = min(k, len(row))
        o = np.argpartition(-row, kk - 1)[:kk]
        return [ids[i] for i in o[np.argsort(-row[o])]]

    dense_lists = [take_top(sims[j], args.top_k) for j in range(len(queries))]

    print("[4/4] CrossEncoder 受控实验 ...", flush=True)
    from sentence_transformers import CrossEncoder

    def evaluate(rank_fn, tag):
        """rank_fn(qi, cand_list) -> 重排后的 cand_list"""
        r5 = r10 = h5 = 0
        mrr = 0.0
        ng = 0
        gold_tot = 0
        for j, q in enumerate(queries):
            g = set(q["gold_union"])
            if not g:
                continue
            ng += 1
            gold_tot += len(g)
            ranked = rank_fn(j, list(dense_lists[j]))
            t5, t10 = set(ranked[:5]), set(ranked[:10])
            r5 += len(g & t5)
            r10 += len(g & t10)
            h5 += 1 if (g & t5) else 0
            for rk, d in enumerate(ranked[:10], start=1):
                if d in g:
                    mrr += 1.0 / rk
                    break
        gt = max(gold_tot, 1)
        return {"tag": tag, "n_queries_with_gold": ng, "n_gold_total": gold_tot,
                "recall@5": round(r5 / gt, 4), "recall@10": round(r10 / gt, 4),
                "hit@5": round(h5 / max(ng, 1), 4), "mrr@10": round(mrr / max(ng, 1), 4)}

    out = {"n_queries": len(queries), "top_k": args.top_k, "split": args.split,
           "results": [], "discrimination": {}}

    out["results"].append(evaluate(lambda j, c: c, "dense_only"))

    # A. 判别力：gold vs 随机负例
    print("      A) 判别力测试", flush=True)
    gold_pairs, neg_pairs = [], []
    gold_meta = []
    for j, q in enumerate(queries):
        g = list(set(q["gold_union"]))
        if not g:
            continue
        for d in g[:3]:
            if d in pos:
                gold_pairs.append((q["query"], texts[pos[d]]))
                gold_meta.append(d)
        negs = [d for d in ids if d not in set(g)]
        for d in rnd.sample(negs, min(args.n_neg, len(negs))):
            neg_pairs.append((q["query"], texts[pos[d]]))
    print("        gold pairs %d / neg pairs %d" % (len(gold_pairs), len(neg_pairs)),
          flush=True)

    for ml in (512, 1024):
        ce = CrossEncoder(os.path.join(root, args.reranker), device=args.device,
                          max_length=ml)
        sg = ce.predict(gold_pairs, batch_size=64, show_progress_bar=False)
        sn = ce.predict(neg_pairs, batch_size=64, show_progress_bar=False)
        sg, sn = np.asarray(sg), np.asarray(sn)
        # AUC：随机取一对 gold/neg，P(gold > neg)
        auc = float((sg[:, None] > sn[None, :]).mean())
        out["discrimination"]["ml%d" % ml] = {
            "mean_gold": round(float(sg.mean()), 4), "mean_neg": round(float(sn.mean()), 4),
            "auc_gold_vs_neg": round(auc, 4),
            "pct_gold_positive": round(float((sg > 0).mean()), 4),
        }
        print("        ml=%d  mean_gold=%.4f mean_neg=%.4f AUC=%.4f"
              % (ml, sg.mean(), sn.mean(), auc), flush=True)
        if ml == 512:
            ce512 = ce
        else:
            ce1024 = ce

    # B. 端到端：不同精排设置
    def make_rank(ce, qmax):
        def f(j, cand):
            q = queries[j]["query"]
            qq = q[:qmax] if qmax else q
            pairs = [(qq, texts[pos[d]]) for d in cand]
            sc = np.asarray(ce.predict(pairs, batch_size=64, show_progress_bar=False))
            order = np.argsort(-sc)
            return [cand[i] for i in order]
        return f

    print("      B) 端到端重排", flush=True)
    for tag, ce, qmax in (("rerank_ml512_qfull", ce512, 0),
                          ("rerank_ml512_q512", ce512, 512),
                          ("rerank_ml512_q256", ce512, 256),
                          ("rerank_ml1024_qfull", ce1024, 0),
                          ("rerank_ml1024_q512", ce1024, 512)):
        out["results"].append(evaluate(make_rank(ce, qmax), tag))
        print("        %s -> %s" % (tag, out["results"][-1]), flush=True)

    # 反向打分（检查方向是否反了）
    out["results"].append(evaluate(
        lambda j, cand: list(reversed(make_rank(ce512, 0)(j, cand))), "rerank_REVERSED"))

    rj = os.path.join(root, args.report_json)
    os.makedirs(os.path.dirname(rj), exist_ok=True)
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===")
    print("  %-22s %8s %8s %8s %8s" % ("setting", "recall@5", "recall@10", "hit@5", "MRR@10"))
    for r in out["results"]:
        print("  %-22s %8.4f %8.4f %8.4f %8.4f"
              % (r["tag"], r["recall@5"], r["recall@10"], r["hit@5"], r["mrr@10"]))
    print("RERANKER_DIAG_DONE")
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
