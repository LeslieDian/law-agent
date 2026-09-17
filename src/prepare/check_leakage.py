"""CLaw 泄漏排查：训练集 vs CLaw 测试集的四级查重。

L1 完全字符串比对
L2 去标点/空白/全半角归一后比对
L3 MinHash + LSH 近似查重（Jaccard > 阈值 → 人工复核）
L4 向量相似度（余弦 > 阈值 → 人工复核，需 sentence-transformers）

用法：
    python -m src.prepare.check_leakage \
        --train data/train/train.jsonl \
        --benchmark data/benchmark/claw_cases.jsonl \
        --text-field-user  user \
        --text-field-answer assistant

输出：reports/leakage_report.md 与 reports/leakage_hits.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io import (
    load_json,
    normalize_text,
    project_path,
    read_jsonl,
    save_json,
    write_jsonl,
)


def extract_texts(path: Path, user_field: str, answer_field: str) -> list[dict]:
    """从 JSONL 中抽取 (user, assistant) 文本，兼容 messages 格式与扁平字段。"""
    rows: list[dict] = []
    for record in read_jsonl(path):
        if "messages" in record:
            user = "\n".join(
                m.get("content", "") for m in record["messages"] if m.get("role") == "user"
            )
            assistant = "\n".join(
                m.get("content", "") for m in record["messages"] if m.get("role") == "assistant"
            )
        else:
            user = str(record.get(user_field, ""))
            assistant = str(record.get(answer_field, ""))
        rows.append({"id": record.get("id") or record.get("case_id"), "user": user, "answer": assistant})
    return rows


def level1_l2(train: list[dict], bench: list[dict]) -> list[dict]:
    """完全匹配 + 归一化匹配。"""
    hits: list[dict] = []
    bench_raw = {(b["user"], b["answer"]) for b in bench}
    bench_norm_user = {normalize_text(b["user"]) for b in bench}
    bench_norm_ans = {normalize_text(b["answer"]) for b in bench}

    for t in train:
        if (t["user"], t["answer"]) in bench_raw:
            hits.append({"level": "L1", "train_id": t["id"], "reason": "完全字符串一致"})
            continue
        if normalize_text(t["user"]) in bench_norm_user:
            hits.append({"level": "L2", "train_id": t["id"], "reason": "归一化后 user 文本一致"})
        elif normalize_text(t["answer"]) in bench_norm_ans:
            hits.append({"level": "L2", "train_id": t["id"], "reason": "归一化后 answer 文本一致"})
    return hits


def level3_minhash(train: list[dict], bench: list[dict], threshold: float = 0.85) -> list[dict]:
    """MinHash + LSH 近似查重。"""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("[SKIP] 未安装 datasketch，跳过 L3。执行 pip install datasketch")
        return []

    def to_minhash(text: str, num_perm: int = 128) -> MinHash:
        m = MinHash(num_perm=num_perm)
        for token in normalize_text(text):
            m.update(token.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for b in bench:
        lsh.insert(str(b["id"]), to_minhash(b["user"] + b["answer"]))

    hits: list[dict] = []
    for t in train:
        for cand in lsh.query(to_minhash(t["user"] + t["answer"])):
            hits.append(
                {"level": "L3", "train_id": t["id"], "benchmark_id": cand,
                 "reason": f"MinHash Jaccard > {threshold}"}
            )
    return hits


def level4_vector(train: list[dict], bench: list[dict], model_name: str, threshold: float = 0.92) -> list[dict]:
    """向量相似度查重（可选，依赖 sentence-transformers + 本地/远程模型）。"""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[SKIP] 未安装 sentence-transformers，跳过 L4。")
        return []

    model = SentenceTransformer(model_name)
    bench_emb = model.encode([b["user"] + b["answer"] for b in bench], normalize_embeddings=True)
    train_emb = model.encode([t["user"] + t["answer"] for t in train], normalize_embeddings=True)

    hits: list[dict] = []
    sims = np.asarray(train_emb) @ np.asarray(bench_emb).T
    for i, row in enumerate(sims):
        j = int(row.argmax())
        if float(row[j]) > threshold:
            hits.append(
                {"level": "L4", "train_id": train[i]["id"], "benchmark_id": bench[j]["id"],
                 "similarity": round(float(row[j]), 4), "reason": f"余弦 > {threshold}"}
            )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="CLaw 泄漏排查")
    parser.add_argument("--train", default="data/train/train.jsonl")
    parser.add_argument("--benchmark", default="data/benchmark/claw_cases.jsonl")
    parser.add_argument("--user-field", default="user")
    parser.add_argument("--answer-field", default="assistant")
    parser.add_argument("--vector-model", default="BAAI/bge-m3")
    parser.add_argument("--minhash-threshold", type=float, default=0.85)
    parser.add_argument("--vector-threshold", type=float, default=0.92)
    parser.add_argument("--skip-vector", action="store_true")
    args = parser.parse_args()

    train_path = project_path(args.train)
    bench_path = project_path(args.benchmark)
    if not train_path.exists() or not bench_path.exists():
        print(f"[ERROR] 文件缺失：{train_path} 或 {bench_path}")
        return 1

    train = extract_texts(train_path, args.user_field, args.answer_field)
    bench = extract_texts(bench_path, args.user_field, args.answer_field)
    print(f"训练样本 {len(train)} 条，CLaw 基准 {len(bench)} 条")

    hits: list[dict] = []
    hits += level1_l2(train, bench)
    print(f"L1/L2 命中 {len(hits)} 条")
    l3 = level3_minhash(train, bench, args.minhash_threshold)
    hits += l3
    print(f"L3 命中 {len(l3)} 条")
    if not args.skip_vector:
        l4 = level4_vector(train, bench, args.vector_model, args.vector_threshold)
        hits += l4
        print(f"L4 命中 {len(l4)} 条")

    write_jsonl(project_path("reports", "leakage_hits.jsonl"), hits)

    by_level: dict[str, int] = {}
    for h in hits:
        by_level[h["level"]] = by_level.get(h["level"], 0) + 1

    report = [
        "# CLaw 泄漏排查报告",
        "",
        f"- 训练样本：{len(train)}",
        f"- CLaw 基准：{len(bench)}",
        f"- MinHash 阈值：{args.minhash_threshold}",
        f"- 向量阈值：{args.vector_threshold}",
        "",
        "## 命中统计",
        "",
        "| 级别 | 命中数 | 处理方式 |",
        "|---|---|---|",
        f"| L1 完全一致 | {by_level.get('L1', 0)} | 直接剔除 |",
        f"| L2 归一化一致 | {by_level.get('L2', 0)} | 直接剔除 |",
        f"| L3 MinHash 近似 | {by_level.get('L3', 0)} | 人工复核 |",
        f"| L4 向量高相似 | {by_level.get('L4', 0)} | 人工复核 |",
        "",
        "> 全部命中明细见 `reports/leakage_hits.jsonl`。人工复核结论需在本文件追加。",
    ]
    report_path = project_path("reports", "leakage_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    save_json(
        project_path("reports", "leakage_summary.json"),
        {"train": len(train), "benchmark": len(bench), "hits_by_level": by_level, "total_hits": len(hits)},
    )
    print(f"\n报告已写出: {report_path}")
    return 0 if not by_level.get("L1") and not by_level.get("L2") else 2


if __name__ == "__main__":
    raise SystemExit(main())
