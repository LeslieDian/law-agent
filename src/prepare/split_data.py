"""训练数据切分：按案件来源分组切分 90/5/5。

核心纪律：
  1. 按 case source 分组，同一案件派生的所有问题必须落在同一分区（避免同源泄漏）
  2. CLaw 的 254 个案例不进入任何分区

用法：
    python -m src.prepare.split_data \
        --input  data/normalized/train_pool.jsonl \
        --group-field source \
        --out-dir data
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io import project_path, read_jsonl, write_jsonl


def split(records: list[dict], group_field: str, ratios: tuple[float, float, float], seed: int = 42):
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(group_field) or "UNKNOWN")].append(r)

    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])

    train_keys = set(keys[:n_train])
    dev_keys = set(keys[n_train : n_train + n_dev])

    train, dev, test = [], [], []
    for key, rows in groups.items():
        (train if key in train_keys else dev if key in dev_keys else test).extend(rows)

    return train, dev, test, len(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description="按来源分组切分训练数据")
    parser.add_argument("--input", default="data/normalized/train_pool.jsonl")
    parser.add_argument("--group-field", default="source", help="分组字段（案件来源）")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.dev_ratio, 1 - args.train_ratio - args.dev_ratio)
    src = project_path(args.input)
    if not src.exists():
        print(f"[ERROR] 输入文件不存在: {src}")
        return 1

    records = list(read_jsonl(src))
    train, dev, test, n_groups = split(records, args.group_field, ratios, args.seed)

    out_dir = project_path(args.out_dir)
    write_jsonl(out_dir / "train" / "train.jsonl", train)
    write_jsonl(out_dir / "dev" / "dev.jsonl", dev)
    write_jsonl(out_dir / "test" / "internal_test.jsonl", test)

    print(f"来源分组数: {n_groups}")
    print(f"train: {len(train)}  dev: {len(dev)}  internal_test: {len(test)}")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
