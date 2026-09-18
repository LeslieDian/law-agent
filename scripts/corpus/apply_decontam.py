#!/usr/bin/env python3
"""阶段 3b：应用去污剔除清单，生成清洗后的语料镜像。

输入  data/corpus/normalized/{qa,statutes}/*.jsonl（只读，不修改）
输出  data/corpus/decontaminated/{qa,statutes}/*.jsonl（剔除命中 uid 后的副本）

原则：
- 原始 normalized 目录保持只读（硬约定：数据只读 + 落盘即登记）
- 每个 uid 只保留第一次出现（与归一化时一致，正常情况下无重复 uid）
- 输出逐文件统计保留/剔除条数，并按域汇总（论文引用口径）
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--uids", required=True, help="DECONTAM_REMOVED_UIDS.txt")
    ap.add_argument("--src", default="", help="默认 <root>/data/corpus/normalized")
    ap.add_argument("--dst", default="", help="默认 <root>/data/corpus/decontaminated")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()

    root = a.root
    src = a.src or os.path.join(root, "data/corpus/normalized")
    dst = a.dst or os.path.join(root, "data/corpus/decontaminated")

    with open(a.uids, encoding="utf-8") as f:
        banned = set(x.strip() for x in f if x.strip())
    print(f"剔除清单: {len(banned)} 个 uid", flush=True)

    per_file = {}
    dom_kept = Counter()
    dom_removed = Counter()
    total_kept = total_removed = 0

    for sub in ("qa", "statutes"):
        sdir = os.path.join(src, sub)
        if not os.path.isdir(sdir):
            continue
        ddir = os.path.join(dst, sub)
        os.makedirs(ddir, exist_ok=True)
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".jsonl") or ".sample." in fn:
                continue
            sp = os.path.join(sdir, fn)
            dp = os.path.join(ddir, fn)
            kept = removed = 0
            t0 = time.time()
            with open(sp, encoding="utf-8") as fin, \
                 open(dp, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("uid") in banned:
                        removed += 1
                        dom_removed[r.get("domain") or "?"] += 1
                        continue
                    kept += 1
                    dom_kept[r.get("domain") or "?"] += 1
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            per_file[f"{sub}/{fn}"] = {
                "kept": kept, "removed": removed,
                "sha256": sha256_file(dp),
            }
            total_kept += kept
            total_removed += removed
            print(f"  {sub}/{fn}: kept={kept} removed={removed} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # 断言：输出中不得残留任何被剔除 uid
    residual = 0
    for sub in ("qa", "statutes"):
        ddir = os.path.join(dst, sub)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(ddir, fn), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if json.loads(line).get("uid") in banned:
                        residual += 1
    if residual:
        print(f"!!! 断言失败：输出中残留 {residual} 条被剔除 uid", flush=True)
        return 2

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uids_file": a.uids,
        "banned_uids": len(banned),
        "src": src, "dst": dst,
        "total_kept": total_kept,
        "total_removed": total_removed,
        "by_domain_kept": dict(dom_kept),
        "by_domain_removed": dict(dom_removed),
        "per_file": per_file,
    }
    out_json = a.out_json or os.path.join(dst, "APPLY_SUMMARY.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n清洗完成: kept={total_kept} removed={total_removed}")
    print(f"按域保留: {dict(dom_kept)}")
    print(f"汇总: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
