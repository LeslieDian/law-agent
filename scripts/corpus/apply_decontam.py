#!/usr/bin/env python3
"""阶段 3b：应用去污剔除清单，生成清洗后的语料镜像。

输入  data/corpus/normalized/{qa,statutes}/*.jsonl（只读，不修改）
输出  data/corpus/decontaminated/{qa,statutes}/*.jsonl（剔除命中 uid 后的副本）

原则：
- 原始 normalized 目录保持只读（硬约定：数据只读 + 落盘即登记）
- 每个 uid 只保留第一次出现，且 **uid 必须全局唯一**（见下方断言）
- 输出逐文件统计保留/剔除条数，并按域汇总（论文引用口径）

======================= 硬约定（改动必须同步 docs 与本文件顶部） =======================
1. 只处理**正式产出文件**：跳过 `_` 开头的合并副本（`_all.jsonl`）与调试残留
   （`*.sample.jsonl`）。历史缺陷：旧实现未跳过 `_all.jsonl`，会在输出目录再生成一份
   合并副本，容易与正式文件双计。
2. **uid 必须全局唯一**，否则「按 uid 剔除」会**整组连坐**：同号但内容不同的样本会被
   一起删掉。阶段 2 曾因 uid 漏了源文件造成 20,947 条撞号 → 本该删 14,218 组、实际删
   16,489 条，连带多删 2,271 条真实样本（方向是多删=保守，无污染风险，但数字被改动）。
   本脚本以恒等式 `剔除条数 == |剔除清单 ∩ 输入 uid 集合|` 做硬断言，不成立即 exit 3。
3. 剔除清单里的 uid 若在输入中不存在，会被计数并打印（清单与语料不同步的信号）。
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
    src_uids = set()          # 输入出现过的全部 uid（用于唯一性断言）
    total_records = 0
    banned_hit_uids = set()   # 实际命中的 uid
    banned_missing = set()    # 清单里有、输入里没有的 uid
    skipped_files = []

    for sub in ("qa", "statutes"):
        sdir = os.path.join(src, sub)
        if not os.path.isdir(sdir):
            continue
        ddir = os.path.join(dst, sub)
        os.makedirs(ddir, exist_ok=True)
        for fn in sorted(os.listdir(sdir)):
            # 跳过合并副本 `_all.jsonl` 与调试残留 `*.sample.jsonl`
            if not fn.endswith(".jsonl") or fn.startswith("_") or ".sample." in fn:
                skipped_files.append(f"{sub}/{fn}")
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
                    total_records += 1
                    u = r.get("uid")
                    if u:
                        src_uids.add(u)
                    if u in banned:
                        removed += 1
                        banned_hit_uids.add(u)
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
    banned_missing = banned - src_uids

    # ---------------- 硬断言 1：uid 全局唯一 ----------------
    print(f"\n输入记录 {total_records} 条 / 唯一 uid {len(src_uids)} 个"
          f"（uid 重复 {total_records - len(src_uids)} 条）", flush=True)
    if total_records != len(src_uids):
        print("!! uid 不唯一 → 「按 uid 剔除」会整组连坐，拒绝继续。"
              "请先修阶段 2 的 uid 生成（须含 source_file）并重跑。", flush=True)
        return 3

    # ---------------- 硬断言 2：剔除条数恒等式 ----------------
    expect_removed = len(banned_hit_uids)
    print(f"剔除清单 {len(banned)} 个 uid；命中 {expect_removed} 个；"
          f"清单里有但输入中不存在 {len(banned_missing)} 个", flush=True)
    print(f"实际剔除记录 {total_removed} 条（恒等式要求 == {expect_removed}）", flush=True)
    if total_removed != expect_removed:
        print("!! 恒等式不成立：剔除条数 != 命中 uid 数 → uid 不唯一，存在整组连坐误删。",
              flush=True)
        return 3

    # 断言：输出中不得残留任何被剔除 uid（只查正式产出文件）
    residual = 0
    dst_stale = []
    for sub in ("qa", "statutes"):
        ddir = os.path.join(dst, sub)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".jsonl"):
                continue
            if fn.startswith("_") or ".sample." in fn:
                dst_stale.append(f"{sub}/{fn}")
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
    if dst_stale:
        print(f"\n!! 警告：输出目录存在本轮未产出的 .jsonl（未参与断言，建议移出）："
              f"{dst_stale}", flush=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uids_file": a.uids,
        "banned_uids": len(banned),
        "banned_hit_uids": len(banned_hit_uids),
        "banned_missing_uids": len(banned_missing),
        "src": src, "dst": dst,
        "skipped_input_files": skipped_files,
        "stale_files_in_dst": dst_stale,
        "input_records": total_records,
        "input_unique_uids": len(src_uids),
        "uid_duplicates": total_records - len(src_uids),
        "uid_unique_assert": total_records == len(src_uids),
        "removal_identity_assert": total_removed == expect_removed,
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
