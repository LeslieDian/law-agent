#!/usr/bin/env python3
"""为单个基准数据集生成独立的 SHA-256 清单（MANIFEST.json）。

设计原则：**一个数据集一份清单，绝不跨数据集合并哈希**。
LexRubric 与 LexEval 各跑一次，各自落到自己的目录里。

用法：
  python build_manifest.py \
      --dataset LexRubric \
      --root /mnt/data/lidian/law-agent/data/benchmarks/lexrubric/repo \
      --out  /mnt/data/lidian/law-agent/data/benchmarks/lexrubric/MANIFEST.json \
      --meta source_repo=https://github.com/foggpoy/LexRubric \
      --meta license=MIT \
      --meta paper=arXiv:2606.09389
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}
SKIP_EXT = {".pyc", ".pyo", ".so", ".o"}

# 文本/数据类扩展名，用于统计
DATA_EXT = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".parquet", ".xlsx"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel_ok(p: Path) -> bool:
    parts = set(p.parts)
    if parts & SKIP_DIRS:
        return False
    if p.suffix.lower() in SKIP_EXT:
        return False
    return True


def git_info(root: Path) -> dict:
    info: dict = {}
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            info["commit_hash"] = r.stdout.strip()
        r = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%cI|%s"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            commit_date, _, subject = r.stdout.strip().partition("|")
            info["commit_date"] = commit_date
            info["commit_subject"] = subject
    except Exception as e:  # noqa: BLE001
        info["git_error"] = str(e)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="数据集名称，如 LexRubric")
    ap.add_argument("--root", required=True, help="要扫描的根目录")
    ap.add_argument("--out", required=True, help="MANIFEST.json 输出路径")
    ap.add_argument("--meta", action="append", default=[], metavar="k=v",
                    help="附加元数据，可重复，如 --meta license=MIT")
    ap.add_argument("--with-hash", action="store_true", default=True,
                    help="计算 SHA-256（默认开启）")
    ap.add_argument("--source-record", default="",
                    help="fetch_repos.py 生成的 source.json；若存在则并入 commit_sha 等溯源信息")
    ap.add_argument("--meta-from", default="",
                    help="从既有 MANIFEST.json 继承元数据键（结构化键自动跳过），"
                         "用于在升级脚本后原样重生成清单、不丢人工登记的 meta")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if not root.is_dir():
        print(f"[FATAL] root 不是目录: {root}", file=sys.stderr)
        return 2

    meta: dict = {}
    for kv in a.meta:
        k, _, v = kv.partition("=")
        if k:
            meta[k.strip()] = v.strip()

    # 从既有清单继承人工登记过的元数据（如 license / paper / venue / note）
    STRUCTURAL = {
        "dataset", "schema_version", "retrieved_at_utc", "root_scanned",
        "hash_algorithm", "summary", "files",
        "commit_hash", "commit_date", "commit_subject",
        "provenance", "provenance_error", "fetch_summary", "excluded_paths_sample",
    }
    if a.meta_from:
        mf = Path(a.meta_from)
        if mf.exists():
            try:
                prev = json.loads(mf.read_text(encoding="utf-8"))
                for k, v in prev.items():
                    if k in STRUCTURAL:
                        continue
                    meta.setdefault(k, v)
                print(f"[meta-from] 继承自 {mf}")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] --meta-from 读取失败: {type(e).__name__}: {e}")
        else:
            print(f"[WARN] --meta-from 不存在: {mf}")

    files: list[dict] = []
    total_bytes = 0
    ext_counter: dict[str, int] = {}
    skipped: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if not p.is_file() or not rel_ok(p):
                skipped.append(p.relative_to(root).as_posix())
                continue
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size
            entry = {"path": rel, "size": size}
            if a.with_hash:
                entry["sha256"] = sha256_of(p)
            files.append(entry)
            total_bytes += size
            ext = p.suffix.lower() or "(noext)"
            ext_counter[ext] = ext_counter.get(ext, 0) + 1

    files.sort(key=lambda x: x["path"])

    manifest = {
        "dataset": a.dataset,
        "schema_version": "1.0",
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root_scanned": str(root),
        "hash_algorithm": "sha256",
        **meta,
        "summary": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "total_mib": round(total_bytes / 1048576, 3),
            "by_extension": dict(sorted(ext_counter.items(), key=lambda kv: -kv[1])),
            "excluded_by_rule": skipped,
            "excluded_rule_dirs": sorted(SKIP_DIRS),
            "excluded_rule_ext": sorted(SKIP_EXT),
        },
        "files": files,
    }
    manifest.update(git_info(root))

    # 非 git 下载（走 API+CDN）时没有 .git，从 source.json 并入 commit 溯源信息
    src_path = Path(a.source_record) if a.source_record else (Path(a.out).parent / "_archive" / "source.json")
    if "commit_hash" not in manifest and src_path.exists():
        try:
            src = json.loads(src_path.read_text(encoding="utf-8"))
            if src.get("commit_sha"):
                manifest["commit_hash"] = src["commit_sha"]
                manifest["commit_date"] = src.get("commit_date", "")
                manifest["commit_subject"] = src.get("commit_message", "")
                manifest["provenance"] = "api+cdn (no .git)"
            if src.get("repo"):
                manifest.setdefault("source_repo", f"https://github.com/{src['repo']}")
            if src.get("summary"):
                manifest["fetch_summary"] = src["summary"]
            if src.get("excluded_paths_sample"):
                manifest["excluded_paths_sample"] = src["excluded_paths_sample"][:50]
        except Exception as e:  # noqa: BLE001
            manifest["provenance_error"] = str(e)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {a.dataset}: {len(files)} 个文件, {manifest['summary']['total_mib']} MiB")
    print(f"     清单 -> {out}")
    for e in files[:15]:
        print(f"     {e['sha256'][:16]}...  {e['size']:>10}  {e['path']}")
    if len(files) > 15:
        print(f"     ... 其余 {len(files) - 15} 个文件见清单")
    return 0


if __name__ == "__main__":
    sys.exit(main())
