"""数据固化：对 data/raw 下文件计算 SHA-256、统计条数、校验数量并更新 manifest。

用法：
    python -m src.prepare.register_data \
        --statutes data/raw/claw_statutes.jsonl \
        --cases    data/raw/claw_cases.jsonl

原则：data/raw 只读，本脚本只读取不修改；manifest 写回 docs/data_manifest.json。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io import (
    load_json,
    project_path,
    read_jsonl,
    save_json,
    sha256_file,
)

MANIFEST_PATH = project_path("docs", "data_manifest.json")


def count_records(path: Path) -> int:
    """统计 JSONL 记录数；非 JSONL 返回 -1。"""
    if path.suffix != ".jsonl":
        return -1
    return sum(1 for _ in read_jsonl(path))


def register(file_arg: str, kind: str, manifest: dict) -> dict:
    path = Path(file_arg)
    if not path.is_absolute():
        path = project_path(file_arg)
    if not path.exists():
        raise FileNotFoundError(f"原始数据不存在: {path}")

    digest = sha256_file(path)
    records = count_records(path)
    size_mb = round(path.stat().st_size / (1 << 20), 2)

    entry = next((f for f in manifest["files"] if f["kind"] == kind), None)
    if entry is None:
        entry = {"kind": kind}
        manifest["files"].append(entry)

    entry.update(
        {
            "file": str(path.relative_to(project_path())).replace("\\", "/"),
            "actual_records": records if records >= 0 else None,
            "size_mb": size_mb,
            "sha256": digest,
            "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    print(f"[OK] {kind}: {entry['file']}  records={records}  sha256={digest[:16]}…  {size_mb}MB")
    return entry


def verify(manifest: dict) -> bool:
    """对照 expected_counts 校验实际条数。"""
    statutes_entry = next((f for f in manifest["files"] if f["kind"] == "statutes"), None)
    cases_entry = next((f for f in manifest["files"] if f["kind"] == "cases"), None)

    ok = True
    if statutes_entry and statutes_entry.get("actual_records") is not None:
        actual = statutes_entry["actual_records"]
        expected = manifest["expected_counts"]["provisions"]
        if actual != expected:
            print(f"[WARN] 法条条数不符：期望 {expected}，实际 {actual}")
            ok = False
        manifest["verified_counts"]["provisions"] = actual

    if cases_entry and cases_entry.get("actual_records") is not None:
        actual = cases_entry["actual_records"]
        expected = manifest["expected_counts"]["cases"]
        if actual != expected:
            print(f"[WARN] 案例数不符：期望 {expected}，实际 {actual}")
            ok = False
        manifest["verified_counts"]["cases"] = actual

    manifest["verification_status"] = "PASS" if ok else "MISMATCH"
    manifest["last_updated"] = dt.datetime.now().date().isoformat()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="CLaw 原始数据固化与校验")
    parser.add_argument("--statutes", help="法条数据路径（JSONL）")
    parser.add_argument("--cases", help="案例数据路径（JSONL）")
    parser.add_argument("--prompts", help="官方提示词文件路径")
    parser.add_argument("--judge-prompt", help="官方 Judge 提示词路径")
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)

    if args.statutes:
        register(args.statutes, "statutes", manifest)
    if args.cases:
        register(args.cases, "cases", manifest)
    if args.prompts:
        register(args.prompts, "prompts", manifest)
    if args.judge_prompt:
        register(args.judge_prompt, "judge_prompt", manifest)

    ok = verify(manifest)
    save_json(MANIFEST_PATH, manifest)
    print(f"\nmanifest 已更新: {MANIFEST_PATH}")
    print(f"校验结果: {manifest['verification_status']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
