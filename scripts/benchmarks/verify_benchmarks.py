#!/usr/bin/env python3
"""两个基准数据集的结构与完整性校验（分别校验，不互相干扰）。

校验内容：
  1. 文件是否都存在、能否解析为 JSON
  2. 记录条数与论文自述规模是否一致
  3. 必需字段是否存在
  4. 关键字段的分布（rubric 维度、rubric 条数、长度等）
  5. 明显的损坏信号（空记录、重复 id、rubric 全零分等）

用法：
  python verify_benchmarks.py \
      --lexrubric /mnt/data/lidian/law-agent/data/benchmark/lexrubric \
      --lexeval   /mnt/data/lidian/law-agent/data/benchmark/lexeval \
      --out-report /mnt/data/lidian/law-agent/data/benchmark/VERIFY_REPORT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

FAILS: list[str] = []
WARNS: list[str] = []


def note_fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def note_warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  [WARN] {msg}")


def load_json(p: Path) -> Any:
    """读 JSON；若失败则回退按 JSONL 解析。

    ⚠️ 实测发现 LexEval 的 data/*.json 其实是 **JSONL**（每行一个对象），
    扩展名有误导性，必须两种都试。
    """
    txt = p.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        pass
    recs = []
    bad = 0
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:  # noqa: BLE001
            bad += 1
    if recs:
        if bad:
            note_warn(f"{p.name} 按 JSONL 解析成功 {len(recs)} 行，但有 {bad} 行无法解析")
        return recs
    # 两种都不行 → 报错
    try:
        json.loads(txt)
    except Exception as e:  # noqa: BLE001
        note_fail(f"{p.name} 既非合法 JSON 也非合法 JSONL: {type(e).__name__}: {e}")
    return None


def as_records(obj: Any) -> list[dict]:
    """把各种可能的外层结构统一成 record 列表。"""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "instances", "records", "items", "examples"):
            v = obj.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # 形如 {"0": {...}, "1": {...}}
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    return []


def describe_schema(recs: list[dict], name: str) -> dict:
    keys: Counter = Counter()
    for r in recs[:500]:
        keys.update(r.keys())
    print(f"  {name}: {len(recs)} 条；字段出现频次前 12 -> {keys.most_common(12)}")
    return {"n": len(recs), "top_keys": dict(keys.most_common(20))}


# --------------------------------------------------------------------------- #
# LexRubric
# --------------------------------------------------------------------------- #

def verify_lexrubric(root: Path) -> dict:
    print("\n=== LexRubric ===")
    repo = root / "repo"
    res: dict = {"root": str(root), "files": {}, "expect": {}}

    expected_files = {
        "code/eval.py": None,
        "code/prompt.txt": None,
        "data/falvzixun.json": None,   # 法律咨询
        "data/sifakaoshi.json": None,  # 司法考试
        "README.md": None,
        "LICENSE": None,
    }
    for rel in expected_files:
        p = repo / rel
        ok = p.exists() and p.stat().st_size > 0
        res["files"][rel] = p.stat().st_size if p.exists() else 0
        if not ok:
            note_fail(f"LexRubric 缺少文件: {rel}")
        else:
            print(f"  [ok] {rel}  {p.stat().st_size} B")

    # --- 数据 ---
    consult = load_json(repo / "data/falvzixun.json")
    exam = load_json(repo / "data/sifakaoshi.json")

    total = 0
    all_recs: list[dict] = []
    per_split: dict[str, list[dict]] = {}
    for label, obj, expect in (("法律咨询 falvzixun", consult, 473),
                               ("司法考试 sifakaoshi", exam, 176)):
        recs = as_records(obj) if obj is not None else []
        res["expect"][label] = {"n": len(recs), "expected": expect}
        print(f"  {label}: {len(recs)} 条（论文自述 {expect}）")
        if expect and abs(len(recs) - expect) > 0:
            note_warn(f"{label} 条数 {len(recs)} != 论文自述 {expect}")
        per_split[label] = recs
        total += len(recs)
        all_recs.extend(recs)

    res["expect"]["total"] = {"n": total, "expected": 649}
    print(f"  合计: {total} 条（论文自述 649）")
    if total != 649:
        note_warn(f"LexRubric 总条数 {total} != 论文自述 649")

    if all_recs:
        res["schema"] = describe_schema(all_recs, "LexRubric")

        # 必需字段。⚠️ 实测字段名是大写 ID（README 示例里写的是小写 id，不可信）
        def pick(r: dict, *names):
            for n in names:
                if n in r:
                    return n
            return None

        id_key = pick(all_recs[0], "ID", "id")
        if not id_key:
            note_fail("LexRubric 记录中找不到 ID / id 字段")
        else:
            print(f"  [ok] ID 字段名为 '{id_key}'")
        res["id_field"] = id_key

        for cand in (("question",), ("answer",), ("rubrics",)):
            if not any(c in all_recs[0] for c in cand):
                note_fail(f"LexRubric 记录缺少字段 {cand}")
        print(f"  [ok] 记录字段: {list(all_recs[0].keys())}")

        # rubric 条目字段（实测 criterion / point / dimension）
        first_rub = next((r["rubrics"][0] for r in all_recs
                          if isinstance(r.get("rubrics"), list) and r["rubrics"]), None)
        if isinstance(first_rub, dict):
            res["rubric_item_fields"] = list(first_rub.keys())
            print(f"  [ok] rubric 条目字段: {list(first_rub.keys())}")
            for want in ("criterion", "point", "dimension"):
                if want not in first_rub:
                    note_fail(f"rubric 条目缺少字段 '{want}'")
        else:
            note_fail("LexRubric 找不到可用的 rubric 条目样例")

        # id 唯一性 —— ★ 必须按 split 分别判定。
        # 实测发现 falvzixun 与 sifakaoshi **共享 ID 111 和 195**，
        # 因此任何「把两个文件合并后按 ID 建索引」的做法都会静默丢数据。
        res["id_uniqueness"] = {}
        ids_by_split: dict[str, set] = {}
        for label, recs in per_split.items():
            ids = [str(r.get(id_key)) for r in recs if id_key and id_key in r]
            ids_by_split[label] = set(ids)
            dup = [k for k, v in Counter(ids).items() if v > 1]
            res["id_uniqueness"][label] = {"n": len(ids), "unique": len(set(ids)),
                                           "duplicates": dup[:10]}
            if dup:
                note_fail(f"{label} 内部存在重复 id {len(dup)} 个: {dup[:5]}")
            else:
                print(f"  [ok] {label}: {len(ids)} 个 id 内部全部唯一")

        labels = list(ids_by_split)
        if len(labels) == 2:
            inter = sorted(ids_by_split[labels[0]] & ids_by_split[labels[1]])
            res["cross_split_id_collision"] = inter
            if inter:
                print(f"  [note] 两个 split **共享 {len(inter)} 个 ID**: {inter}")
                print("         → 评测键必须写成 (split, ID)，否则会互相覆盖")
            else:
                print("  [ok] 两个 split 的 ID 无交集")

        # rubric 统计
        n_items = [len(r.get("rubrics") or []) for r in all_recs]
        total_items = sum(n_items)
        empty_rub = sum(1 for n in n_items if n == 0)
        res["rubrics"] = {
            "total_items": total_items,
            "expected_total_items": 12337,
            "mean_items": round(sum(n_items) / len(n_items), 2) if n_items else 0,
            "min_items": min(n_items) if n_items else 0,
            "max_items": max(n_items) if n_items else 0,
            "records_with_no_rubric": empty_rub,
        }
        print(f"  rubric 细则总数: {total_items}（论文自述 12,337）")
        print(f"  rubric 条数/题: 平均 {res['rubrics']['mean_items']}, "
              f"范围 [{res['rubrics']['min_items']}, {res['rubrics']['max_items']}]")
        if total_items != 12337:
            note_warn(f"rubric 总数 {total_items} != 论文自述 12,337"
                      f"（差 {total_items - 12337}，差额位置见下方逐维度对账）")
        if empty_rub:
            note_fail(f"有 {empty_rub} 条记录的 rubrics 为空")

        # 维度分布
        # 维度分布与分值分布
        # ⚠️ 字段名是单数 'point'（README 示例里误写成 'points'）
        dims: Counter = Counter()
        pts_pos = pts_neg = pts_zero = 0
        pt_values: Counter = Counter()
        for r in all_recs:
            for it in (r.get("rubrics") or []):
                if not isinstance(it, dict):
                    continue
                dims[str(it.get("dimension", ""))] += 1
                try:
                    p = float(it.get("point", it.get("points", 0)) or 0)
                except Exception:  # noqa: BLE001
                    p = 0.0
                pt_values[p] += 1
                if p > 0:
                    pts_pos += 1
                elif p < 0:
                    pts_neg += 1
                else:
                    pts_zero += 1
        res["dimensions"] = dict(dims.most_common())
        res["points"] = {
            "positive_items": pts_pos,
            "negative_items": pts_neg,
            "zero_items": pts_zero,
            "distinct_values": len(pt_values),
        }
        print(f"  维度分布: {dims.most_common()}")
        print(f"  正分项 {pts_pos} / 负分项 {pts_neg} / 零分项 {pts_zero}")
        if pts_neg == 0:
            note_fail("未发现任何负分 rubric 项（论文明确存在负分项）")
        if pts_zero:
            note_warn(f"存在 {pts_zero} 条 0 分 rubric 项（对总分无影响）")
        # 满分上限合计（用于归一化）
        total_max = sum(v * n for v, n in pt_values.items() if v > 0)
        total_min = sum(v * n for v, n in pt_values.items() if v < 0)
        res["score_range_sum"] = {"sum_positive": total_max, "sum_negative": total_min}
        print(f"  全部题目的正分合计 {total_max:.1f} / 负分合计 {total_min:.1f}")

        # 与论文表格逐维度对账（论文给出的是两个 split 的合计）
        PAPER_DIMS = {
            "法律准确性": 3947,
            "逻辑推理与分析": 4151,
            "全面性": 1505,
            "表达与结构": 593,
            "指令/题意遵从": 1641,
            "伦理安全": 500,
        }
        res["dimension_check"] = {}
        for d, exp in PAPER_DIMS.items():
            got = dims.get(d, 0)
            res["dimension_check"][d] = {"got": got, "paper": exp, "delta": got - exp}
            if got != exp:
                note_warn(f"维度「{d}」实测 {got} != 论文 {exp}（差 {got - exp}）")
        if all(v["delta"] == 0 for v in res["dimension_check"].values()):
            print("  [ok] 六个维度的细则条数与论文表格完全一致")

    return res


# --------------------------------------------------------------------------- #
# LexEval
# --------------------------------------------------------------------------- #

LEXEVAL_TASKS = [
    "1_1", "1_2", "1_3",
    "2_1", "2_2", "2_3", "2_4", "2_5",
    "3_1", "3_2", "3_3", "3_4", "3_5", "3_6",
    "4_1", "4_2",
    "5_1", "5_2", "5_3", "5_4",
    "6_1", "6_2", "6_3",
]


def verify_lexeval(root: Path) -> dict:
    print("\n=== LexEval ===")
    repo = root / "repo"
    res: dict = {"root": str(root), "tasks": {}, "expect": {}}

    for rel in ("code/evaluation/evaluate.py", "code/main.py", "README.md", "LICENSE"):
        p = repo / rel
        if p.exists() and p.stat().st_size > 0:
            print(f"  [ok] {rel}  {p.stat().st_size} B")
        else:
            note_fail(f"LexEval 缺少文件: {rel}")

    data_dir = repo / "data"
    if not data_dir.is_dir():
        note_fail("LexEval 缺少 data/ 目录")
        return res

    # 排除 model_output 是刻意为之，这里显式记录
    mo = repo / "model_output"
    res["model_output_present"] = mo.is_dir()
    if not mo.is_dir():
        print("  [note] model_output/ 已按计划排除（占整仓 3.1GB 的绝大部分，"
              "本次仅需 data/ + code/）")

    grand_total = 0
    found_tasks = []
    for t in LEXEVAL_TASKS:
        p = data_dir / f"{t}.json"
        if not p.exists():
            note_fail(f"LexEval 缺少任务文件 {t}.json")
            continue
        obj = load_json(p)
        recs = as_records(obj) if obj is not None else []
        # few-shot 示例通常是对象而非题目列表，注意不要混入计数
        n = len(recs)
        found_tasks.append(t)
        res["tasks"][t] = {"n": n, "size": p.stat().st_size}
        grand_total += n
        print(f"  {t}: {n:>6} 条  ({p.stat().st_size:>9} B)")

    res["expect"]["tasks_found"] = len(found_tasks)
    res["expect"]["tasks_expected"] = 23
    res["expect"]["total_questions"] = grand_total
    res["expect"]["total_expected"] = 14150
    print(f"  任务数 {len(found_tasks)}/23，题目合计 {grand_total}（论文自述 14,150）")

    if len(found_tasks) != 23:
        note_fail(f"LexEval 任务文件数 {len(found_tasks)} != 23")
    if grand_total != 14150:
        note_warn(f"LexEval 题目合计 {grand_total} != 论文自述 14,150"
                  "（差额可能来自 few-shot 文件或被合并的题目）")

    # 抽样看一条记录的结构
    if found_tasks:
        p = data_dir / f"{found_tasks[0]}.json"
        recs = as_records(load_json(p) or [])
        if recs:
            res["sample_keys"] = list(recs[0].keys())
            print(f"  样例字段 ({found_tasks[0]}): {list(recs[0].keys())}")

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexrubric", default="")
    ap.add_argument("--lexeval", default="")
    ap.add_argument("--out-report", default="")
    a = ap.parse_args()

    report: dict = {"tool": "verify_benchmarks", "datasets": {}}

    if a.lexrubric:
        p = Path(a.lexrubric)
        report["datasets"]["LexRubric"] = verify_lexrubric(p) if p.is_dir() \
            else note_fail(f"LexRubric 目录不存在: {p}") or {}
    if a.lexeval:
        p = Path(a.lexeval)
        report["datasets"]["LexEval"] = verify_lexeval(p) if p.is_dir() \
            else note_fail(f"LexEval 目录不存在: {p}") or {}

    report["failures"] = FAILS
    report["warnings"] = WARNS
    report["verdict"] = "PASS" if not FAILS else "FAIL"

    print("\n" + "=" * 62)
    print(f"校验结论: {report['verdict']}   "
          f"失败 {len(FAILS)} 项 / 警告 {len(WARNS)} 项")
    for f in FAILS:
        print(f"  FAIL: {f}")
    for w in WARNS:
        print(f"  WARN: {w}")
    print("=" * 62)

    if a.out_report:
        op = Path(a.out_report)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告 -> {op}")

    print("MARKER_VERIFY_DONE")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
