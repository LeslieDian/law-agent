#!/usr/bin/env bash
# =============================================================================
# 阶段 2b：法条切条 + 质检门禁（幂等包装）
# =============================================================================
# 用法：
#   bash scripts/corpus/prepare_split_statutes.sh                 # 全量
#   bash scripts/corpus/prepare_split_statutes.sh --max-docs 80   # 冒烟（只切前 80 部）
#   bash scripts/corpus/prepare_split_statutes.sh --types ALL     # 不过滤类型（不推荐）
#
# 分工：
#   split_statutes.py       切条（只读 data/corpus/decontaminated/statutes）
#   verify_statute_items.py 质检门禁 C1–C7，verdict 必须 PASS
#
# ★ 2026-09-18 加固：把质检从「手工另跑」并入本包装脚本 —— 切条与质检必须成对，
#   否则会出现「切完了但没质检」的静默漏检。
# =============================================================================
set -euo pipefail

ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$ROOT/docs/corpus"

mkdir -p "$DOCS"

echo "=== [1/4] 语法自检 ==="
"$PY" -m py_compile "$HERE/split_statutes.py"
"$PY" -m py_compile "$HERE/verify_statute_items.py"

echo "=== [2/4] 跑法条切条 ==="
"$PY" "$HERE/split_statutes.py" \
  --root "$ROOT" \
  --out-json "$DOCS/statute_items_STATS.json" \
  --out-md   "$DOCS/STATUTE_SPLIT_REPORT.md" \
  "$@"

echo "=== [3/4] 质检门禁（C1–C7） ==="
"$PY" "$HERE/verify_statute_items.py" \
  --root "$ROOT" \
  --out-json "$DOCS/statute_items_VERIFY.json" \
  --out-md   "$DOCS/statute_items_VERIFY.md"

echo "=== [4/4] 结论 ==="
"$PY" - <<'PYEOF'
import json, os, sys
docs = os.environ.get("ROOT", "/mnt/data/lidian/law-agent") + "/docs/corpus"
s = json.load(open(os.path.join(docs, "statute_items_STATS.json"), encoding="utf-8"))
v = json.load(open(os.path.join(docs, "statute_items_VERIFY.json"), encoding="utf-8"))
d = s["docs"]["twang2218__chinese-law-and-regulations"]
print("kept_types   =", s["keep_types"])
print("docs_kept    =", d["type_kept"], "=> sum", sum(d["type_kept"].values()))
print("docs_dropped =", d["type_dropped"])
print("items_total  =", s["items"]["total"], "(doc-level", d["items_out"], ")")
print("by_domain    =", s["items"]["domain_items"])
print("status_dirty =", s["items"]["status_dirty_dist"])
print("status_after =", s["items"]["status_after_norm"])
print("char_stats   =", s["items"]["char_stats"])
print("dup_uids     =", s["items"]["duplicate_uids"], "(被跳过的重复条号数；产物 uid 唯一)")
print("diag         =", d["diagnostics"])
print("C1_schema    =", {k: x for k, x in v["C1_schema"].items() if k != "dup_uid_samples"})
print("VERDICT      =", v["verdict"])
for b in v.get("blockers", []):
    print("BLOCKER:", b)
if v["verdict"] != "PASS":
    sys.exit(3)
PYEOF

echo "DONE_PREPARE_SPLIT_STATUTES"
