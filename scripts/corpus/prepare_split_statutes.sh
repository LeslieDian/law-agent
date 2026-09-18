#!/usr/bin/env bash
# 阶段 2b：法条切条（幂等包装）。语法自检 -> 跑 split_statutes.py -> 打印结论。
# 用法：
#   bash scripts/corpus/prepare_split_statutes.sh                 # 全量
#   bash scripts/corpus/prepare_split_statutes.sh --max-docs 80   # 冒烟（只切前 80 部）
#   bash scripts/corpus/prepare_split_statutes.sh --types ALL     # 不过滤类型（不推荐）
set -euo pipefail

ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$ROOT/docs/corpus"

mkdir -p "$DOCS"

echo "=== [1/3] 语法自检 ==="
"$PY" -m py_compile "$HERE/split_statutes.py"

echo "=== [2/3] 跑法条切条 ==="
"$PY" "$HERE/split_statutes.py" \
  --root "$ROOT" \
  --out-json "$DOCS/statute_items_STATS.json" \
  --out-md   "$DOCS/STATUTE_SPLIT_REPORT.md" \
  "$@"

echo "=== [3/3] 结论 ==="
"$PY" - <<'PYEOF'
import json, os
p = os.path.join(os.environ.get("ROOT", "/mnt/data/lidian/law-agent"),
                 "docs/corpus/statute_items_STATS.json")
r = json.load(open(p, encoding="utf-8"))
d = r["docs"]["twang2218__chinese-law-and-regulations"]
print("kept_types   =", r["keep_types"])
print("docs_kept    =", d["type_kept"], "=> sum", sum(d["type_kept"].values()))
print("docs_dropped =", d["type_dropped"])
print("items_total  =", r["items"]["total"], "(doc-level", d["items_out"], ")")
print("by_domain    =", r["items"]["domain_items"])
print("status_dirty =", r["items"]["status_dirty_dist"])
print("status_after =", r["items"]["status_after_norm"])
print("char_stats   =", r["items"]["char_stats"])
print("diag         =", d["diagnostics"])
print("dup_uids     =", r["items"]["duplicate_uids"])
PYEOF

echo "DONE_PREPARE_SPLIT_STATUTES"
