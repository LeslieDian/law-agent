#!/usr/bin/env bash
# =============================================================================
# 阶段 2c 幂等包装：程序法条文任务派生（补程序法域任务多样性）
# =============================================================================
# 用法：
#   bash scripts/corpus/prepare_derive_statute_tasks.sh
#   bash scripts/corpus/prepare_derive_statute_tasks.sh --types statute_recall
#
# 前置：阶段 2b 已产出 data/corpus/statute_items/*.items.jsonl
# 产出：data/corpus/derived/statute_tasks.jsonl
#       docs/corpus/DERIVE_STATS.json + docs/corpus/DERIVE_REPORT.md
# 只读：不修改 statute_items 下任何文件
# =============================================================================
set -euo pipefail

export ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
export PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCS="$ROOT/docs/corpus"
export ITEMS="$ROOT/data/corpus/statute_items"

mkdir -p "$DOCS"

echo "=== 阶段 2c：程序法条文任务派生 ==="
date '+%Y-%m-%d %H:%M:%S'
echo "参数: $*"
echo

echo "=== [1/4] 语法自检 ==="
"$PY" -m py_compile "$HERE/derive_statute_tasks.py"

echo "=== [2/4] 前置检查：法条条目存在且非空 ==="
[ -d "$ITEMS" ] || { echo "!! 缺少 $ITEMS（先跑阶段 2b）"; exit 1; }
ls -l "$ITEMS"
"$PY" - <<'PYEOF'
import glob, os
items = os.environ["ITEMS"]
fs = sorted(glob.glob(os.path.join(items, "*.items.jsonl")))
assert fs, "没有 *.items.jsonl（先跑阶段 2b）"
tot = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in fs)
assert tot > 0, "法条条目为空"
print("items_files=%d items_rows=%d" % (len(fs), tot))
PYEOF

echo "=== [3/4] 派生 ==="
"$PY" "$HERE/derive_statute_tasks.py" \
  --root "$ROOT" \
  --out-json "$DOCS/DERIVE_STATS.json" \
  --out-md   "$DOCS/DERIVE_REPORT.md" \
  "$@"

echo "=== [4/4] 结论 ==="
"$PY" - <<'PYEOF'
import json, os
p = os.path.join(os.environ["DOCS"], "DERIVE_STATS.json")
r = json.load(open(p, encoding="utf-8"))
print("verdict      =", r["verdict"])
print("by_task      =", r["output"]["by_task"])
print("total        =", r["output"]["total"])
print("unique sha1  =", r["output"]["unique_content_sha1"])
print("unique uid   =", r["output"]["unique_uid"])
print("verbatim_bad =", r["verbatim_check"]["mismatch"])
print("nested 书名号 =", r["title_wrap_check"]["nested_brackets"])
print("out          =", r["output"]["path"])
PYEOF

echo DONE_PREPARE_DERIVE_STATUTE_TASKS
