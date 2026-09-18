#!/usr/bin/env bash
# 阶段 3：双向去污（幂等包装）。语法自检 -> 跑 decontaminate.py -> 打印结论。
set -euo pipefail

ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$ROOT/docs/corpus"

mkdir -p "$DOCS"

echo "=== [1/3] 语法自检 ==="
"$PY" -m py_compile "$HERE/decontaminate.py"

echo "=== [2/3] 跑去污 ==="
"$PY" "$HERE/decontaminate.py" \
  --root "$ROOT" \
  --out-json "$DOCS/DECONTAMINATION_REPORT.json" \
  --out-md   "$DOCS/DECONTAMINATION_REPORT.md"

echo "=== [3/3] 结论 ==="
"$PY" - <<'PYEOF'
import json, os
p = os.path.join(os.environ.get("ROOT", "/mnt/data/lidian/law-agent"),
                 "docs/corpus/DECONTAMINATION_REPORT.json")
r = json.load(open(p, encoding="utf-8"))
print("verdict        =", r["verdict"])
print("blacklist      =", r["blacklist"]["total"], r["blacklist"]["by_tag"])
print("claw_loaded    =", r["blacklist"]["claw_loaded"])
print("checked_rows   =", r["checked"]["total_rows"])
print("exact_hits     =", r["exact_hits"]["count"])
print("near_hits      =", r["near_hits"]["count"])
print("same_source    =", {k: v for k, v in r["same_source_check"].items() if k != "samples"})
for b in r["blockers"]:
    print("BLOCKER:", b)
PYEOF

echo "DONE_PREPARE_DECONTAM"
