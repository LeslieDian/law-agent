#!/usr/bin/env bash
# =============================================================================
# 阶段 2 幂等包装：归一化 + 域打标
# =============================================================================
# 用法：
#   bash scripts/corpus/prepare_normalize.sh --list
#   bash scripts/corpus/prepare_normalize.sh --limit-per-file 300 --suffix .sample
#   bash scripts/corpus/prepare_normalize.sh
#
# 分工：
#   configs/corpus_sources.yaml   ← 合规层（能不能采、license、体积、等级）
#   configs/corpus_adapters.yaml  ← 解析层（怎么读、进哪条流、阈值与权重）
#   raw 只读：本脚本不修改 data/corpus/raw 下任何文件
# 产出：data/corpus/normalized/{qa,statutes}/*.jsonl + STATS.json + DEDUP_REPORT.json
# =============================================================================
set -euo pipefail

ROOT=/mnt/data/lidian/law-agent
cd "$ROOT"

PY="$ROOT/envs/main/bin/python"
[ -x "$PY" ] || { echo "!! 找不到 $PY"; exit 1; }

echo "=== 阶段 2：归一化 + 域打标 ==="
date '+%Y-%m-%d %H:%M:%S'
echo "参数: $*"
echo

# 语法自检（防止上传截断）
"$PY" - <<'PYEOF'
import ast, sys
p = "scripts/corpus/normalize_corpus.py"
ast.parse(open(p, encoding="utf-8").read())
print("SYNTAX_OK", p)
PYEOF

# YAML 自检
"$PY" - <<'PYEOF'
import yaml
for p in ("configs/corpus_adapters.yaml", "configs/corpus_sources.yaml"):
    d = yaml.safe_load(open(p, encoding="utf-8"))
    print("YAML_OK", p, "| top keys:", list(d.keys()))
PYEOF

echo
echo "--- raw 现状（只读核对） ---"
du -sh data/corpus/raw
find data/corpus/raw -maxdepth 1 -type d | sort

echo
"$PY" scripts/corpus/normalize_corpus.py "$@"

echo
echo "--- 产出 ---"
ls -la data/corpus/normalized data/corpus/normalized/qa data/corpus/normalized/statutes 2>/dev/null || true
du -sh data/corpus/normalized 2>/dev/null || true

echo MARKER_NORMALIZE_DONE
