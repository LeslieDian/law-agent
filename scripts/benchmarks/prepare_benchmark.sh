#!/usr/bin/env bash
# 评测基准的"拉取 -> 验完整性 -> 登记 SHA-256"标准流程。
#
# 用法：
#   bash scripts/benchmarks/prepare_benchmark.sh lexrubric
#   bash scripts/benchmarks/prepare_benchmark.sh lexeval
#
# 原则：
#   1. 一个数据集一份 MANIFEST.json + 一份 VERIFY_REPORT.json，绝不跨数据集合并。
#   2. 反复执行幂等：清单与报告整体重写，不追加。
#   3. 源信息（repo / commit / license / paper）在下面的 CASE 里声明，改这里即可。
set -euo pipefail

DATASET="${1:-}"
[[ -z "$DATASET" ]] && { echo "用法: $0 <lexrubric|lexeval>"; exit 2; }

PROJ="${LAW_AGENT_ROOT:-/mnt/data/lidian/law-agent}"
PY="$PROJ/envs/main/bin/python"
TOOL="$(cd "$(dirname "$0")" && pwd)"
BASE="$PROJ/data/benchmark/$DATASET"
REPO_DIR="$BASE/repo"
MANIFEST="$BASE/MANIFEST.json"
REPORT="$BASE/VERIFY_REPORT.json"

case "$DATASET" in
  lexrubric)
    SRC_REPO="https://github.com/foggpoy/LexRubric"
    SRC_REF="main"
    LICENSE="MIT"
    PAPER="arXiv:2606.09389"
    VERIFY_ARG="--lexrubric"
    ;;
  lexeval)
    SRC_REPO="https://github.com/CSHaitao/LexEval"
    SRC_REF="main"
    LICENSE="MIT"
    PAPER="arXiv:2409.20288"
    VERIFY_ARG="--lexeval"
    ;;
  *)
    echo "[FATAL] 未登记的数据集: $DATASET（请在本脚本 CASE 中补充）"; exit 2 ;;
esac

[[ -d "$REPO_DIR" ]] || { echo "[FATAL] 源目录不存在: $REPO_DIR（请先跑 fetch_repos.py 拉取）"; exit 2; }

echo "===== 1/2 生成 SHA-256 清单: $DATASET ====="
# 若已有清单，先继承其中人工登记的 meta，避免重生成时丢失
META_FROM_ARGS=()
if [[ -f "$MANIFEST" ]]; then
  cp "$MANIFEST" "/tmp/${DATASET}.manifest.bak.json"
  META_FROM_ARGS=(--meta-from "/tmp/${DATASET}.manifest.bak.json")
fi

"$PY" "$TOOL/build_manifest.py" \
  --dataset "$DATASET" \
  --root "$REPO_DIR" \
  --out "$MANIFEST" \
  --meta "source_repo=$SRC_REPO" \
  --meta "source_ref=$SRC_REF" \
  --meta "license=$LICENSE" \
  --meta "paper=$PAPER" \
  "${META_FROM_ARGS[@]}"

echo
echo "===== 2/2 结构与完整性校验: $DATASET ====="
"$PY" "$TOOL/verify_benchmarks.py" "$VERIFY_ARG" "$BASE" --out-report "$REPORT"

echo
echo "[DONE] $DATASET"
echo "  清单: $MANIFEST"
echo "  报告: $REPORT"
echo "  拉回仓库: docs/benchmarks/${DATASET}.MANIFEST.json / ${DATASET}.VERIFY_REPORT.json"
