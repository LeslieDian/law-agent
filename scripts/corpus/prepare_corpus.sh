#!/usr/bin/env bash
# 训练语料「阶段 1：分批采集 + SHA-256 登记」标准流程。
#
# 用法：
#   bash scripts/corpus/prepare_corpus.sh --list          # 只看计划，不下载
#   bash scripts/corpus/prepare_corpus.sh                 # 采 A 级（默认）
#   bash scripts/corpus/prepare_corpus.sh --only ShengbinYue/DISC-Law-SFT
#   bash scripts/corpus/prepare_corpus.sh --grade A --threads 12
#
# 原则：
#   1. 数据集清单**只在 configs/corpus_sources.yaml 里维护**，本脚本不硬编码任何数据集名。
#   2. license_grade=A 才允许进论文实验集；B 级须显式指定并标注"仅内部探索"。
#   3. 体积护栏兜底：单数据集超过 --max-gb（默认 3GB）直接拒绝，
#      MNBVC-judgment（124GB）这样的超大语料靠这条拦下，不靠 usage 标签一刀切。
#   4. 反复执行幂等：已在且 SHA-256 校验通过的文件会跳过。
set -euo pipefail

PROJ="${LAW_AGENT_ROOT:-/mnt/data/lidian/law-agent}"
PY="$PROJ/envs/main/bin/python"
cd "$PROJ"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "===== 阶段 1：采集训练语料（license_grade=A） ====="
echo "  项目根 : $PROJ"
echo "  端点   : $HF_ENDPOINT"
echo "  源登记 : configs/corpus_sources.yaml"
echo "  落盘   : data/corpus/raw/<dataset>/"
echo "  清单   : data/corpus/MANIFEST.json"
echo

"$PY" scripts/corpus/fetch_corpus.py \
  --sources configs/corpus_sources.yaml \
  --out-root data/corpus/raw \
  --manifest data/corpus/MANIFEST.json \
  "$@"

echo
echo "[DONE] 语料采集"
echo "  清单: data/corpus/MANIFEST.json"
echo "  落盘: data/corpus/raw/"
echo "  验收: 每个数据集 status 必须为 OK；再把 MANIFEST 拉回仓库 docs/corpus/"
