#!/usr/bin/env bash
# =============================================================================
# law-agent 服务器环境一键重建脚本
#
# 适用：Ubuntu 20.04（或相近版本），**无 sudo 权限**，大陆网络环境
# 用法：bash scripts/server_env.sh            # 全量安装（幂等，可重复执行）
#       bash scripts/server_env.sh --torch-only   # 只重装 PyTorch
#
# 设计要点（都是踩过坑之后的结论，改动前请先读 docs/env_setup.md）：
#   1. 不用 conda —— 清华镜像的 Miniforge 安装包实测 md5 校验失败
#   2. 用 uv 拉 python-build-standalone，无需编译、无需 sudo
#   3. PyTorch 必须按 nvidia-smi 报告的驱动 CUDA 版本钉死，且**不能用
#      unsafe-best-match**（会跨索引挑到 CUDA 13 版，导致 GPU 完全不可用）
#   4. huggingface.co 不可达，一律走 hf-mirror.com
#   5. Neo4j 官方 tarball 分发返回 403，改用 Docker 镜像
# =============================================================================
set -uo pipefail

PROJ_DEFAULT=/mnt/data/lidian/law-agent
PROJ="${LAW_ROOT:-$PROJ_DEFAULT}"
CACHE_ROOT=/mnt/data/lidian/.cache
PREFIX="$PROJ/envs"
PY="$PREFIX/main/bin/python"
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple
UV=/tmp/uvboot/bin/uv

export UV_CACHE_DIR="$CACHE_ROOT/uv"
mkdir -p "$PREFIX" "$PROJ/logs" "$UV_CACHE_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] 失败: $*" >&2; exit 1; }

TORCH_ONLY=0
[ "${1:-}" = "--torch-only" ] && TORCH_ONLY=1

# -----------------------------------------------------------------------------
# 0. 前置检查
# -----------------------------------------------------------------------------
log "=== 0. 前置检查 ==="
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi"
command -v curl >/dev/null || die "找不到 curl"

DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
DRV_MAJOR="${DRV_CUDA%%.*}"
log "驱动支持的 CUDA 版本: ${DRV_CUDA:-未知}（主版本 $DRV_MAJOR）"
[ -z "$DRV_CUDA" ] && die "无法解析驱动 CUDA 版本"

# -----------------------------------------------------------------------------
# 1. 引导 uv（用系统 python3，仅此一次）
# -----------------------------------------------------------------------------
if [ "$TORCH_ONLY" = "0" ]; then
  log "=== 1. 引导 uv ==="
  if [ ! -x "$UV" ]; then
    python3 -m venv /tmp/uvboot || die "创建 uv 引导 venv 失败"
    /tmp/uvboot/bin/pip install -q -i "$TUNA" --upgrade pip
    /tmp/uvboot/bin/pip install -i "$TUNA" uv || die "安装 uv 失败"
  fi
  [ -x "$UV" ] || die "uv 不可用"
  log "uv: $("$UV" --version)"

  # ---------------------------------------------------------------------------
  # 2. 安装 Python 3.11 运行时
  # ---------------------------------------------------------------------------
  log "=== 2. 安装 Python 3.11 ==="
  PYBASE=""
  for M in \
    "https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download" \
    "https://mirrors.tuna.tsinghua.edu.cn/github-release/astral-sh/python-build-standalone/releases/download" \
    "" ; do
    if [ -n "$M" ]; then
      export UV_PYTHON_INSTALL_MIRROR="$M"; log "  镜像: $M"
    else
      unset UV_PYTHON_INSTALL_MIRROR; log "  回退官方源"
    fi
    "$UV" python install 3.11 >/dev/null 2>&1 || continue
    PYBASE=$("$UV" python find 3.11 2>/dev/null) && [ -n "$PYBASE" ] && break
  done
  unset UV_PYTHON_INSTALL_MIRROR
  [ -n "$PYBASE" ] || die "Python 3.11 安装失败"
  log "  Python: $PYBASE"

  # ---------------------------------------------------------------------------
  # 3. 创建主环境
  # ---------------------------------------------------------------------------
  log "=== 3. 创建 venv: $PREFIX/main ==="
  rm -rf "$PREFIX/main"
  "$UV" venv --python 3.11 "$PREFIX/main" || die "创建 venv 失败"
  "$PY" -V
fi

[ -x "$PY" ] || die "环境不存在，请先不带 --torch-only 运行一次"

# -----------------------------------------------------------------------------
# 4. PyTorch —— 按驱动 CUDA 主版本钉死（本脚本最关键的一步）
# -----------------------------------------------------------------------------
log "=== 4. 安装 PyTorch（驱动 CUDA $DRV_CUDA）==="
if [ "$DRV_MAJOR" -ge 13 ]; then
  PT_INDEX=https://download.pytorch.org/whl/cu130
  TORCH_CANDIDATES="torch"
elif [ "$DRV_MAJOR" -eq 12 ]; then
  PT_INDEX=https://download.pytorch.org/whl/cu124
  # cu124 索引上 cp311 的已知最高版本；若索引更新可放宽
  TORCH_CANDIDATES="torch==2.6.0+cu124 torch==2.5.1+cu124 torch==2.5.1+cu121"
else
  die "驱动 CUDA $DRV_CUDA 过低，无法运行现代 PyTorch，请升级驱动"
fi
log "  索引: $PT_INDEX"

# 先清掉可能存在的错误版本（历史上曾被 cu13 污染过）
"$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE '^(torch|triton|nvidia-|pytorch-)' > /tmp/_t_rm.txt || true
if [ -s /tmp/_t_rm.txt ]; then
  log "  卸载已有的 torch 相关包 $(wc -l < /tmp/_t_rm.txt) 个"
  "$UV" pip uninstall --python "$PY" -r /tmp/_t_rm.txt >/dev/null 2>&1 || true
fi

TORCH_OK=0
for SPEC in $TORCH_CANDIDATES; do
  log "  尝试 $SPEC"
  "$UV" pip install --python "$PY" \
      --default-index "$PT_INDEX" \
      --index-strategy first-index \
      "$SPEC" 2>&1 | tail -4
  if "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    TORCH_OK=1; log "  torch 安装成功: $("$PY" -c 'import torch;print(torch.__version__)')"; break
  fi
  log "  该版本不可用，试下一个"
done
[ "$TORCH_OK" = "1" ] || die "PyTorch 安装失败 —— 检查驱动 CUDA $DRV_CUDA 与索引 $PT_INDEX 是否匹配"
[ "$TORCH_ONLY" = "1" ] && { log "TORCH_ONLY 完成"; exit 0; }

# -----------------------------------------------------------------------------
# 5. 其余全栈依赖
# -----------------------------------------------------------------------------
log "=== 5. 安装训练 / 检索 / 图谱 / 评测 全栈 ==="
"$UV" pip install --python "$PY" --default-index "$TUNA" \
  transformers datasets accelerate peft trl bitsandbytes \
  sentencepiece protobuf safetensors einops \
  sentence-transformers faiss-cpu rank-bm25 jieba datasketch rapidfuzz \
  neo4j \
  rouge-score sacrebleu bert-score nltk evaluate \
  numpy scipy scikit-learn pandas pyarrow \
  pyyaml python-dotenv tqdm rich loguru tenacity tabulate \
  openai google-genai \
  tensorboard ipykernel jupyterlab 2>&1 | tail -10
# 注意：不装 FlagEmbedding（与新版 transformers 冲突）、
#       不装 flash-attn（无 nvcc，用 torch 原生 sdpa 代替）

log "=== 6. 依赖一致性 ==="
"$UV" pip check --python "$PY" || log "  pip check 有告警，见上"

log "=== 7. 冻结环境 ==="
"$UV" pip freeze --python "$PY" > "$PROJ/requirements-lock.txt"
log "  $PROJ/requirements-lock.txt ($(wc -l < "$PROJ/requirements-lock.txt") 行)"
log "  环境体积: $(du -sh "$PREFIX/main" | cut -f1)"

# -----------------------------------------------------------------------------
# 8. Neo4j（Docker）
# -----------------------------------------------------------------------------
log "=== 8. Neo4j ==="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  NAME=law-agent-neo4j
  PW="${NEO4J_PASSWORD:-lawagent2026claw}"
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    log "  $NAME 已在运行，跳过"
  else
    mkdir -p "$PROJ/neo4j"/{data,logs,import,plugins,conf}
    chmod -R 777 "$PROJ/neo4j"
    IMG=""
    for C in neo4j:5.26-community neo4j:community; do
      docker pull "$C" >/dev/null 2>&1 && docker image inspect "$C" >/dev/null 2>&1 && { IMG="$C"; break; }
    done
    if [ -n "$IMG" ]; then
      docker run -d --name "$NAME" --restart unless-stopped \
        -p 7474:7474 -p 7687:7687 \
        -v "$PROJ/neo4j/data":/data -v "$PROJ/neo4j/logs":/logs \
        -v "$PROJ/neo4j/import":/var/lib/neo4j/import -v "$PROJ/neo4j/plugins":/plugins \
        -e NEO4J_AUTH="neo4j/$PW" \
        -e NEO4J_server_memory_heap_initial__size=8G \
        -e NEO4J_server_memory_heap_max__size=16G \
        -e NEO4J_server_memory_pagecache_size=8G \
        "$IMG" >/dev/null && log "  已启动 $NAME ($IMG)"
    else
      log "  !! 拉取 Neo4j 镜像失败，跳过"
    fi
  fi
else
  log "  !! 无 docker 权限，跳过 Neo4j"
fi

# -----------------------------------------------------------------------------
# 9. 自检
# -----------------------------------------------------------------------------
log "=== 9. 自检 ==="
source "$PROJ/scripts/activate.sh" >/dev/null 2>&1 || true
"$PY" "$PROJ/scripts/selfcheck.py" || log "  自检有失败项，见上"

log "ENV_SETUP_DONE"
