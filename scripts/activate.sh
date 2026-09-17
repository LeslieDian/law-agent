#!/usr/bin/env bash
# law-agent 环境激活脚本
#
# 用法（必须 source，不能直接执行，否则环境变量不会留在当前 shell）：
#     source scripts/activate.sh
#
# 作用：把 envs/main 置入 PATH，并设置大陆网络环境所需的关键变量。

_LAW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LAW_ROOT="$(cd "$_LAW_SCRIPT_DIR/.." && pwd)"
LAW_CACHE_ROOT="/mnt/data/lidian/.cache"

export LAW_ROOT
export LAW_ENV="$LAW_ROOT/envs/main"
export PATH="$LAW_ENV/bin:$PATH"

# --- 大陆网络环境必需 ---
# huggingface.co 在本网络不可达，必须走镜像，否则所有模型下载失败
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$LAW_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$LAW_CACHE_ROOT/uv}"

# --- 运行稳定性 ---
export TOKENIZERS_PARALLELISM=false     # 多进程下避免 tokenizer 竞争告警
export PYTHONUTF8=1                     # 强制 UTF-8，避免中文输出乱码
export PYTHONUNBUFFERED=1               # 日志实时落盘，便于远程 tail

# --- 模型路径快捷变量 ---
export LAW_BASE_MODEL="${LAW_BASE_MODEL:-$LAW_ROOT/models/Qwen3-8B}"
export LAW_EMB_MODEL="${LAW_EMB_MODEL:-$LAW_ROOT/models/Qwen3-Embedding-0.6B}"

# --- 图谱连接 ---
export NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-lawagent2026claw}"

# --- GPU 选择（默认两张卡都可见）---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "law-agent 环境已激活"
echo "  LAW_ROOT       = $LAW_ROOT"
echo "  python         = $(command -v python)"
echo "  python 版本    = $(python -V 2>&1 | awk '{print $2}')"
echo "  HF_ENDPOINT    = $HF_ENDPOINT"
echo "  CUDA_VISIBLE   = $CUDA_VISIBLE_DEVICES"
