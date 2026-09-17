#!/usr/bin/env bash
# ============================================================
# law-agent 环境自检（实测版，不读配置）
#
# 用途：换机器 / 换环境 / 训练前，确认 GPU 链路真的可用。
#       会真跑 CUDA 计算并分配显存，因此能识别
#       「装了 CPU-only torch」「驱动与 CUDA 版本不匹配」这类
#       光靠 import 测不出来的问题。
#
# 用法：  bash scripts/verify_env.sh
# 退出码：0 = 全部通过；1 = 有检查项失败
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/envs/main/bin/python"

FAIL=0
ok()   { echo "  [ OK ] $*"; }
warn() { echo "  [WARN] $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }

echo "===================== 1. 主机与路径 ====================="
echo "  host      : $(hostname)"
echo "  user      : $(whoami)"
echo "  date      : $(date '+%F %T %Z')"
echo "  os        : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "  kernel    : $(uname -r)"
echo "  project   : $ROOT"
[ -d "$ROOT/envs/main" ] && ok "venv 存在 $ROOT/envs/main" || bad "venv 缺失"
[ -x "$PY" ] && ok "解释器可执行" || { bad "解释器不可执行: $PY"; exit 1; }

echo
echo "===================== 2. 磁盘 ====================="
df -h "$ROOT" | sed 's/^/  /'

echo
echo "===================== 3. GPU 硬件 ====================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total \
             --format=csv,noheader | sed 's/^/  /'
else
  bad "未找到 nvidia-smi"
fi

echo
echo "===================== 4. torch / CUDA（核心） ====================="
"$PY" - <<'PYEOF' || exit 1
import sys, os, torch
print("  torch                 :", torch.__version__)
print("  torch.version.cuda    :", torch.version.cuda)
print("  cuda is_built         :", torch.backends.cuda.is_built())
print("  torch 安装路径        :", os.path.dirname(torch.__file__))
print("  python 解释器         :", sys.executable)

fail = False
if not torch.backends.cuda.is_built():
    print("  [FAIL] torch 是 CPU-only 构建，必须重装 GPU 版")
    fail = True
if not torch.cuda.is_available():
    print("  [FAIL] CUDA 不可用：多为驱动支持的 CUDA 主版本 < torch 编译版本")
    print("         用 nvidia-smi 看驱动上限，据此重装匹配的 torch")
    fail = True
if fail:
    sys.exit(1)

print("  cuda available        : True")
print("  device_count          :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("    GPU{}: {} | {:.1f} GiB | cc {}.{} | SMs {}".format(
        i, p.name, p.total_memory / 1024**3, p.major, p.minor, p.multi_processor_count))
PYEOF
[ $? -eq 0 ] && ok "CUDA 可用" || bad "CUDA 检查未通过"

echo
echo "===================== 5. GPU 真算 + 显存实测 ====================="
"$PY" - <<'PYEOF' || exit 1
import torch, time

# --- 5.1 张量落在 GPU 上 ---
a = torch.randn(4096, 4096, dtype=torch.bfloat16)
a = a.cuda()
print("  张量 device           :", a.device, "|", torch.cuda.get_device_name(a.device))

# --- 5.2 大矩阵 bf16 与 CPU 对照（消除 kernel 启动开销） ---
N = 8192
ac = torch.randn(N, N, dtype=torch.bfloat16)
bc = torch.randn(N, N, dtype=torch.bfloat16)
ag, bg = ac.cuda(), bc.cuda()

def bench(fn, n, warmup):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n

t_gpu = bench(lambda: ag @ bg, 20, 5)
t_cpu = bench(lambda: ac @ bc, 2, 1)
flops = 2 * N**3
print("  {}^3 bf16  GPU  : {:.4f} s -> {:8.1f} TFLOPS".format(N, t_gpu, flops/t_gpu/1e12))
print("  {}^3 bf16  CPU  : {:.4f} s -> {:8.1f} TFLOPS".format(N, t_cpu, flops/t_cpu/1e12))
print("  加速比                : {:.1f}x".format(t_cpu/t_gpu))
if t_cpu / t_gpu < 10:
    print("  [WARN] 加速比偏低，确认没有回落到 CPU")
del ac, bc, ag, bg
torch.cuda.empty_cache()

# --- 5.3 显存分配/释放 ---
torch.cuda.empty_cache()
free_before, _ = torch.cuda.mem_get_info(0)
x = torch.zeros(5 * 1024**3 // 2, dtype=torch.bfloat16, device="cuda")  # 5 GiB
free_after, _ = torch.cuda.mem_get_info(0)
print("  显存分配 5 GiB 实测   : {:.2f} GiB 占用".format((free_before - free_after)/1024**3))
del x; torch.cuda.empty_cache()
print("  释放后占用            : {:.3f} GiB".format(torch.cuda.memory_allocated()/1024**3))
PYEOF
[ $? -eq 0 ] && ok "GPU 计算与显存正常" || bad "GPU 实测失败"

echo
echo "===================== 6. 关键依赖 ====================="
"$PY" - <<'PYEOF'
import importlib, importlib.metadata as md

# (包名, 导入名, 是否必需)
PKGS = [
    ("torch",                "torch",                True),
    ("transformers",         "transformers",         True),
    ("peft",                 "peft",                 True),
    ("trl",                  "trl",                  True),
    ("bitsandbytes",         "bitsandbytes",         True),
    ("accelerate",           "accelerate",           True),
    ("datasets",             "datasets",             True),
    ("sentence-transformers","sentence_transformers",True),
    ("faiss-cpu",            "faiss",                True),
    ("rank_bm25",            "rank_bm25",            True),
    ("jieba",                "jieba",                True),
    ("neo4j",                "neo4j",                True),
    ("langgraph",            "langgraph",            True),
    ("langchain-core",       "langchain_core",       True),
    ("scikit-learn",         "sklearn",              True),
    ("rouge-score",          "rouge_score",          True),
    ("sacrebleu",            "sacrebleu",            True),
    ("bert-score",           "bert_score",           True),
    ("openai",               "openai",               True),
    ("rapidfuzz",            "rapidfuzz",            True),
    ("datasketch",           "datasketch",           False),
    ("google-genai",         "google.genai",         False),
]
missing = []
for pkg, mod, required in PKGS:
    try:
        importlib.import_module(mod)
        try:
            v = md.version(pkg)
        except Exception:
            v = "?"
        print("  {:<24} {:<12} OK".format(pkg, v))
    except Exception:
        tag = "MISSING" if required else "missing(optional)"
        print("  {:<24} {:<12} {}".format(pkg, "-", tag))
        if required:
            missing.append(pkg)
if missing:
    print("\n  [FAIL] 必需依赖缺失: " + ", ".join(missing))
    raise SystemExit(1)
PYEOF
[ $? -eq 0 ] && ok "依赖齐备" || bad "依赖缺失"

echo
echo "===================== 7. 依赖一致性 ====================="
"$PY" -m pip check 2>&1 | sed 's/^/  /' | head -20

echo
echo "===================== 8. 模型资产 ====================="
for d in models/Qwen3-8B models/Qwen3-Embedding-0.6B; do
  p="$ROOT/$d"
  if [ -d "$p" ]; then
    sz=$(du -sh "$p" 2>/dev/null | cut -f1)
    n=$(ls -1 "$p"/*.safetensors 2>/dev/null | wc -l)
    ok "$d  ($sz, $n 个 safetensors 分片)"
  else
    warn "$d 不存在（首次使用前需下载，见 docs/env_setup.md）"
  fi
done

echo
echo "===================== 9. Neo4j 图谱服务 ====================="
if docker ps --filter name=law-agent-neo4j --format '{{.Names}}' 2>/dev/null | grep -q neo4j; then
  ok "容器运行中"
  docker ps --filter name=law-agent-neo4j --format '  {{.Names}} | {{.Image}} | {{.Status}}'
  "$PY" - <<'PYEOF'
import os
try:
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "lawagent2026claw")))
    d.verify_connectivity()
    with d.session() as s:
        v = s.run("CALL dbms.components() YIELD name, versions "
                  "RETURN name + ' ' + versions[0] AS v").single()["v"]
        idx = s.run("SHOW INDEXES YIELD name, type, state "
                    "RETURN count(*) AS total, "
                    "sum(CASE WHEN type='VECTOR' THEN 1 ELSE 0 END) AS vec, "
                    "sum(CASE WHEN state='ONLINE' THEN 1 ELSE 0 END) AS online").single()
    print("  服务端        :", v)
    print("  索引总数      :", idx["total"], "| 向量索引:", idx["vec"],
          "| ONLINE:", idx["online"])
    d.close()
except Exception as e:
    print("  [FAIL] 连接失败:", type(e).__name__, e)
    raise SystemExit(1)
PYEOF
  [ $? -eq 0 ] && ok "Neo4j 连通，索引就绪" || bad "Neo4j 连接失败"
else
  warn "law-agent-neo4j 容器未运行（启动方式见 docs/env_setup.md）"
fi

echo
echo "===================== 汇总 ====================="
if [ "$FAIL" -eq 0 ]; then
  echo "  [PASS] 环境检查全部通过，可以进行训练与推理实验"
  exit 0
else
  echo "  [FAIL] 存在失败项，请按上面的 [FAIL] 提示处理"
  exit 1
fi
