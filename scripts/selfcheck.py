#!/usr/bin/env python
"""law-agent 环境自检。

用法：
    source scripts/activate.sh
    python scripts/selfcheck.py

逐项检查 GPU / PyTorch / 量化 / 依赖包 / 模型权重 / Neo4j 连通性，
输出 [OK] / [!!] 标记，任一致命项失败则退出码非 0。
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

OK, BAD, WARN = "[OK]", "[!!]", "[--]"
failures: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    print("\n=== {} ===".format(title))


def check(cond: bool, msg: str, critical: bool = True) -> bool:
    if cond:
        print("  {} {}".format(OK, msg))
    else:
        print("  {} {}".format(BAD if critical else WARN, msg))
        (failures if critical else warnings).append(msg)
    return cond


# ---------------------------------------------------------------- 1. GPU
section("1. GPU / 驱动")
try:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            print("  {} {}".format(OK, line.strip()))
    else:
        check(False, "nvidia-smi 执行失败: " + out.stderr.strip()[:120])
except FileNotFoundError:
    check(False, "找不到 nvidia-smi")

# ---------------------------------------------------------------- 2. PyTorch
section("2. PyTorch / CUDA")
try:
    import torch

    print("  torch 版本        : {}".format(torch.__version__))
    print("  torch 编译 CUDA   : {}".format(torch.version.cuda))
    check(torch.cuda.is_available(), "torch.cuda.is_available()")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        check(n >= 1, "可见 GPU 数量 = {}".format(n))
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print("    GPU{}: {} | {:.1f} GB | sm_{}{}".format(
                i, p.name, p.total_memory / 1024 ** 3, p.major, p.minor))

        # 驱动 CUDA 版本 vs torch 编译版本 —— 本环境最容易出错的点
        drv = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        import re

        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", drv)
        if m:
            drv_cuda = m.group(1)
            tc = torch.version.cuda or "?"
            print("  驱动支持 CUDA     : {} / torch 编译 CUDA: {}".format(drv_cuda, tc))
            check(
                tc.split(".")[0] == drv_cuda.split(".")[0],
                "CUDA 主版本一致（不一致时 torch.cuda 必然不可用）",
            )

        # 真实算一遍，确认 kernel 可执行
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        _ = a @ b
        torch.cuda.synchronize()
        check(True, "bf16 4096x4096 矩阵乘可执行")
        if n > 1:
            x = torch.randn(1024, 1024, device="cuda:1", dtype=torch.float16)
            _ = x @ x
            torch.cuda.synchronize()
            check(True, "cuda:1 可用（多卡）")
except Exception as exc:  # noqa: BLE001
    check(False, "PyTorch 检查异常: {}: {}".format(type(exc).__name__, exc))

# ---------------------------------------------------------------- 3. 量化
section("3. bitsandbytes 4bit 量化（QLoRA 前提）")
try:
    import bitsandbytes as bnb
    import torch

    print("  bitsandbytes      : {}".format(bnb.__version__))
    lin = bnb.nn.Linear4bit(512, 512, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
    y = lin(torch.randn(8, 512, device="cuda", dtype=torch.bfloat16))
    check(tuple(y.shape) == (8, 512), "Linear4bit(nf4) 前向可执行")
    from transformers import BitsAndBytesConfig

    BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    check(True, "BitsAndBytesConfig(4bit NF4 + 双重量化) 可构造")
except Exception as exc:  # noqa: BLE001
    check(False, "量化检查异常: {}: {}".format(type(exc).__name__, exc))

# ---------------------------------------------------------------- 4. 依赖包
section("4. 关键依赖包")
PKGS = [
    ("transformers", "训练/推理"),
    ("peft", "QLoRA"),
    ("trl", "SFTTrainer"),
    ("datasets", "数据集"),
    ("accelerate", "分布式/设备映射"),
    ("sentence_transformers", "向量与重排"),
    ("faiss", "向量索引"),
    ("rank_bm25", "关键词检索"),
    ("jieba", "中文分词"),
    ("datasketch", "MinHash 查重"),
    ("rapidfuzz", "模糊匹配"),
    ("neo4j", "图谱驱动"),
    ("rouge_score", "评测"),
    ("sacrebleu", "评测"),
    ("bert_score", "评测"),
    ("numpy", "数值"),
    ("scipy", "统计/置信区间"),
    ("sklearn", "指标"),
    ("pandas", "数据处理"),
    ("pyarrow", "列式存储"),
    ("yaml", "配置"),
    ("openai", "Judge API"),
    ("tenacity", "API 重试"),
]
for mod, why in PKGS:
    try:
        m = importlib.import_module(mod)
        print("  {} {:<22} {:<10} {}".format(OK, mod, getattr(m, "__version__", ""), why))
    except Exception as exc:  # noqa: BLE001
        check(False, "{} 导入失败 ({}) — {}".format(mod, why, type(exc).__name__))

# ---------------------------------------------------------------- 5. 模型
section("5. 模型权重")
root = Path(os.environ.get("LAW_ROOT") or Path(__file__).resolve().parent.parent)
for name in ("Qwen3-8B", "Qwen3-Embedding-0.6B"):
    d = root / "models" / name
    if not d.is_dir():
        check(False, "{} 目录不存在: {}".format(name, d), critical=False)
        continue
    files = list(d.rglob("*"))
    size_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 ** 3
    n_weights = len(list(d.glob("*.safetensors")))
    check(n_weights > 0, "{}: {} 个 safetensors, 合计 {:.1f} GB".format(name, n_weights, size_gb))
    # 检查分片索引与实际分片是否齐全
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        import json

        wm = json.loads(idx.read_text())["weight_map"]
        needed = sorted(set(wm.values()))
        missing = [s for s in needed if not (d / s).is_file()]
        check(not missing, "{} 分片齐全（{}/{}{}）".format(
            name, len(needed) - len(missing), len(needed),
            "" if not missing else ", 缺: " + ", ".join(missing[:3])))

# ---------------------------------------------------------------- 6. Neo4j
section("6. Neo4j 连通性")
try:
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "lawagent2026claw")
    drv = GraphDatabase.driver(uri, auth=(user, pw), connection_timeout=10)
    with drv.session() as s:
        rec = s.run("CALL dbms.components() YIELD name, versions, edition "
                    "RETURN name, versions[0] AS v, edition").single()
        print("  {} {} {} {}".format(OK, rec["name"], rec["v"], rec["edition"]))
        ver = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
        if ver >= "5":
            print("  {} 版本 {} 支持向量索引（需 >= 5.11）".format(OK, ver))
        else:
            check(False, "版本 {} 过低，不支持向量索引".format(ver), critical=False)
    drv.close()
except Exception as exc:  # noqa: BLE001
    check(False, "Neo4j 连接失败 ({}) — 检查容器是否运行: {}".format(
        type(exc).__name__, str(exc)[:100]), critical=False)

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 60)
if failures:
    print("自检失败 {} 项：".format(len(failures)))
    for f in failures:
        print("  - " + f)
if warnings:
    print("警告 {} 项（不阻塞）：".format(len(warnings)))
    for w in warnings:
        print("  - " + w)
if not failures:
    print("全部关键项通过，环境可用。")
print("=" * 60)
sys.exit(1 if failures else 0)
