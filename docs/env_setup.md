# 实验环境说明（服务器侧）

> 本文档记录 **实际使用** 的环境，而非论文原稿中的描述。
> 论文中「Windows 11 + WSL2 + 12GB 显存」的设备条件已作废 —— 实验实际在实验服务器上完成。

---

## 1. 硬件与系统

| 项 | 实测值 |
|---|---|
| 主机 | `192.168.195.61`，用户 `lidian`，端口 22 |
| GPU | **2 × NVIDIA A800 80GB PCIe**（共 160GB 显存，sm_80） |
| 驱动 | `550.163.01` → 驱动侧 **CUDA 12.4** 为上限 |
| CPU / 内存 | 128 核 / 503GB（约 244GB 可用） |
| 系统盘 | `/` 1.8T（可用 973G） |
| 数据盘 | `/mnt/data` 8.7T（可用 5.2T）← **项目、模型、环境全放这里** |
| OS | Ubuntu 20.04.6 LTS，kernel 5.4，glibc 2.31 |
| 编译器 | gcc/g++ 9.4.0，make 可用；**无 nvcc / 无 CUDA Toolkit** |
| 权限 | **无 sudo**，一切只能装在 `$HOME` 或 `/mnt/data` |

### 服务器为共享机器（重要）

该机器上已有其他生产容器在运行（Kafka、Elasticsearch、Kibana、Harbor registry 等），
`/mnt/data` 是全组共用目录。因此本项目的所有操作遵循：

- 不做 `sudo` 级系统改动，不修改既有容器
- 项目独占目录 `/mnt/data/lidian/law-agent/`，新增容器单独命名并单独映射端口
- GPU 使用前用 `nvidia-smi` 确认空闲

---

## 2. 目录布局（服务器）

```
/mnt/data/lidian/law-agent/           # 项目根（实体，大数据卷）
├─ envs/
│  └─ main/                           # Python 3.11 主环境（训练 + 检索 + 评测 全栈）
├─ models/
│  ├─ Qwen3-8B/                       # 底座（16.4G，5 分片，sha256 已校验）
│  └─ Qwen3-Embedding-0.6B/           # 向量模型（1.2G）
├─ neo4j/                             # Neo4j 容器挂载卷
│  ├─ data/  logs/  import/  plugins/  conf/
├─ data/  corpus/  indexes/  outputs/  logs/  scripts/  configs/  docs/
└─ requirements-lock.txt              # 环境冻结清单

~/law-agent -> /mnt/data/lidian/law-agent   # 软链接，两个路径等价
```

---

## 3. Python 环境方案

### 3.1 为什么不用 conda

系统只有 Python 3.8.10（对 2026 年的 ML 栈太旧），且无 sudo。
实测从清华镜像下载的 Miniforge 安装包 **md5 校验失败**（tar 截断），试两次均坏。
改用 **`uv` + python-build-standalone**：单文件二进制、无需编译、下载快。

### 3.2 分层结构

```
system python3.8  →  /tmp/uvboot          （仅用于装 uv 本身，一次性）
uv 0.12.15        →  ~/.local/share/uv    （下载并管理 3.11 运行时）
CPython 3.11.16   →  envs/main            （项目唯一环境，OpenSSL 3.5.7 自带）
```

> 只保留 **一个** 环境 `envs/main`。不按「训练/推理」拆环境，避免 torch 版本分叉。
> 若后续确需 vLLM 高速推理，再单开 `envs/vllm`（vLLM 对 torch 版本有硬钉死要求）。

### 3.3 PyTorch 版本：**必须按驱动 CUDA 版本钉死**（本环境最大的坑）

`nvidia-smi` 显示的 `CUDA Version: 12.4` 是**驱动支持的上限**，不是已安装的 toolkit。

- CUDA **主版本**升级（12 → 13）需要更新驱动。驱动 550 最高支持 CUDA 12.4，
  所以 **cu130 及以上的 wheel 一律不可用**。
- CUDA **次版本**兼容（12.1 / 12.4 / 12.6 / 12.8 之间）由 NVIDIA minor version
  compatibility 保证，同一驱动可混用。

**事故记录**：初次安装时 `--index-strategy unsafe-best-match` 让 uv 跨索引挑了
PyPI 上最新的 `torch 2.14.0+cu130`，直接导致：

```
UserWarning: The NVIDIA driver on your system is too old (found version 12040)
torch.cuda.is_available() -> False
```

**正确做法**：只指定 PyTorch 官方索引，用 `first-index` 策略，并**显式钉版本号**：

```bash
uv pip install --python "$PY" \
  --default-index https://download.pytorch.org/whl/cu124 \
  --index-strategy first-index \
  "torch==2.6.0+cu124"
```

- `cu124` 索引上 `cp311` 可用最高版本：**torch 2.6.0+cu124**
- ⚠️ 不要用 `unsafe-best-match`，也不要加 PyPI 作 extra-index —— 会跨索引取最新版

环境重建脚本会自动读 `nvidia-smi` 的 CUDA 版本并选择对应索引，
见 [`../scripts/server_env.sh`](../scripts/server_env.sh)。

---

## 4. 依赖清单（按用途分组）

| 用途 | 包 |
|---|---|
| 训练 | `torch==2.6.0+cu124`、`transformers` 5.17、`datasets`、`accelerate`、`peft` 0.21、`trl` 1.13、`bitsandbytes` 0.50 |
| 向量 / 检索 | `sentence-transformers` 6.0、`faiss-cpu`、`rank-bm25`、`jieba`、`datasketch`、`rapidfuzz` |
| 图谱 | `neo4j` 6.3（Python 驱动）+ Neo4j 5.26 社区版（Docker） |
| 编排 | `langgraph` 1.2（自带 `langchain-core` 1.6 / `langgraph-checkpoint` / `langgraph-sdk`） |
| 评测 | `rouge-score`、`sacrebleu`、`bert-score`、`nltk`、`evaluate` |

**未安装及原因**：

- `FlagEmbedding` —— 其旧版本对 `transformers` 有上界约束，会与 Qwen3 所需的新版冲突。
  BGE-M3 与 bge-reranker-v2-m3 改由 `sentence-transformers`（`SentenceTransformer` /
  `CrossEncoder`）加载，功能等价且无版本冲突。
- `langchain`（元包）—— **不需要**。`langgraph` 独立可用，装元包只会引入大量无关依赖。
- `flash-attn` —— 服务器**无 nvcc**，编译需要本地 CUDA Toolkit。
  默认使用 PyTorch 原生 `sdpa` 注意力（A800 上性能已足够）。
  若确需 flash-attn，可先用 `pip install nvidia-cuda-nvcc-cu12` 提供 nvcc 再源码编译。
- 未装 vLLM —— 其对 torch 版本有硬钉死要求，会与主环境冲突；如需高速批量推理单开环境。

> ⚠️ **venv 里没有 `pip` 是正常的**（uv 建的环境默认不带）。需要时用
> `uv pip install --python <venv>/bin/python <pkg>`，或先补 `pip`。
> `uv` 二进制在 `~/.local/bin/uv`，**不在默认 PATH 里**，脚本中需用全路径。

---

## 5. 网络与镜像（大陆网络环境）

| 目标 | 状态 | 用法 |
|---|---|---|
| `huggingface.co` | ❌ 被墙（SSL reset / 000） | 禁用 |
| `hf-mirror.com` | ✅ | `export HF_ENDPOINT=https://hf-mirror.com` |
| `pypi.tuna.tsinghua.edu.cn` | ✅ 快 | pip / uv 默认索引 |
| `download.pytorch.org` | ✅ 可达 | torch 专用索引（偶发限流，需重试） |
| `dist.neo4j.org` | ❌ 403（地域封锁） | **改用 Docker 镜像** |
| `mirrors.tuna.tsinghua.edu.cn/Adoptium` | ✅ | 需要 JDK 时用 |
| `repo1.maven.org` | ✅ | Neoj4 插件（APOC）jar 可用 |
| Docker Hub 直连 | ❌ | 走 daemon 已配置的加速器 |

---

## 6. 关键环境变量

运行任何脚本前必须设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com          # 否则模型下载全部失败
export UV_CACHE_DIR=/mnt/data/lidian/.cache/uv    # uv 缓存放数据盘，别撑爆 /
export HF_HOME=/mnt/data/lidian/.cache/huggingface # HF 缓存同理
export TOKENIZERS_PARALLELISM=false               # 避免多进程 tokenizer 警告/竞争
```

已封装在 [`../scripts/activate.sh`](../scripts/activate.sh)：

```bash
source scripts/activate.sh          # 激活环境 + 设置全部变量
```

---

## 7. 环境自检

```bash
source /mnt/data/lidian/law-agent/scripts/activate.sh
bash scripts/verify_env.sh        # 一键全量自检（推荐，实测版）
python scripts/selfcheck.py       # 细项自检
```

`scripts/verify_env.sh` 是**实测版**自检：会真跑 CUDA 计算、真分配显存、真连 Neo4j，
因此能识别「装了 CPU-only torch」「torch 编译的 CUDA 版本高于驱动上限」这类
**光靠 `import torch` 完全测不出来**的问题。退出码 0 = 全通过。

`scripts/selfcheck.py` 逐项检查：GPU 数量与显存、torch/CUDA 版本一致性、
bf16 矩阵乘、bitsandbytes NF4 前向、各关键包导入、模型权重完整性、Neo4j 连通性。

### 7.1 端到端冒烟测试（2026-09-17 实测通过）

`scripts/smoke_test.py` 验证「加载模型 → 用 GPU → 4bit 量化 → 挂 LoRA 反向传播」全链路。
**只 import 成功不算数，必须跑通这个。**

```
[OK] 1. GPU 与 PyTorch 版本匹配        torch=2.6.0+cu124 torch.cuda=12.4 驱动CUDA=12.4 可用=True 卡数=2
[OK] 2. Qwen3-Embedding-0.6B 编码中文  6.5s  dim=1024
[OK] 3. 4bit NF4 加载 Qwen3-8B         11s   显存 5.66 GB
[OK] 4. 生成（/no_think）              1.8s
[OK] 5. LoRA 挂载 + 单步反向传播        可训练 43.6M/4761.5M (0.917%)
                                       loss=4.162  有梯度层 252  峰值显存 9.14 GB
[OK] 6. 释放显存
通过 6/6
```

**由此得到的容量结论**（160GB 显存下的实测边界）：

- Qwen3-8B 4bit 常驻仅 **5.66 GB**，LoRA r=16 单步峰值 **9.14 GB**
- 单卡 80GB 意味着：可同时把**两个 adapter 训练进程**各占一卡，或大幅提高
  `lora_r` / `batch_size` / `max_seq_length`，而不必再受 12GB 时代的参数裁剪
- 也意味着 **14B / 32B 稠密、30B-A3B MoE 都可直接 QLoRA**，底座规模不必锁死在 8B
- ⚠️ 但论文的对比实验要求**只换一个变量**：底座一旦定下，E0–E5 全部用同一个底座重跑

### 7.2 Neo4j 图层实测（2026-09-17 已建）

```
7 个唯一约束 + 2 个范围索引（effective_from/to、law_name+article+paragraph）
4 个向量索引  1024d(Qwen3-Embedding-0.6B) / 1024d(bge-m3) / 768d(bert-wwm) / 2560d(Qwen3-Embedding-4B)
2 个全文索引  provision_fulltext / case_fulltext
全部状态 ONLINE；Python 驱动 bolt://127.0.0.1:7687 连通正常
```

> 实测复核（`SHOW INDEXES` 聚合）：索引总数 **17**，其中向量索引 **4**，ONLINE **17**。

### 7.3 环境真伪核验（2026-09-17 全量实测）

`bash scripts/verify_env.sh` 全部 PASS。以下为**实测证据**，不是配置读取：

| 检查项 | 实测值 | 说明 |
|---|---|---|
| `torch.__version__` | `2.6.0+cu124` | 编译 CUDA 12.4 |
| `torch.backends.cuda.is_built()` | `True` | **不是 CPU-only 构建** |
| `torch.cuda.is_available()` | `True` | |
| `device_count` | `2` | A800 80GB PCIe × 2，cc 8.0，各 108 SM |
| 进程内已加载的 CUDA 运行库 | `libcudart.so.12` `libcublas.so.12` `libcublasLt.so.12` `libnccl.so.2` `libnvrtc.so.12` | `/proc/<pid>/maps` 实读 |
| **8192³ bf16 矩阵乘（GPU）** | **249.7 TFLOPS** | 0.0044 s/次 |
| 8192³ bf16 矩阵乘（CPU 128 核） | 0.8 TFLOPS | 1.3260 s/次 |
| **GPU / CPU 加速比** | **301x** | CPU 硬件上限约 1–2 TFLOPS，不可能达到 249.7 |
| 显存分配实测 | 申请 5 GiB → 实占 **5.07 GiB** | `torch.cuda.mem_get_info` 差值 |
| `nvidia-smi` 交叉验证 | 进程占 10730 MiB，`--query-compute-apps` 能列出 PID | 与 torch 自述一致，无监控盲区 |
| 依赖完整性 | 22 个关键包全部 OK，`pip check` 无冲突 | `langgraph 1.2.11` 就位 |
| 模型资产 | `Qwen3-8B` 16G / 5 分片；`Qwen3-Embedding-0.6B` 1.2G | |
| Neo4j | Kernel 5.26.30，索引 17 个全 ONLINE | |

**结论：GPU 链路完全打通，不是 CPU 回退。**

关于 `nvidia-smi` 的一处历史困惑（早期探测曾看到 `1 MiB, 0%`）：
原因是当时的测试脚本用了后台 + heredoc 的写法导致采样时机错位，
**不是监控盲区**。用「后台进程占住 10 GiB → 独立连接采样」的方式复核，
`nvidia-smi` 与 `torch.cuda.mem_get_info` 数值一致（10739 MiB vs 10.00 GiB）。
训练时可以直接用 `nvidia-smi` 或 `watch -n1 nvidia-smi` 观察显存。

### 7.4 环境资产落点（诚实清点）

题目是「是否都装在项目目录下」。结论：**主体在项目目录，有一处例外**。

| 资产 | 路径 | 体积 | 是否在项目目录 |
|---|---|---|---|
| Python 包（site-packages） | `$ROOT/envs/main` | 6.4 G | ✅ |
| 模型权重 | `$ROOT/models` | 17 G | ✅ |
| Neo4j 数据/日志/插件 | `$ROOT/neo4j` | 517 M | ✅ |
| **Python 解释器本体** | `~/.local/share/uv/python/cpython-3.11.16-…` | 98 M | ❌ 在 `$HOME` |
| uv 包缓存（安装期用，可删） | `/mnt/data/lidian/.cache/uv` | 18 G | ❌ 在 `$HOME` 同级 |
| pip 缓存 | `~/.cache` | 1.2 G | ❌ 在 `$HOME` |

- 解释器本体 98M 在 `$HOME` 是 `uv` 的设计（多 venv 共享同一解释器），`$HOME/.local` 合计 230M。
- `uv` 缓存 18G 是**安装过程的中间缓存，可以随时删除**：`rm -rf /mnt/data/lidian/.cache/uv`。
  删掉不影响已装好的环境，只是下次装包会重新下载。
- 三者都在 `/mnt/data` 卷（8.7T，剩余 5.1T）或 `/` 卷（1.8T，剩余 972G）上，**没有撑爆任何分区**。
- **系统 Python 3.8 完全未被污染**：`/usr/lib/python3/dist-packages` 下无 torch、无 transformers。

---

## 8. 与论文原方案的差异（需在论文中同步修订）

| 项 | 论文原稿 | 实际采用 | 原因 |
|---|---|---|---|
| 设备 | Windows 11 + WSL2 + 12GB 显存 | Ubuntu 20.04 + 2×A800 80GB | 实际算力设备 |
| 底座 | Qwen2.5-7B-Instruct | **Qwen3-8B** | 同尺寸更新一代；`-Instruct` 变体在 HF 上不存在 |
| 向量 | Chinese-BERT-wwm-ext | **Qwen3-Embedding-0.6B**（主力）+ BGE-M3 等四路消融 | 原方案为 PLM 非检索嵌入模型 |
| 知识层 | Neo4j（未说明部署方式） | Neo4j 5.26 社区版（Docker） | 官方 tarball 分发被 403 |
| 精度 | QLoRA 4bit NF4 | 不变 | 训练方法不变，仅底座升级 |

> ⚠️ 底座与向量模型一旦确定就 **不要再改**：换底座需重训全部适配器，
> 换向量模型需全量重建索引，前面所有实验作废。
