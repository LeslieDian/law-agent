# 法律领域智能体：混合专家模型 + 知识图谱

> 论文《基于混合专家模型和知识图谱的法律领域智能体研究与构建》实验代码仓库

面向**民法、刑法、程序法**三法域的法律智能问答系统。以 Qwen3-8B 为共享底座，
训练三个 QLoRA 领域适配器；以 Neo4j 统一存储法律知识图谱与案例向量索引；在线链路采用
「向量召回 → 图谱关系扩展 → RRF 融合 → 重排序 → 请求级适配器路由」的混合检索增强生成。

---

## 1. 技术路线

| 层 | 组件 | 说明 |
|---|---|---|
| 底座模型 | `Qwen/Qwen3-8B` | 固定 commit hash。注意：HF 上**不存在** `Qwen3-8B-Instruct`。选型分析见 [`docs/model_selection.md`](docs/model_selection.md) |
| 领域适配 | QLoRA（4bit NF4 + 双重量化 + BF16） | 民法 / 刑法 / 程序法 三个适配器 + 请求级路由 |
| 知识层 | Neo4j 5.26 Community（Docker） | Law / LawVersion / Provision / Case / Cause / Court… |
| 检索 | Qwen3-Embedding-0.6B 向量 + BM25/全文 + 图谱扩展 | 排名融合 RRF(k=60)；向量模型四路消融（BERT-wwm / BGE-M3 / Qwen3-Emb-0.6B / -4B） |
| 重排 | `BAAI/bge-reranker-v2-m3` | 精排 Top-5 / Top-8 |
| 编排 | LangGraph | 事实整理 → 证据检索 → 专家路由 → 适配器调用 → 结果聚合 |
| 评测 | CLaw 基准 | 306 部法律 / 64,849 条法条 / 254 个最高法案例 |

## 2. 实验设计（两条线，不可混排）

**闭卷线（Closed-book）**：不允许检索外部知识库 → 用于与 CLaw 官方模型结果比较
**开卷线（RAG）**：允许查询法条向量库 → 验证系统实际应用能力，成绩必须注明增强条件

| 编号 | 模型训练 | 检索系统 | 用途 |
|---|---|---|---|
| E0 | 原始 Qwen3-8B | 无 | 基础闭卷基线 |
| E1 | QLoRA 法律微调 | 无 | 验证训练效果 |
| E2 | 原始 Qwen3-8B | 纯向量 RAG | 验证向量库效果 |
| E3 | 原始 Qwen3-8B | 向量 + 关键词 + 知识图谱 | 验证混合检索 |
| E4 | QLoRA 法律微调 | 纯向量 RAG | 验证训练 + 检索结合 |
| E5 | QLoRA 法律微调 | 混合检索 + 知识图谱 | 论文完整系统 |

> 底座统一为 `Qwen3-8B`。论文原稿写的是 Qwen2.5-7B，已升级为同尺寸级别的更新一代，
> 需在论文中同步修订（见 `docs/env_setup.md` 第 8 节）。

消融：去重排序器 / 去时间版本过滤 / 去知识图谱 / 去领域适配器路由 / 换向量模型（四路：Chinese-BERT-wwm-ext vs BGE-M3 vs Qwen3-Embedding-0.6B vs Qwen3-Embedding-4B）。

详见 [`docs/experiment_matrix.md`](docs/experiment_matrix.md)。

## 3. 目录结构

```
law_agent/
├─ data/
│  ├─ raw/          # 原始数据，只读，永不修改
│  ├─ normalized/   # 规范化后的法条 / 案例
│  ├─ train/        # 训练集
│  ├─ dev/          # 调参验证集
│  └─ benchmark/    # CLaw 测试集（严禁进入训练）
├─ corpus/
│  ├─ statutes/     # 法律法规文本
│  ├─ cases/        # 非 CLaw 训练案例
│  └─ metadata/     # 来源 / 日期 / 版本元数据
├─ configs/         # QLoRA、检索、评测配置
├─ models/
│  ├─ base/         # 底座模型权重
│  └─ adapters/     # 训练输出的 LoRA 适配器
├─ indexes/         # 向量库索引 / 图谱导出
├─ src/
│  ├─ prepare/      # 数据清洗、切分、固化
│  ├─ retrieval/    # 混合检索、RRF、重排序、图谱查询
│  ├─ train/        # QLoRA 训练
│  ├─ inference/    # 闭卷 / RAG 推理
│  ├─ evaluation/   # 检索指标、法条指标、Judge 评分
│  └─ common/       # 通用工具（IO、日志、hash）
├─ outputs/
│  ├─ closed_book/  # 闭卷输出
│  ├─ rag/          # RAG 输出
│  └─ judge/        # Judge 评分结果
├─ scripts/         # 环境脚本：activate / selfcheck / smoke_test / server_env
├─ reports/         # 最终结果表
└─ docs/            # 实验设计、环境说明、数据清单、防泄漏规则、模型选型
```

## 4. 环境（实验服务器）

> 论文原稿写的「Windows 11 + WSL2 + 12GB 显存」已作废。实验实际在实验服务器上完成。
> 完整环境说明见 **[`docs/env_setup.md`](docs/env_setup.md)**。

| 项 | 实测值 |
|---|---|
| GPU | **2 × NVIDIA A800 80GB PCIe**（160GB 显存，sm_80） |
| 驱动 | 550.163.01 → 驱动侧 **CUDA 12.4** 为上限（**决定 torch 版本**） |
| OS / 权限 | Ubuntu 20.04.6，**无 sudo** |
| Python | 3.11.16（uv 安装的 python-build-standalone，系统只有 3.8） |
| PyTorch | **`torch==2.6.0+cu124`**（不能用 cu13x，驱动会拒绝） |
| 图谱 | Neo4j 5.26 Community，Docker 部署，端口 7474 / 7687 |

### 快速开始

```bash
# 服务器上
source /mnt/data/lidian/law-agent/scripts/activate.sh   # 激活环境 + 设置全部变量
python scripts/selfcheck.py                            # 环境自检
python scripts/smoke_test.py                           # 端到端冒烟（加载模型 + LoRA 反向传播）
```

一键重建环境（幂等，可重复执行）：

```bash
bash scripts/server_env.sh
```

### 实测性能（Qwen3-8B）

| 指标 | 数值 |
|---|---|
| 4bit NF4 加载耗时 | 11 s |
| 4bit 常驻显存 | 5.66 GB |
| LoRA r=16 可训练参数 | 43.6 M（占 0.917%） |
| 单步训练峰值显存 | 9.14 GB |

单卡 80GB 意味着显存**不再是约束**：可提高 LoRA rank、放大 batch 与序列长度，
底座也完全可以换成 14B/32B。但对比实验只允许换一个变量——**底座一旦定下不要中途改**。

### 大陆网络注意

- `huggingface.co` **不可达** → 必须走 `hf-mirror.com`（已写入 `activate.sh`）
- `dist.neo4j.org` 返回 403 → Neo4j 改用 Docker 镜像
- `pypi.nvidia.com` 不可达 → torch 从清华 PyPI 装，不要用 PyTorch 官方索引单独装
- Docker 镜像加速器已在 daemon 配好，直接 `docker pull` 即可

> ⚠️ **PyTorch 版本必须按驱动 CUDA 主版本钉死**，且安装时**不要**用
> `--index-strategy unsafe-best-match`（会跨索引挑到 CUDA 13 版，导致 GPU 完全不可用）。
> 详见 `docs/env_setup.md` 第 3.3 节的事故记录。

## 5. 数据与合规（强制）

- `data/raw/` **只读**：下载后计算 SHA-256，记录来源、日期、版本，写入 `docs/data_manifest.json`。
- **CLaw 254 个案例的题目、参考答案、改写版本、Judge 评分解释，一律不得进入训练/验证集。**
- 训练集与测试集需做四级查重（完全 / 规范化 / MinHash / 向量），余弦 >0.92 人工复核。
- 法条必须保留**历史版本与生效/失效区间**，不得只存"最新版"。
- 仓库默认 **Private**：CLaw 数据存在再分发限制，论文发表前不公开。

详见 [`docs/leakage_rules.md`](docs/leakage_rules.md)。

## 6. 执行顺序

见 [`docs/run_order.md`](docs/run_order.md)。核心原则：**先跑通评测链路，再训练模型；先用 10 个案例验证 Judge，再跑 254 个。**

## 7. 结果完整性要求

每次运行必须保存**检索结果 + 最终答案 + 运行元数据**（模型哈希、适配器哈希、提示词哈希、
token 数、延迟），否则无法区分错误来源（未检索到 / 检索到未使用 / 版本引用错 / 模型推理错 / Judge 异常）。
