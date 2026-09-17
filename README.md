# 法律领域智能体：混合专家模型 + 知识图谱

> 论文《基于混合专家模型和知识图谱的法律领域智能体研究与构建》实验代码仓库

面向**民法、刑法、程序法**三法域的法律智能问答系统。以 Qwen2.5-7B-Instruct 为共享底座，
训练三个 QLoRA 领域适配器；以 Neo4j 统一存储法律知识图谱与案例向量索引；在线链路采用
「向量召回 → 图谱关系扩展 → RRF 融合 → 重排序 → 请求级适配器路由」的混合检索增强生成。

---

## 1. 技术路线

| 层 | 组件 | 说明 |
|---|---|---|
| 底座模型 | `Qwen/Qwen2.5-7B-Instruct` | 固定版本/commit hash |
| 领域适配 | QLoRA（4bit NF4 + 双重量化 + BF16） | 民法 / 刑法 / 程序法 三个适配器 + 请求级路由 |
| 知识层 | Neo4j（Community / Enterprise） | Law / LawVersion / Provision / Case / Cause / Court… |
| 检索 | BGE-M3 向量 + BM25/全文 + 图谱扩展 | 排名融合 RRF(k=60) |
| 重排 | `BAAI/bge-reranker-v2-m3` | 精排 Top-5 / Top-8 |
| 编排 | LangGraph | 事实整理 → 证据检索 → 专家路由 → 适配器调用 → 结果聚合 |
| 评测 | CLaw 基准 | 306 部法律 / 64,849 条法条 / 254 个最高法案例 |

## 2. 实验设计（两条线，不可混排）

**闭卷线（Closed-book）**：不允许检索外部知识库 → 用于与 CLaw 官方模型结果比较
**开卷线（RAG）**：允许查询法条向量库 → 验证系统实际应用能力，成绩必须注明增强条件

| 编号 | 模型训练 | 检索系统 | 用途 |
|---|---|---|---|
| E0 | 原始 Qwen2.5-7B | 无 | 基础闭卷基线 |
| E1 | QLoRA 法律微调 | 无 | 验证训练效果 |
| E2 | 原始 Qwen2.5-7B | 纯向量 RAG | 验证向量库效果 |
| E3 | 原始 Qwen2.5-7B | 向量 + 关键词 + 知识图谱 | 验证混合检索 |
| E4 | QLoRA 法律微调 | 纯向量 RAG | 验证训练 + 检索结合 |
| E5 | QLoRA 法律微调 | 混合检索 + 知识图谱 | 论文完整系统 |

消融：去重排序器 / 去时间版本过滤 / 去知识图谱 / 去领域适配器路由 / 换向量模型（Chinese-BERT-wwm-ext vs BGE-M3）。

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
├─ reports/         # 最终结果表
└─ docs/            # 实验设计、数据清单、防泄漏规则
```

## 4. 环境

Windows 11 + WSL2 Ubuntu，Python 3.10/3.11，CUDA 版 PyTorch，12GB 显存。
模型与向量模型**分阶段加载**，不同时常驻显存。

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

环境冻结：`pip freeze > requirements-lock.txt`

自检：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

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
