# 智能体（Agent）设计：检索 → 生成 → 引用核验 → 补检

> 论文题目《基于混合专家模型和知识图谱的法律领域智能体研究与构建》里，
> 「智能体」的工程落点就是本模块：**不是一条 RAG 管线，而是一个由工具调用
> 与自我核验组成的多步循环**。

## 1. 循环结构

```
        ┌────────────────────────────────────────────┐
        │                  循环器 LawAgent            │
        │                                            │
  问题 ─┼─▶ [工具1 检索] hybrid(dense+BM25+图谱+热度) │
        │        │ top-8 法条                         │
        │        ▼                                   │
        │   [工具2 生成] A0/专家 adapter，引用格式强制 │
        │        │ 答案                              │
        │        ▼                                   │
        │   [工具3 核验] 提取《法名》第X条 → 对库核验  │
        │        │                                   │
        │        ├─ 全部落实 ──────────▶ 终答         │
        │        └─ 有缺失/编造/出上下文              │
        │              ▼                             │
        │        补检：用缺失的法名条号重构查询        │
        │        合并候选 → 重新生成（≤ max_steps）    │
        └────────────────────────────────────────────┘
```

三个工具的边界与复用关系：

| 工具 | 实现 | 复用 |
|---|---|---|
| 检索 `RetrievalTool` | 加权 RRF（dense 1.0 / BM25 0.70 / 图谱 0.25）+ 热度，与 4b-5 最优 `hybrid` 配置逐参数一致 | 直接 import `smoke_retrieval.py` 的加载器/融合函数——同一份索引、单一事实源 |
| 生成 `GenerationTool` | Qwen3-8B + LoRA 适配器（nf4，adapter 目录自带 chat template，`enable_thinking=False`） | 与 `run_inference.py` 同口径，保证与训练可比 |
| 核验 `CitationVerifier` | 正则抽《法名》第X条（含中文条号换算）→ 三态判定：落实 / 疑似编造（库里无）/ 出上下文（库里有但没检到） | provision_index 复用检索报告同一构建器 |

## 2. 关键参数与理由

| 参数 | 值 | 为什么 |
|---|---|---|
| `max_steps` | 2 | 首轮 + 至多一轮补检。法律问答一轮补检即可覆盖"引用漏检"主场景；更多轮次的边际收益低、延迟翻倍。留作消融。 |
| `final_k` | 8 | 检索消融中 `hybrid` hit@5=0.669、hit@10=0.746；取 8 平衡上下文长度与覆盖。 |
| RRF 权重 | 1.0/0.70/0.25 | 与 4b-5 dev 扫描最优值一致，**不重新调参**——保证与检索报告可对照。 |
| `max_new_tokens` | 1024 | 与 inference.yaml 的 1536 相比收窄：RAG 答案比闭卷短。 |
| 量化 | nf4 4bit | 与训练/推理口径一致（bf16 只快 1.57 倍，已实测否决）。 |
| 补检查询构造 | `"原问题 + 缺失法名条号"` | 缺失引用本身就是最好的检索线索（比让模型自由改写可控）。 |

## 3. 与论文各章的对应

- **第 X 章（智能体设计）**：本 README 的循环结构与工具表。
- **实验**：`scripts/agent/run_agent_eval.py` 产出逐题 trace（JSONL）：
  补检触发率、引用三态分布（落实/编造/出上下文）、修复前后引用命中率、逐轮延迟
  ——即"智能体行为分析"的数据源，也是与"单轮 RAG"基线（max_steps=1）的消融对照。
- **trace 字段**：`steps[]`（每次工具调用的输入摘要与输出规模）、`citations_first/final`、
  `requery_triggered/reason`、`latency_sec`。

## 4. 运行

```bash
# 通路自检（不加载模型，秒级）
envs/main/bin/python -m src.agent.law_agent --dry-run

# 单题冒烟（GPU 空闲时）
envs/main/bin/python -m src.agent.law_agent --adapter models/adapters/A0_unified_qwen3_8b

# 批量评测（内部 test 抽样；JSONL trace + md 报告）
envs/main/bin/python scripts/agent/run_agent_eval.py \
  --adapter models/adapters/A0_unified_qwen3_8b \
  --limit 50 --max-steps 2 \
  --report-md docs/agent/AGENT_EVAL_A0.md
```

显存：检索 embedding 编码 + 生成 4bit 同卡约 10GB，另一卡空闲时也可分置
（`--device cuda:1` 给生成器）。
