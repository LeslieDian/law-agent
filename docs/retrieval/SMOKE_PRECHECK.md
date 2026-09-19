# 阶段 4b-5 检索链路 + 消融矩阵

> 生成时间：2026-09-19T14:13:51　脚本：`scripts/retrieval/smoke_retrieval.py`

## 1. 主结果（gold 口径 = `union`）

| 配置 | 链路 | query | gold | recall@5 | recall@10 | hit@5 | hit@10 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vector` | 纯向量（Qwen3-Embedding-0.6B） | 20 | 60 | **0.3500** | 0.4667 | 0.5500 | 0.6500 | 0.4467 |

> 目标：`recall@5 ≥ 0.85`、`recall@10 ≥ 0.90`。最佳配置 **`vector`**：recall@5 = **0.3500**（未达标）；recall@10 = **0.4667**（未达标）

## 2. 三种 gold 口径全表

| 配置 | 口径 | query | gold | recall@5 | recall@10 | hit@5 |
|---|---|---:|---:|---:|---:|---:|
| `vector` | union | 20 | 60 | 0.3500 | 0.4667 | 0.5500 |
| `vector` | system | 20 | 60 | 0.3500 | 0.4667 | 0.5500 |
| `vector` | output | 20 | 60 | 0.3500 | 0.4667 | 0.5500 |

## 3. 链路配置

- dense：`/mnt/data/lidian/law-agent/models/Qwen3-Embedding-0.6B`，1024 维，索引 72088 行（**只用 level=item**，doc 级排除）
- BM25：jieba 分词 + scipy 稀疏**精确**实现（词表 24172 / df≥2 / nnz 3008557）
- 图谱：1 跳，种子 20；`NEXT` 权重 1.0（主力）、`CITES` 权重 0.5（仅覆盖少数条文）
- RRF：k = 60
- 热度：`s *= (1 + 0.05 * hotness)`，热度表 72440 条（**只在检索库内统计，零泄漏**）
- 重排：**未启用**

## 4. gold 从哪来

- `system(tierA)`：数据集自带 `system` 的相关法律条文清单 —— 权威口径
- `output(tierB)`：从**参考答案** `output` 解析法条引用 —— 覆盖大，但是**下界**
- `union`：tierA ∪ tierB（主口径）

- dev∪test 原始行数 **2000**；有可评测 gold 的 **841** 条
- `system` 口径 **104** 条；`output` 口径 **840** 条
- gold 引用解析：`system` 命中 315 / 库里缺 0；`output` 命中 2241 / 库里缺 3
- 法名简称解析（exact → 唯一后缀，最小 2 字）：{'alias': 1964, 'exact': 128, 'ambiguous': 20, 'miss': 26}

## 5. 评测安全性

- 检索库与 train/dev/test/router 四份 split **强隔离**（4b-0 门禁 PASS），不存在「检索到题目本身」。

- 热度只用 CITES 入度与法源位阶（法条侧）/ 检索库内同案复现次数（案件侧），**不接触评测集**。

**耗时 42.8s**

