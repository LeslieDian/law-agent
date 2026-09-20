# 论文图索引（全部由真实报告数据生成，可溯源）

生成脚本模式：matplotlib（数据写死在脚本里，每个数字都能在仓库报告里找到出处）。
训练类图在 `docs/train/figures/`，其余分类如下。

## 数据集（dataset/）

| 图 | 内容 | 数据源 |
|---|---|---|
| `fig_corpus_pipeline.png` | 语料管线各阶段规模：312,365 → 308,767 → 294,499（去污）→ QA 270,989 + 法条 23,510 + 派生 13,484 | `docs/corpus/DECONTAMINATION_REPORT_PASS.*`、`SPLIT_STATS.json` |
| `fig_dataset_domain_mix.png` | train/test/router 四域配比（对数轴）；C1–C14 质检 PASS | `docs/corpus/SPLIT_STATS.json`、`SPLIT_VERIFY.md` |

## 向量库 / 知识图谱 / 检索（retrieval/）

| 图 | 内容 | 数据源 |
|---|---|---|
| `fig_knowledge_base.png` | Neo4j 节点（Law 1,647 / Provision 72,449 / Case 145,339）+ 案例四域分布；关系边 NEXT 69,530 / CITES 7,224 / SAME_CASE 10,742；9 索引 ONLINE | `docs/retrieval/NEO4J_IMPORT.md` |
| `fig_retrieval_ablation.png` | 检索消融主结果：5 配置 × hit@5 / recall@10 / MRR@10 + 达标线；hybrid = 深召回最优，dense_bm25 = 头部最优 | `docs/retrieval/SMOKE_RETRIEVAL.md`（2026-09-19 复现版） |
| `fig_dense_ceiling.png` | 纯 dense recall@k 天花板曲线（k=1…1000 → 0.8981），论证原目标作废依据 | `SMOKE_RETRIEVAL.md` §1c |

## 训练（../train/figures/）

| 图 | 内容 | 数据源 |
|---|---|---|
| `fig_a0_loss_curve.png` | A0 统一基线 train/dev loss（13,276 步） | `checkpoint-13276/trainer_state.json` |
| `fig_experts_training_progress.png` | 三专家训练进度快照（2026-09-20 11:25） | `logs/train/A0_criminal.log`、`A0_civil.log` |
| `fig_gpu_training_scene.png` | 双 A800 利用率/显存现场 | nvidia-smi 快照 |
| `TRAINING_SNAPSHOT_20260920.md`（同目录上级） | 原始现场证据（进程/命令行/GPU） | 远程服务器实况 |

**待补**：`fig_loss_comparison.png`（A0 vs 三专家 loss 对比）——等各专家 epoch-1 checkpoint
的 `trainer_state.json` 出来后生成（loss 不落控制台日志）。
