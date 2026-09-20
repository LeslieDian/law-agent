### 阶段进度（自动汇总）

> **本段由自动化回填**（`scripts/eval/stage_status.py` 扫描产物/日志/标志后生成，经 `patch_readme.py --tag AUTO_STAGE` 贴入）。每完成一个阶段刷新一次，并自动提交推送。机读版：`docs/eval/STAGE_STATUS.json`。
>
> 生成时间 **2026-09-20 22:56:01** ｜ 已完成 **2/19** 项

| 阶段 | 项 | 状态 | 进度 | 备注 |
|---|---|---|---|---|
| A0 评测链 | LexRubric 649 | ✅ 已完成 | 649/649 | cap=1536；AI 裁判判分 |
| A0 评测链 | LexEval 客观 11,400 + 生成 2,750 | 🔄 进行中 | 832/14150 | 客观 cap=256 / 生成 cap=1536 |
| 阶段 6 门控 | L2 门控训练 | 🔄 进行中 | 745/1125 | 72 gates / 2.36M 参数；→ 等 gate_weights.pt |
| MoE 队列 | Q1 MoE 冒烟 | ⬜ 待跑 | 0/2 | 等门控权重 |
| MoE 队列 | Q2 MoE 内部集 1k | ⬜ 待跑 | 0/1000 | — |
| MoE 队列 | Q3 MoE LexRubric 649 | ⬜ 待跑 | 0/649 | — |
| MoE 队列 | Q4 MoE 客观 11,400 | ⬜ 待跑 | 0/11400 | — |
| MoE 队列 | Q5 MoE 生成 2,750 | ⬜ 待跑 | 0/14150 | — |
| MoE 队列 | Q6 A0 内部集 cap=1024 | ⬜ 待跑 | 0/1000 | — |
| base 评测 | LexRubric 649 | ⬜ 待跑 | 0/649 | cap=1536 |
| base 评测 | LexEval 客观+生成 14,150 | ⬜ 待跑 | 0/14150 | — |
| 三专家内部集 | internal_criminal 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_civil 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_procedure 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 判分(API) | MiniMax-M3 × A0_unified_qwen3_8b | 🔄 进行中 | — | 649 题 / 22 维度；冒烟 8 条中 |
| 判分(API) | MiniMax-M3 × moe_L2 | 🔄 进行中 | — | 649 题 / 22 维度；冒烟 8 条中 |
| 判分(API) | MiniMax-M3 × base | 🔄 进行中 | — | 649 题 / 22 维度；冒烟 8 条中 |
| 收尾 | 汇总 + 出图 | ✅ 已完成 | — | collect_results.py + make_figures.py；最近一次 09-20 20:33 |
| 收尾 | 过夜链整链 | 🔄 进行中 | — | MARKER_OVERNIGHT_DONE |

<sub>状态来源：答案文件行数 / 日志 `rc=` 与 `[n/N]` 进度 / 完成标志 / 报告文件。未到位一律如实标注，不做推测。</sub>
