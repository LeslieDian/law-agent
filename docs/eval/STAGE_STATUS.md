<!-- AUTO_STAGE_BEGIN 由 scripts/eval/stage_status.py 自动生成，勿手改 -->

### 阶段进度（自动汇总）

> **本段由自动化回填**（`scripts/eval/stage_status.py` 扫描产物/日志/标志后生成，经 `patch_readme.py --tag AUTO_STAGE` 贴入）。每完成一个阶段刷新一次，并自动提交推送。机读版：`docs/eval/STAGE_STATUS.json`。
>
> 生成时间 **2026-09-22 09:54:25** ｜ 已完成 **21/21** 项

| 阶段 | 项 | 状态 | 进度 | 备注 |
|---|---|---|---|---|
| A0 评测链 | LexRubric 649 | ✅ 已完成 | 649/649 | cap=1536；AI 裁判判分 |
| A0 评测链 | LexEval 客观 11,400 + 生成 2,750 | ✅ 已完成 | 14150/14150 | 客观 cap=256 / 生成 cap=1536 |
| A0 评测链 | 整链标志 | ✅ 已完成 | — | MARKER_FULL_INFER_A0_DONE |
| 阶段 6 门控 | L2 门控训练 | ✅ 已完成 | 1125/1125 | 72 gates / 2.36M 参数；→ gate_weights.pt 已出 |
| MoE 队列 | Q1 MoE 冒烟 | ✅ 已完成 | 2/2 | — |
| MoE 队列 | Q2 MoE 内部集 1k | ✅ 已完成 | 1000/1000 | — |
| MoE 队列 | Q3 MoE LexRubric 649 | ✅ 已完成 | 649/649 | — |
| MoE 队列 | Q4 MoE 客观 11,400 | ✅ 已完成 | 11400/11400 | — |
| MoE 队列 | Q5 MoE 生成 2,750 | ✅ 已完成 | 14150/14150 | — |
| MoE 队列 | Q6 A0 内部集 cap=1024 | ✅ 已完成 | 1000/1000 | — |
| MoE 队列 | 整队列标志 | ✅ 已完成 | — | MARKER_GPU1_QUEUE_DONE（本轮 MoE 队列） |
| base 评测 | LexRubric 649 | ✅ 已完成 | 649/649 | cap=1536 |
| base 评测 | LexEval 客观+生成 14,150 | ✅ 已完成 | 14150/14150 | — |
| 三专家内部集 | internal_criminal 1k | ✅ 已完成 | 1000/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_civil 1k | ✅ 已完成 | 1000/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_procedure 1k | ✅ 已完成 | 1000/1000 | cap=1024；域专业化分析 |
| 判分(API) | MiniMax-M3 × A0_unified_qwen3_8b | ✅ 已完成 | 649/649 | 649 题 / 22 维度 |
| 判分(API) | MiniMax-M3 × moe_L2 | ✅ 已完成 | 649/649 | 649 题 / 22 维度 |
| 判分(API) | MiniMax-M3 × base | ✅ 已完成 | 649/649 | 649 题 / 22 维度 |
| 收尾 | 汇总 + 出图 | ✅ 已完成 | — | collect_results.py + make_figures.py；最近一次 09-22 09:44 |
| 收尾 | 过夜链整链 | ✅ 已完成 | — | MARKER_OVERNIGHT_DONE |

<sub>状态来源：答案文件行数 / 日志 `rc=` 与 `[n/N]` 进度 / 完成标志 / 报告文件。未到位一律如实标注，不做推测。</sub>

<!-- AUTO_STAGE_END -->
