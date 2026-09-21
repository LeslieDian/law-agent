<!-- AUTO_RESULTS_BEGIN 由 scripts/eval/collect_results.py 自动生成，勿手改 -->
### 终评结果（自动汇总）

| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L | 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |
|---|---:|---:|---:|---:|---:|---|
| A0 (统一适配器) | 12.85 | — | — | 0.5378 | 0.5386 | 0 / 0 / 1000 |
| MoE-L2 (L2 门控混合) | — | — | — | — | — | 32 / 0 / 1000 |

> 生成口径（全系统统一）：LexEval 客观题 cap=256 / LexEval 生成题 cap=1536 / LexRubric cap=1536 / 内部验证集 cap=1024；
> 判分 MiniMax-M3（`configs/judge.yaml`）；完整机读数据见 `docs/eval/RESULTS_MATRIX.json`，图见 `docs/figures/`。
<!-- AUTO_RESULTS_END -->
