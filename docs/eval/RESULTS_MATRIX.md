# law-agent 终评结果矩阵

| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L | 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |
|---|---:|---:|---:|---:|---:|---|
| base (Qwen3-8B, 无微调) | 31.91 | — | — | — | — | 649 / 14150 / 0 |
| A0 (统一适配器) | 12.90 | 0.5412 | 0.0934 | 0.5378 | 0.5386 | 0 / 0 / 1000 |
| MoE-L2 (L2 门控混合) | 13.10 | 0.5340 | 0.2050 | — | — | 649 / 14150 / 1000 |

> LexRubric 归一化 % = mean_total / max_score × 100（逐 case 平均）；LexEval 客观 Acc 为 exact_match；ROUGE-L 为字符级 LCS f，cap=800。

## 缺失项

- `base`：缺 outputs/score/lexeval_base.json
- `base`：缺 outputs/score/internal_base.json
- `moe_L2`：缺 outputs/score/internal_moe_L2.json
