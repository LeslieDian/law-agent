# law-agent 终评结果矩阵

| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L | 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |
|---|---:|---:|---:|---:|---:|---|
| base (Qwen3-8B, 无微调) | 31.92 | 0.5389 | 0.5626 | — | — | 649 / 14150 / 0 |
| A0 (统一适配器) | 12.90 | 0.5412 | 0.0934 | 0.5400 | 0.5330 | 0 / 0 / 1000 |
| MoE-L2 (L2 门控混合) | 13.10 | 0.5340 | 0.2050 | 0.4923 | 0.5210 | 649 / 14150 / 1000 |
| 专家: criminal | — | — | — | 0.4872 | 0.5051 | 0 / 0 / 1000 |
| 专家: civil | — | — | — | 0.5028 | 0.4928 | 0 / 0 / 1000 |
| 专家: procedure | — | — | — | 0.4442 | 0.2975 | 0 / 0 / 1000 |

> LexRubric 归一化 % = mean_total / max_score × 100（逐 case 平均）；LexEval 客观 Acc 为 exact_match；ROUGE-L 为字符级 LCS f，cap=800。

## 缺失项

- `base`：缺 outputs/score/internal_base.json
- `criminal`：缺 outputs/score/lexeval_criminal.json
- `criminal`：缺 outputs/judge/lexrubric/criminal/SUMMARY.json
- `civil`：缺 outputs/score/lexeval_civil.json
- `civil`：缺 outputs/judge/lexrubric/civil/SUMMARY.json
- `procedure`：缺 outputs/score/lexeval_procedure.json
- `procedure`：缺 outputs/judge/lexrubric/procedure/SUMMARY.json
