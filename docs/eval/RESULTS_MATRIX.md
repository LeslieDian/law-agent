# law-agent 终评结果矩阵

| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L | 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |
|---|---:|---:|---:|---:|---:|---|
| A0 (统一适配器) | — | — | — | 0.5378 | 0.5386 | 0 / 0 / 1000 |

> LexRubric 归一化 % = mean_total / max_score × 100（逐 case 平均）；LexEval 客观 Acc 为 exact_match；ROUGE-L 为字符级 LCS f，cap=800。

## 缺失项

- `A0`：缺 outputs/score/lexeval_A0.json
- `A0`：缺 outputs/judge/lexrubric/A0/SUMMARY.json
