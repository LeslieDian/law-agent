# 阶段 4：分层下采样 + 切分报告

> 生成脚本 `scripts/corpus/downsample_split.py`（确定性、幂等、输入只读）
> 生成时间 2026-09-18 19:50:19（服务器 aidda80061）

## 0. 口径

- 输入：`data/corpus/decontaminated/qa/*.jsonl`（真实 QA 流）+ `data/corpus/derived/*.jsonl`（阶段 2c 派生流）；已排除 `无`（合并副本，读它必然双计）
- 池记录：**284473** = 真实 **270989** + 派生(合成) **13484**（`content_sha1` 全局唯一 → 样本键）
- `uid` 撞号（旧口径 `uid_legacy`）：**0** 条 —— 阶段 2 已把 `uid` 修为`<dataset>__<file_stem>:<source_index>`，与 `uid_g` 同口径，不再撞号；`uid_legacy` 仅作血缘回溯。
- **派生样本只进 train**：router/val/test 全部来自真实样本（评测公平性）。
- 切分方式：sha1 排序确定性抽取（无随机数）；四份 split 两两不相交
- 目标配比：criminal 30000、civil 40000、procedural 20000、general 10000

## 1. 四份 split 规模（目标 vs 实得）

| split | 文件 | 目标 | 实得 |
|---|---|---|---|
| 训练集（A0 全域） | `data/train/train.jsonl` | 100000 | **100000** |
| 路由集 | `data/router/router_train.jsonl` | 20000 | **20000** |
| 验证集 | `data/dev/dev.jsonl` | 1000 | **1000** |
| 测试集 | `data/test/test.jsonl` | 1000 | **1000** |
| **合计占用** | | | **122000** |

池剩余（未使用，可回溯）：**162473**

## 2. 训练集分层（域）

| 域 | 目标 | 实得 | 占比 | 达成率 | 池可用 | 池使用率 |
|---|---|---|---|---|---|---|
| criminal | 30000 | **30000** | 30.0% | 100.0% | 84013 | 35.7% |
| civil | 40000 | **40000** | 40.0% | 100.0% | 91504 | 43.7% |
| procedural | 20000 | **20000** | 20.0% | 100.0% | 35464 | 56.4% |
| general | 10000 | **10000** | 10.0% | 100.0% | 51492 | 19.4% |

## 3. 训练集分层（域 × 任务）—— 三层约束的留痕

- 单任务软上限 `--task-max-share 0.35`：单任务不得超过域配额的该比例。
- **派生组上限** `--derived-max-share 0.15`：合成样本合计不得超过域配额的该比例（防止「程序法专家只会背法条」）。
- 放宽顺序：① 单任务上限 → ② 派生组上限；每级放宽都记进本表。

| 域 | 任务数 | 最大单任务份额 | 最大份额任务 | 单任务上限放宽 |
|---|---|---|---|---|
| criminal | 18 | **31.5%** | judgement_predit | 否 |
| civil | 17 | **35.0%** | legal_question_answering | 否 |
| procedural | 18 | **35.0%** | legal_question_answering | 否 |
| general | 14 | **35.0%** | legal_question_answering | 否 |

| 域 | 训练集 | 其中派生(合成) | 派生份额 | 派生池可用 | 派生上限 | 派生上限放宽 |
|---|---|---|---|---|---|---|
| criminal | 30000 | 0 | 0.0% | 0 | - | 否 |
| civil | 40000 | 0 | 0.0% | 0 | - | 否 |
| procedural | 20000 | 3000 | 15.0% | 13484 | 3000 | 否 |
| general | 10000 | 0 | 0.0% | 0 | - | 否 |

**程序法口径**：目标 20000 条，训练池 35464 条（其中派生 13484 条，占配额 15.0%）。
程序法**本轮无需放宽单任务上限**：阶段 2c 已用程序法条文派生「程序法条文任务」把池子加宽（README 3.1 的「诉讼法条文任务」来源），头号任务份额从 49.4% 降到 35.0%（上限 35%）。

| 域\|任务 | 配额 | 池可用（训练池口径） |
|---|---|---|
| procedural|legal_question_answering | 7000 | 11842 |
| procedural|exam | 5209 | 5288 |
| procedural|jud_read_compre | 1510 | 1532 |
| procedural|statute_locate | 1228 | 5605 |
| procedural|statute_recall | 1227 | 5605 |
| procedural|op_sum | 760 | 770 |
| procedural|judical_examination_v2 | 601 | 608 |
| procedural|statute_structure | 545 | 2274 |
| procedural|legal_advice | 452 | 458 |
| procedural|judical_examination | 421 | 426 |
| procedural|sim_case_match | 359 | 362 |
| procedural|leg_eve_detec | 310 | 313 |
| procedural|qa | 247 | 250 |
| procedural|leg_case_cls | 49 | 49 |
| procedural|leg_ele_extra | 36 | 36 |
| procedural|legal_counsel_multi_turn | 32 | 32 |
| procedural|legal_counsel | 10 | 10 |
| procedural|judgement_predit | 4 | 4 |
| civil|legal_question_answering | 14000 | 38117 |
| criminal|judgement_predit | 9441 | 27126 |
| civil|jud_read_compre | 6894 | 14501 |
| criminal|jud_read_compre | 5533 | 15818 |
| civil|leg_eve_detec | 4461 | 9345 |
| criminal|legal_question_answering | 3883 | 11045 |
| criminal|sent_pred | 3521 | 9997 |
| general|legal_question_answering | 3500 | 20109 |
| civil|jud_doc_sum | 3463 | 7230 |
| civil|exam | 2948 | 6136 |
| civil|sim_case_match | 2862 | 5955 |
| general|leg_ele_extra | 2414 | 13229 |
| criminal|leg_case_cls | 1848 | 5158 |
| criminal|leg_ele_extra | 1766 | 4921 |
| civil|leg_ele_extra | 1619 | 3320 |
| criminal|leg_eve_detec | 1524 | 4221 |
| criminal|exam | 1121 | 3053 |
| civil|legal_advice | 970 | 1943 |
| general|leg_eve_detec | 959 | 4978 |
| general|exam | 929 | 4805 |
| general|legal_advice | 572 | 2778 |
| civil|op_sum | 549 | 1052 |
| criminal|op_sum | 516 | 1304 |
| civil|legal_counsel | 459 | 861 |
| civil|judical_examination_v2 | 451 | 845 |
| civil|qa | 385 | 705 |
| general|qa | 367 | 1615 |
| general|jud_read_compre | 309 | 1283 |
| civil|judical_examination | 280 | 481 |
| civil|leg_case_cls | 235 | 385 |
| general|op_sum | 232 | 849 |
| civil|judgement_predit | 226 | 367 |
| general|judical_examination_v2 | 216 | 759 |
| criminal|judical_examination_v2 | 209 | 415 |
| criminal|crime_concept | 195 | 375 |
| general|judical_examination | 188 | 598 |
| civil|legal_counsel_multi_turn | 155 | 218 |
| criminal|judical_examination | 153 | 252 |
| general|sim_case_match | 124 | 240 |
| criminal|legal_advice | 120 | 158 |
| general|legal_counsel_multi_turn | 113 | 172 |
| criminal|qa | 84 | 84 |
| general|leg_case_cls | 73 | 73 |
| civil|sent_pred | 43 | 43 |
| criminal|jud_doc_sum | 40 | 40 |
| criminal|sim_case_match | 30 | 30 |
| criminal|legal_counsel | 13 | 13 |
| general|judgement_predit | 4 | 4 |
| criminal|legal_counsel_multi_turn | 3 | 3 |

## 4. 训练集分层（来源）

| 来源 | 条数 | 占比 |
|---|---|---|
| ShengbinYue/DISC-Law-SFT | 85095 | 85.09% |
| Skepsun/lawyer_llama_data | 6048 | 6.05% |
| Dusker/lawyer-llama | 5857 | 5.86% |
| derived/statute-items | 3000 | 3.00% |

## 5. 验证集 / 测试集分布（按目标配比分层）

| 域 | val | test |
|---|---|---|
| criminal | 300 | 300 |
| civil | 400 | 400 |
| procedural | 200 | 200 |
| general | 100 | 100 |

## 6. 路由集

- 规模 **20000**（= 训练目标的 20%）
- 每条带 `router_label`（多标签 `domains`）与 `router_query`（user 侧问题文本）
- 域分布：civil 7401、criminal 6787、general 4104、procedural 1708
- 多标签（跨域）占比：5.43%
- ⚠️ 阶段 7 所需的「贴近 CLaw 254 案分布的 **200–500 条人工标注**路由评测集」**必须新建**，不在本阶段产物内。

## 7. 法条条目流（检索语料，不进 SFT）

- 条目总数 **65037**（`data/corpus/statute_items/`，阶段 2b 产物）
- 逐域：civil 12356、general 42834、criminal 3246、procedural 6601
- 逐 level：item 64700、doc 337
- 逐 status：未知 982、有效 46810、已修改 12547、已废止 4635、尚未生效 63
- **不做下采样**：它是检索语料（向量库主料），不是 SFT 样本 ——检索侧语料越多越好，砍它没有收益只会掉召回。
- 跨流重叠检查：QA 四份 split 的 `content_sha1` ∩ 法条条目 `content_sha1` = **0**
- 若将来要「法条任务」样本（README 3.1 的程序法来源之一），须**另派生**并保持与 val/test disjoint —— 本阶段未做。

## 8. 输出文件（逐个 SHA-256）

| 文件 | 行数 | 字节 | SHA-256 |
|---|---|---|---|
| `/mnt/data/lidian/law-agent/data/dev/civil.jsonl` | 400 | 2529737 | `54c1cf0fcaca01bc3e9e3b83a110ff57…` |
| `/mnt/data/lidian/law-agent/data/dev/criminal.jsonl` | 300 | 1874492 | `730d6f17e98aa591c7c12c39a2c2d5a6…` |
| `/mnt/data/lidian/law-agent/data/dev/dev.jsonl` | 1000 | 5860423 | `b911b5dec2031ccc901337bbc5cce749…` |
| `/mnt/data/lidian/law-agent/data/dev/general.jsonl` | 100 | 440134 | `fa18a4f2e0c1ce32c2e50ec653c10777…` |
| `/mnt/data/lidian/law-agent/data/dev/procedure.jsonl` | 200 | 1016060 | `ab48f2027169ef3b8d1f83796e4593cf…` |
| `/mnt/data/lidian/law-agent/data/router/router_train.jsonl` | 20000 | 139299672 | `e730712c17c17ec6aaea1ea7571106be…` |
| `/mnt/data/lidian/law-agent/data/test/civil.jsonl` | 400 | 2407133 | `c1f2b3dcec8f2e13a1f6fd08fa635a81…` |
| `/mnt/data/lidian/law-agent/data/test/criminal.jsonl` | 300 | 1921846 | `f538623da3a0dadd1489514d72d10049…` |
| `/mnt/data/lidian/law-agent/data/test/general.jsonl` | 100 | 424190 | `9af5bb4da10c1522b8cbe8a786d24cca…` |
| `/mnt/data/lidian/law-agent/data/test/procedure.jsonl` | 200 | 1066047 | `136fe8a4dcc56d7dba89ee85711f4cf0…` |
| `/mnt/data/lidian/law-agent/data/test/test.jsonl` | 1000 | 5819216 | `a8fd21a2b8031c146dd30fae66cbcdb4…` |
| `/mnt/data/lidian/law-agent/data/train/civil.jsonl` | 40000 | 256964375 | `9f9171cd433455b61653db772795c5e1…` |
| `/mnt/data/lidian/law-agent/data/train/criminal.jsonl` | 30000 | 189321106 | `5eb1e81b210f80741571c4449b60bd76…` |
| `/mnt/data/lidian/law-agent/data/train/general.jsonl` | 10000 | 42035706 | `b7316a21d15fa154bab7df0856e6ef6c…` |
| `/mnt/data/lidian/law-agent/data/train/procedure.jsonl` | 20000 | 98401757 | `9b8bd0abf735f22ecbeb39f1146084d9…` |
| `/mnt/data/lidian/law-agent/data/train/train.jsonl` | 100000 | 586722944 | `3a3b2e9c0135c3c580883c5afb3f2f1c…` |

完整 SHA-256 见 `SPLIT_STATS.json` 的 `outputs`。

