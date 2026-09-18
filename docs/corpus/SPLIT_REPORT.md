# 阶段 4：分层下采样 + 切分报告

> 生成脚本 `scripts/corpus/downsample_split.py`（确定性、幂等、输入只读）
> 生成时间 2026-09-18 16:22:58（服务器 aidda80061）

## 0. 口径

- 输入：`data/corpus/decontaminated/qa/*.jsonl`（已排除 `_all.jsonl`，防合并副本双计）
- 池记录：**268768**（`content_sha1` 全局唯一 → 样本键）
- `uid` 撞号：20947 条记录共用已存在的 uid（内容不同）→ 新增 `uid_g` 作唯一键；原 `uid` 原样保留
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

池剩余（未使用，可回溯）：**146768**

## 2. 训练集分层（域）

| 域 | 目标 | 实得 | 占比 | 达成率 | 池可用 | 池使用率 |
|---|---|---|---|---|---|---|
| criminal | 30000 | **30000** | 30.0% | 100.0% | 83717 | 35.8% |
| civil | 40000 | **40000** | 40.0% | 100.0% | 90330 | 44.3% |
| procedural | 20000 | **20000** | 20.0% | 100.0% | 21635 | 92.4% |
| general | 10000 | **10000** | 10.0% | 100.0% | 51086 | 19.6% |

## 3. 训练集分层（域 × 任务）—— 份额软上限的留痕

软上限 `--task-max-share 0.35`（单任务不得超过域配额的该比例）。
**程序法必须放宽**，原因写在表后。

| 域 | 任务数 | 实测最大单任务份额 | 最大份额任务 | 上限绑定 | 上限放宽 |
|---|---|---|---|---|---|
| criminal | 18 | **31.6%** | judgement_predit | 是 | 否 |
| civil | 17 | **35.0%** | legal_question_answering | 是 | 否 |
| procedural | 15 | **49.4%** | legal_question_answering | 是 | **是** |
| general | 14 | **35.0%** | legal_question_answering | 是 | 否 |

为什么程序法必须放宽：程序法训练池只有 **21635** 条，目标是 **20000** 条，仅 **1.08 倍** —— 可用量已吃掉池子的 92.4%。
而域内头号任务 `legal_question_answering` 独占池子的 45.6%。

**若强制 35% 上限，程序法 20% 配额在数学上不可达**：
其余任务的可用量之和不足以补上被砍掉的量。本脚本的处理是：
**先按上限分配 → 分配不完的部分自动放宽上限补齐 → 把 `cap_relaxed` 与实测最大份额如实写进本报告**。既不静默改口径，也不静默丢数据。

要真正把程序法单任务份额压下来，只有一条路：**扩大程序法来源** ——
阶段 2b 已切出 6601 条程序法条文（`data/corpus/statute_items/`），可派生「程序法条文任务」补量（README 3.1 已把「诉讼法条文任务」列为程序法来源之一），属**扩采/派生**动作，不在本阶段范围内。

| 域\|任务 | 配额 | 池可用 |
|---|---|---|
| procedural|legal_question_answering | 9871 | 11506 |
| procedural|exam | 5281 | 5281 |
| procedural|jud_read_compre | 1530 | 1530 |
| procedural|op_sum | 770 | 770 |
| procedural|judical_examination_v2 | 608 | 608 |
| procedural|legal_advice | 458 | 458 |
| procedural|judical_examination | 426 | 426 |
| procedural|sim_case_match | 362 | 362 |
| procedural|leg_eve_detec | 313 | 313 |
| procedural|qa | 250 | 250 |
| procedural|leg_case_cls | 49 | 49 |
| procedural|leg_ele_extra | 36 | 36 |
| procedural|legal_counsel_multi_turn | 32 | 32 |
| procedural|legal_counsel | 10 | 10 |
| procedural|judgement_predit | 4 | 4 |
| civil|legal_question_answering | 14000 | 36977 |
| criminal|judgement_predit | 9468 | 27105 |
| civil|jud_read_compre | 6894 | 14493 |
| criminal|jud_read_compre | 5548 | 15806 |
| civil|leg_eve_detec | 4461 | 9340 |
| criminal|legal_question_answering | 3812 | 10801 |
| criminal|sent_pred | 3531 | 9992 |
| general|legal_question_answering | 3500 | 19720 |
| civil|jud_doc_sum | 3463 | 7224 |
| civil|exam | 2949 | 6134 |
| civil|sim_case_match | 2862 | 5953 |
| general|leg_ele_extra | 2414 | 13223 |
| criminal|leg_case_cls | 1854 | 5156 |
| criminal|leg_ele_extra | 1772 | 4919 |
| civil|leg_ele_extra | 1617 | 3315 |
| criminal|leg_eve_detec | 1528 | 4217 |
| criminal|exam | 1124 | 3051 |
| civil|legal_advice | 970 | 1942 |
| general|leg_eve_detec | 960 | 4975 |
| general|exam | 929 | 4801 |
| general|legal_advice | 572 | 2778 |
| civil|op_sum | 549 | 1051 |
| criminal|op_sum | 517 | 1303 |
| civil|legal_counsel | 459 | 860 |
| civil|judical_examination_v2 | 451 | 844 |
| civil|qa | 386 | 705 |
| general|qa | 367 | 1614 |
| general|jud_read_compre | 308 | 1281 |
| civil|judical_examination | 280 | 481 |
| civil|leg_case_cls | 234 | 385 |
| general|op_sum | 232 | 849 |
| civil|judgement_predit | 226 | 365 |
| general|judical_examination_v2 | 217 | 759 |
| criminal|judical_examination_v2 | 209 | 413 |
| criminal|crime_concept | 195 | 375 |
| general|judical_examination | 187 | 598 |
| civil|legal_counsel_multi_turn | 156 | 218 |
| criminal|judical_examination | 152 | 251 |
| general|sim_case_match | 124 | 239 |
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
| ShengbinYue/DISC-Law-SFT | 88059 | 88.06% |
| Skepsun/lawyer_llama_data | 6070 | 6.07% |
| Dusker/lawyer-llama | 5871 | 5.87% |

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
- 域分布：civil 7383、criminal 6828、general 4089、procedural 1700
- 多标签（跨域）占比：5.46%
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
| `/mnt/data/lidian/law-agent/data/dev/civil.jsonl` | 400 | 2498907 | `d9d53db5901b0db7f39c6a482c3fc849…` |
| `/mnt/data/lidian/law-agent/data/dev/criminal.jsonl` | 300 | 1861890 | `8b281e5df892bb62e0d0e0b1bc03f70c…` |
| `/mnt/data/lidian/law-agent/data/dev/dev.jsonl` | 1000 | 5799576 | `19ca397eaad37e9471ea38e0cee59572…` |
| `/mnt/data/lidian/law-agent/data/dev/general.jsonl` | 100 | 438759 | `075002a4d187177b4e19aa05f62d77f7…` |
| `/mnt/data/lidian/law-agent/data/dev/procedure.jsonl` | 200 | 1000020 | `3f67a47044213265faddcf75fcf422e8…` |
| `/mnt/data/lidian/law-agent/data/router/router_train.jsonl` | 20000 | 137946564 | `c3a39443fbb6ee2d3199a5b52703aa36…` |
| `/mnt/data/lidian/law-agent/data/test/civil.jsonl` | 400 | 2372087 | `c4daadcea6407c0df2dd62290d95ae48…` |
| `/mnt/data/lidian/law-agent/data/test/criminal.jsonl` | 300 | 1898644 | `eb763a8c44bb8924dcd6bcb662e262df…` |
| `/mnt/data/lidian/law-agent/data/test/general.jsonl` | 100 | 413945 | `40ebbbc385319ac006a925664c232f2a…` |
| `/mnt/data/lidian/law-agent/data/test/procedure.jsonl` | 200 | 1051898 | `25e9f2bfb64c1a84586ed00c7d62fcfb…` |
| `/mnt/data/lidian/law-agent/data/test/test.jsonl` | 1000 | 5736574 | `0e7cc806209f2073dccbf8acfafb035e…` |
| `/mnt/data/lidian/law-agent/data/train/civil.jsonl` | 40000 | 253730660 | `8a0261332942fc59fe59521231b18ed2…` |
| `/mnt/data/lidian/law-agent/data/train/criminal.jsonl` | 30000 | 187185294 | `1bd5007705537bc98970809beacdba20…` |
| `/mnt/data/lidian/law-agent/data/train/general.jsonl` | 10000 | 41211235 | `33fed806d70ea72408370220d1ca5616…` |
| `/mnt/data/lidian/law-agent/data/train/procedure.jsonl` | 20000 | 103860553 | `d54a751f2249bdff9237a269859663d6…` |
| `/mnt/data/lidian/law-agent/data/train/train.jsonl` | 100000 | 585987742 | `f18a62025892c053e8c4ac2c606aa937…` |

完整 SHA-256 见 `SPLIT_STATS.json` 的 `outputs`。

