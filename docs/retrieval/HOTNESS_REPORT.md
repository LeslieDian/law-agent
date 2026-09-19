# 阶段 4b-2b 热度权重报告

> 生成时间：2026-09-19T16:19:09　脚本：`scripts/retrieval/build_hotness.py`　结论：**PASS**

## 1. 公式（论文实现细节，口径不得改动而不声明）

```
norm_log1p             log1p(x)/log1p(max_x)
provision_hotness      0.85*norm_log1p(in_cites) + 0.15*rank_norm
case_hotness           norm_log1p(n_same_case)
ranking                final = rrf_score * (1 + W_HOT * hotness)   [W_HOT 默认 0.05]
```

## 2. 规模

| 项 | 值 |
|---|---:|
| provision | 72440 |
| case | 145339 |
| provision_meta_rows | 72440 |
| case_meta_rows | 145339 |
| cites_edges | 7224 |
| unique_case_sha1 | 134597 |

| 入度分档 | 条文数 |
|---|---:|
| in_cites==0 | 68413 |
| in_cites>=1 | 4027 |
| in_cites>=5 | 147 |
| in_cites>=20 | 5 |

| 法源 | 条文数 |
|---|---:|
| doc | 352 |
| item | 72088 |

| 域 | 条文数 |
|---|---:|
| civil | 15169 |
| criminal | 4326 |
| general | 45511 |
| procedural | 7434 |

## 3. 热点条文 Top20（按热度）

| provision_id | 热度 | 被引 | 法律 | 法源 |
|---|---:|---:|---|---|
| `中华人民共和国民法典-a510` | 1.0000 | 26 | 中华人民共和国民法典 | 法律 |
| `中华人民共和国合同法-a61` | 1.0000 | 26 | 中华人民共和国合同法 | 法律 |
| `中华人民共和国民用航空法-a128` | 0.9472 | 21 | 中华人民共和国民用航空法 | 法律 |
| `中华人民共和国民用航空法-a128-v2` | 0.9472 | 21 | 中华人民共和国民用航空法 | 法律 |
| `中华人民共和国民用航空法-a128-v3` | 0.9472 | 21 | 中华人民共和国民用航空法 | 法律 |
| `中华人民共和国民事诉讼法-a200` | 0.8954 | 17 | 中华人民共和国民事诉讼法 | 法律 |
| `中华人民共和国民事诉讼法-a200-v2` | 0.8954 | 17 | 中华人民共和国民事诉讼法 | 法律 |
| `中华人民共和国食品安全法-a50` | 0.8651 | 15 | 中华人民共和国食品安全法 | 法律 |
| `中华人民共和国行政许可法-a12` | 0.8651 | 15 | 中华人民共和国行政许可法 | 法律 |
| `中华人民共和国食品安全法-a50-v2` | 0.8651 | 15 | 中华人民共和国食品安全法 | 法律 |
| `中华人民共和国食品安全法-a50-v3` | 0.8651 | 15 | 中华人民共和国食品安全法 | 法律 |
| `中华人民共和国行政许可法-a12-v2` | 0.8651 | 15 | 中华人民共和国行政许可法 | 法律 |
| `中华人民共和国刑法-a234` | 0.8306 | 13 | 中华人民共和国刑法 | 法律 |
| `中华人民共和国刑法-a234-v2` | 0.8306 | 13 | 中华人民共和国刑法 | 法律 |
| `中华人民共和国行政诉讼法-a91` | 0.8306 | 13 | 中华人民共和国行政诉讼法 | 法律 |
| `中华人民共和国土地管理法-a44` | 0.8115 | 12 | 中华人民共和国土地管理法 | 法律 |
| `中华人民共和国土地管理法-a44-v2` | 0.8115 | 12 | 中华人民共和国土地管理法 | 法律 |
| `中华人民共和国民用航空法-a119` | 0.8115 | 12 | 中华人民共和国民用航空法 | 法律 |
| `中华人民共和国民用航空法-a129` | 0.8115 | 12 | 中华人民共和国民用航空法 | 法律 |
| `中华人民共和国民用航空法-a119-v2` | 0.8115 | 12 | 中华人民共和国民用航空法 | 法律 |

## 4. 热点案件 Top20（按同案复现次数）

| uid | 热度 | 同案复现 |
|---|---:|---:|
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159137` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159189` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159207` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159273` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159426` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159585` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159867` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:159911` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160007` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160110` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160161` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160172` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160209` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160339` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160390` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160559` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160570` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:160991` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:161017` | 1.0000 | 82 |
| `ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:161328` | 1.0000 | 82 |

## 5. 泄漏说明

案件复现次数**只在检索库内**统计，不含 `train` / `dev` / `test` / `router` 的任何样本；
provision 热度只用 CITES 边与法源位阶，与评测集无关。**本条须写进论文。**
