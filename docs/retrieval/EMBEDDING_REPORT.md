# 阶段 4b-3 向量化报告

> 生成时间：2026-09-19T16:19:05　脚本：`scripts/retrieval/build_embeddings.py`

模型：`/mnt/data/lidian/law-agent/models/Qwen3-Embedding-0.6B`　**1024d**　max_seq_length **2048**　fp16=True　L2 归一化=是

> ★ 查询侧加 `Instruct: …\nQuery: …` 前缀，**文档侧不加**（Qwen3-Embedding 是 instruction-aware 模型）。

## 1. 库分层：case 4 库（1 通用 + 3 子库）+ 法条 1 库

| collection | 类型 | 域 | 条数 | 分片 | 维度 | 超长截断 | 最长 token | 耗时 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `provision_embedding` | 法条 | - | **72440** | 15 | None | 0 | 0 | 0.0s |
| `case_embedding_general` | 案件 | general | **22263** | 5 | 1024 | 48 | 2748 | 233.2s |
| `case_embedding_civil` | 案件 | civil | **45638** | 10 | 1024 | 1613 | 3617 | 923.7s |
| `case_embedding_criminal` | 案件 | criminal | **74170** | 15 | 1024 | 131 | 14583 | 440.7s |
| `case_embedding_procedural` | 案件 | procedural | **3268** | 1 | 1024 | 56 | 4694 | 41.7s |
| **合计** | | | **217779** | | | | | |

| 库 | 条数 |
|---|---:|
| `provision_embedding`（item 72088 + doc 352） | 72440 |
| `case_embedding_general`（general） | 22263 |
| `case_embedding_civil`（civil） | 45638 |
| `case_embedding_criminal`（criminal） | 74170 |
| `case_embedding_procedural`（procedural） | 3268 |

> 检索主入口 = 4 个 case 库（以 case 为节点）；法条库是命中后沿
> `HAS_PROVISION` / `CITES` / `NEXT` 扩展的对象。四域与 MoE 专家一一对齐。

## 2. 长文本处置

- item > 1000 字 → 去表 + 截断到 2000 字
- 去表后 < 200 字 → **只入图谱、不入向量索引**：**9 条**
  - `中华人民共和国资源税法-a17`（5587 字 → 74 字）
  - `中华人民共和国船舶吨税法-a22`（1871 字 → 65 字）
  - `中华人民共和国车船税法-a13`（2387 字 → 66 字）
  - `中华人民共和国耕地占用税法-a16`（1105 字 → 79 字）
  - `中华人民共和国车船税法-a13-v2`（2345 字 → 65 字）
  - `中华人民共和国环境保护税法-a28`（1018 字 → 32 字）
  - `中华人民共和国消费税暂行条例-a17`（2819 字 → 38 字）
  - `最高人民法院、最高人民检察院关于办理非法制造、买卖、运输、储存毒鼠强等禁用剧毒化学品刑事案件`（2499 字 → 58 字）
  - `最高人民法院关于审理破坏野生动物资源刑事案件具体应用法律若干问题的解释-a12`（26980 字 → 93 字）
- doc 级 352 条单独成 collection（不与 item 同池）

## 3. 案件池

| 项 | 值 |
|---|---:|
| 入库条数（uid 唯一） | **145339** |
| 其中唯一 `case_sha1` | 134597 |

> `case_sha1` 是语料的案件正文指纹（尾部 400 字），**不是**语义案件 ID ——
> 同一案件的不同任务视角（判决预测 / 文书摘要 / 要素抽取）会产生多条样本。
> 本阶段**按样本级入库**（口径与 4b-0 门禁一致，与四份 split 强隔离），
> 同案多视角在 4b-4 用 `SAME_CASE` 关系显式关联，**不删节点**。

### 3.1 案件池划分账本（不变量，替代旧写死的 82,821）

| 项 | 条数 |
|---|---:|
| `qa_rows` | 270989 |
| `kept` | 145339 |
| `skip_not_case` | 124643 |
| `skip_in_splits` | 1007 |

> 恒等式 `kept + Σskip_* == qa_rows` = **true**。
> 案件池大小随阶段 4 切分变化（split 一重跑，被排除的 uid 就变），因此**不写死具体数字**，
> 只断言账本自洽 + 落在 3 万～12 万的合理带内；准确数字见上表与报告 JSON。

## 4. 断言

- `provision_items_complete` = true
- `case_pool_accounted` = true
- `case_pool_le_all_case_analysis` = true
- `case_pool_plausible_band` = true
- `case_not_in_eval_splits` = true
- `provision_not_in_eval_splits` = true
- `eval_scope_is_dev_test_only` = true
- `no_empty_text` = true
- `inputs_are_official_only` = true
- `case_domains_cover_pool` = true
- `case_domains_all_nonempty` = true
- `provision_pool_complete` = true

**verdict = PASS**　总耗时 1662.0s

