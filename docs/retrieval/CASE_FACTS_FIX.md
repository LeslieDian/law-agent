# 案例 facts 挂载口径错位事故（2026-09-20 发现并修复）

## 现象

Neo4j 导入报告（9/19 版）显示 `case_rows_without_text = 62,519`（占 145,339 条案例的 **43%**）。
这些 Case 节点：**有真实向量**（单位范数、彼此余弦 0.36–0.47，非零向量）、
能被向量检索命中，但 `facts / question / output` 全空——命中后送生成器的是**空案例**。

## 根因：两处排除口径不一致

| 环节 | 排除口径 | 结果 |
|---|---|---|
| `build_embeddings.py`（建向量，scope=`eval`） | **只排 dev/test** | train/router 的 case_analysis 样本**进了向量库**（145,339 条全量） |
| `import_neo4j.py`（挂 facts） | **排 train/dev/test/router 四份** | 同一批样本的 `case_text` 查不到 → facts 空串 |

即：向量说"这些案例在库里"，挂载说"它们是训练数据不给正文"——**口径错位**，
且导入时无门禁，62,519 条空 facts **静默入库**。抽查实证：uid
`ShengbinYue__DISC-Law-SFT__DISC-Law-SFT-Pair:8240` 在 `train.jsonl` 命中、
qa 流中 `input` 435 字、向量 meta `tokens=326`（有真文本编码），Neo4j 中 facts 却为空。

## 影响

- **检索主结果不受影响**：4b-5 消融的 gold 全部是法条（union 口径），案例库只作
  SAME_CASE 图谱扩展与热度统计的输入，不直接出 gold。
- **受影响的是案例内容消费**：任何把命中案例正文喂给生成器的链路（含后续智能体
  tool-loop）会在 43% 的命中上拿到空文本。
- 图谱结构（SAME_CASE 6,991 条边涉及无正文节点）、热度表均不受影响。

## 修复（2026-09-20）

1. `scripts/retrieval/import_neo4j.py`：facts 挂载的排除口径改为**只排 dev/test**，
   与 `build_embeddings` 的 scope=`eval` 对齐（KB 含 train/router 案例——检索库是
   知识库不是训练数据，泄漏风险只有"评测 query 原文入库"，排 dev/test 即已堵死）。
2. 新增**硬门禁**：`case_rows_without_text / Case > 0.5%` 时 fail-fast 拒绝入库，
   空转 PASS 不可能再次静默通过。
3. 全量重导（`--step all`，CPU，~10 分钟，与训练无资源冲突），
   验收标准：`case_rows_without_text = 0`、9 个索引 ONLINE、
   `Case = 145,339`、`Law = 1,647`、`Provision = 72,449` 不变。

## 教训（并入论文工程实践一节）

- **两个模块共享同一批数据时，排除口径必须来自同一处定义**——本次是复制粘贴的
  排除清单各自演化导致的漂移。
- "挂载失败率"这类**覆盖率指标必须有门禁阈值**，否则 43% 的缺失会被 `verdict=PASS`
  掩盖（与 9/19 复扫空转 PASS 同一类教训）。
