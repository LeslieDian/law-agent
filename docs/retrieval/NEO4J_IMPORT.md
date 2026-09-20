# 阶段 4b-4 Neo4j 导入报告

> 生成时间：2026-09-20T13:17:13　脚本：`scripts/retrieval/import_neo4j.py`　step=`all`

## 0. 库布局（case 4 库 + 法条 1 库）

| 磁盘 collection | Neo4j 向量索引 | 属性 |
|---|---|---|
| `provision_embedding` (level=item) | `provision_embedding` | `Provision.embedding` |
| `provision_embedding` (level=doc) | `provision_doc_embedding` | `Provision.embedding_doc` |
| `case_embedding_general` | `case_embedding_general` | `Case.embedding_general` |
| `case_embedding_civil` | `case_embedding_civil` | `Case.embedding_civil` |
| `case_embedding_criminal` | `case_embedding_criminal` | `Case.embedding_criminal` |
| `case_embedding_procedural` | `case_embedding_procedural` | `Case.embedding_procedural` |

## 1. 计数

| 项 | 数量 |
|---|---:|
| `Law` | 1647 |
| `Provision_item_with_vec` | 72088 |
| `Provision_doc_with_vec` | 352 |
| `Provision_no_vector` | 0 |
| `Provision` | 72449 |
| `Case` | 145339 |
| `case_rows_by_domain` | general=22263, civil=45638, criminal=74170, procedural=3268 |
| `case_rows_without_text` | 0 |
| `HAS_PROVISION` | 72449 |
| `IN_DOMAIN` | 72449 |
| `OF_TYPE` | 72440 |
| `FROM_SOURCE` | 72449 |
| `NEXT` | 69530 |
| `CITES` | 7224 |
| `SAME_CASE` | 10742 |
| `hotness_provision` | 72440 |
| `hotness_case` | 145339 |
| `hotness_formula` | norm_log1p=log1p(x)/log1p(max_x), provision_hotness=0.85*norm_log1p(in_cites) + 0.15*rank_norm, case_hotness=norm_log1p(n_same_case), ranking=final = rrf_score * (1 + W_HOT * hotness)   [W_HOT 默认 0.05] |
| `W_HOT_default` | 0.05 |
| `index_states` | case_embedding=ONLINE, case_embedding_civil=ONLINE, case_embedding_criminal=ONLINE, case_embedding_general=ONLINE, case_embedding_procedural=ONLINE, case_fulltext=ONLINE, provision_doc_embedding=ONLINE, provision_embedding=ONLINE, provision_fulltext=ONLINE |
| `Domain` | 4 |
| `LawType` | 4 |
| `SourceDataset` | 2 |
| `provision_embedding_rows` | 72088 |
| `provision_doc_embedding_rows` | 352 |
| `case_embedding_rows` | 145339 |
| `case_embedding_rows_by_domain` | general=22263, civil=45638, criminal=74170, procedural=3268 |
| `case_hotness_rows` | 145339 |
| `case_embedding` | ONLINE |
| `case_embedding_civil` | ONLINE |
| `case_embedding_criminal` | ONLINE |
| `case_embedding_general` | ONLINE |
| `case_embedding_procedural` | ONLINE |
| `provision_doc_embedding` | ONLINE |
| `provision_embedding` | ONLINE |

## 2. 断言（期望值从输入产物现算，不写死）

| 项 | 期望（输入行数） | 实测（Neo4j） | 通过 |
|---|---:|---:|:--:|
| `Law` | 1647 | 1647 | ✅ |
| `Provision` | 72449 | 72449 | ✅ |
| `HAS_PROVISION` | 72449 | 72449 | ✅ |
| `NEXT` | 69530 | 69530 | ✅ |
| `CITES` | 7224 | 7224 | ✅ |
| `provision_embedding_rows` | 72088 | 72088 | ✅ |
| `provision_doc_embedding_rows` | 352 | 352 | ✅ |
| `Case` | 145339 | 145339 | ✅ |
| `case_embedding_rows` | 145339 | 145339 | ✅ |
| `hotness_provision` | 72440 | 72440 | ✅ |
| `hotness_case` | 145339 | 145339 | ✅ |
| `vector_indexes_online` | — | — | ✅ |
| `case_rows_by_domain_match` | — | — | ✅ |

分库明细：`{"provision_by_level": {"item": 72088, "doc": 352}, "case_rows_by_domain": {"general": 22263, "civil": 45638, "criminal": 74170, "procedural": 3268}, "hotness_available": true}`

**verdict = PASS**　耗时 528.8s

## 3. 执行轨迹

```
[   0.0s] 建约束与索引
[   0.0s] 导 Law / LawType / Domain / SourceDataset
[   0.2s] 导 Provision（item → p.embedding；doc → p.embedding_doc）
[ 163.8s] 补齐无向量的 Provision 节点（只入图谱）
[ 165.6s] 导 Case（4 库 → c.embedding_<domain4>）
[ 514.1s] 导 HAS_PROVISION
[ 514.8s] 回填 Provision.law_type ← Law.law_type_std（补 pandalla 的分类串）
[ 514.9s] 导 IN_DOMAIN / OF_TYPE / FROM_SOURCE
[ 516.7s] 导 NEXT
[ 519.5s] 导 CITES
[ 519.8s] 导 SAME_CASE（同案多视角，星形连接）
[ 520.1s] 导热度权重（hotness.json → 节点属性）
[ 527.1s] 建向量索引（数据就绪后建，更快）
[ 527.1s] 等待索引 ONLINE
[ 527.1s] 验收计数
```

