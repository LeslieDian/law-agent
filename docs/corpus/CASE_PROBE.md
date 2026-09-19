# 案例侧语料探查（支撑论文 3.2「Case 入口 + 四子图」）

> 生成时间：2026-09-19　探查对象：`data/corpus/normalized/qa/`（285,257 行，三源）
> 脚本：`.`（服务器内联只读扫描，不产生产物、不修改语料）
> 用途：核实论文 **3.2 多法域知识图谱模式层设计**（通用图 + 刑法/民法/程序法子图）
> 与 **3.3/3.4**（Case 节点向量化 + 内嵌向量索引）在**现有语料上能实例化到什么程度**。
>
> ⚠️ 本文修正了 [`retrieval_design.md`](../retrieval_design.md) 早期版本的一个判断偏差：
> 该文曾称「`:Case` 无法构建」。**实测证明该结论错误** —— 语料含大量案件正文，Case 入口可落地。
> 详见本文件第 5 节的勘误说明。

---

## 1. `case_sha1` 的真实口径（重要）

```python
# scripts/corpus/normalize_corpus.py:558
rec["case_sha1"] = sha1(inp[-400:]) if inp else ""
```

**它是 `input` 最后 400 个字符的 SHA-1，不是「案件正文指纹」。**

- 对 `jud_doc_sum`（input = 完整判决书）≈ 判决结果 + 落款 → 近似案件指纹，可用；
- 对 `legal_question_answering`（input = 题目/选项）→ 尾部是选项，**不是案件**；
- 对 `Skepsun` 全源为空（input 为空）→ 0 条。

⇒ 因此**不能**用「全量 distinct case_sha1」当案件数。逐 task 计算才有意义（见第 3 节）。

## 2. 任务分布（285,257 行）

| task | rows | 唯一 `case_sha1` | rows/uniq | 能否作 Case 来源 |
|---|---:|---:|---:|---|
| `legal_question_answering` | 92,381 | 91,152 | 1.01 | ❌ 题目/选项，非案件 |
| `jud_read_compre` | 38,214 | 38,191 | 1.00 | ✅ 含完整案情描述 |
| `judgement_predit` | 32,000 | 28,653 | 1.12 | ✅ 案情事实 → 判决理由（刑/民） |
| `leg_ele_extra` | 23,809 | 22,218 | 1.07 | ❌ **是安卓隐私政策 BIO 标注，与法律子图无关** |
| `exam` | 22,054 | 22,051 | 1.00 | ❌ 试题 |
| `leg_eve_detec` | 21,260 | 21,255 | 1.00 | ⚠️ 案情片段 → 标签（如"婚后有子女"），**可作 LegalIssue 来源** |
| `sent_pred` | 11,657 | 11,649 | 1.00 | ✅ 刑事案情 → 罪名 + 量刑建议 |
| `jud_doc_sum` | 8,233 | 8,118 | 1.01 | ✅ **完整判决书**（含法院名/案号/案由） |
| `sim_case_match` | 7,987 | 1,177 | 6.79 | ✅ 案例 A/B/C 三元组 → **RELATED_CASE 监督信号** |
| `leg_case_cls` | 6,509 | 6,509 | 1.00 | ❌ 风险类型分类（灰产推广等），与子图无关 |
| `legal_advice` | 5,998 | 1,000 | 6.00 | ⚠️ 一案多问 |
| `op_sum` | 5,251 | 5,251 | 1.00 | ✅ 操作摘要 |
| `qa` / `judical_examination*` / `legal_counsel*` | 8,469 | 0 | — | ❌ 无 input（Skepsun 源） |
| `crime_concept` | 435 | 435 | 1.00 | ✅ 罪名概念 → **Crime 节点候选** |

## 3. 案件类去重规模

取案件正文类 task（`jud_read_compre` / `judgement_predit` / `leg_ele_extra` /
`leg_eve_detec` / `sent_pred` / `jud_doc_sum` / `sim_case_match` / `leg_case_cls` /
`op_sum`）：

| 项 | 值 |
|---|---:|
| 案件类样本总数 | **154,920** |
| 跨 task 去重后唯一（`case_sha1`） | **143,009** |
| 其中 `input >= 300` 字（真实案件正文） | **99,626** |

## 4. Case 节点结构化属性可得性

分母 = 99,626（去重 + `input >= 300`）：

| 属性 | 可得 | 覆盖率 |
|---|---:|---:|
| 日期（`YYYY年M月D日`） | 75,188 | **75.5%** |
| 案由（"XX纠纷"） | 40,851 | **41.0%** |
| 法院名（"XX人民法院"） | 16,219 | 16.3% |
| 案号（"(2017)粤0306民初3474号"） | 9,884 | 9.9% |
| 文书类型=民事判决书 | 9,754 | 9.8% |
| 文书类型=刑事判决书 | 2,244 | 2.3% |
| 文书类型=裁定书（各类） | 563 | 0.6% |
| 文书类型=行政判决书 | 52 | 0.1% |

> 对照论文表 3.5 的 Case 属性（`case_id` / `title` / `content` / `domain` /
> `case_type` / `court` / `judgment_date`）：
> `content` / `domain` 可得（100% / 已有域标签）；`case_type` 可部分从案由与罪名推；
> **`court` 与 `judgment_date` 只有 16.3% / 75.5%，`case_id` 需另生成**。

## 5. 论文 3.2 各实体的可实例化程度（实测判定）

| 子图 | 实体 | 判定 | 依据 |
|---|---|---|---|
| 通用 | `Case` | ✅ 可建 ~99,626 | 案件正文样本 |
| 通用 | `Law` | ✅ 1,579 | 阶段 2b 法条切条（规范化后） |
| 通用 | `Domain` | ✅ 4 | 已有 `domain` 字段 |
| 通用 | `LegalIssue` | ⚠️ 弱 | `leg_eve_detec` 21,260 条标签可作焦点标签，非规范争议焦点 |
| 通用 | `Evidence` | ❌ | 无证据类型标注 |
| 刑法 | `Crime` | ✅ 可建 | `sent_pred` / `judgement_predit` 输出含罪名（"销售假药罪""强迫劳动罪"） |
| 刑法 | `CrimeElement` | ❌ | 无构成要件标注 |
| 刑法 | `SentencingFactor` | ⚠️ 弱 | 判决书含"自首/立功/累犯/如实供述"等词，可规则抽取 |
| 民法 | `CauseOfAction` | ✅ 可建 ~40,851 | 判决书标题含"XX纠纷" |
| 民法 | `LegalRelation` | ❌ | 无标注 |
| 民法 | `ClaimBasis` | ❌ | 无标注 |
| 民法 | `CivilLiability` | ❌ | 无标注 |
| 程序法 | `ProcedureStage` | ❌ | 无标注（仅法条文本可作规则来源，无案件侧实例） |
| 程序法 | `JurisdictionRule` | ❌ | 同上 |
| 程序法 | `EvidenceRule` | ❌ | 同上 |
| 程序法 | `TimeLimit` | ❌ | 同上 |
| 程序法 | `Remedy` | ❌ | 同上 |

**小结**：17 类实体中，**5 类可建 + 2 类弱可建 + 10 类无料**。

| 判定 | 数量 | 实体 |
|---|---:|---|
| ✅ 可建 | **5** | `Case` / `Law` / `Domain` / `Crime` / `CauseOfAction` |
| ⚠️ 弱可建 | **2** | `LegalIssue` / `SentencingFactor` |
| ❌ 无料 | **10** | `Evidence` / `CrimeElement` / `LegalRelation` / `ClaimBasis` / `CivilLiability` / `ProcedureStage` / `JurisdictionRule` / `EvidenceRule` / `TimeLimit` / `Remedy` |
论文 3.2.6 声明的「100,674 节点 / 192,304 关系」需要能对上的口径：
Case ~99,626 + Law 1,579 + Domain 4 ≈ **101,209**（数量级吻合，但**具体口径必须复现**）。

## 6. 与论文表 3.1「数据来源」的差异（必须处理）

| 论文表 3.1 声明 | 实际采购情况 |
|---|---|
| 中国裁判文书网（裁判文书） | ❌ 未采；案件正文来自开源 SFT 数据集 |
| 国家法律法规数据库 | ⚠️ 部分等价（twang2218 / pandalla） |
| CAIL2018（刑事 + 罪名标注） | ❌ **未采**（`docs/corpus/README.md:57` 标注未采；license=unknown，B 级） |
| LeCaRD（类案检索） | ❌ 未采；功能由 `sim_case_match` 部分顶替 |
| 最高人民法院司法解释 | ⚠️ 通过法条源间接覆盖 |
| 人工整理案例 | ❌ 未见 |
| （论文未写） | ✅ **实际使用**：`ShengbinYue/DISC-Law-SFT` 252,355 / `Dusker/lawyer-llama` 18,435 / `Skepsun/lawyer_llama_data` 14,467 / `twang2218/chinese-law-and-regulations` 22,510 / `pandalla/chinese_law_examples` 1,000 |

⇒ **能力上大体覆盖，但论文的数据来源表述与实际不一致**，需按实际改写（含许可证分级的如实说明）。
