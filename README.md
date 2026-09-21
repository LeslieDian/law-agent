# 法律领域智能体：混合专家模型 + 知识图谱

> 论文《基于混合专家模型和知识图谱的法律领域智能体研究与构建》实验代码仓库
> 最后更新：2026-09-20

面向**民法、刑法、程序法**三法域的法律智能问答系统。以 Qwen3-8B 为共享底座，
训练三个 QLoRA 领域适配器；以 Neo4j 统一存储法律知识图谱与案例向量索引；在线链路采用
「向量召回 → 图谱关系扩展 → RRF 融合 → 重排序 → 请求级适配器路由」的混合检索增强生成。

---

## 目录

1. [技术路线与模型清单](#1-技术路线与模型清单)
2. [实验设计（两条线，不可混排）](#2-实验设计两条线不可混排)
3. [训练语料方案](#3-训练语料方案)
   - 3.1 [目标配比](#31-目标配比)
   - 3.2 [数据集来源](#32-数据集来源)
   - 3.3 [数据处理流程](#33-数据处理流程)
   - 3.4 [统一数据 Schema](#34-统一数据-schema)
   - 3.5 [MoE 路由对数据的额外要求](#35-moe-路由对数据的额外要求)
   - 3.6 [数据红线与去污](#36-数据红线与去污)
4. [评测基准](#4-评测基准)
5. [目录结构](#5-目录结构)
6. [实验环境](#6-实验环境)
7. [数据与合规（强制）](#7-数据与合规强制)
8. [进度与执行顺序](#8-进度与执行顺序)
9. [结果完整性要求](#9-结果完整性要求)
10. [文档维护约定](#10-文档维护约定)

---

## 1. 技术路线与模型清单

| 层 | 模型 / 组件 | 规格 | 说明 |
|---|---|---|---|
| **底座模型** | `Qwen/Qwen3-8B` | 8.19B，双模式 | 固定 commit hash。⚠️ HF 上**不存在** `Qwen3-8B-Instruct`。非思考模式用 `/no_think` 关闭 |
| **领域适配** | QLoRA × 3 | 4bit NF4 + 双重量化 + BF16 + paged_adamw_8bit | 民法 / 刑法 / 程序法三个适配器 + 请求级路由 |
| **MoE 路由（主方案）** | `Qwen/Qwen3-Embedding-0.6B`（冻结）+ 线性/MLP 头 | 1024d | R1 方案，便宜、可复现、可解释 |
| **MoE 路由（上界参考）** | `Qwen/Qwen3-0.6B` 域分类 | — | R2 方案 |
| **向量检索（四路消融）** | `Qwen/Qwen3-Embedding-0.6B` | 1024d | **主力**（MTEB 64.33 > BGE-M3 62.34）。instruction-aware，查询侧须用 `Instruct: {task}\nQuery: {q}` |
| | `BAAI/bge-m3` | 1024d | 对照 |
| | `hfl/chinese-bert-wwm-ext` | 768d | 退化对照 |
| | `Qwen/Qwen3-Embedding-4B` | 2560d | 大模型对照 |
| **重排** | `BAAI/bge-reranker-v2-m3` | 568M，MIT | ⚠️ **实测对本任务无效**（判别力 AUC 0.888 但 top-k 覆盖反而下降）→ **不进入主链路**，见 §4b-5b |
| **知识层** | Neo4j 5.26 Community（Docker） | — | Law / LawVersion / Provision / Case / Cause / Court / Date |
| **检索融合** | **加权 RRF（k=10）** | — | 不同检索源各自排名后融合，**不比原始分数**；权重 dense 1.0 / BM25 0.7 / 图谱 0.25（dev 扫出）。等权 RRF 会打崩精度，见 §4b-5b |
| **编排** | LangGraph | 1.2.11 | 事实整理 → 证据检索 → 专家路由 → 适配器调用 → 结果聚合 |
| **判分器（双盲）** | `MiniMax-M3`（主，国内可达）+ `gemini-2.5-pro` / `deepseek-r1`（可选 provider） | 外部 API | 匿名化编号 + 洗自指标签 + 打乱顺序（seed=42），见 `configs/judge.yaml`；LexRubric 判分走 `--judges minimax-m3` |
| **评测基准** | LexRubric / LexEval | — | 见 [第 4 节](#4-评测基准) |

**备选底座对照**（只用于消融与审稿人问询，不参与主线）：

| 模型 | 说明 |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 原方案，已被 Qwen3-8B 取代 |
| `Qwen/Qwen3.5-9B` | Gated DeltaNet 新架构，对 peft/bitsandbytes 兼容性需先冒烟验证 |
| LegalOne-R1-8B（清华，2026-01 开源） | 法律专用对照基线，用于回答「你的 QLoRA 贡献了什么」 |

> ⚠️ **底座一旦定下不可中途更换** —— 对比实验只允许换一个变量。
> 完整选型分析见 [`docs/model_selection.md`](docs/model_selection.md)。

---

## 2. 实验设计（两条线，不可混排）

**闭卷线（Closed-book）**：不允许检索外部知识库 → 测底座/适配器**自身**的法律知识上限
**开卷线（RAG）**：允许查询法条向量库 → 验证系统实际应用能力，**成绩必须注明增强条件**

| 编号 | 模型训练 | 检索系统 | 用途 |
|---|---|---|---|
| E0 | 原始 Qwen3-8B | 无 | 基础闭卷基线 |
| E1 | QLoRA 法律微调 | 无 | 验证训练效果 |
| E2 | 原始 Qwen3-8B | 纯向量 RAG | 验证向量库效果 |
| E3 | 原始 Qwen3-8B | 向量 + 关键词 + 知识图谱 | 验证混合检索 |
| E4 | QLoRA 法律微调 | 纯向量 RAG | 验证训练 + 检索结合 |
| E5 | QLoRA 法律微调 | 混合检索 + 知识图谱 | 论文完整系统 |

**MoE 消融矩阵**（论文必备）：

| 编号 | 配置 | 作用 |
|---|---|---|
| A0 | 单 LoRA 全域训练 | baseline |
| A1 | 3 专家 + **oracle 路由** | **上界**（用真标签路由） |
| A2 | 3 专家 + **学习式路由（R1）** | **主方法** |
| A3 | 3 专家 + 无路由（权重平均融合） | 证明「路由有用」而非「多专家有用」 |
| A4 | 3 专家 + **随机路由** | **下界** |
| A5 | **token 级 MoE-LoRA**（MixLoRA 式：FFN 旁挂 LoRA 专家 + 层内可学习门控，token 级路由 + 负载均衡损失） | **可选扩展**：与 A2 对比两种路由粒度（请求级 vs token 级）。预期单域样本打平、混合域样本 token 级占优——对比本身就是论文贡献 |

> A1 与 A2 的差值 = 路由器质量损失；A2 与 A3/A4 的差值 = 路由机制本身的增益。
> **没有 A1 和 A4，审稿人一定会问「你怎么知道不是多专家本身带来的增益」。**

**★ MoE 表述红线（2026-09-18 与用户确认）**：本方案的专家路由是**请求级的 MoLE
（Mixture-of-LoRA-Experts）**——LoRA 专家挂在冻结底座外、请求级路由、路由器独立训练；
**不是**模型内 token 级的原生 MoE。论文必须写「借鉴混合专家思想构建 LoRA 专家混合架构」，
**禁止**写「训练了混合专家模型」。主方法选请求级而非 token 级的理由：
与 LangGraph 跨域聚合架构契合、专家语义（三法域）可解释、训练无路由坍缩风险；
token 级以 A5 作为对比消融，实现与否**待阶段 6 结果后决定**。

检索侧消融：去重排序器 / 去时间版本过滤 / 去知识图谱 / 去领域适配器路由 / 换向量模型（四路）。

推理参数全局统一：`do_sample=false, temperature=0, top_p=1`。
`max_new_tokens` **分档**（对所有系统统一，不破坏可比性；批内最长序列支配整批墙钟时间，
客观题给 1536 会把吞吐拖慢 ~6 倍）：**LexEval 客观题 = 256，LexEval 生成题与 LexRubric = 1536**。
注意 LexEval 5_2（长文摘要，gold p50≈1004 字）在 cap 1536 下会有截断（相对比较仍成立，论文注明）。

详见 [`docs/experiment_matrix.md`](docs/experiment_matrix.md)。

---

## 3. 训练语料方案

> 完整方案（含实测依据与风险处置）见 [`docs/finetune_data_plan.md`](docs/finetune_data_plan.md)。
> 候选源清单同时以**机器可读**形式维护在 [`configs/corpus_sources.yaml`](configs/corpus_sources.yaml)，
> **采集脚本直接读该文件，不得在脚本里硬编码数据集名**。

### 3.1 目标配比

**目标不是「凑 10 万条」，而是「凑 10 万条带域标签的数据」** —— 因为还要适配 MoE 路由。
专家 LoRA 需要 `(instruction, input) → output`，路由器需要 `query → domain`，
**这是两批数据，且必须互不相交**。

| 域 | 目标条数 | 占比 | 主要来源 |
|---|---|---|---|
| **刑法** | **30,000** | 30% | 刑事判决（罪名 + 法条 + 刑期）+ 犯罪知识图谱转问答 + 刑法典法条任务 |
| **民法** | **40,000** | 40% | 民事判决（信息抽取/预测/摘要）+ 民事问答 + 民法典法条任务 |
| **程序法** | **20,000**（合成 8,000 **已降为可选**，见 3.1.1） | 20% | 三大诉讼法司考题 + 诉讼法**条文任务**（切条后）+ 合成补量（可选） |
| **通用回放** | **10,000** | 10% | 通用指令数据抽样，**防灾难性遗忘**，不是凑数 |
| **合计** | **100,000** | 100% | |

**两个必须解释的设计**：

1. **为什么要有 10% 通用回放**：纯法律数据微调会打崩底座通用能力。评测集里有相当比例
   题目需要常识推理与语言组织，只喂法律数据会让这部分指标**不升反降**。
2. **为什么曾计划程序法合成 40%（最终定为 15% 的"题型补量"）**：程序法曾是三域里公开语料
   **最稀薄**的（公开 SFT 集里占比普遍 <10%），而目标占 20%，故原计划自建补量。
   **阶段 2 实测证明程序法真实数据已够（25,445 > 20,000）→ "凑量型"合成取消**；
   但重跑发现程序法**题型极度单一**（头号任务 `legal_question_answering` 独占 49.3%，
   连 35% 单任务份额软上限都守不住，见 3.1.3），故保留**"题型补量型"合成 3,000 条**：
   - 来源：阶段 2b 切出的 **6,601 条程序法条文**（116 部），**逐字派生**三种任务型
     （`statute_recall` / `statute_locate` / `statute_structure`）；
   - 上限：占程序法域配额 **15%**（`--derived-max-share 0.15`），实测正好取满 3,000；
   - 合成规范写死、不许绕过：
     - 答案**逐字取自条文原文**（`verbatim_check` 逐条回查，13,484 / 13,484 通过、不符 0）
     - 每条登记 `synthetic: true` / `derived: true` / `derivation` / `derived_from_uid`
     - **合成数据一律不进验证集与测试集**（质检 C12 断言：实测只出现在 train）
   - 论文里如实披露「程序法域 15% 为条文任务派生样本」。

### 3.1.1 配比可得性核算（★ 阶段 2 实测结果：**每域都超额，总缺口 = 0**）

**在定配比之后、动手清洗之前，必须先做一次「可得量核算」**，否则到切分阶段才发现凑不齐。
以下是阶段 2 归一化 + 域打标后的**实测**数字（raw 349,665 条 → qa 流 **285,257 条**）：

| 域 | 目标 | **实得** | 实得占比 | 达成率 |
|---|---|---|---|---|
| 刑法 criminal | 30,000 | **97,414** | 34.1% | **325%** |
| 民法 civil | 40,000 | **104,691** | 36.7% | **262%** |
| 程序法 procedural | 20,000 | **25,445** | 8.9% | **127%** |
| 通用/其他 general | 10,000 | 57,707 | 20.2% | 577% |
| **合计** | 100,000 | **285,257** | | **缺口 0** |

**三条结论（都推翻了原先的假设）**：

1. **程序法不是"凑不齐"，而是"够用"** —— 实得 25,445 > 目标 20,000。
   → 8,000 条合成**不再是必需**：可选「25,445 真实中取 20,000」或「12,000 真实 + 8,000 合成」。
   **实际决策（2026-09-18）**：真实数据足够，**不做"凑量型"合成**；只保留
   **3,000 条"题型补量型"合成**（占程序法域配额的 15%，由阶段 2b 的 6,601 条程序法条文**逐字派生**而来，
   见 3.1.3 与 [`docs/corpus/README.md`](docs/corpus/README.md) 第八节）——
   目的是补**题型多样性**而不是补量，因为有派生时程序法的头号任务独占 49.3%、
   会把 35% 单任务份额软上限打破（见 3.1.3）。论文里如实披露「程序法域 15% 为条文任务派生样本」。
2. **不需要再单独找通用数据集** —— `general` 桶（宪法 / 行政法 / 法治理论 / 职业道德等
   **域不明确但仍属法律**的样本）已有 57,707 条，超过「通用回放 1 万」的需求。
   ⚠️ **但要注意它和"通用回放"不是一回事**：`general` 仍是**法律领域**内容（只是域不明确），
   而「通用回放」本意是**非法律的通用指令数据**（防止底座通用能力崩塌）。
   → 阶段 4 需二选一：(a) 用 `general` 桶充数（便宜，但对"防遗忘"效果存疑）；
   (b) 另采一份通用中文指令数据（更符合设计意图）。**建议 (b)**，或两者都加并做消融。
3. **真正的工作量是"下采样"而不是"补数据"** —— 285,257 → 100,000，需砍掉约 65%。
   下采样必须**按 `task` × `source` 分层**，否则某个任务型（如 `legal_question_answering` 92,381 条）
   可能垄断某个专家，让该专家只学会一种题型。

#### 3.1.3 阶段 4 实测结果（2026-09-18 修复 uid 后全链路重跑，配比 100% 命中）

> 完整报告：[`docs/corpus/SPLIT_REPORT.md`](docs/corpus/SPLIT_REPORT.md) +
> [`docs/corpus/SPLIT_VERIFY.md`](docs/corpus/SPLIT_VERIFY.md)。

| split | 落点 | 条数 |
|---|---|---|
| 训练集（A0 全域） | `data/train/train.jsonl` | **100,000** |
| 训练集（分域视图） | `data/train/{civil,criminal,procedure,general}.jsonl` | 40,000 / 30,000 / 20,000 / 10,000 |
| 验证集 | `data/dev/dev.jsonl`（+ 逐域） | **1,000**（每域 400/300/200/100） |
| 测试集 | `data/test/test.jsonl`（+ 逐域） | **1,000**（同配比） |
| 路由集 | `data/router/router_train.jsonl` | **20,000**（query → domain 多标签） |
| 合计占用 | | **122,000**（`uid_g` 唯一 122,000；池剩余 162,473 可回溯） |

- 输入池 **284,473 条** = 真实 QA **270,989** + 派生 **13,484**；真实部分 `content_sha1`
  **全局唯一 0 重复**，`uid_collision_records = 0`。
- 逐域达成率 **100%**（30,000 / 40,000 / 20,000 / 10,000 精确命中）；
  质检 verdict = **PASS**，四份 split 两两不相交（`uid_g` 与 `content_sha1` 双口径交集六对**全 0**）。
- **路由集口径澄清**：不是「从 10 万里抽 20%」（那会把训练集削到 8 万），
  而是**先从 28.4 万池子里预留路由集与 val/test，再对余下训练池下采样到 10 万** ——
  「训练集 10 万」与「路由集 disjoint」两件事因此不必二选一。
- ✅ **程序法的软上限问题已解决**（原为待决策项）。程序法域接入 **3,000 条**条文任务派生样本
  （占该域配额 **15%**，上限 0.15，正好取满）后：训练池 21,980 → **35,464**，
  头号任务 `legal_question_answering` 49.3% → **35.0%**，`cap_relaxed` **true → false**。
  → **派生不只"补量"，它把 35% 软上限从"数学上不可达"变回"成立"。**
  新增题型：`statute_recall` 1,227 / `statute_locate` 1,228 / `statute_structure` 545。
  四个域**全部** `cap_relaxed=false`、`derived_cap_relaxed=false`。
- **通用回放仍是缺口（待决策）**：池里 `replay` 标记 **全部为 0** —— DISC-Law-SFT 内置的
  Alpaca-GPT4 / Firefly 通用回放**不在已下载的 4 个文件内**，故「通用回放 10%」目前
  由 `general` 桶（法律领域内域不明确的样本）代充。**待决策**：是否另采非法律中文通用指令数据。
- **法条流不进 SFT**：72,449 条法条检索单元是检索语料（向量库主料；2026-09-19 重跑后口径，
  原 65,037 → +7,412），不下采样、不参与配比；
  跨流检查 QA ∩ 法条 `content_sha1` = **0**（`cross_flow_overlap = 0`）。
- 训练集来源分布：DISC-Law-SFT 85,095（85.10%）/ Skepsun 6,048（6.05%）/
  Dusker 5,857（5.86%）/ **derived/statute-items 3,000（3.00%）**。

#### 3.1.2 法条流（statutes）实测结构 —— 一处必须纠正的表述 + 一处必须补的工序

> ⚠️ 此前 README 与方案里写的「法条库 **23,510 条**」是**不准确**的表述，实测后必须纠正。
> 阶段 3 前置探测（2026-09-17）的实测结构如下。

| 来源 | 条数 | 实际粒度 | `task` | 证据 |
|---|---|---|---|---|
| `twang2218/chinese-law-and-regulations` | **22,510** | **整部法律法规全文** | `statute_doc` | output 以「第X条」开头 = **0%**；行数 > 8 的整部文本 = **99.5%** |
| `pandalla/chinese_law_examples` | **1,000** | **逐条法条** | `statute_item` | output 以「第X条」开头 = **100%**；`statute.article_no` 有值 |

→ 所以「23,510 条」= **22,510 部法规全文 + 1,000 条法条**，**不是 23,510 条法条**。

**按域分布（当前打标，仅供参考 —— 整部法全文会让打标失真）**：
`general` 20,199 / `civil` 3,132 / `procedural` **568** / `criminal` **411**。

**twang2218 的 `statute.type` 分布（22,510 条）**：
地方性法规 19,733（87.7%）/ 司法解释 788 / 行政法规 693 / 修改、废止的决定 661 /
法律 429 / 有关法律问题和重大问题的决定 172 / 法律解释 26 / 宪法 7 / 监察法规 1。

**三大诉讼法 + 两大法典的全文都在（已按 `statute.title` 核实）**：

| 法律 | 字符数 | 备注 |
|---|---|---|
| 中华人民共和国民事诉讼法 | 33,309 / 32,664 | **两个版本 → 历史版本已在库** |
| 中华人民共和国刑事诉讼法 | 40,914 / 37,477 | 同上 |
| 中华人民共和国行政诉讼法 | 12,520 | — |
| 最高人民法院关于适用《民事诉讼法》的解释 | 61,667 / 62,006 | 程序法条文大户 |
| 最高人民法院关于适用《行政诉讼法》的解释 | 26,018 | — |
| 中华人民共和国民法典 | 113,346 | 民法条文大户 |
| 中华人民共和国刑法（+ 13 个修正案） | 60,936 | 刑法条文大户 |

**结论：程序法的法条「不是少，而是还没切条」。**
切条后可得：民诉 ≈ 284 条 + 刑诉 ≈ 308 条 + 行政诉讼 ≈ 103 条
+ 三大司法解释（民诉解释 ≈ 552 / 刑诉解释 ≈ 655 / 行诉解释 ≈ 163）≈ **2,000+ 条程序法条文**，
足以支撑程序法域的「法条任务」。

**⇒ 必须补的工序（阶段 2b：法条切条）**：
1. 把 `statute_doc` 按「第X条」**切分为逐条法条**（保留 `title` / `article_no` / `status` / `effective_from`）；
2. 法条域打标**改为按 `statute.title` 判定**（不再靠全文关键词 —— 否则整部法律里出现
   「执行 / 管辖 / 上诉」就会被误判成程序法，这正是上面 procedural 568 条不可信的原因）；
3. 地方性法规（19,733 部）量极大且与三域弱相关，**切条后可只保留「法律 / 司法解释 / 行政法规」
   三类**（≈ 2,000 部），其余留档不参与 10 万条配比。

### 3.2 数据集来源

**分级语义**：`A` = 协议明确（Apache-2.0 / MIT），可用于论文实验；
`B` = 协议不明 / unknown / gated，**仅限内部探索**，进论文实验集前必须先确认授权。

以下体积、协议均为 **2026-09-17 在实验服务器上实测所得**（hf-mirror.com API 逐条探测），
探测脚本 `scripts/benchmarks/probe_hf_datasets{,2,3,4}.py` 可随时重跑复核。

**A 级 —— 已采集（阶段 1 完成，合计 1.2 GB / 23 文件）**

> 服务器落点 `data/corpus/raw/<dataset>/`，逐文件 SHA-256 已登记，
> 验收 **PASS**（磁盘实测与清单逐文件比对无缺失、无大小不符，LFS 大文件 sha256 与 HF `lfs.oid` 全部一致）。
> 登记归档见 [`docs/corpus/`](docs/corpus/)。

| 数据集 | 体积 | 文件 | 协议 | 状态 | 用途 |
|---|---|---|---|---|---|
| `ShengbinYue/DISC-Law-SFT` | 552.4 MiB | 6 | Apache-2.0 | ✅ OK | **主力**：判决预测/信息抽取/摘要/问答/司考，403K 样本；内置 48K Alpaca-GPT4 + 60K Firefly 通用回放 |
| `Dusker/lawyer-llama` | 425.7 MiB | 7 | MIT | ✅ OK | 含 `kg_crime_llama.json`（**犯罪知识图谱**，可转问答）。⚠️ 内含 DISC-Law-SFT 副本 → 阶段 2 三指纹核验：`DISC-Law-SFT-Pair.json` **100% 重复已丢弃**，`-Triplet.json` 仅 20.5% 重复故**保留** |
| `twang2218/chinese-law-and-regulations` | 152.8 MiB | 4 | Apache-2.0 | ✅ OK | **法条原文**（带元数据，协议干净可发表） |
| `Skepsun/lawyer_llama_data` | 22.9 MiB | 3 | Apache-2.0 | ✅ OK | 司法考试题，**含《民诉》《刑诉》程序法题**，带 `source` 字段 |
| `pandalla/chinese_law_examples` | 519.8 KiB | 3 | Apache-2.0 | ✅ OK | 仅作格式参考 |
| `Aiiluo/Chinese-Law-SFT-Dataset` | 2.6 MiB | 6 | Apache-2.0 | ⏭️ 跳过 | **文件名已按民事/商事/刑事分好**；`gated=auto`，需配置 `HF_TOKEN` 后单独补采 |

**A 级 —— 体积超限，只允许按需抽样（禁止整下）**

| 数据集 | 体积 | 协议 | 说明 |
|---|---|---|---|
| `wormtooth/MNBVC-judgment` | **121.5 GB** / 996 文件 | MIT | 判决书大规模语料。由采集脚本的体积护栏（`--max-gb`，默认 3 GB）自动拒绝 |

**B 级 —— 仅内部探索，不进论文实验集**（须先确认授权）

| 数据集 | 体积 | 协议 | 说明 |
|---|---|---|---|
| `china-ai-law-challenge/cail2018` | 1167.8 MB | unknown | 刑事一审判决：罪名 + 法条 + 刑期，**刑法域的黄金标注** |
| `Brench/chinese_law_data_rag_ft` | 506.9 MB | 未声明 | `laws_data_281k.jsonl`（302 MB），28.1 万条法条数据，RAG 侧可用 |
| `Kuugo/Chinese_Law` | 7.6 MB | 未声明 | **238 个 .txt，文件名即法律名** → 最省事的法律原文来源 |
| `Dusker/chinese-laws-pretrain` | 5.5 MB | 未声明 | 按编拆分（刑法.json / 合同编.json / 物权编.json）→ **结构化最好**，便于切条 |

**已证伪 / 不可用（省得再试）**

| 目标 | 结果 |
|---|---|
| `OpenBMB/LawBench`（HF） | 不存在（HF 上是 `doolayer/LawBench` 等非官方镜像） |
| `THUIR/LEEC`、`CSHaitao/LegalAgentBench`（HF） | 不在 HF 镜像上（后者 GitHub 有，6.0 MB） |
| `china-ai-law-challenge/cail2019`（HF） | 镜像无；**GitHub 有**（178.6 MB），覆盖民事+刑事 |
| HF 搜索接口搜 `法律` / `刑法` / `判决` / `司法考试` / `法条` | **全部返回 0 条** → 中文关键词搜不出来，只能靠英文或已知 ID 定位 |
| `FudanDISC/DISC-Law-SFT` | **不存在**，真实 ID 是 `ShengbinYue/DISC-Law-SFT` |

> ⚠️ **两个实测陷阱，踩过才知道**：
> 1. **hf-mirror 对「不存在的仓库」返回 `HTTP 401` 而不是 `404`** —— 按记忆猜 ID 探测会把
>    不存在的仓库误判成「无权访问」。必须**先用搜索接口反查真实 ID** 再探测。
> 2. **HF 搜索接口对中文关键词完全无效**，只能靠英文关键词或已知 ID 定位。

### 3.3 数据处理流程

九个阶段，**每阶段有验收卡口，不达标不许进下一阶段**。
`──────────────── 以下才动 GPU ────────────────` 是硬边界：

```
阶段 0  源合规审查        → configs/corpus_sources.yaml（A/B 级分级，A 级才可进论文实验集）✅
阶段 1  分批采集          → data/corpus/raw/<dataset>/ + data/corpus/MANIFEST.json（逐文件 SHA-256）✅
阶段 2  归一化 + 域打标   → data/corpus/normalized/{qa,statutes}/*.jsonl + STATS.json + DEDUP_REPORT.json ✅
                          （uid 全局唯一由构造成立 + 当场断言；合并流只并本轮正式产出）
阶段 3  去污（双向）      → DECONTAMINATION_REPORT_PASS.json（复扫 verdict = PASS）★硬门禁 ✅
                          （两条不变量：uid 唯一、removed == hit_uids）
阶段 2b 法条切条 + 法条域打标 → data/corpus/statute_items/*.jsonl（整部法 → 72,449 条）✅
                          （切条与质检在同一包装脚本内，verdict ≠ PASS 即失败）
阶段 2c 程序法条文任务派生 → data/corpus/derived/statute_tasks.jsonl（13,484 条，补题型多样性）✅
阶段 4  切分 + 分层下采样  → data/{train,dev,test,router}/（122,000 条，四份两两不相交）✅
───────────────────────────────  以下才动 GPU ───────────────────────────────
阶段 5  基线：单 LoRA 全域训练                 → A0
阶段 6  专家 LoRA ×3（刑法 / 民法 / 程序法）    → A1/A2/A3/A4 的组件
阶段 7  路由器训练（R0 → R1）                  → ROUTER_EVAL.json
阶段 8  MoE 组合 + 消融矩阵                    → 四档消融表
阶段 9  用内部验证集选 checkpoint              → 选定后才允许跑外部基准 ★红线
```

**阶段验收卡口**：

| 阶段 | 卡口（不达标不许进下一阶段） |
|---|---|
| 0 | 每个源都有 `license` + `license_grade`；B 级源明确标注「仅内部探索」 |
| 1 | 每批落盘后立即算 SHA-256；`data/corpus/MANIFEST.json` 与磁盘实测一致 |
| 2 | ✅ **每域实得量 ≥ 目标量**（本次总缺口 0）；`domain_source` = `fallback` 占比 **< 15%**；人工抽检 50 条准确率 ≥ 95%；**uid 全局唯一（当场断言）** |
| 2b | ✅ 三大诉讼法条文齐备；法条域按法名判定；切条质检 C1–C7 全过、verdict PASS（残留条号 0 / 目录块 0） |
| 2c | ✅ 派生素材全为 `synthetic=true`；答案**逐字取自原文**（不符 0）；`domain=procedural` 100%；**只进 train** |
| 3 | ✅ **去污 verdict = PASS**（复扫 294,499 行 / 精确 0 / 近似 0；2026-09-19 修复版证据，见 `docs/corpus/DECONTAMINATION_REPORT_PASS.*` 与 `APPLY_SUMMARY.json`），报告里有近重复命中明细；**两条硬断言**：uid 唯一、实际剔除行数 == 命中 uid 数 |
| 4 | ✅ 路由集 ∩ 专家训练集 = **0**（`uid_g` 与 `content_sha1` **双口径**断言）；训练集逐域配比 = 目标（**精确相等，不是「接近」**）；val/test 与训练集也 disjoint；**派生份额 ≤ 15% 且零放宽** |
| 5–8 | 每档消融都有独立 config + 独立输出目录，**不许共用目录覆盖** |
| 9 | ★ checkpoint 只能由内部验证集选定；外部基准（LexRubric / LexEval）各只跑一次终评 |

**域标签怎么打 —— 用「加权打分」，不能用优先级链短路**：

实测两种短路顺序都会系统性错标（程序法优先 → 判决预测被附带引用带偏；实体法优先 → 刑诉考题判成刑法）。
六路证据加权求和取最高分，权重写在 `configs/corpus_adapters.yaml`：

| 信号 | 权重 | 说明 |
|---|---|---|
| 问句里引用的法条 | ×2.0 | 最可靠，可解释可审计 |
| 答案/解析里引用的法条 | ×1.0 | 同上 |
| 文书类型（刑事/民事判决书） | ×2.5 | 判决类主力 |
| 任务型强指示（刑期预测→刑法） | ×1.5 | 少数任务 |
| 罪名释义类 | ×3.0 | — |
| **议题关键词**（只在问句里找） | 程序0.8/个，刑民0.5/个 | **必须有，否则 fallback 高达 35%** |

> **关键设计 1：程序法不能靠判决书类型区分** —— 一份民事判决书既涉民事实体法也涉民诉程序法。
> 必须走「内容议题」判断（管辖 / 送达 / 举证责任 / 时效 / 上诉 / 再审 / 执行）。
>
> **关键设计 2：议题关键词兜底必须有** —— 大量司考/问答样本根本不引用具体法条
> （「下列哪个选项不属于法官应当遵守的司法礼仪？」），只靠引用法条打标会让 **fallback 占 35%**；
> 补上关键词信号后降到 **11.7%**。
>
> **关键设计 3：判决类任务给程序法引用降权 ×0.35** —— 判决预测问的是实体结论，附带引用民诉法/刑诉法不应主导标签。
>
> 每条约记 `domain_source` + 完整 `scores` / `evidence` / `contrib_primary`，可事后审计。

### 3.4 统一数据 Schema

所有来源归一化为**同一格式**，落 `data/corpus/normalized/`：

```jsonc
{
  "uid": "ShengbinYue__DISC-Law-SFT:jud_doc_sum-1",  // 全局唯一：来源前缀 + 原始 id
  "domain": "civil",                          // criminal | civil | procedural | general（主域）
  "domains": ["civil", "procedural"],         // 跨域多标签（路由器需要；专家 LoRA 仍按主域分配）
  "domain_source": "statute_ref_in_answer",   // 见下（★ 对"胜出域"贡献最大的那一路信号）
  "domain_evidence": {                        // 留痕：凭什么这么打标，可事后审计
    "scores": {"civil": 4.5, "procedural": 0.35},
    "evidence": {"statute_ref_in_answer": ["《合同法》→civil", "《民事诉讼法》→procedural"],
                 "doc_type": ["民事判决书→civil"]},
    "contrib_primary": {"statute_ref_in_answer": 2.0, "doc_type": 2.5}
  },
  "task": "jud_doc_sum",                      // 原始任务型（与 domain 正交）
  "task_kind": "case_analysis",               // case_analysis | qa | exam | concept
  "system": "",
  "instruction": "请大致描述这篇文书的内容。",
  "input": "...",
  "output": "...",
  "refs_question": ["《民法典》"],             // 问句里引用的法条（最高权重信号）
  "refs_answer": ["《民法典》第1165条"],       // 答案/解析里引用的法条
  "messages": [                                // 规范 chat 渲染，消除下游拼接歧义
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source_dataset": "ShengbinYue/DISC-Law-SFT",
  "source_file": "DISC-Law-SFT-Pair.jsonl",
  "source_license": "apache-2.0",
  "source_grade": "A",
  "synthetic": false,
  "replay": false,
  "content_sha1": "...",                       // 全局精确去重键
  "case_sha1": "..."                           // 案件正文指纹（尾部 400 字），跨源判重用
}
```

`domain_source` 枚举：`statute_ref_in_question` / `statute_ref_in_answer` / `doc_type` /
`task_hint` / `crime_concept` / `topic_keyword` / `fallback`

`task` 枚举（实测出现）：`jud_doc_sum` / `jud_read_compre` / `leg_ele_extra` / `leg_eve_detec` /
`leg_case_cls` / `sim_case_match` / `op_sum` / `sent_pred` / `judgement_predit` /
`legal_question_answering` / `exam` / `judical_examination(_v2)` / `legal_advice` /
`legal_counsel(_multi_turn)` / `crime_concept` / `statute_doc` / `statute_item`

> **注意 1**：`task` 必须**正交于** `domain`。不能出现「刑法任务集」这种把两个维度揉一起的命名，
> 否则消融实验无法拆解是域的作用还是任务的作用。
>
> **注意 2**：法条库（`statutes/`）单独一条流，**保留 `effective_from` / `effective_period` /
> `status`**，直接支撑「法条必须保留历史版本」这条硬约定。
> 实测（22,510 条）`status` 分布：**有效 15,282 / 已修改 4,719 / 已废止 1,681 /
> 尚未生效 1 / 脏值 `"7"` 827**。
> ⚠️ 该源的 `status` 混有脏值 `"7"`，**阶段 2b 切条时一并清洗**（把 `"7"` 归为未知并标注）。
> ⚠️ 该流的粒度是**整部法规全文**（不是逐条法条），必须先切条 —— 见 [3.1.2](#312-法条流statutes实测结构--一处必须纠正的表述--一处必须补的工序)。

### 3.5 MoE 路由对数据的额外要求

凑够 10 万条 SFT 数据，**不等于**能训出路由器：

| | 专家 LoRA 训练 | 路由器训练 |
|---|---|---|
| 需要什么 | `(instruction, input) → output` | `query → domain` |
| 数据量 | 10 万条 | 2–5 万条（可从专家数据派生） |
| **关键约束** | — | **必须与专家训练集 disjoint** |

**三条硬约束**：

1. **路由集与专家集必须切分** —— 按 uid 哈希抽取 **20,000 条**作为路由集（= 训练目标的 20%），
   **从专家训练集中移除**。否则路由器学到的是「专家见过的样本」，实际查询上泛化崩塌。
   **实施口径（2026-09-18 定）**：先从 26.8 万池子里预留路由集与 val/test，**再**对余下训练池
   下采样到 10 万 —— 这样「训练集 10 万」与「路由集 disjoint」同时成立。
   ⚠️ 若按字面「从 10 万里抽 20%」，训练集会被削到 8 万，与论文的 10 万训练量不符。
2. **必须支持跨域查询** —— 例如「合同诈骗」同时涉民法（合同）+ 刑法（诈骗罪）。
   单标签路由器必然错 → 用**多标签 + top-2 路由**，并做成消融项。
3. **路由评测集必须贴近真实「案件咨询形态」的分布** —— 人工标 **200–500 条**「案件咨询形态」的
   query + 域标签。**这 200–500 条必须新建，不能从任何已有数据集里借。**

**路由方案阶梯**（逐级做，每级留消融）：R0 关键词/法条名规则 → **R1 Qwen3-Embedding-0.6B
冻结 + 线性/MLP 头（推荐主方案）** → R2 Qwen3-0.6B 域分类 → R3 MixLoRA / MoLE 式层内路由。

### 3.6 数据红线与去污

**绝对不许进训练/验证集**：

1. **LexRubric 649 题 + 12,335 条 rubric**
2. **LexEval 23 任务 / 14,150 题**

> 原第 1 条列的是「CLaw 254 个最高法案例」，**已于 2026-09-19 删除** ——
> 该数据从未落地（`data/raw/` 无文件、`docs/data_manifest.json` 全 null），
> 属**空规则**。见第 4 节的退役说明。

**必须双向去污** —— 不仅查「训练集里有没有混入评测题」，还要查「评测集与训练语料是否同源」。
例如 `Skepsun/lawyer_llama_data` 本身是「司法考试题」，而 LexRubric 的 176 条就是 `sifakaoshi`
（司法考试）—— **两者存在同源风险，必须核查**。

```
对每个训练样本的 (instruction + input) 计算 simhash(64bit) 与精确哈希，
与黑名单库（254 + 649 + 14150 条）做：
  ① 精确哈希比对          → 命中即剔除
  ② simhash 汉明距离 ≤ 3  → 命中即剔除并记录
输出 docs/corpus/DECONTAMINATION_REPORT.json：
  {checked: N, exact_hits: [...], near_hits: [...], verdict: "PASS|FAIL"}
verdict 必须为 PASS 才允许进入训练。
```

---

## 4. 评测基准

**筛选口径**：① 够新（不落后于当前模型代际）② 我能自测评 ③ 已有多家模型公开得分。
（LawBench 官方榜停在 GPT-3/GPT-4 时代，最后提交 2023-11-13，已**降级为数据源与题型参考**。）

> ### ⛔ CLaw 已退役（2026-09-19）
>
> CLaw 曾列为主基准，现**已从本项目中彻底移除**。退役理由（均为实测核实的硬事实）：
>
> | 核查项 | 实测结果 |
> |---|---|
> | `data/raw/` 下 claw 文件 | **不存在**（`find` 返回空） |
> | `docs/data_manifest.json` | 4 个文件全 `sha256: null`、`verification_status: PENDING`、`actual_records: null` |
> | 官方评测脚本 | 未开放 → **无法自测** |
> | 实际引用它的文件 | 却多达 **26 个**（含 README 的「主基准」表述） |
>
> **结论：CLaw 在本项目里是一条「幽灵基准」——无法获取、无法验证、无法自测。**
> 以它为基础的一切表述（「主基准」「兼容性复现」）**均不成立，已删除**。
> 连带影响：原「CLaw 254 个案例不得进训练集」这条隔离规则是**空规则**（无数据可隔离），已一并删除。
>
> **现在的基准 = LexRubric（主锚点）+ LexEval（辅锚点）** —— 两者均可自测、均有公开基线，
> 登记见 [`docs/benchmarks/`](docs/benchmarks/)。
>
> > 附：语料**从来不是**从 CLaw 来的。实际语料源是 5 个 HF 数据集
> > （`ShengbinYue/DISC-Law-SFT` 主力、`twang2218/chinese-law-and-regulations` 法条、
> > `pandalla/chinese_law_examples`、`Skepsun/lawyer_llama_data`、`Dusker/lawyer-llama`），
> > 见 [`configs/corpus_sources.yaml`](configs/corpus_sources.yaml)。CLaw 只影响过「目标形状」的表述。

| 基准 | 年份 / 会议 | 规模 | 协议 | 定位 |
|---|---|---|---|---|
| **LexRubric** | 2026 / EMNLP Main | 649 题 / 12,335 条 rubric / 6 维度 | MIT | **主锚点**；18 个模型已有完整分数表 |
| **LexEval** | 2024 / NeurIPS | 23 任务 / 14,150 题 | MIT | 辅锚点；38 个模型已测 |

**已在服务器完成落地与登记**（两个基准在**目录 / 清单 / 校验报告 / 评分口径**四个层面完全隔离）：

| | LexRubric | LexEval |
|---|---|---|
| 服务器路径 | `data/benchmark/lexrubric/` | `data/benchmark/lexeval/` |
| commit | `9141ee4b` | `044c695f` |
| 规模 | 649 题 / 12,335 rubric / 9 文件 | 23 任务 / 14,150 题 / 65 文件 / 37.1 MiB |
| 校验 | **PASS**（0 失败 / 2 警告） | **PASS**（0 失败 / 0 警告） |
| 仓库归档 | [`docs/benchmarks/lexrubric.*`](docs/benchmarks/) | [`docs/benchmarks/lexeval.*`](docs/benchmarks/) |

> 唯一差异：LexRubric rubric 总数 12,335 对论文 12,337（差 2，全落在「法律准确性」维度），
> 推断论文含未公开私有 split。**论文里必须如实写这个差异，不得声称「与论文完全一致」。**

**评测侧的两个角色**（容易混淆，务必分清）：

| 角色 | 是谁 | 跑在哪 |
|---|---|---|
| **被评方**（答题） | Qwen3-8B（+ LoRA 适配器） | **服务器本地**（4bit 常驻 5.66 GB） |
| **判分器**（打分） | Gemini-2.5-Pro + DeepSeek-R1 | **外部 API**（权重不公开，规模远超本地可部署范围） |

> 「必须走外部 API」讲的**只是判分器**，与被评方是否本地部署无关。

详见 [`docs/benchmarks/README.md`](docs/benchmarks/README.md)。

---

## 5. 目录结构

```
law-agent/
├─ data/
│  ├─ corpus/
│  │  ├─ raw/<dataset>/    # 阶段 1：原始语料，只读，落盘即算 SHA-256
│  │  ├─ normalized/
│  │  │  ├─ qa/*.jsonl         # 阶段 2：问答/判决类统一 schema（三份源合计 285,257 条）
│  │  │  ├─ qa/_all.jsonl      # 阶段 2：上述三份的合并副本（285,257 行；uid 全局唯一，当场断言）
│  │  │  └─ statutes/*.jsonl   # 阶段 2：法条流（整部法规全文，23,510 部）
│  │  ├─ decontaminated/   # 阶段 3：去污清洗镜像（只读基线；qa 270,989 + statutes 22,771）
│  │  ├─ statute_items/    # 阶段 2b：切条后的逐条法条（72,449 条 = 向量库主料）
│  │  ├─ derived/          # 阶段 2c：程序法条文任务派生（13,484 条，synthetic，只进 train）
│  │  └─ MANIFEST.json     # 逐文件 SHA-256 清单
│  ├─ train/               # 阶段 4：train.jsonl(100,000) + {civil,criminal,procedure,general}.jsonl
│  ├─ dev/                 # 阶段 4：dev.jsonl(1,000) + 逐域视图
│  ├─ test/                # 阶段 4：test.jsonl(1,000) + 逐域视图
│  ├─ router/              # 阶段 4：router_train.jsonl(20,000，query → domain 多标签)
│  ├─ benchmark/           # 评测基准（LexRubric / LexEval），严禁进入训练
│  └─ _obsolete_*/         # 早期残留，保留可回溯
├─ configs/                # qlora_unified / adapters_router / retrieval / inference / judge
│                          # corpus_sources.yaml（合规层）+ corpus_adapters.yaml（解析层）
├─ models/                 # 底座权重 + 训练输出的 LoRA 适配器
├─ neo4j/                  # Neo4j 数据卷
├─ indexes/
│  └─ retrieval/            # 阶段 4b 检索层产物
│     ├─ law_nodes.jsonl           # :Law（L2 法名层）
│     ├─ provision_nodes.jsonl     # :Provision（item + doc）
│     ├─ edges_{has_provision,next,cites}.jsonl
│     └─ embeddings/{provision,provision_doc,case}/shard_*.{npy,jsonl,done}
│                                  # 1024d 向量分片 + meta + `.done` 断点标记
├─ src/
│  ├─ prepare/             # 数据清洗、切分、固化
│  ├─ retrieval/           # 混合检索、RRF、重排序、图谱查询
│  ├─ train/               # QLoRA 训练
│  ├─ inference/           # 闭卷 / RAG 推理
│  ├─ evaluation/          # 检索指标、法条指标、双盲 Judge 评分
│  └─ common/              # 通用工具
├─ scripts/
│  ├─ activate.sh          # 激活环境 + 设置全部变量
│  ├─ selfcheck.py         # 环境自检
│  ├─ smoke_test.py        # 端到端冒烟（加载模型 + LoRA 反向传播）
│  ├─ verify_env.sh        # 实测版全量自检（真跑 CUDA + 真分配显存 + 连 Neo4j）
│  ├─ corpus/              # 训练语料：采集(0/1) → 归一化+域打标(2) → 法条切条(2b) → 去污(3) → 切分(4)
│  │                       # 配套 prepare_*.sh 幂等包装，每步自带质检门禁
│  ├─ retrieval/           # 阶段 4b：法名归一(4b-1) → 边抽取(4b-2) → 热度(4b-2b) → 向量化(4b-3)
│  │                       #           → 导入 Neo4j(4b-4) → 检索消融(4b-5)
│  │                       #           → 融合权重扫描(4b-5b tune_fusion) → 重排诊断(4b-5c diag_reranker)
│  └─ benchmarks/          # 评测基准「拉取 → 验完整性 → 登记 SHA-256」
├─ outputs/                # 闭卷 / RAG / judge 输出
└─ docs/
   ├─ experiment_matrix.md  # 实验矩阵与消融
   ├─ run_order.md          # 执行顺序与阶段卡口
   ├─ leakage_rules.md      # 防泄漏红线与四级查重
   ├─ model_selection.md    # 模型选型分析
   ├─ env_setup.md          # 环境搭建与事故记录
   ├─ finetune_data_plan.md # 10 万条语料方案（人读的完整版）
   ├─ retrieval_design.md   # 阶段 4b 检索层设计（节点/边/指标口径）
   ├─ neo4j_setup.cypher    # Neo4j 约束与索引 DDL
   ├─ corpus/               # 训练语料登记（MANIFEST + 溯源 + 判重报告 + 阶段 2 统计）
   ├─ retrieval/            # 检索层报告（EDGE_EXTRACT / EMBEDDING_REPORT / NEO4J_IMPORT / SMOKE_*）
   ├─ benchmarks/           # 评测基准登记（LexRubric / LexEval，各自独立）
   └─ data_manifest.json    # **自建语料源**固化清单（HF 源逐文件 sha256；不混入评测基准）
```

---

## 6. 实验环境

> 论文原稿写的「Windows 11 + WSL2 + 12GB 显存」**已作废**。实验实际在实验服务器上完成。
> 完整环境说明见 [`docs/env_setup.md`](docs/env_setup.md)。

| 项 | 实测值 |
|---|---|
| GPU | **2 × NVIDIA A800 80GB PCIe**（160GB 显存，sm_80） |
| CPU / 内存 | 128 核 / 503 GB |
| 驱动 | 550.163.01 → 驱动侧 **CUDA 12.4** 为上限（**决定 torch 版本**） |
| OS / 权限 | Ubuntu 20.04.6，**无 sudo** |
| Python | 3.11.16（uv 安装的 python-build-standalone，系统只有 3.8） |
| PyTorch | **`torch==2.6.0+cu124`**（不能用 cu13x，驱动会拒绝） |
| 图谱 | Neo4j 5.26 Community，Docker 部署，端口 7474 / 7687 |
| 磁盘 | `/mnt/data` 8.7T，项目根 `/mnt/data/lidian/law-agent` |

### 快速开始

```bash
source /mnt/data/lidian/law-agent/scripts/activate.sh   # 激活环境 + 设置全部变量
python scripts/selfcheck.py                             # 环境自检
python scripts/smoke_test.py                            # 端到端冒烟
bash scripts/verify_env.sh                              # 实测版全量自检
bash scripts/corpus/prepare_corpus.sh --list            # 看看阶段 1 要采什么
```

一键重建环境（幂等）：`bash scripts/server_env.sh`

### 实测性能（Qwen3-8B）

| 指标 | 数值 |
|---|---|
| 4bit NF4 加载耗时 | 11 s |
| 4bit 常驻显存 | 5.66 GB |
| LoRA r=16 可训练参数 | 43.6 M（占 0.917%） |
| 单步训练峰值显存 | 9.14 GB |

单卡 80GB 意味着显存**不再是约束**：可提高 LoRA rank、放大 batch 与序列长度，
底座也完全可以换成 14B/32B。但对比实验只允许换一个变量 —— **底座一旦定下不要中途改**。

### 大陆网络注意

- `huggingface.co` **不可达** → 必须走 `hf-mirror.com`（已写入 `activate.sh`）
- `dist.neo4j.org` 返回 403 → Neo4j 改用 Docker 镜像
- `pypi.nvidia.com` 不可达 → torch 从清华 PyPI 装，不要用 PyTorch 官方索引单独装
- 本机到 `github.com` 极不稳定（`502` / `Connection reset`），push 需重试多次

> ⚠️ **PyTorch 版本必须按驱动 CUDA 主版本钉死**，且安装时**不要**用
> `--index-strategy unsafe-best-match`（会跨索引挑到 CUDA 13 版，导致 GPU 完全不可用）。
> 详见 `docs/env_setup.md` 第 3.3 节的事故记录。

---

## 7. 数据与合规（强制）

- `data/corpus/raw/` **只读**：下载后立即计算 SHA-256，记录来源、日期、版本，写入 `data/corpus/MANIFEST.json`。
- **LexRubric 649 题、LexEval 14,150 题，一律不得进入训练/验证集。**
- 训练集与测试集需做四级查重（完全 / 规范化 / MinHash / 向量），余弦 > 0.92 人工复核。
- 法条必须保留**历史版本与生效/失效区间**，不得只存「最新版」。
- **仓库保持 Private**：语料含协议未明的 B 级源（见 `configs/corpus_sources.yaml`），论文发表前不公开。
- **论文表述红线**：不得出现任何以 CLaw 为基准的比较或「复现」表述（该基准数据从未获取，
  已于 2026-09-19 退役，见第 4 节）。基准成绩只报 LexRubric / LexEval，且两者**分节分表**。

详见 [`docs/leakage_rules.md`](docs/leakage_rules.md)。

---

## 8. 进度与执行顺序

执行顺序见 [`docs/run_order.md`](docs/run_order.md)。
核心原则：**先跑通评测链路，再训练模型；先用少量案例验证 Judge，再整体开跑。**
（原表述为「再跑 254 个」= CLaw 254 案 —— **CLaw 已于 2026-09-19 退役**，见下方专用段。）

### 总实验流程（2026-09-20 定稿 · 按序执行 · 状态实时更新）

> 顺序即依赖序：上游不出 PASS，下游不开工。GPU 侧"训练占 GPU1、推理/评测占 GPU0"可并行。

| # | 步骤 | 内容 / 验收标准 | 状态 |
|---|---|---|---|
| P0 | 语料与检索层 | 阶段 0–4 + 4b 全链（去污复扫 PASS、四份 split、Neo4j 导入、检索消融达标） | ✅ PASS |
| P1 | A0 统一适配器 QLoRA | Qwen3-8B 4bit + r16/α32 七投影，13,276 步 / 2ep，`verdict=PASS` 11/11 | ✅ PASS（train_loss 0.6756 / dev eval_loss 0.3896） |
| P2 | 三域专家 QLoRA | criminal / civil / procedure，各自 `verdict=PASS` 11/11 检查 | ✅ PASS（criminal 0.5434 / civil 0.7009 / procedure 0.7730） |
| P3 | A0 全量基准生成 | GPU0 分档链：lexrubric 649 (cap1536) → lexeval 客观 11,400 (cap256) → lexeval 生成 2,750 (cap1536)，单系统 ≈14h | 🔄 2026-09-20 20:24 以**口径对齐后代码**重启（旧口径 504 条已归档 `answers.bf16norm_STALE.jsonl`） |
| P4 | A0 判分 | LexEval 客观秒级本地打分；LexRubric 走 `--judges minimax-m3`（12,335 条 rubric / 系统） | ⬜ 等 P3 |
| P5 | L1 请求级路由 | Qwen3-Embedding 冻结 + 逻辑回归头，独立集 top1_acc / 回退率验收 | ✅ PASS（top1_acc 0.8370 / macro_f1 0.7922 / routed_acc 0.8976） |
| P6 | L2 层内门控 MoLE | 4 专家齐备后：① `verify_mixture.py` **在目标 GPU** 全层全专家等价自检 → ② `train_moe_gate.py`（lr 1e-3，balance α=0.01，fp32 门控，≈2.36M 参数），验收 `expert_utilization` 无坍缩 | 🔄 ① PASS（2026-09-20 GPU/4bit 全量：四专家 max\|Δ\|=0 逐位一致）；② 训练中（GPU1，1,125 步，`logs/train/L2_gate.log`） |
| P7 | MoE 推理分支 | `run_inference.py --moe-gate`（底座+K 专家+门控装配），否则 MoE 行无法进评测 | ✅ 代码已写并提交（commit 56ebc2a / 767715b） |
| P8 | checkpoint 选择 | 用 dev（内部验证集）在 ckpt-6638(1ep) vs ckpt-13276(2ep) 间选点；**论文干净数字必须来自 test(1k) 或基准集**，dev 训练时被用作 eval 不可当"未见数据" | ⬜ |
| P9 | 消融 + 终评 | E0/E5 主对比 + A1/A3/A4（oracle/平均/随机路由）+ A7/A8/A9；各系统按 P3 分档口径生成、按 P4 判分；完整矩阵双卡 ≈1.75–2.6 天 | ⬜ |

**论文数字红线**：① 生成侧指标只能来自 P3/P4 的评测产物（loss 不是论文指标）；② MoE 行评测前 P6+P7 必须双 PASS；
③ 所有系统的 `max_new_tokens` 分档口径必须一致；④ **凡 4bit 底座，推理与门控训练都必须调
`prepare_model_for_kbit_training`**（LN/lm_head upcast fp32）—— 训练调了而推理不调，
36 层累积后 logit 相对差 ~2e-2，等价性自检 GPU/4bit 实锤（四专家 1.8–2.5e-2 FAIL → 修后 max\|Δ\|=0）。
LoRA 增量同样必须保持 fp32 加进底座输出（bf16+fp32 提升，与 peft 同型）。


### 已完成

> **✅ 2026-09-19：阶段 3 误删事故已修复，下游全链路重跑完成 —— 本节数字已全部回填为重跑后实测值**
>
> **事故**：`decontaminate.py` 对法条流做 simhash 近似去污时，uid 粒度是**整部法规**而
> 指纹只取**正文前 800 字** → 一处近似命中就**整部法典连坐删除**。实测误删
> **739 行 / 646 个法名 / 4,191,231 字**（民法典、刑法、刑诉法、民法总则、民法通则、
> 劳动合同法、期货和衍生品法…），且**剔除全部来自近似匹配，精确（逐字复制）命中为 0**。
> 该事故被一份**空转 PASS** 掩盖：复扫扫的是已删干净的镜像，只能证幂等、证不了删得对。
>
> **根因**：法条流是**检索语料**（3.1.3 明确「法条流不进 SFT」），且评测集必然引用法条原文
> → 拿法条去比评测题，近似误报是**结构性必然**；删法条等于删掉 RAG 的开卷依据。
>
> **修复**：`--statutes-near {remove,audit,ignore}`（**默认 `audit`**）—— 法条流**只做精确 sha1 剔除**，
> 近似命中只记录不剔除（报告单列 4b 节留痕）；`remove` 可复现历史行为。
>
> **独立复核（第三方口径）**：拿**事故期**剔除清单直接撞法条流 uid（不是比对目录差集，是撞清单）——
> 事故期清单 **15,007** 行 ∩ `normalized/statutes` 唯一 uid（23,510）= **恰好 739**，
> 与 `normalized − decontaminated = 739` 完全一致。**事故量化闭环，且这条交集本身就是最好的门禁断言。**
>
> ⚠️ **口径提醒**：上面的 15,007 行是**修复前的中间产物**。仓库里 checked-in 的
> [`docs/corpus/DECONTAM_REMOVED_UIDS.txt`](docs/corpus/DECONTAM_REMOVED_UIDS.txt) 是重跑后的**最终**清单
> —— **14,268 行**，与 [`APPLY_SUMMARY.json`](docs/corpus/APPLY_SUMMARY.json) 的 `banned_uids = 14268`
> 逐字一致。**不要拿仓库里这一份去复算 739**（那是另一代清单）。
>
> **重跑结果（4b 链 v4，2026-09-19 13:23 起）**：
> `STEP0` 四断言全过（★ 剔除清单 ∩ 法条流 = **0**）→ apply 294,499 保留 / 14,268 剔除
> → 2b `items=72,449` PASS → 2c `15,535` PASS → 阶段 4 四份 split PASS → 4b-2/3/4/5 全部按新语料重跑。
> **法条切条 65,037 → 72,449（+7,412，+11.4%）**，其中程序法 +833。
> 另：`STEP 6` 曾因 `extract_edges.py` 参数名写错（`--out-json` 应为 `--out-dir/--report-json`）
> 而 fail-fast 中断，已单独重跑并 PASS —— 链条的 fail-fast 起到了应有作用，**没有带着错误继续往下跑**。


- ✅ 实验环境全栈（`envs/main`，torch 2.6.0+cu124，Neo4j 17 索引 ONLINE）
- ✅ 底座与向量模型下载（`Qwen3-8B` 16 GB / `Qwen3-Embedding-0.6B`）
- ✅ 评测基准 LexRubric + LexEval 下载、校验、SHA-256 登记与归档
- ✅ 双盲判分层（`src/evaluation/judge/`）+ LexRubric 评测器改造 + 统计层
- ✅ **阶段 1：A 级训练语料采集** —— 5 个数据集 / **23 文件 / 1.2 GB** → `data/corpus/raw/`，
  逐文件 SHA-256 已登记，验收 **PASS**（[`docs/corpus/`](docs/corpus/)）；
  采集链路幂等：`bash scripts/corpus/prepare_corpus.sh`
- ⛔ **检索指标原目标 `Recall@5 ≥ 0.85` / `Recall@10 ≥ 0.90` 已作废（2026-09-19，实测不可达）**：
  纯 dense 在 72,088 条法条上把候选深度开到 **1000**，recall 也只有 **0.8981**
  → 该目标在数学上不可达（瓶颈是 0.6B embedding 的表示能力，非融合/排序）。
  **新主指标（可达）**：`hit@5 ≥ 0.60` / `recall@10 ≥ 0.50` / `MRR@10 ≥ 0.45`，
  实测（test 414 query，`hybrid`）：**hit@5 0.6691 / recall@10 0.5216 / MRR@10 0.5328 —— 全部达标**；
  历史版本准确率 ≥ 0.95 目标保留
- ✅ **阶段 2：归一化 + 域打标** —— raw 349,665 条 → **qa 流 285,257 条** + 法条库 23,510 条，
  统一 schema（含 `messages` 规范 chat 渲染与 `domain_evidence` 审计链）；
  **配比核算结果：刑法 97,414 / 民法 104,691 / 程序法 25,445 / 通用 57,707 —— 每域都超额，总缺口 0**；
  跨源判重完成（`Dusker/.../DISC-Law-SFT-Pair.json` 100% 重复已丢弃，`-Triplet.json` 仅 20.5% 重复故保留）；
  **uid 全局唯一**：`uid = <dataset>__<file_stem>:<source_index>`（含源文件词干，与阶段 2b/4 同口径），
  构造成立即断言（`uid_unique_assert`：308,767 行 → 308,767 个唯一 uid，duplicates = 0）；
  合并流只并本轮 `stats["sources"]` 登记的产出，`*.sample.jsonl` 残留与陈旧 `_all.jsonl` 已被守卫拦截；
  幂等链路 `bash scripts/corpus/prepare_normalize.sh`（[`docs/corpus/`](docs/corpus/)）

- ✅ **阶段 3：双向去污（★ 硬门禁，两遍式）—— 按新口径全链路重跑完成**（2026-09-19）。
  首扫 308,767 行：精确命中 **0** / 近似命中 **17,837 处**（汉明距离分布 0:**493** / 1:981 / 2:2,445 / 3:13,918）
  → **剔除 14,268 个唯一 uid**（来源分布：`_all.jsonl` 合并副本口径 + twang2218 734 + pandalla 16；
  其中 `DISC-Law-SFT` −12,598 / `Dusker` −1,117 / `Skepsun` −553）。
  **法条流豁免生效**：法条流近似命中 **750 处 / 739 个法名**，`policy=audit` → **只记录不剔除**；
  ★ 门禁断言 `剔除清单 ∩ 法条流 uid = 0`（历史上这里是 739，现在必须是 0）。
  apply 生成清洗镜像 `decontaminated/`（normalized 保持只读）：**kept 294,499 / removed 14,268**
  （实测剔除行数 == 清单 uid 数，恒等式成立）。
  **恒等式闭环**：`首扫 308,767 − 清洗后 294,499 = 14,268 = 剔除 uid 数`。
  清洗后按域（整个镜像，含法条流）：刑法 **91,806** / 民法 **102,736** / 程序法 **24,642** / 通用 **75,315**。
  同源风险专项：Skepsun 司考 vs LexRubric `sifakaoshi` 首扫精确 0 / 近似 1，该条已落剔除集。
  黑名单 = LexEval 14,150 + LexRubric 649（**CLaw 已退出论文，不在黑名单**）。
  报告：[`docs/corpus/DECONTAMINATION_REPORT.md`](docs/corpus/DECONTAMINATION_REPORT.md)。
  复扫门禁（独立、只读、**与数据管线并行**跑，最后 `wait` 收口）：
  `prepare_decontaminate_pass.sh` 扫**清洗镜像**（294,499 行），要求 **精确 0 / 近似 0**，
  证据文件 `DECONTAM_REMOVED_UIDS_PASS.txt` 必须为**空文件**；
  **✅ 已收口（2026-09-19）**：复扫报告 `verdict = PASS`，被查 **294,499** 行、
  精确命中 **0** / 近似命中 **0**，证据文件实测 **0 字节**（空文件）；
  黑名单 **14,799** 条（LexEval 14,150 + LexRubric `zixun` 473 + `sikao` 176）。
  法条流在这一遍同样带 `--statutes-near audit`，其近似命中只进报告不改清单 ——
  **实测 750 个近似命中事件、涉及 739 部法规，按硬约定「只记录不剔除」**
  （法条流不进 SFT，不存在评测泄漏，拿法条正文去比评测题结构上必然假阳性）。
  ⚠️ 注意：复扫只能证明**幂等**，**证明不了删得对** —— 上次的「空转 PASS」正是栽在这里；
  删得对不对只能靠**首扫阶段清单 ∩ uid 的独立复核**（本次 = 0）来保证。

- ✅ **阶段 2b：法条切条（整部法规 → 逐条法条）**（2026-09-19 重跑，质检 verdict = PASS，6.6s）。
  input 23,510 行 → **72,449 条检索单元**（twang2218 + pandalla 两源；
  按 level：`item` 72,097 / `doc` 兜底 **352**）；条目 uid = `<dataset>__<file_stem>:<source_index>#<条号>`，
  与阶段 2/4 同口径，产物 uid 唯一（质检 C1 = 0）；文档内重复条号 75 处**被跳过不重复入库**。
  只留国家级规范（司法解释 788 / 行政法规 693 / 法律 429 / 法律解释 26 = **1,936 部文档**）；
  地方性法规 **19,733 部**（占源 87%）丢弃，另丢「修改、废止的决定」661 /
  「有关法律问题和重大问题的决定」172 / 宪法 7 / 监察法规 1。
  `status` 落值 有效 1,471 / 已修改 284 / 已废止 180 / 尚未生效 1；脏值 **0**。
  长度分布：p10 38 / p50 91 / p90 221 / **max 61,387**（最长者是国务院行政许可决定所附 500 项目录）。
  质检 C1–C7 全过（残留条号 0 / 目录块 0 / 短文占比 0.08% / 章级占比 0.489）。
  **法名去重口径**：切条产物里 distinct 法规标题 **1,904**；经 4b-1 的 L2 规范化（剥书名号 + 剥版本后缀）
  后落到 `:Law` 节点 **1,647** —— 两个数不是缺口，是「原始标题 vs 规范化法名」两个口径。
  脚本 `split_statutes.py` + 门禁 `verify_statute_items.py`，报告
  [`docs/corpus/STATUTE_SPLIT_REPORT.md`](docs/corpus/STATUTE_SPLIT_REPORT.md)。
  三个实测坑：锚点必须盯**行首**（交叉引用虚高 9.8 万）、目录后正文重启会吞首章、
  无条号批复需整篇入库但修正案必须排除。

- ✅ **阶段 2c：程序法条文任务派生（补题型多样性）**（2026-09-19 重跑，质检 verdict = PASS）。
  用阶段 2b 切出的 **7,355 条程序法条级 item**（覆盖 **121 部**程序法）
  派生出 **15,535 条**题型样本：`statute_recall` 6,392 / `statute_locate` 6,392 /
  `statute_structure` 2,751（丢弃 6,530 条：不适用 4,498 + 正文重复 2,032）。
  **三条硬原则**：
  ① 答案**逐字取自条文原文**（`verbatim_check` 15,535/15,535 通过、不符 0）；
  ② 只派生程序法（`domain=procedural` 100%，`domain_source=derived_from_statute_item`）；
  ③ 一律 `synthetic=true` + `derived=true`，**只进 train**，不进 val/test/router。
  脚本 `derive_statute_tasks.py` + 门禁 `prepare_derive_statute_tasks.sh`，报告
  [`docs/corpus/DERIVE_REPORT.md`](docs/corpus/DERIVE_REPORT.md)。
  一个实测坑：源 `law_title` 自带书名号，模板再包一层会产出 `《《…》》` →
  加 `unwrap_title()` 剥壳 + `title_wrap_check` 硬断言（实测嵌套书名号 = **0**）。
  **确定性边界（必须说清）**：脚本无随机数，相同输入必得**相同的样本集合 / uid /
  `content_sha1` / 行序**；但每条记录写 `normalized_at`（运行时刻），这是**唯一的非确定性来源**
  → 要逐字节复现产物必须带同一 `--stamp`。`content_sha1` 不含时间戳，故**重跑 2c 不影响阶段 4**。

- ✅ **阶段 4：分层下采样 + 切分（★ 硬卡口）**（2026-09-19 重跑，质检 verdict = PASS，38.8s）。
  池 **286,524**（真实 QA 流 270,989 + 派生 15,535）→
  **122,000 条四份 split，两两不相交（uid 口径交集全为 0）**：
  训练集 **100,000**（刑法 30,000 / 民法 40,000 / 程序法 20,000 / 通用 10,000，**精确命中**）
  + 验证集 1,000 + 测试集 1,000 + 路由集 20,000；池剩余 **164,524** 可回溯。
  质检 C1–C14 全过（含 C14 `uid == uid_g` **122,000/122,000 行全覆盖比对、0 处不一致**，
  覆盖不全同样判 FAIL）；`failures = []`、`warnings = []`。
  跨流隔离：法条流 72,449 与四份 split 交叠 **0**。
  产物直接落在 `configs/` 写死的路径上（`data/train/train.jsonl`、`data/dev/dev.jsonl`、
  `data/test/test.jsonl`、`data/router/router_train.jsonl`），**配置零改动**。
  切分口径：样本键 = `content_sha1`（实测全局唯一）、全 sha1 排序确定性抽取
  （**无随机数、不写任何时间戳**）→ 同一份输入重跑阶段 4 逐字节一致。
  先预留 router/val/test 再对训练池下采样。报告
  [`docs/corpus/SPLIT_REPORT.md`](docs/corpus/SPLIT_REPORT.md) + [`SPLIT_VERIFY.md`](docs/corpus/SPLIT_VERIFY.md)。
  **三层约束全部满足、零放宽**：各域头号任务份额均 ≤ 35%（程序法 `legal_question_answering`
  29.1% / 各域 `cap_relaxed` **全为 false**）；程序法派生实得 3,000 = 域内 **15.0%**，
  `derived_relaxed` 亦为 false。

### 决策项记录

- ✅ **决策项 A｜程序法任务多样性 —— 已解决**：曾因 `legal_question_answering` 占程序法 49.3% 触发软上限放宽。
  已按 README 3.1 规划用阶段 2b 的程序法条文派生「程序法条文任务」
  （见上「阶段 2c」），阶段 4 实得 3,000 条派生进训练集后份额降到 35% 以下，**不再需要放宽**。
- ⬜ **决策项 B｜通用回放**（唯一未决项）：池里 `replay` 标记**全为 0** —— DISC-Law-SFT 内置的 Alpaca-GPT4/Firefly
  通用回放**不在已下载的文件内**，现由 `general` 桶（法律领域内样本）代充 10%。
  **用户 2026-09-19 指示：暂不动**，留作论文局限性一节如实说明。
- ✅ **决策项 C｜阶段 2/3 缺陷根治 —— 已完成**：`normalize_corpus.py` 的 `uid` 已加入 `source_file`
  词干、合并流只并本轮正式产出、`*.sample.jsonl` 残留已剔除，阶段 2/3/2b/4 **全链路已重跑**。
  详见 [`docs/corpus/README.md`](docs/corpus/README.md) 第十节「已修复缺陷与前后对照」。
- ✅ **决策项 D｜`case_embedding_procedural` 仅 37 条 —— 已解决**（2026-09-19）：
  **原记录的根因是错的**（曾写「`domain4_of()` 只看 `domain` 字段」）。实测交叉验证：
  程序法 `case_analysis` 共 **3,333** 条，`domain` 字段**就是** `procedural`，
  其中 **3,268 条（98.1%）被阶段 4 的分层切分吃进 train/router/dev/test**，
  而索引侧为防泄漏把四份 split **全部**排除 → 案件池只剩 65 条（实入 37）。
  被排除的 3,268 条在 split 里确实以 `case_analysis`/`procedural` 身份存在，
  **不是 uid 撞号误伤**。**用户拍板**：改为**只排除评测集 dev/test**（`--case-scope eval`）
  —— 检索库是知识库不是训练数据，IR 标准做法是「语料固定、只排除被查询项本身」。
  **改后程序法案件库 65 → 3,268**（四库齐全），新案件池 145,339 条。

### 已完成（续）：4b 检索层 + 阶段 5 训练

- ✅ **阶段 4b：检索层（向量库 + 知识图谱）—— 已建成并按新语料全量重跑**（2026-09-19）。
  设计定稿 → [`docs/retrieval_design.md`](docs/retrieval_design.md)；
  **已拍板（D1–D5 全按「A」）**：① 时间版本过滤无字段可依 → A2 消融改用 `status`；
  ② 款/项粒度为 0 → 评测表删「条+款/项」两列；③ 图谱扩展主力改 `NEXT`（`CITES` 仅覆盖少数条文）；
  ④ 多版本语料不补，论文声明「2026-09 快照、单版本」；
  ⑤ 论文区分「模式层设计（17 类全保留）」与「数据层实例化（实测 5 可建 + 2 弱 + 10 无料）」。
  **长文本隐患已排除**：item 最长 26,980 字**不是切条遗漏**，而是条文自带的附表
  （[`docs/corpus/LONG_TEXT_PROBE.md`](docs/corpus/LONG_TEXT_PROBE.md)），不需重跑阶段 2b。
  **论文改写稿已备** → [`docs/paper_revision_notes.md`](docs/paper_revision_notes.md)。

  #### 4b-0 泄漏门禁 —— PASS
  四份 split 两两交集 **0**；实测**评测集 22,000 条 100% 落在 qa 流，其中案件分析类独占 11,811 条**
  —— 不做集合差就直接向量化，近 1.2 万条评测样本会进检索库（模型对考题「开卷检索」）。
  **检索库范围定案（2026-09-19 修订）**：案件 **145,339** + 法条 **72,449** = **217,788** 条。
  ⚠️ **排除口径已改**：原先把 train/dev/test/router **四份全排除**，导致程序法案件几乎归零
  （3,333 条里 3,268 条被切分吃掉 → 索引只剩 65 条，实入 37 条）。
  现改为**只排除评测集 dev/test**（`--case-scope eval`）——检索库是**知识库**不是训练数据，
  IR 的标准做法是「语料固定、只排除被查询项本身」；train/router 不参与评测，入库零泄漏。
  改后程序法案件库 **65 → 3,268**，四库齐全（general 22,263 / civil 45,638 / criminal 74,170 /
  procedural 3,268）。证据 → [`docs/corpus/RETRIEVAL_SCOPE.md`](docs/corpus/RETRIEVAL_SCOPE.md)。

  #### 4b-1 法名两级规范化 —— PASS
  `raw 1,807 → L1(剥书名号) 1,727 → L2(再剥版本后缀)`；跨源交集 `0 → 80 → 228` ——
  不归一化两源在字符串层面**一个法都连不上**，图谱会被撕成两半。
  ⚠️ 原 H4 记「228 个书名号重复」**归因有误**，实为 **80 书名号 + 148 版本后缀**。
  新语料下落到 `:Law` 节点 **1,647**。

  #### 4b-2 图谱边抽取 —— PASS（4.0s）
  | 产出 | 实测 |
  |---|---:|
  | `:Law`（L2 法名） | **1,647** |
  | `:Provision`（item 72,097 + doc 352） | **72,449** |
  | `HAS_PROVISION` | 72,449 |
  | `NEXT`（★ 扩展主力，覆盖率 100%） | **69,530** |
  | `CITES`（辅助） | **7,224** |

  三个关键实现判断（都是实测出来的，不是想当然）：
  ① **`NEXT` 的分组键是「来源文档」而不是 `law_id`** —— 同一 L2 法名可对应多个源文档，
     按 law_id 连会把不同版本的第 2 条接在一起；
  ② **`CITES` 自指引用必须写成 `本(法|条例|规定|解释|规则|办法|细则|准则|意见|批复|通知|决定)第X条`**
     （只写「本法」会漏一大半：自指引用 1,826 → 3,894，CITES 边 +78%）；
  ③ **`provision_id` 必须消歧**（`-v{k}`）→ 论文表 3.2 的「唯一键 = 法律名称 + 条文编号」**不成立**，
     须补消歧规则说明。
  两个新发现：**`content_sha1` 不能用来判跨源重复**（它含法名+条号前缀，前缀写法不同必然不同 →
  假阴性），须用**正文**指纹（精确 430 组 / 宽松 846 组跨源重复，暂定「检索时折叠、不删节点」）；
  产物 → [`docs/retrieval/EDGE_EXTRACT.md`](docs/retrieval/EDGE_EXTRACT.md)。

  #### 4b-3 向量化 —— PASS（Qwen3-Embedding-0.6B / 1024d / L2 归一化；**决策项 D 重建后的最终口径**）
  **按用户拍板改为「case 4 库 + 法条 1 库」**（与论文主从关系一致：入口是裁判文书 Case）：
  | 库 | 行数 |
  |---|---:|
  | `provision_embedding`（法条，item 72,088 + doc 352） | **72,440** |
  | `case_embedding_general` | **22,263** |
  | `case_embedding_civil` | **45,638** |
  | `case_embedding_criminal` | **74,170** |
  | `case_embedding_procedural` | **3,268** |
  合计 **217,779** 条向量（法条 72,440 + 案件 145,339）。item 与 doc **同库不同语义**
  （doc 最长 6 万字 vs item 中位 91 字）→
  入库时分流到两个属性、两个向量索引。文档侧不加 instruction、查询侧加
  `Instruct: …\nQuery: …`（Qwen3-Embedding 是 instruction-aware）。分片 5000/片 + `.done` 断点续跑。
  ✅ **`case_embedding_procedural` 37 → 3,268（决策项 D 已闭环）**：根因**不是** `domain4_of()`
  看错字段，而是**旧排除口径把 train/router 的案件样本也一并排掉** —— 程序法 `case_analysis`
  共 3,333 条，其中 **3,268（98.1%）落在 train split**，索引侧只剩 65、实际入库 37。
  改为「只排除评测集（dev/test）」后四库齐全（口径依据见 §4b-0）。
  重建必须带 `--clean-cases`：池子条数一变，旧 `.done` 分片会**静默错位跳过**。

  #### 4b-2b 热度权重（用户拍板新增）—— PASS
  `Provision.hotness = 0.85·norm_log1p(in_cites) + 0.15·rank_norm(法源位阶)`；
  `Case.hotness = norm_log1p(n_same_case 在**检索库内**的复现次数)`。
  实测：CITES 入度 > 0 的条文 **4,027** 条（≥5 次 147 条 / ≥20 次 5 条）；
  唯一 `case_sha1` **134,597** 个，同案最大复现 **82**。
  **排序用法**：`final = rrf_score × (1 + W_HOT × hotness)`，`W_HOT = 0.05`
  —— 热度只做 ≤5% 的先验微调，**不允许盖过相关性本身**。
  🔒 **零泄漏**：案件复现次数**只在检索库内**统计，绝不把 train/val/test/router 的样本算进来。
  报告 [`docs/retrieval/HOTNESS_REPORT.md`](docs/retrieval/HOTNESS_REPORT.md)。

  #### 4b-4 导入 Neo4j —— PASS（539.3s，**决策项 D 重建后的最终入库**）
  Law **1,647** / Provision **72,449**（item 72,088 + doc 352，无向量 0）/ Case **145,339**
  （general 22,263 / civil 45,638 / criminal 74,170 / procedural 3,268）；关系
  `HAS_PROVISION` 72,449、`IN_DOMAIN` 72,449、`OF_TYPE` 72,440、`FROM_SOURCE` 72,449、
  `NEXT` 69,530、`CITES` 7,224、`SAME_CASE` **10,742**；
  **7 个向量索引 + 2 个全文索引全部 ONLINE**；热度 provision 72,440 / case 145,339。
  13 项断言全 True、`verdict = PASS`。报告
  [`docs/retrieval/NEO4J_IMPORT.md`](docs/retrieval/NEO4J_IMPORT.md)。
  （历史：重建前旧口径下 Case 82,820 / `SAME_CASE` 4,470 —— 那是因为旧口径把 train/router
  的案件样本也一并排除，案件库被削到只剩 8.2 万；口径修正见 §4b-0 / §4b-3。）

  #### 4b-5 检索链路 + 消融矩阵 —— 已跑通，**并推翻了「加料越多越好」的直觉**
  一键命令见 [§10.8](#108-阶段-4b--检索层向量库--知识图谱)。test 414 query / union 口径：

  | 配置 | recall@5 | recall@10 | hit@5 | MRR@10 | 三目标 |
  |---|---:|---:|---:|---:|:---:|
  | `vector`（纯向量） | 0.4040 | 0.4922 | 0.6739 | 0.5296 | — |
  | `dense_hot`（+热度） | 0.4040 | 0.4922 | 0.6739 | 0.5296 | — |
  | `dense_bm25`（+BM25） | 0.4242 | 0.5069 | **0.6836** | **0.5516** | ✅ |
  | `dense_graph`（+图谱注入） | 0.4059 | 0.5032 | 0.6473 | 0.4878 | ✅ |
  | **`hybrid`（加权 RRF：向量+BM25+图谱）★主链路** | **0.4298** | **0.5216** | 0.6691 | 0.5328 | ✅ |
  | `full`（hybrid + 热度 + bge-reranker 精排） | 0.3646 | 0.4683 | 0.6377 | 0.5165 | — |

  ★ **两条「最优」在不同轴上，报告必须分开写**（合成一个「最佳」就是把结论说错）：
  **深召回轴** `argmax recall@5` = **`hybrid`**（recall@5 0.4298 / recall@10 0.5216）——
  决定送进生成器的候选池；**头部队列轴** `argmax (hit@5, MRR@10)` = **`dense_bm25`**
  （hit@5 0.6836 / MRR@10 0.5516）—— 决定用户/评测直接看的前 5 条。
  三目标**同时**达标的配置 = `dense_bm25` / `dense_graph` / `hybrid`。
  主链路取 `hybrid`（论文架构要求图谱参与，且深召回更好），`dense_bm25` 作为
  「图谱不参与」的头部精度对照。

  **三条必须写进论文的结论**：
  1. **等权 RRF + 图谱「注入式」扩展会把精度打崩**（首版 `dense_graph` recall@5 仅 0.2545
     vs 纯向量 0.3961）。根因：`graph_expand` 把种子节点的全部邻居当**同权有序列表**丢进 RRF，
     邻近条文与 dense rank-1 拿同样的 `1/(k+rank)` 质量 → top-5 被无关邻条占满。
     修正：**加权 RRF + 图谱列表限长降权**（dense 1.0 / BM25 0.7 / KG 0.25，k 60→10）。
  2. **图谱的真实作用是「候选扩展器」而非「排序信号」**：`w_graph` 从 0 → 0.5 时
     recall@5 **掉**（0.3912→0.3383），但 recall@50 **涨**（0.6899→0.7128）。
     即它把远离的正确答案拉进候选池、同时污染头部 —— 这正好论证了
     「宽召回 + 精排」的两段式架构，是论文把知识图谱讲圆的关键机制。
  3. **重排器对本任务无效（负结果，已用受控实验证伪「实现有 bug」）**：
     `bge-reranker-v2-m3` 的判别力其实很好（gold vs 随机负例 **AUC 0.888**，
     gold 平均分 0.575 vs 负例 0.116，反向打分对照直接崩到 0.0 → 实现无误），
     但端到端重排 dense top-50 **从不优于纯 dense**（recall@5 0.3534 vs 0.3534，
     recall@10 0.4297 vs **0.4418**），且池越深越差（pool 50 → 200 再掉 0.02）。
     原因是 gold 是「参考答案里引用的一**组**法条」，重排锐化头部 → hit@5 略升但
     多目标覆盖下降。**结论：重排不进入主链路**。

  三个新工具：`scripts/retrieval/tune_fusion.py`（dev 上一次缓存扫 504 组融合权重，89s）、
  `scripts/retrieval/diag_reranker.py`（重排器受控诊断 + 反向打分对照）、
  `smoke_retrieval.py` 新增 `dense_ceiling`（天花板曲线）与 `--eval-split`（dev 调参 / test 报告分离）。
  报告：[`SMOKE_RETRIEVAL.md`](docs/retrieval/SMOKE_RETRIEVAL.md) /
  [`SMOKE_RETRIEVAL_DEV.md`](docs/retrieval/SMOKE_RETRIEVAL_DEV.md) /
  [`FUSION_SWEEP.md`](docs/retrieval/FUSION_SWEEP.md) /
  [`RERANKER_DIAG.json`](docs/retrieval/RERANKER_DIAG.json)。
- ✅ **阶段 5：A0 统一适配器 QLoRA 训练 —— 已完成，`verdict = PASS`**（2026-09-20 03:47）。
  **13,276 步 / 2 epochs 全量跑完**（13h45m49s，3.96 s/it），`train_loss = 0.6756`、
  dev `eval_loss = 0.3896`（末轮），峰值显存 **46.44 GB**（A800 80GB）；**11 项检查全 True**。
  适配器落盘 `models/adapters/A0_unified_qwen3_8b`（43,646,976 个可训练参数），
  报告 [`docs/train/A0_unified_report.md`](docs/train/A0_unified_report.md)。
  **★ 冒烟轮（500 条）的 FAIL 是「假阴性」，已定性**：`checks.rendered_gt_0 = false`
  并非模板渲染失败 —— `dataloader_num_workers > 0` 时 `__getitem__` 在**子进程**执行，
  数据集内 `self.stats` 的累加**不回传父进程** → 父进程读到 `rendered = 0`，
  把「渲染全成功」误判成失败（同期 loss 从 1.142 正常降到 0.513）。
  修法：加 `preview(n=2000)` 在**父进程**单独跑一遍拿真实统计，
  `rendered_gt_0` 改读 `dataset_preview`；`dataset_stats` 保留但标注「num_workers>0 时可能为 0，以 preview 为准」。
  教训：**凡是靠 DataLoader 子进程累加的统计量，都不能当验收依据。**

  配置：Qwen3-8B 4bit（nf4 + double quant）/ LoRA r=16 α=32（7 个投影矩阵，
  可训练 **43,646,976 / 4,761,498,624 = 0.9167%**）/ seq 2048 / lr 1e-4 / 2 epochs / cosine / paged_adamw_8bit。
  **assistant-only loss 用「前缀边界」实现**（TRL 的 `assistant_only_loss=True` 要求模板带
  `{% generation %}` 标记，而 Qwen3-8B 模板**没有**）→ 改成「渲染全文后定位
  `<|im_start|>assistant\n` 前缀，前缀之前一律 mask 成 -100」，并强制 `enable_thinking=False`
  （Qwen3 模板默认插空 think 块）。实测 assistant token 占比 **0.310**。

  **★ 两个必须记住的训练工程坑（都实测踩过）**：
  1. **长度分桶把 80GB 打爆**：`transformers 5.17` 删掉了 `group_by_length`（字段表只剩
     `length_column_name`），Trainer 退回 RandomSampler → 一个 2000 token 的样本会把同批 15 个
     短样本一起 padding。**自建分桶采样器后**同批长度一致、padding 消失，但又带来新问题：
     同批 token 数可达 `16 × 2048 = 32,768`，光 LM head 的 logits 就是
     `32768 × 151936 × 2B ≈ 9.9 GB`（再加梯度/上采样约 20 GB）→
     **实测 `torch.OutOfMemoryError: Tried to allocate 18.55 GiB`（step 44）**。
     **修法**：改成 **token 预算批采样器**（`LengthBucketedBatchSampler`）——
     桶内累积到 `max_batch_tokens = 16384` 或 `batch_size = 16` 任一先到就封批
     （短样本照样 16 条一批，长样本自动降到 8 条）。**必须是 `batch_sampler` 而不是 `sampler`**
     —— DataLoader 在 `batch_size` 已给定时会无视 sampler 的分组、自行每 N 个切一批。
     实测：批大小 1–16、均值 15.06、**每批 token 峰值 16,384 精确等于上限**，
     显存峰值 44.05 GB，500 条冒烟 loss 0.94 → 0.62。
  2. **`--max-seq-length` 别按 4096 配**：实测 token 长度 p50 368 / p90 907 / p99 2048（上限），
     配 4096 只是白占显存。**改 2048，零截断**。

<!-- AUTO_STAGE_BEGIN 由 scripts/eval/stage_status.py 自动生成，勿手改 -->

### 阶段进度（自动汇总）

> **本段由自动化回填**（`scripts/eval/stage_status.py` 扫描产物/日志/标志后生成，经 `patch_readme.py --tag AUTO_STAGE` 贴入）。每完成一个阶段刷新一次，并自动提交推送。机读版：`docs/eval/STAGE_STATUS.json`。
>
> 生成时间 **2026-09-21 14:40:56** ｜ 已完成 **9/21** 项

| 阶段 | 项 | 状态 | 进度 | 备注 |
|---|---|---|---|---|
| A0 评测链 | LexRubric 649 | ✅ 已完成 | 649/649 | cap=1536；AI 裁判判分 |
| A0 评测链 | LexEval 客观 11,400 + 生成 2,750 | ✅ 已完成 | 14150/14150 | 客观 cap=256 / 生成 cap=1536 |
| A0 评测链 | 整链标志 | ✅ 已完成 | — | MARKER_FULL_INFER_A0_DONE |
| 阶段 6 门控 | L2 门控训练 | ✅ 已完成 | 1125/1125 | 72 gates / 2.36M 参数；→ gate_weights.pt 已出 |
| MoE 队列 | Q1 MoE 冒烟 | 🔄 进行中 | 0/2 | — |
| MoE 队列 | Q2 MoE 内部集 1k | ✅ 已完成 | 1000/1000 | — |
| MoE 队列 | Q3 MoE LexRubric 649 | ✅ 已完成 | 649/649 | — |
| MoE 队列 | Q4 MoE 客观 11,400 | 🔄 进行中 | 4928/11400 | — |
| MoE 队列 | Q5 MoE 生成 2,750 | 🔄 进行中 | 4928/14150 | — |
| MoE 队列 | Q6 A0 内部集 cap=1024 | ✅ 已完成 | 1000/1000 | — |
| MoE 队列 | 整队列标志 | ✅ 已完成 | — | MARKER_GATE_QUEUE_DONE |
| base 评测 | LexRubric 649 | 🔄 进行中 | 384/649 | cap=1536 |
| base 评测 | LexEval 客观+生成 14,150 | ⬜ 待跑 | 0/14150 | — |
| 三专家内部集 | internal_criminal 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_civil 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 三专家内部集 | internal_procedure 1k | ⬜ 待跑 | 0/1000 | cap=1024；域专业化分析 |
| 判分(API) | MiniMax-M3 × A0_unified_qwen3_8b | 🔄 进行中 | — | 649 题 / 22 维度；冒烟 8 条中 |
| 判分(API) | MiniMax-M3 × moe_L2 | 🔄 进行中 | — | 649 题 / 22 维度；冒烟 8 条中 |
| 判分(API) | MiniMax-M3 × base | ⬜ 待跑 | — | 649 题 / 22 维度 |
| 收尾 | 汇总 + 出图 | ✅ 已完成 | — | collect_results.py + make_figures.py；最近一次 09-21 13:36 |
| 收尾 | 过夜链整链 | 🔄 进行中 | — | MARKER_OVERNIGHT_DONE |

<sub>状态来源：答案文件行数 / 日志 `rc=` 与 `[n/N]` 进度 / 完成标志 / 报告文件。未到位一律如实标注，不做推测。</sub>

<!-- AUTO_STAGE_END -->

## 剩余路线与时间表（全自动执行 · 2026-09-21 10:10 定稿）

> 本节为静态路线图；上方两张自动表（阶段进度 / 终评结果）每小时刷新并推送。

### 剩余步骤（无需任何人工介入）

| 顺序 | 步骤 | 在哪跑 | 预计完成 |
|---|---|---|---|
| 1 | A0 LexEval 生成题 2,750（1,632/2,750） | GPU0 | 09-21 ~14:45 |
| 2 | MoE Q2 内部集 → Q3 LexRubric → Q4 客观 11,400 | GPU1 | 09-21 ~21:50（Q3 完 ~16:10 后 AI 裁判自动判分） |
| 3 | base 三阶段（rubric 649 / 客观 11,400 / 生成 2,750） | GPU0（v2 通道：A0 完成且显存 <10GB 才起跑） | 09-22 ~10:30 |
| 4 | ★ **Q5 助推**：GPU0 空出后用断点续跑分担 MoE 生成题剩余部分，合并去重（unique≥14148 才原子替换，否则回退自然完成零损失） | GPU0+GPU1 | 09-22 ~14:20（比不分担提前约 2h） |
| 5 | 最终打分 → 汇总 → 出图 → `MARKER_OVERNIGHT_DONE` | GPU0 | 09-22 ~14:30 |
| 6 | 自动收割：打包→下载→回填 README→commit→push | 本机 | 09-22 ~15:00-15:30 |
| 7 | 三专家内部集（criminal/civil/procedure 各 1,000，**只喂域专业化分析图，不挡主表**，后置执行） | GPU0 | 09-22 ~17:30（分数由每小时同步自动补进下方表格） |

### 实验取舍说明（2026-09-21 拍板）

- **主表 9 格（A0 / MoE / base × LexRubric / 客观 / 生成）一格不砍**——砍任何一格，三系统对比即不成立。
- **唯一后置项**：三专家内部集从主链挪到主标志之后（分析图素材，非主表）。
- 已评估并放弃的提速项：两卡对半分片（总工作量守恒仅省 ~1.5h）、加大 batch（峰值显存 59.5GB→~72GB，过于贴近 80GB 上限）。

### 全部跑完后你可以做什么

`MARKER_OVERNIGHT_DONE` + `MARKER_GATE_QUEUE_DONE` 齐 → 本机自动跑收割脚本回填 README 并推送 GitHub。**主表数字在 09-22 下午 3 点前全部落袋**，即可开始写论文实验章；方法 / 相关工作 / 实验设置章节可提前撰写。

### 论文与实验并行（2026-09-21 拍板：现在就可以开写）

**结论：可行，立即开写。** 实验链全自动跑，论文不用等数字齐再动笔——

- **照常写的部分**：摘要骨架、引言、相关工作、方法章（MoE 两级架构 / 门控设计 / QLoRA 配置 / 检索与智能体循环）、实验设置（数据、判分器、硬件、超参——全部已知且已在 README 落定）、实现细节、结论与展望。
- **留空待填的部分（只留空白，其余照常）**：
  1. **主表 9 格**（A0 / MoE / base × LexRubric / 客观 Acc / 生成 ROUGE-L）——占位写 `【填：主表】`；
  2. **MoE 消融数字**（门控路由准确率已有独立集数字 0.8370/0.7922，终评侧留空）；
  3. **损失曲线 / 域配比 / 检索消融等图**——图已在 `docs/figures/` 陆续产出，可直接先用；
  4. **三专家内部集域专业化分析段**（09-22 晚补齐）。
- **填数时间点**：09-22 ~15:00-15:30 收割脚本自动把最终数字回填到本 README 的「终评结果」表并推送 GitHub——照表抄进论文占位处即可。
- **防返工红线**：论文不得出现 CLaw 基准比较；基准写 LexRubric（649 题 / 12,335 rubric，主）+ LexEval（辅）；模型配置以实测为准（Qwen3-8B / r=16 α=32 七模块 / seq 2048 / A800），不写旧版（Qwen2.5-7B / r=8 / seq512 / RTX5070Ti）。
- 判分口径、cap 限制、ROUGE/归一化公式均已在上文「终评结果」表脚注固化，实验设置章可直接引用。

<!-- AUTO_RESULTS_BEGIN 由 scripts/eval/collect_results.py 自动生成，勿手改 -->
### 终评结果（自动汇总）

| 系统 | LexRubric (归一化 %) | LexEval 客观 Acc | LexEval 生成 ROUGE-L | 内部集 ROUGE-L | 内部集法条命中 | 已生成答案 (rubric/eval/internal) |
|---|---:|---:|---:|---:|---:|---|
| base (Qwen3-8B, 无微调) | — | — | — | — | — | 384 / 0 / 0 |
| A0 (统一适配器) | 12.85 | — | — | 0.5378 | 0.5386 | 0 / 0 / 1000 |
| MoE-L2 (L2 门控混合) | — | — | — | — | — | 649 / 4928 / 1000 |

> 生成口径（全系统统一）：LexEval 客观题 cap=256 / LexEval 生成题 cap=1536 / LexRubric cap=1536 / 内部验证集 cap=1024；
> 判分 MiniMax-M3（`configs/judge.yaml`）；完整机读数据见 `docs/eval/RESULTS_MATRIX.json`，图见 `docs/figures/`。
<!-- AUTO_RESULTS_END -->

### 待办

- ✅ **阶段 5 收口 —— 已完成**：A0 统一适配器 13,276 步 / 2 epochs 跑完（13h45m49s），
  `verdict = PASS`、11 项检查全 True、峰值显存 46.44 GB、`train_loss 0.6756` /
  dev `eval_loss 0.3896` → [`docs/train/A0_unified_report.md`](docs/train/A0_unified_report.md)。
- ✅ **阶段 4b-4 / 4b-5 已跑完并回填数字**：Neo4j 导入报告
  [`docs/retrieval/NEO4J_IMPORT.md`](docs/retrieval/NEO4J_IMPORT.md)、检索消融矩阵
  [`docs/retrieval/SMOKE_RETRIEVAL.md`](docs/retrieval/SMOKE_RETRIEVAL.md)（test）/
  [`SMOKE_RETRIEVAL_DEV.md`](docs/retrieval/SMOKE_RETRIEVAL_DEV.md)（dev）。
  主链路 = 加权 RRF（dense 1.0 + BM25 0.7 + KG 0.25，k=10），**不带重排器**。
- ✅ **决策项 D｜`case_embedding_procedural` 只有 37 条 —— 已定性并修复**（2026-09-19 闭环）：
  **真正的根因不是 `domain4_of()`**（它没问题，程序法样本的 `domain` 字段就是 `procedural`），
  而是**阶段 4 的分层切分把 3,333 条程序法 `case_analysis` 吃掉了 3,268 条（98.1%）**，
  叠加「索引侧排除四份 split」的旧口径 → 池子只剩 65 条。
  **处置（用户拍板）**：`load_split_uids(root, scope="eval")` → 只排除 dev/test；
  新增 `--clean-cases` 强制清空旧案件库（案件池条数一变→分片错位，旧 `.done` 会**静默跳过**）。
  顺带修掉 `build_embeddings.py` 里写死的合理带 `30000 ≤ cases ≤ 120000`
  （新池 145,339 必然误判 FAIL）→ 改为从 `case_analysis` 总行数现算。
- ✅ **三域专家 QLoRA 全部 PASS（2026-09-20）**：criminal 30k / 3,874 步 / 4h26m / loss 0.5434；
  civil 40k / 5,628 步 / 6h25m / loss 0.7009；procedure 20k / 2,606 步 / 2h02m（7,333.9s）/ loss 0.7730。
  三者 `verdict=PASS`、11 项检查全 True、峰值显存 55.15 GB（A800）。报告 `docs/train/A0_{domain}_report.*`。
  ⚠️ 实际步数 > rows/16（token 预算裁剪长样本批次所致），论文训练配置表用实际值。
- ⬜ **阶段 6–9**：L2 门控自检+训练（4 专家已齐）→ MoE 等价验证 → 消融 → checkpoint 选择 → 终评
  （LexEval / LexRubric 各只跑一次；A0 评测链 2026-09-20 已在 GPU0 启动）。
- ⬜ （可选）配置 `HF_TOKEN` 后补采 `Aiiluo/Chinese-Law-SFT-Dataset`（2.6 MB，gated）
  与 `Brench/chinese_law_data_rag_ft`、`wormtooth/MNBVC-judgment`（按需抽样），扩充法条/案例侧。
- ✅ **阶段 3 复扫门禁 —— 已收口**：`prepare_decontaminate_pass.sh` 跑完（与数据管线并行），
  `verdict = PASS`，被查 **294,499** 行 / 精确 **0** / 近似 **0**；
  证据文件 `DECONTAM_REMOVED_UIDS_PASS.txt` 实测 **0 字节**（空文件）。
- ✅ **案件库按新口径重建（决策项 D）—— 已闭环**（2026-09-19 16:28）：
  `build_embeddings.py --case-scope eval --clean-cases`（1662s）→ `build_hotness.py`（3.1s）
  → `import_neo4j.py --step all`（539.3s，`verdict = PASS`，13 项断言全 True）。
  案件池 **82,821 → 145,339**、程序法子库 **37 → 3,268**；
  Neo4j 侧 `Case` **145,339** / `SAME_CASE` **10,742** / `hotness_case` 145,339 全部同步到位
  （`docs/retrieval/NEO4J_IMPORT.md`）。


---

## 9. 结果完整性要求

每次运行必须保存**检索结果 + 最终答案 + 运行元数据**（模型哈希、适配器哈希、提示词哈希、
token 数、延迟），否则无法区分错误来源（未检索到 / 检索到未使用 / 版本引用错 /
模型推理错 / Judge 异常）。

---

## 10. 操作手册：从头到尾每一步怎么跑

> 本节写给「隔几周回来还要复现」的场景（也方便写论文的「实验设置」章节）。
> 全部命令在实验服务器 `/mnt/data/lidian/law-agent` 下执行；先 `source scripts/activate.sh`。
> 本机（Windows）只做开发与文档，**跑数据一律在服务器**，用 `_remote/ssh_run.py` 上传+远程执行。

### 10.1 本机 → 服务器的两条链路

```powershell
# 远程执行（本机 PowerShell；注意本机 PowerShell 的 stdout 不回显，务必用 --out 落到文件再读）
python C:\...\_remote\ssh_run.py --cmd "ls /mnt/data/lidian/law-agent"

# 上传 + 后台跑长任务 + 轮询（长任务不要一次等到超时，SSH 通道约 2 分钟会断）
python C:\...\_remote\ssh_run.py `
  --upload "D:\vs project\law\scripts\corpus\xxx.py:/mnt/data/lidian/law-agent/scripts/corpus/xxx.py" `
  --script "D:\vs project\law\scripts\corpus\prepare_xxx.sh" `
  --remote "/mnt/data/lidian/law-agent/scripts/corpus/prepare_xxx.sh" `
  --log "/tmp/xxx.log" --out "C:\...\_remote\_xxx_start.txt"
# 之后单独轮询：  --cmd "tail -n 30 /tmp/xxx.log"
```

### 10.2 阶段 0 — 源合规审查

**手动**，产物是 [`configs/corpus_sources.yaml`](configs/corpus_sources.yaml)。
逐源登记 `license` / `license_grade`（A 可进论文，B 仅内部）/ 体积 / 文件数。
**卡口**：没有 `license_grade` 的源不许进采集。

### 10.3 阶段 1 — 分批采集

```bash
bash scripts/corpus/prepare_corpus.sh --list   # 干跑：看要采什么、多大
bash scripts/corpus/prepare_corpus.sh          # 实采（默认 --max-gb 3 体积护栏）
```

- 脚本**直读 `configs/corpus_sources.yaml`，零硬编码**；按 `license_grade` 选源。
- 产物：`data/corpus/raw/<dataset>/`（`/` 换成 `__`）+ `data/corpus/MANIFEST.json`（逐文件 SHA-256）。
- **卡口**：清单与磁盘**逐文件比对**一致（大小 + SHA-256，LFS 大文件对齐 HF `lfs.oid`）。

### 10.4 阶段 2 — 归一化 + 域打标

```bash
bash scripts/corpus/prepare_normalize.sh        # 幂等：先语法+YAML 自检，再跑归一化
/mnt/data/lidian/law-agent/envs/main/bin/python scripts/corpus/inspect_normalized.py   # 只读复核
```

- 读两份 YAML：`corpus_sources.yaml`（合规层）+ [`corpus_adapters.yaml`](configs/corpus_adapters.yaml)（解析层：
  `drop_files` / `keep_files` / 适配器 / 过滤阈值 / 打标权重）。
- 产物：`data/corpus/normalized/{qa,statutes}/*.jsonl`、`qa/_all.jsonl`（下游只读这个）、
  `normalized_STATS.json`、`normalized_DEDUP_REPORT.json`。
- **判重必须三指纹交叉**（严格 / 宽松 / 案件正文）—— 详见 [`docs/corpus/DEDUP_REPORT.md`](docs/corpus/DEDUP_REPORT.md)。
  本次即靠三指纹发现「同名同条数却是 100% 重复」。
- **卡口**：每域实得量 ≥ 目标量；`domain_source = fallback` 占比 < 15%。

### 10.5 阶段 2b — 法条切条（整部法规 → 逐条法条）

> 见 [3.1.2](#312-法条流statutes实测结构--一处必须纠正的表述--一处必须补的工序)。当前 `statutes/` 是**整部法规全文**，
> 必须先按「第X条」切条，并把法条域**按法名判定**（不能靠全文关键词）。

```bash
bash scripts/corpus/prepare_split_statutes.sh   # ✅ 已实现（阶段 2b）
```

> 产物落点更正：不是 `data/corpus/normalized/articles/`，而是 **`data/corpus/statute_items/`**
> （`twang2218__chinese-law-and-regulations.items.jsonl` + `pandalla__chinese_law_examples.items.jsonl`，
> 合计 **72,449** 条；分文件条数随语料版本变化，以 `docs/corpus/STATUTE_SPLIT_REPORT.json` 为准）。
> 质检：`python scripts/corpus/verify_statute_items.py --root $PWD`

**切条规则（写死，不许绕过）**：

1. **按「第X条」正则切分**：一部法规 → N 条记录，每条 = 一个条文；序号无法解析的碎片并入上一条。
2. **类别过滤**：只保留「法律 / 司法解释 / 行政法规」三类（≈1,910 部 → 数万条法条）；
   **地方性法规 19,733 部留档不进配比、不进主向量库**（各省规定互相冲突，会污染检索 top-k；
   将来若需要，单独建子 collection 带省份元数据）。
3. **法条域按法名判定**：`民法典/合同编... → civil`、`刑法/刑法修正案 → criminal`、
   `民事诉讼法/刑事诉讼法/行政诉讼法 + 三大诉讼法司法解释 → procedural`，其余国家法按内容归 civil/criminal，
   归不了的标 `general`；**禁止用全文关键词判断**（整部法规里什么词都有，必然错标）。
4. **历史版本硬约定**：每条保留 `title` / `article_no` / `status` / `effective_from` / `effective_period`；
   同名法规多版本（已修改/已废止）**全部保留**，不得只存最新版。
5. **顺带清洗**：`status` 脏值 `"7"`（829 条）归为 `unknown` 并标注，不静默丢弃。

- 产物：`data/corpus/normalized/articles/*.jsonl`，每条 = 一部法的一个条文
  （同时就是**向量库主料**：入库粒度从「整部法规」细化到「条文」）。
- **卡口**：民诉 ≈284 / 刑诉 ≈308 / 行诉 ≈103 条不得缺少；抽查 50 条域标签准确率 ≥ 95%。

### 10.6 阶段 3 — 双向去污（★ 硬门禁，两遍式）

> **✅ 本节数字已于 2026-09-19 全链路重跑后更新**。2026-09-18 那轮复扫出的 PASS 是**空转证明**：
> 它扫的是已经删干净的镜像，只能证明「剔除已生效」，**证明不了「剔除删得对」** ——
> 而它恰好掩盖了一次误删：法条流 uid 粒度是**整部法规**、simhash 只看正文前 800 字，
> 导致**一处近似命中就整部法典连坐**，实测**误删 739 部法规 / 646 个法名 / 419 万字**
> （民法典 113,346 字、刑法 75,291+60,936 字、刑诉法 40,914+37,477 字都在内）。
> 修复方式：法条流豁免近似去污（`--statutes-near audit`，见下方「法条流豁免」条目）。
>
> **重跑后实测（唯一权威口径）**：
> 首扫 **308,767** 行 → 精确命中 **0** / 近似命中 **17,837 处** → 剔除 **14,268** 个唯一 uid；
> 清洗镜像 **294,499** 行；**恒等式 `308,767 − 294,499 = 14,268`** 成立；
> ★ 门禁断言 `剔除清单 ∩ 法条流 uid = 0`（历史口径下这里是 739）。
> 下游全部按 294,499 重跑：2b → **72,449** 条、2c → **15,535** 条、阶段 4 → 122,000 条四份 split。

**实测证明必须跑两遍**：第一遍在全量语料上找出所有命中并生成剔除清单；应用剔除后**第二遍必须复扫出 PASS**，
这个复扫 PASS 才是论文里能引用的门禁证据。

```bash
# 第一遍：全量扫描，产出 DECONTAMINATION_REPORT.json + DECONTAM_REMOVED_UIDS.txt
# 注意带上 --statutes-near audit（默认值，包装脚本里已显式写出）：
#   法条流只做精确 sha1 剔除，近似命中只记录不剔除 —— 见下面「法条流豁免」条目
bash scripts/corpus/prepare_decontaminate.sh

# 第二遍 a：应用剔除清单，生成清洗镜像（normalized 保持只读，硬约定）
python scripts/corpus/apply_decontam.py \
  --uids docs/corpus/DECONTAM_REMOVED_UIDS.txt

# 第二遍 b：对清洗镜像复扫，verdict 必须 = PASS（★ 门禁，包装脚本自带三条数值断言）
bash scripts/corpus/prepare_decontaminate_pass.sh
```

> **第二遍 b 必须与第一遍带同一个 `--statutes-near`**（默认 `audit`）。复扫镜像里
> **仍然存在**那 739 部法规的近似命中（按设计保留）；若复扫改用 `remove`，它们会被重新算成命中
> → 复扫必然 FAIL（假故障）。

- 脚本：`scripts/corpus/decontaminate.py`（扫描 + 出报告）、`scripts/corpus/apply_decontam.py`（按 uid 剔除并逐文件登记 SHA-256）。**双向**比对：
  - 正向 = 训练集里有没有混入评测题（含改写）
  - 反向 = 评测集里有没有混入训练语料**来源**（同源风险，如 Skepsun 司考 vs LexRubric `sifakaoshi`）
- 黑名单：LexRubric 649（473 咨询 + 176 司考）+ LexEval 14,150（**23 个任务文件全部自动扫描**）。
  **CLaw 已退役**（官方数据从未公开发布，无法获取也无法验证）→ 不在黑名单，也不是漏做去污；
  理由与证据见第 8 节「⛔ CLaw 已退役」段与 `docs/data_manifest.json:retired_datasets`。
- 两种指纹缺一不可：**精确** `sha1(normalize(text))`；**近似** `simhash64(char-3gram, crc32)`，
  汉明距离 ≤ 3 判近重。近似比对用 **4×16bit 分段索引**先取候选，把 O(N×M) 降成线性。
- 产物：`docs/corpus/DECONTAMINATION_REPORT.json`（含 `verdict`）+ 同名 `.md`（人读）
  + `DECONTAM_REMOVED_UIDS.txt`（剔除清单，一行一个 uid）。
- **卡口**：复扫 `verdict = PASS`（且 `DECONTAM_REMOVED_UIDS_PASS.txt` 为空）。
  首扫命中不是失败，是流程的一部分 —— 剔除后复扫 PASS 才算过关。
  复扫包装脚本 `scripts/corpus/prepare_decontaminate_pass.sh` 自带**三条数值断言**
  （verdict=PASS / 二次清单为空 / 恒等式 `首扫 checked − 复扫 checked == 首扫剔除 uid 数`），
  只打结论不看数值不算过关。
- ⚠️ **剔除清单里的数字只认 `removal_summary.unique_uids_to_remove`**：
  历史字段 `near_hits.unique_train_uids` 名字有歧义，实际是**精确+近似合计**的唯一 uid 数
  （`hit_uids` 由两条路径共同写入），数值等同但名字误导；新报告两者都给。
- ⚠️ **上游语料一变，`indexes/retrieval/embeddings/` 必须先删**：4b-3 的断点续跑
  **按分片序号**判完成（看 `.done`），**不看内容**。法条数从 65,037 变 72,449 后，
  旧分片里的条目全部错位，但 `.done` 还在 → 会被**静默跳过**，向量库与新图谱对不上。
- ★ **法条流豁免近似去污**（`--statutes-near audit`，2026-09-19 修复事故后新增）：
  法条流的 uid 粒度是**整部法规**，而 simhash 只看正文前 800 字 → **开头几条近似命中就整部法典连坐**；
  且法条流是**检索语料**（3.1.3：法条流不进 SFT），评测集必然引用法条原文 → 误报是结构性必然。
  故法条流**只做精确 sha1 剔除**，近似命中在报告第 4b 节 `statutes_near_audit` 里**只记录不剔除**。
  历史行为用 `--statutes-near remove` 可复现。
- ⚠️ **「复扫 PASS」的适用边界**：复扫必须扫**应用剔除后的镜像**才有意义。
  若拿它当「删除是否正确」的证据 —— **它做不到**，它只能证明「剔除已生效且无残留」。
  2026-09-19 那次误删 739 部法规，就是被一份这样的空转 PASS 掩盖过去的
  （**事故期镜像** 293,760 行 / 0 命中，因为它已经删干净了）。
  ⚠️ **口径提醒**：293,760 是**事故期**那份镜像；重跑后的最终镜像为 **294,499** 行
  （`docs/corpus/DECONTAMINATION_REPORT_PASS.json` → `checked.total_rows`，2026-09-19T15:00:10，
  与 `APPLY_SUMMARY.json` 的 `total_kept = 294499` 一致）。**两份不要混用。**
- ✅ CLaw 已退出论文（2026-09-18 用户决策），黑名单 = LexEval + LexRubric，复扫可直接出**纯 PASS**；
  论文中不再出现 CLaw，评测基准以内部验证集 + LexEval/LexRubric 闭卷线为准。
- 经验教训（2026-09-18 实测）：首版报告的 `near_hits.by_bench_tag` 是从**被截断的样例列表**
  算出来的，导致「总数 18,587 vs 按基准 300」的自相矛盾 —— 统计口径必须**永远对全量命中集合**算，
  样例列表只用于人读展示。

### 10.7 阶段 4 — 切分 + 分层下采样

```bash
bash scripts/corpus/prepare_downsample_split.sh              # 切分 + 质检门禁（C1–C14）
bash scripts/corpus/prepare_downsample_split.sh --dry-run    # 只算配额，不落盘
```

- 分层下采样 **286,524** → 100,000（真实 QA 270,989 + 阶段 2c 派生 15,535），
  **按 `task` × `source` 分层**（防单一任务型垄断某专家）；派生样本只进 train，受 15% 组份额上限约束。
- 切 train / val / test；**路由集按 uid 哈希抽 20%，与专家训练集 disjoint**（脚本断言交集 = 0）。
- **卡口**：`verify_split.py` verdict = PASS（四份 split 两两不相交，C1–C14 全过）。
  池剩余 **164,524** 可回溯。

### 10.8 阶段 4b — 检索层：向量库 + 知识图谱

> 设计见 [`docs/retrieval_design.md`](docs/retrieval_design.md)；口径门禁见
> [`docs/corpus/RETRIEVAL_SCOPE.md`](docs/corpus/RETRIEVAL_SCOPE.md)。

```bash
# 4b-0 / 4b-1：检索池范围门禁（与四份 split 强隔离）+ 法名两级规范化
python scripts/retrieval/probe_pool_scope.py --root <ROOT>
python scripts/retrieval/normalize_law_title.py --root <ROOT>

# 4b-2：知识图谱边抽取（只读语料 → indexes/retrieval/*.jsonl）
#   ⚠️ 参数名是 --out-dir / --report-json / --report-md，**不是** --out-json / --out-md
python scripts/retrieval/extract_edges.py --root <ROOT> \
    --out-dir indexes/retrieval \
    --report-json docs/retrieval/EDGE_EXTRACT.json --report-md docs/retrieval/EDGE_EXTRACT.md

# 4b-2b：热度权重（读者拍板新增；只读检索库自身，零泄漏）
python scripts/retrieval/build_hotness.py --root <ROOT>
#   → indexes/retrieval/hotness.json + docs/retrieval/HOTNESS_REPORT.{json,md}

# 4b-3：向量化（Qwen3-Embedding-0.6B / 1024d / L2 归一化；分片 5000 + .done 断点续跑）
#   库布局 = **case 4 库（general/civil/criminal/procedural）+ 法条 1 库**
#   ⚠️ 上游语料一换，**必须先 rm -rf indexes/retrieval/embeddings**：
#      断点续跑只认 `shard_%05d.done` 文件名，不看内容 —— 条数变了会静默跳过导致库与图谱错位
python scripts/retrieval/build_embeddings.py --root <ROOT> \
    --out-dir indexes/retrieval/embeddings \
    --report-json docs/retrieval/EMBEDDING_REPORT.json \
    --report-md   docs/retrieval/EMBEDDING_REPORT.md
# 只想重跑某一侧：--only provision / --only case

# 4b-4：导入 Neo4j（约束 → 节点 → 边 → 热度 → 向量索引 → 计数验收；幂等 MERGE，可反复重跑）
python scripts/retrieval/import_neo4j.py --step all \
    --uri bolt://127.0.0.1:7687 --user neo4j --password <PWD> \
    --report-md docs/retrieval/NEO4J_IMPORT.md
#    干净重灌 = --step reset --drop-legacy → --step all
#    `--drop-legacy` 顺带删掉旧实验遗留的向量索引（provision_embedding_4b/_bert768/_bge_m3）
#      与旧约束（LawVersion/Cause/Court/LegalConcept），否则 schema 里会混着两套口径
#    验收期望值**从输入产物现算**（不写死），所以语料一变不会误报 FAIL

# 4b-5：检索链路 + **消融矩阵**（一次跑完多组配置，给出论文检索侧数字）
#   ★ 调参与报告必须分离：dev 选型、test 出数字（--eval-split）
python scripts/retrieval/smoke_retrieval.py --root <ROOT> \
    --configs vector,dense_hot,dense_bm25,dense_graph,hybrid,full \
    --eval-split test \
    --report-md docs/retrieval/SMOKE_RETRIEVAL.md
#    重排模型未就绪时可 --no-rerank（会明确标注「本次自动降级」）
#    输出含 §1c「纯 dense 全量索引天花板曲线」—— 用来判断目标未达标是排序问题还是召回天花板

# 4b-5b：**融合权重扫描**（只在 dev 上扫；一次缓存 dense/BM25/图谱候选，504 组组合约 90s）
python scripts/retrieval/tune_fusion.py --root <ROOT> \
    --split dev --top-k 500 --report-md docs/retrieval/FUSION_SWEEP.md

# 4b-5c：**重排器受控诊断**（判别力 AUC + 反向打分对照 + 截断敏感性；用于证伪「实现有 bug」）
python scripts/retrieval/diag_reranker.py --root <ROOT> \
    --split dev --n-queries 100 --report-json docs/retrieval/RERANKER_DIAG.json
```

- **4b-2 三个边**：`HAS_PROVISION`（法→条）/ `NEXT`（同法上下条，图谱扩展主力）/
  `CITES`（交叉引用）。`NEXT` 的分组键是**来源文档**而非 `law_id`（同一 L2 法名多版本会错连）。
- **4b-3 库布局（用户 2026-09-19 拍板）**：**case 4 库 + 法条 1 库** ——
  `provision_embedding`（法条 item + doc 同库，入库时按 `level` 分流到两个属性/两个向量索引）+
  `case_embedding_{general,civil,criminal,procedural}`（一域一库一索引）。
  Neo4j 侧对应 `Provision.embedding` / `.embedding_doc` 与
  `Case.embedding_general` / `_civil` / `_criminal` / `_procedural`。
  **泄漏排除口径（2026-09-19 修订）**：只排除**评测集 dev/test**（`--case-scope eval`），
  不再排除 train/router —— 检索库是知识库，只有「评测 query 自身在库里」才算泄漏。
  重建前须加 `--clean-cases`（案件池条数一变，旧 `.done` 会被静默跳过 → 库与图谱错位）。
- **4b-2b 热度**：`final = rrf_score × (1 + W_HOT × hotness)`，`W_HOT = 0.05`；零泄漏（只在检索库内统计）。
  ⚠️ **实测：在 top-50 上 reweight 对 recall@5/@10 完全无影响**（`dense_hot` 与 `vector` 数字逐位相同）
  —— 它只在**已召回集合内部**微调次序，而 gold 的瓶颈是「有没有被召回」。论文按此如实说明。
- **4b-5 完整混合链路（已定稿）**：dense（0.6B，instruction-aware）→ BM25（jieba 分词 +
  **scipy 稀疏精确实现**，比 `rank_bm25` 逐条打分快两个数量级、结果逐位等价）→
  图谱 1 跳扩展（NEXT 1.0 / CITES 0.5，**列表限长 60 且只给 0.25 权重**）→
  **加权 RRF（k=10，dense 1.0 / BM25 0.7 / KG 0.25）** → 热度 reweight。
  ⛔ **重排器不进主链路**（受控实验证明其净收益为负，见 §4b-5）。
- **4b-5 gold 双口径**（必须分开报）：`system` = 数据集自带「相关法律条文」清单（最权威，
  但只有 case_analysis 样本有）；`output` = 从**参考答案**解析（覆盖大，是**下界**）；
  `union` = 两者并集（主口径）。法名是**简称**（《刑法》），必须先按「exact → 唯一后缀」解析成
  `law_id`（后缀最小长度 **2 字**，卡 3 字会让刑法域全挂）。
- **断言只认不变量，不认绝对数字**（2026-09-19 连续踩了 4 处）：上游语料一重跑，
  写死的期望值就会把**正确结果判成 FAIL**。已全部改造为「从输入产物现算」或「账本恒等式」——
  `extract_edges`（节点数 == 输入行数）、`build_embeddings`（条文项数 == `provision_nodes` 的 item 节点数；
  案件池改用**划分账本** `kept + Σskip_* == qa_rows` + **相对**合理带
  `0.3·n_case_analysis ≤ len(cases) ≤ n_case_analysis`，替代写死的 82,821）、
  `import_neo4j`（Law/Provision/边/向量行数全部从输入 JSONL/npy 元信息现算）。
- **卡口**：4b-0 池子与评测集交集 = 0；4b-4 计数与输入自洽；4b-5 三条主指标
  （`hit@5`/`recall@10`/`MRR@10`，口径与作废说明见 §4b-5）全部达标才算 PASS。

### 10.9 阶段 5–9 — GPU 阶段（见 [docs/run_order.md](docs/run_order.md)）

A0 单 LoRA → A1/A2/A3/A4 三专家 + 路由（A5 token 级 MoE-LoRA 可选扩展）→ 路由器 R0→R1 →
MoE 组合与消融 → **内部验证集选 checkpoint**（选完才允许跑终评，★红线）。
**终评 = LexEval / LexRubric 各只跑一次**（CLaw 已退役，不再出现在任何终评环节）。

```bash
# 阶段 5：A0 统一适配器 QLoRA（assistant-only loss 用前缀边界实现；token 预算批采样器防 OOM）
#   ⚠️ 不要加 --no-bucket：分桶是为了消除 padding（实测 9.79 → 2.2 s/step）
#   ⚠️ --max-batch-tokens 是**显存闸门**：不设上限会被 16×2048 token 的批次打爆 80GB
python scripts/train/train_qlora.py \
    --micro-batch 16 --grad-accum 1 --max-batch-tokens 16384 \
    --max-seq-length 2048 --num-epochs 2 --gpu 1 \
    --output-dir models/adapters/A0_unified_qwen3_8b \
    --report-json docs/train/A0_unified_report.json --report-md docs/train/A0_unified_report.md

# 先冒烟再全量（512 条 / 25 步，验证模板、掩码、显存、loss 下降）
python scripts/train/train_qlora.py --smoke 512 --max-steps 25 --no-eval \
    --micro-batch 16 --max-batch-tokens 16384 --max-seq-length 2048 --gpu 1
```

### 10.10 常用复核命令（只读，随时可重跑）

| 目的 | 命令 |
|---|---|
| 语料清点 | `bash scripts/corpus/prepare_corpus.sh --list` |
| 法条/域分布 | `python scripts/corpus/probe_statutes{,2,3}.py` |
| 打标质量抽检 | `python scripts/corpus/inspect_normalized.py` |
| 判重证据 | `docs/corpus/DEDUP_REPORT.md` |
| 去污结论 | `python -c "import json;print(json.load(open('docs/corpus/DECONTAMINATION_REPORT.json'))['verdict'])"` |
| 检索层产物清点 | `ls -l indexes/retrieval/ indexes/retrieval/embeddings/*/ \| head -40` |
| 检索消融矩阵 | `docs/retrieval/SMOKE_RETRIEVAL.md` |
| 热度权重 | `docs/retrieval/HOTNESS_REPORT.md` |
| 图谱计数（Neo4j） | `cypher-shell -u neo4j -p <PWD> "MATCH (n) RETURN labels(n), count(n)"` |
| 训练进度 | `tail -f logs/train/A0_full.20260919.log` |
| 基准完整性 | `python scripts/benchmarks/verify_benchmarks.py` |
| 环境自检 | `bash scripts/verify_env.sh` |

---

## 11. 文档维护约定

> **本 README 必须随每一次实质变更同步更新，这是长期约定。**

凡发生以下任一情况，**同一次提交内**必须更新 README 对应章节：

| 变更类型 | 必须更新 README 的哪里 |
|---|---|
| 新增 / 更换**模型**（底座、向量、重排、判分器） | 第 1 节模型清单表 |
| 新增 / 调整**数据集来源**（含协议、体积、分级） | 3.2 节；同步 `configs/corpus_sources.yaml` |
| 调整**配比**或合成配额 | 3.1 节；同步 `configs/corpus_sources.yaml` 的 `target_mix` |
| **实测可得量 / 配比核算结果变化** | 3.1.1 节 |
| 修改**处理流程 / 阶段卡口** | 3.3 节 |
| 修改**归一化 schema** 或域打标权重 | 3.4 节；同步 `configs/corpus_adapters.yaml` |
| 新增**评测基准** | 第 4 节；同步 `docs/benchmarks/` |
| 环境 / 依赖版本变化 | 第 6 节 |
| 阶段推进（完成 / 开始） | 第 8 节进度 |

**同时保持三份记录同步**：
- 人读的完整方案 → `docs/`（如 [`docs/finetune_data_plan.md`](docs/finetune_data_plan.md)）
- 机器读的登记表 → `configs/`
  （[`corpus_sources.yaml`](configs/corpus_sources.yaml) 管**合规**、
  [`corpus_adapters.yaml`](configs/corpus_adapters.yaml) 管**解析**）
- README 是**索引与摘要**，不是唯一真相源 —— 细节永远以 `docs/` 与 `configs/` 为准。
