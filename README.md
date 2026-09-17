# 法律领域智能体：混合专家模型 + 知识图谱

> 论文《基于混合专家模型和知识图谱的法律领域智能体研究与构建》实验代码仓库
> 最后更新：2026-09-17

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
| **重排** | `BAAI/bge-reranker-v2-m3` | 568M，MIT | 协议最干净，精排 Top-5 / Top-8 |
| **知识层** | Neo4j 5.26 Community（Docker） | — | Law / LawVersion / Provision / Case / Cause / Court / Date |
| **检索融合** | RRF(k=60) | — | 不同检索源各自排名后融合，**不比原始分数** |
| **编排** | LangGraph | 1.2.11 | 事实整理 → 证据检索 → 专家路由 → 适配器调用 → 结果聚合 |
| **判分器（双盲）** | `gemini-2.5-pro` + `deepseek-r1` | 外部 API | 匿名化编号 + 洗自指标签 + 打乱顺序（seed=42），见 `configs/judge.yaml` |
| **评测基准** | CLaw / LexRubric / LexEval | — | 见 [第 4 节](#4-评测基准) |

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

**闭卷线（Closed-book）**：不允许检索外部知识库 → 用于与 CLaw 官方模型结果比较
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

> A1 与 A2 的差值 = 路由器质量损失；A2 与 A3/A4 的差值 = 路由机制本身的增益。
> **没有 A1 和 A4，审稿人一定会问「你怎么知道不是多专家本身带来的增益」。**

检索侧消融：去重排序器 / 去时间版本过滤 / 去知识图谱 / 去领域适配器路由 / 换向量模型（四路）。

推理参数全局统一：`do_sample=false, temperature=0, top_p=1, max_new_tokens=1536`。

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
| **程序法** | **20,000**（**含合成 8,000**） | 20% | 三诉讼法司考题 + 诉讼法原文任务 + 合成补量 |
| **通用回放** | **10,000** | 10% | 通用指令数据抽样，**防灾难性遗忘**，不是凑数 |
| **合计** | **100,000** | 100% | |

**两个必须解释的设计**：

1. **为什么要有 10% 通用回放**：纯法律数据微调会打崩底座通用能力。CLaw 254 案里有相当比例
   需要常识推理与语言组织，只喂法律数据会让这部分指标**不升反降**。
2. **为什么程序法要合成 40%**：程序法是三域里公开语料**最稀薄**的（公开 SFT 集里占比普遍 <10%），
   而目标占 20%，缺口只能自建。合成规范（写死，不许绕过）：
   - 必须以**三大诉讼法原文条文**为唯一事实来源，禁止模型自由发挥
   - 每条登记 `synthetic: true` / `generator_model` / `prompt_sha256` / `source_articles[]`
   - **合成数据一律不进验证集与测试集**
   - 合成占比在论文里如实披露（程序法域 8000/20000 = 40%）

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
| `Dusker/lawyer-llama` | 425.7 MiB | 7 | MIT | ✅ OK | 含 `kg_crime_llama.json`（**犯罪知识图谱**，可转问答）。⚠️ 内含 DISC-Law-SFT 副本，阶段 2 必须去重 |
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
阶段 0  源合规审查        → docs/corpus/candidates.json（A/B 级分级，A 级才可进论文实验集）
阶段 1  分批采集          → data/corpus/raw/<dataset>/ + data/corpus/MANIFEST.json（逐文件 SHA-256）
阶段 2  归一化 + 域打标   → data/corpus/normalized/*.jsonl + DOMAIN_LABEL_REPORT.json
阶段 3  去污              → DECONTAMINATION_REPORT.json（verdict 必须 PASS）★硬门禁
阶段 4  切分              → train / val / test + 路由集（与专家训练集 disjoint）
───────────────────────────────  以下才动 GPU ───────────────────────────────
阶段 5  基线：单 LoRA 全域训练                 → A0
阶段 6  专家 LoRA ×3（刑法 / 民法 / 程序法）    → A1/A2/A3/A4 的组件
阶段 7  路由器训练（R0 → R1）                  → ROUTER_EVAL.json
阶段 8  MoE 组合 + 消融矩阵                    → 四档消融表
阶段 9  用内部验证集选 checkpoint              → 选定后才允许跑 CLaw ★红线
```

**阶段验收卡口**：

| 阶段 | 卡口（不达标不许进下一阶段） |
|---|---|
| 0 | 每个源都有 `license` + `license_grade`；B 级源明确标注「仅内部探索」 |
| 1 | 每批落盘后立即算 SHA-256；`data/corpus/MANIFEST.json` 与磁盘实测一致 |
| 2 | 三域占比落在目标 **±15%**；`domain_source` 分布可解释；人工抽检 50 条准确率 **≥ 95%** |
| 3 | **去污 verdict = PASS**，且报告里能看到近重复命中明细 |
| 4 | 路由集与专家训练集 uid 交集 = **0**（脚本断言） |
| 5–8 | 每档消融都有独立 config + 独立输出目录，**不许共用目录覆盖** |
| 9 | ★ checkpoint 只能由内部验证集选定；CLaw 只跑一次终评 |

**域标签怎么打 —— 用「引用法条名」反推，而不是用判决书类型**：

```
1. 答案/解析中出现的法条全名            ← 最可靠，可解释、可审计
   《中华人民共和国刑法》 / 《民法典》 → 实体法域
   《民事诉讼法》/《刑事诉讼法》/《行政诉讼法》 → 程序法域
2. 文书类型 + 案由                       ← 判决类兜底
   (民事|刑事|行政)(判决书|裁定书) + 「X纠纷」
3. 任务 id 前缀                          ← 最后兜底（如 jud_doc_sum-*）
4. 分类器兜底 + 人工抽检                 ← 仅用于前面全落空的样本
```

> **关键设计**：程序法**不能**靠判决书类型区分 —— 一份民事判决书既涉民事实体法也涉民诉程序法。
> 必须走「内容议题」判断（管辖 / 送达 / 举证责任 / 时效 / 上诉 / 再审 / 执行）。
> 打标必须**双路交叉验证**（规则 + 分类器），不一致的进人工抽检队列，
> 并在清单里记录每条的 `domain_source`，方便论文里做域标签质量分析。

### 3.4 统一数据 Schema

所有来源归一化为**同一格式**，落 `data/corpus/normalized/`：

```jsonc
{
  "uid": "disc_lawsft_pair:jud_doc_sum-1",   // 全局唯一，来源前缀 + 原始 id
  "domain": "civil",                          // criminal | civil | procedural | general
  "domain_source": "doc_type",                // statute_ref|doc_type|id_prefix|clf|manual
  "subdomain": "侵权责任",                     // 案由 / 罪名 / 编章（可空）
  "task_type": "judgment_summary",            // 见下
  "instruction": "请大致描述这篇文书的内容。",
  "input": "...",
  "output": "...",
  "law_refs": ["《中华人民共和国民法典》第1165条"],  // 引用法条，用于域校验与检索监督
  "source": {
    "dataset": "ShengbinYue/DISC-Law-SFT",
    "file": "DISC-Law-SFT-Pair.jsonl",
    "license": "apache-2.0",
    "license_grade": "A",
    "url": "https://hf-mirror.com/datasets/ShengbinYue/DISC-Law-SFT"
  },
  "synthetic": false,
  "split": "train",                            // train | val | test
  "content_sha256": "..."                      // 归一化后内容哈希，用于去重
}
```

`task_type` 枚举：`judgment_summary` / `judgment_prediction` / `info_extraction` /
`case_classification` / `legal_qa` / `statute_recall` / `statute_application` /
`exam_qa` / `general_replay`

> **注意**：`task_type` 必须**正交于** `domain`。不能出现「刑法任务集」这种把两个维度揉一起的命名，
> 否则消融实验无法拆解是域的作用还是任务的作用。

### 3.5 MoE 路由对数据的额外要求

凑够 10 万条 SFT 数据，**不等于**能训出路由器：

| | 专家 LoRA 训练 | 路由器训练 |
|---|---|---|
| 需要什么 | `(instruction, input) → output` | `query → domain` |
| 数据量 | 10 万条 | 2–5 万条（可从专家数据派生） |
| **关键约束** | — | **必须与专家训练集 disjoint** |

**三条硬约束**：

1. **路由集与专家集必须切分** —— 从 10 万条里**按 uid 哈希抽取 20%** 作为路由集，
   这 20% **从专家训练集中移除**。否则路由器学到的是「专家见过的样本」，实际查询上泛化崩塌。
2. **必须支持跨域查询** —— 例如「合同诈骗」同时涉民法（合同）+ 刑法（诈骗罪）。
   单标签路由器必然错 → 用**多标签 + top-2 路由**，并做成消融项。
3. **路由评测集必须贴近 CLaw 254 案的分布** —— 人工标 **200–500 条**「案件咨询形态」的
   query + 域标签。**这 200–500 条必须新建，不能从任何已有数据集里借。**

**路由方案阶梯**（逐级做，每级留消融）：R0 关键词/法条名规则 → **R1 Qwen3-Embedding-0.6B
冻结 + 线性/MLP 头（推荐主方案）** → R2 Qwen3-0.6B 域分类 → R3 MixLoRA / MoLE 式层内路由。

### 3.6 数据红线与去污

**绝对不许进训练/验证集**：

1. **CLaw 254 个最高法案例**的题目 / 参考答案 / 改写版本 / Judge 评分解释
2. **LexRubric 649 题 + 12,335 条 rubric**
3. **LexEval 23 任务 / 14,150 题**

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

| 基准 | 年份 / 会议 | 规模 | 协议 | 定位 |
|---|---|---|---|---|
| **CLaw** | 2025 / EMNLP-Findings | 306 部法律 / 64,849 条法条 / 254 个最高法案例 | 未公开发布 | **主基准**；官方数据无公开下载渠道，走自建路线 |
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
│  │  ├─ normalized/       # 阶段 2：归一化 + 域打标后的统一 schema
│  │  └─ MANIFEST.json     # 逐文件 SHA-256 清单
│  ├─ benchmark/           # 评测基准（LexRubric / LexEval），严禁进入训练
│  └─ _obsolete_*/         # 早期残留，保留可回溯
├─ configs/                # qlora_unified / adapters_router / retrieval / inference / judge / corpus_sources
├─ models/                 # 底座权重 + 训练输出的 LoRA 适配器
├─ neo4j/                  # Neo4j 数据卷
├─ indexes/                # 向量索引 / 图谱导出
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
│  ├─ corpus/              # 训练语料采集（阶段 1）
│  └─ benchmarks/          # 评测基准「拉取 → 验完整性 → 登记 SHA-256」
├─ outputs/                # 闭卷 / RAG / judge 输出
└─ docs/
   ├─ experiment_matrix.md  # 实验矩阵与消融
   ├─ run_order.md          # 执行顺序与阶段卡口
   ├─ leakage_rules.md      # 防泄漏红线与四级查重
   ├─ model_selection.md    # 模型选型分析
   ├─ env_setup.md          # 环境搭建与事故记录
   ├─ finetune_data_plan.md # 10 万条语料方案（人读的完整版）
   ├─ corpus/               # 训练语料登记（MANIFEST + 逐数据集溯源）
   ├─ benchmarks/           # 评测基准登记（LexRubric / LexEval，各自独立）
   └─ data_manifest.json    # CLaw 语料固化清单（**不混入上面两者**）
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
- **CLaw 254 个案例的题目、参考答案、改写版本、Judge 评分解释，一律不得进入训练/验证集。**
- **LexRubric 649 题、LexEval 14,150 题，同样一律不得进入训练/验证集。**
- 训练集与测试集需做四级查重（完全 / 规范化 / MinHash / 向量），余弦 > 0.92 人工复核。
- 法条必须保留**历史版本与生效/失效区间**，不得只存「最新版」。
- **仓库保持 Private**：CLaw 数据存在再分发限制，论文发表前不公开；无 License。
- **论文表述红线**：只能写「依据 CLaw 公开论文所描述协议构建的**兼容性复现实验**」，
  不能写「完全复现 CLaw 官方实验」。

详见 [`docs/leakage_rules.md`](docs/leakage_rules.md)。

---

## 8. 进度与执行顺序

执行顺序见 [`docs/run_order.md`](docs/run_order.md)。
核心原则：**先跑通评测链路，再训练模型；先用 10 个案例验证 Judge，再跑 254 个。**

### 已完成

- ✅ 实验环境全栈（`envs/main`，torch 2.6.0+cu124，Neo4j 17 索引 ONLINE）
- ✅ 底座与向量模型下载（`Qwen3-8B` 16 GB / `Qwen3-Embedding-0.6B`）
- ✅ 评测基准 LexRubric + LexEval 下载、校验、SHA-256 登记与归档
- ✅ 双盲判分层（`src/evaluation/judge/`）+ LexRubric 评测器改造 + 统计层
- ✅ **阶段 1：A 级训练语料采集** —— 5 个数据集 / **23 文件 / 1.2 GB** → `data/corpus/raw/`，
  逐文件 SHA-256 已登记，验收 **PASS**（[`docs/corpus/`](docs/corpus/)）；
  采集链路幂等：`bash scripts/corpus/prepare_corpus.sh`
- ✅ 检索指标目标：Recall@5 ≥ 0.85 / Recall@10 ≥ 0.90 / 历史版本准确率 ≥ 0.95

### 待办

- ⬜ **阶段 2：归一化 + 域打标**（域打标优先级链见 3.3 节；须先按内容哈希去掉
  `Dusker/lawyer-llama` 里与主源重复的样本）
- ⬜ 阶段 3：去污（硬门禁，`verdict` 必须 PASS）
- ⬜ 阶段 4：切分 train/val/test + 路由集
- ⬜ 阶段 5–9：LoRA 训练 → 路由 → MoE 消融 → 内部验证集选 checkpoint → CLaw 终评
- ⬜ `indexes/` 为空，尚无任何向量索引
- ⬜ （可选）配置 `HF_TOKEN` 后补采 `Aiiluo/Chinese-Law-SFT-Dataset`（2.6 MB，gated）

---

## 9. 结果完整性要求

每次运行必须保存**检索结果 + 最终答案 + 运行元数据**（模型哈希、适配器哈希、提示词哈希、
token 数、延迟），否则无法区分错误来源（未检索到 / 检索到未使用 / 版本引用错 /
模型推理错 / Judge 异常）。

---

## 10. 文档维护约定

> **本 README 必须随每一次实质变更同步更新，这是长期约定。**

凡发生以下任一情况，**同一次提交内**必须更新 README 对应章节：

| 变更类型 | 必须更新 README 的哪里 |
|---|---|
| 新增 / 更换**模型**（底座、向量、重排、判分器） | 第 1 节模型清单表 |
| 新增 / 调整**数据集来源**（含协议、体积、分级） | 3.2 节；同步 `configs/corpus_sources.yaml` |
| 调整**配比**或合成配额 | 3.1 节；同步 `configs/corpus_sources.yaml` 的 `target_mix` |
| 修改**处理流程 / 阶段卡口** | 3.3 节 |
| 新增**评测基准** | 第 4 节；同步 `docs/benchmarks/` |
| 环境 / 依赖版本变化 | 第 6 节 |
| 阶段推进（完成 / 开始） | 第 8 节进度 |

**同时保持两份记录同步**：
- 人读的完整方案 → `docs/`（如 [`docs/finetune_data_plan.md`](docs/finetune_data_plan.md)）
- 机器读的登记表 → `configs/`（如 [`configs/corpus_sources.yaml`](configs/corpus_sources.yaml)）
- README 是**索引与摘要**，不是唯一真相源 —— 细节永远以 `docs/` 与 `configs/` 为准。
