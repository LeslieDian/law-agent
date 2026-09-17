# 法律领域 LoRA + MoE 微调 —— 数据方案与执行流程

> 编制日期：2026-09-17
> 编制依据：**服务器实测**（`192.168.195.61`，hf-mirror.com API 逐条探测，未依赖任何记忆或二手描述）
> 探测脚本：`scripts/benchmarks/probe_hf_datasets{,2,3,4}.py`（只读，可随时重跑复核）

---

## 0. 先回答两个问题

### Q1：LexEval / LexRubric 下载好了吗？

**都下好了，都校验通过，都已登记 SHA-256，都已归档进仓库。**

| | LexRubric | LexEval |
|---|---|---|
| 服务器路径 | `data/benchmark/lexrubric/` | `data/benchmark/lexeval/` |
| commit | `9141ee4b` | `044c695f` |
| 规模 | 649 题 / 12,335 rubric / 9 文件 | 23 任务 / 14,150 题 / 65 文件 / 37.1 MiB |
| 校验 | **PASS**（0 失败 / 2 警告） | **PASS**（0 失败 / 0 警告） |
| 仓库归档 | `docs/benchmarks/lexrubric.{MANIFEST,VERIFY_REPORT}.json` | `docs/benchmarks/lexeval.{MANIFEST,VERIFY_REPORT}.json` |

已在 GitHub `main`（`b8ae35a`）。两个基准在**目录 / 清单 / 校验报告 / 评分口径**四个层面完全隔离。

> 唯一差异：LexRubric rubric 总数 12,335 对论文 12,337（差 2，全部落在「法律准确性」维度）。
> 推断论文含未公开私有 split。**论文里必须如实写这个差异，不得声称"与论文完全一致"。**

### Q2：下一步该干什么？

你说的对：**下一步是造训练语料，不是继续做评测**。但要把话说得更准一点——

> 目标不是"凑 10 万条"，而是"凑 10 万条**带域标签**的数据，装得下 3 个专家 + 喂得饱 1 个路由器"。

差一个词，工程量差一倍。原因见第 5 节。

---

## 1. 实测结论：候选源清单（2026-09-17 探测）

### 1.1 一个必须先知道的探测陷阱

**hf-mirror 的 API 对「不存在」的仓库返回 `HTTP 401`，不是 `404`。**
第一轮我按记忆猜 ID，14 个里 12 个报 401，差点全判成"不可用"。改用**搜索接口反查真实 ID** 后才对上。

顺带纠正一个我上一条说错的 ID：

| 我上一条说的 | 实际正确 ID |
|---|---|
| `FudanDISC/DISC-Law-SFT` | **`ShengbinYue/DISC-Law-SFT`** |

### 1.2 判决/案例/问答类（训练语料主力）

| 数据集 | 体积 | 文件 | 协议 | 等级 | 用途 |
|---|---|---|---|---|---|
| `ShengbinYue/DISC-Law-SFT` | 579.2 MB | 6 | **Apache-2.0** | **A** | 主力：判决预测/信息抽取/摘要/问答/司考，含 48K Alpaca-GPT4 + 60K Firefly 通用回放 |
| `Skepsun/lawyer_llama_data` | 24.0 MB | 3 | **Apache-2.0** | **A** | 司法考试题，**含《民诉》《刑诉》程序法题**，带 `source` 字段 |
| `Dusker/lawyer-llama` | 446.3 MB | 7 | **MIT** | **A** | 含 `kg_crime_llama.json`（7.14 MB **犯罪知识图谱**，可转问答） |
| `wormtooth/MNBVC-judgment` | **124.5 GB** | 996 | **MIT** | **A** | 判决书大规模语料，**按需抽样**，不要整下 |
| `china-ai-law-challenge/cail2018` | 1167.8 MB | 11 | **unknown** | **B** | 刑事一审判决：罪名 + 法条 + 刑期，**刑法域的黄金标注** |
| `pandalla/chinese_law_examples` | 0.5 MB | 3 | Apache-2.0 | A | 小，参考格式用 |
| `Aiiluo/Chinese-Law-SFT-Dataset` | 2.6 MB | 6 | Apache-2.0 | A | **文件名已按民事/商事/刑事分好**，体量小但协议干净；`gated=auto` 需登录 |

### 1.3 法条原文类（知识库 + 法条任务）

| 数据集 | 体积 | 文件 | 协议 | 等级 | 用途 |
|---|---|---|---|---|---|
| `Kuugo/Chinese_Law` | 7.6 MB | **238 个 .txt** | 不明 | **B** | 文件名即《中华人民共和国民法典.txt》《刑法.txt》《刑事诉讼法.txt》→ **最省事的 306 部原文来源** |
| `Dusker/chinese-laws-pretrain` | 5.5 MB | 119 json | 不明 | **B** | 按编拆分（刑法.json / 合同编.json / 物权编.json）→ **结构化最好**，便于切条 |
| `twang2218/chinese-law-and-regulations` | 160.3 MB | 4 | **Apache-2.0** | **A** | 大体量带元数据法条库，**协议干净可发表** |
| `Brench/chinese_law_data_rag_ft` | 506.9 MB | 5 | 不明 | **B** | `laws_data_281k.jsonl`（302 MB）→ 28.1 万条，RAG 侧可用 |

### 1.4 已被证伪／不可用的（省得再试）

| 目标 | 结果 |
|---|---|
| `OpenBMB/LawBench`（HF） | 不存在（HF 上是 `doolayer/LawBench` 等镜像，非官方） |
| `THUIR/LEEC`、`CSHaitao/LegalAgentBench`（HF） | 不在 HF 镜像上 |
| `china-ai-law-challenge/cail2019`（HF） | 镜像无；**GitHub 有**（178.6 MB） |
| HF 搜索 `法律` / `刑法` / `判决` / `司法考试` / `法条` | **全部返回 0 条** → 中文关键词搜不出来，**只能靠英文或已知 ID 定位** |
| `china-ai-law-challenge/CAIL2018`（GitHub） | 仓库 0 MB，只放外链，**数据要走 HF 那份 parquet** |

---

## 2. 三域划分：先把"域"定义清楚

**这是整个方案的地基，定义错了后面全白干。**

| 域 | 覆盖范围 | 关键词锚点 |
|---|---|---|
| **刑法** | 刑法典、刑事司法解释、罪名、量刑 | 罪名（盗窃/故意伤害）、《中华人民共和国刑法》 |
| **民法** | 民法典七编、合同/物权/侵权/婚姻家庭/继承、商事 | 案由「X 纠纷」、《民法典》《合同法》 |
| **程序法** | 刑诉 / 民诉 / 行政诉讼法、证据规则、管辖、送达、时效、执行 | 《民事诉讼法》《刑事诉讼法》《行政诉讼法》、「管辖」「举证」「上诉」「再审」 |

### 2.1 域标签怎么打——**用"引用法条名"反推，而不是用判决书类型**

这是本次探测最有价值的发现，也是我在样本里亲眼验证的：

- **证据 1**：`ShengbinYue/DISC-Law-SFT` 的判决类样本，`input` 里**自带案由与文书类型**
  （实测样本：`侵权责任纠纷`、`...一审民事判决书`）→ 民法域可直接正则抽取。
- **证据 2**：`Skepsun/lawyer_llama_data` 的样本，`instruction` 直接就是
  「下列选项属于**《民事诉讼法》**直接规定、具有简易程序特点的内容?」
  「关于**补充侦查**，下列说法是错误的?」→ **答案里引用的法条名 = 最可靠的域标签**。

**打标优先级链（高 → 低）**：

```
1. 答案/解析中出现的法条全名            ← 最可靠，可解释、可审计
   《中华人民共和国刑法》 / 《民法典》 → 实体法域
   《民事诉讼法》/《刑事诉讼法》/《行政诉讼法》 → 程序法域
2. 文书类型 + 案由                       ← 判决类兜底
   (民事|刑事|行政)(判决书|裁定书) + 「X纠纷」
3. 任务 id 前缀                          ← 最后兜底（如 jud_doc_sum-*）
4. 分类器兜底 + 人工抽检                 ← 仅用于前面全落空的样本
```

**关键设计**：程序法**不能**靠判决书类型区分——一份民事判决书既涉民事实体法也涉民诉程序法。
必须走「内容议题」判断（管辖/送达/举证责任/时效/上诉/再审/执行）。

**落地要求**：打标必须**双路交叉验证**（规则 + 分类器），不一致的样本进人工抽检队列，
并在 `MANIFEST` 里记录每条的 `domain_source`（`statute_ref` / `doc_type` / `id_prefix` / `clf` / `manual`），
**方便论文里做域标签质量分析**。

---

## 3. 10 万条怎么配

| 域 | 目标条数 | 主要来源 | 占比 |
|---|---|---|---|
| 刑法 | **30,000** | CAIL2018 刑事抽样 + DISC-Law-SFT 刑事判决 + `kg_crime_llama` 转问答 + 刑法典 452 条法条任务 | 30% |
| 民法 | **40,000** | DISC-Law-SFT 民事判决（抽取/预测/摘要）+ 民事问答 + 民法典 1260 条法条任务 | 40% |
| 程序法 | **20,000** | `Skepsun` 司考程序法题 + 三诉讼法原文任务 + **合成补量 ~8,000** | 20% |
| 通用回放 | **10,000** | DISC-Law-SFT 内置 Alpaca-GPT4 / Firefly 抽样（**防灾难性遗忘**） | 10% |
| **合计** | **100,000** | | |

### 3.1 为什么必须有「通用回放」这 10%

纯法律数据微调会**打崩底座通用能力**。CLaw 254 案里有相当比例需要常识推理与语言组织，
只喂法律数据会让这部分指标**不升反降**。这 10k 是保险，不是凑数。

### 3.2 程序法为什么要合成

程序法是三域里公开语料**最稀薄**的：公开 SFT 集里程序法题占比普遍 <10%，
而用户需求里它占 20%。缺口只能自建：

- **合成规范**（写死，不许绕过）：
  1. 必须以**三大诉讼法原文条文**为唯一事实来源，禁止让模型自由发挥
  2. 每条合成样本登记：`synthetic: true`、`generator_model`、`prompt_sha256`、`source_articles[]`
  3. **合成数据一律不进验证集与测试集**
  4. 合成占比在论文里如实披露（程序法域 8000/20000 = 40%）

---

## 4. 统一数据 Schema

所有来源归一化为**同一格式**，落 `data/corpus/normalized/`：

```jsonc
{
  "uid": "disc_lawsft_pair:jud_doc_sum-1",   // 全局唯一，来源前缀 + 原始 id
  "domain": "civil",                          // criminal | civil | procedural | general
  "domain_source": "doc_type",                // statute_ref|doc_type|id_prefix|clf|manual
  "subdomain": "侵权责任",                     // 案由 / 罪名 / 编章（可空）
  "task_type": "judgment_summary",            // 见 4.1
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

### 4.1 `task_type` 枚举（与论文实验对齐）

`judgment_summary` / `judgment_prediction` / `info_extraction` / `case_classification` /
`legal_qa` / `statute_recall` / `statute_application` / `exam_qa` / `general_replay`

> **注意**：`task_type` 必须**正交于** `domain`。不能出现"刑法任务集"这种把两个维度揉一起的命名，
> 否则消融实验无法拆解是域的作用还是任务的作用。

---

## 5. MoE 路由对数据的额外要求（**最容易被忽略的一节**）

凑够 10 万条 SFT 数据，**不等于**能训出路由器。两者对数据的要求不同：

| | 专家 LoRA 训练 | 路由器训练 |
|---|---|---|
| 需要什么 | `(instruction, input) → output` | `query → domain` |
| 数据量 | 10 万条 | 2–5 万条（可从专家数据派生） |
| **关键约束** | — | **必须与专家训练集 disjoint** |

### 5.1 三条硬约束

1. **路由集与专家集必须切分，不能同一批数据既训专家又训路由。**
   否则路由器学到的是"专家见过的样本"，在实际查询上泛化崩塌。
   → 从 10 万条里**按 uid 哈希抽取 20%** 作为路由集，这 20% **从专家训练集中移除**。

2. **必须支持跨域查询。**
   实测样本里就有「合同诈骗」这种**同时涉民法（合同）+ 刑法（诈骗罪）**的题。
   单标签路由器在这里必然错。→ 用**多标签 + top-2 路由**，并把它做成消融项。

3. **下游任务是 CLaw 254 案（最高法案例），路由评测集必须贴近该分布。**
   → 人工标 **200–500 条**「案件咨询形态」的 query + 域标签，专门做路由评测。
   **这 200–500 条必须新建，不能从任何已有数据集里借。**

### 5.2 路由方案阶梯（逐级做，每级留消融）

| 级别 | 方案 | 说明 |
|---|---|---|
| R0 | 关键词/法条名规则路由 | 可解释 baseline，必须有 |
| **R1** | **Qwen3-Embedding-0.6B（冻结）+ 线性/MLP 头** | **推荐主方案**：便宜、可复现、可解释 |
| R2 | Qwen3-0.6B 做域分类 | 上界参考 |
| R3 | MixLoRA / MoLE 式层内路由（单模型多专家） | 论文"方法创新"升级项 |

### 5.3 MoE 消融矩阵（论文必备）

| 编号 | 配置 | 作用 |
|---|---|---|
| A0 | 单 LoRA 全域训练 | baseline |
| A1 | 3 专家 + **oracle 路由** | **上界**（用真标签路由） |
| A2 | 3 专家 + **学习式路由（R1）** | **主方法** |
| A3 | 3 专家 + 无路由（权重平均融合） | 证明"路由有用"而非"多专家有用" |
| A4 | 3 专家 + **随机路由** | **下界** |

> A1 与 A2 的差值 = 路由器质量损失；A2 与 A3/A4 的差值 = 路由机制本身的增益。
> **没有 A1 和 A4 的论文，审稿人一定会问"你怎么知道不是多专家本身带来的增益"。**

---

## 6. 红线与去污（**不可协商**）

沿用项目既有约定，且本方案新增一条：

1. **CLaw 254 案的题目/参考答案/改写版/判分解释，绝不进训练或验证集。**
2. **LexRubric 649 题、LexEval 14,150 题，同样绝不进训练或验证集。**
3. **新增**：上述三者之间以及与训练语料之间，**必须做双向去污**——
   不仅查"训练集里有没有混入评测题"，还要查"评测集里有没有混入训练语料来源"
   （例如 `Skepsun/lawyer_llama_data` 本身就是"司法考试题"，
   而 LexRubric 的 176 条就是 `sifakaoshi`（司法考试）—— **这两者有同源风险，必须查**）。

**去污实现**（阶段 3 强制卡口）：

```
对每个训练样本的 (instruction + input) 计算 simhash(64bit) 与 精确哈希；
与黑名单库（254 + 649 + 14150 条）做：
  ① 精确哈希比对        → 命中即剔除
  ② simhash 汉明距离 ≤ 3 → 命中即剔除并记录
输出 docs/corpus/DECONTAMINATION_REPORT.json：
  {checked: N, exact_hits: [...], near_hits: [...], verdict: "PASS|FAIL"}
**verdict 必须为 PASS 才允许进入训练。**
```

---

## 7. 执行流程（分阶段，每阶段有验收卡口）

```
阶段 0  源合规审查        → docs/corpus/candidates.json（A/B 级分级，A 级才可进论文实验集）
阶段 1  分批采集          → data/corpus/raw/<dataset>/  + 每批 SHA-256 登记
阶段 2  归一化 + 域打标   → data/corpus/normalized/*.jsonl + DOMAIN_LABEL_REPORT.json
阶段 3  去污              → DECONTAMINATION_REPORT.json（verdict 必须 PASS）
阶段 4  切分              → train/val/test + 路由集（与专家集 disjoint）
───────────────────────────────  以下才动 GPU ───────────────────────────────
阶段 5  基线：单 LoRA 全域训练                 → A0
阶段 6  专家 LoRA ×3（刑法/民法/程序法）        → A1/A2/A3/A4 的组件
阶段 7  路由器训练（R0 → R1）                  → ROUTER_EVAL.json
阶段 8  MoE 组合 + 消融矩阵                    → 四档消融表
阶段 9  用内部验证集选 checkpoint              → 选定后才允许跑 CLaw ★红线
```

### 阶段验收卡口

| 阶段 | 卡口（不达标不许进下一阶段） |
|---|---|
| 0 | 每个源都有 `license` + `license_grade`；B 级源明确标注"仅内部探索" |
| 1 | 每批落盘后立即算 SHA-256；`docs/corpus/MANIFEST.json` 与磁盘实测一致 |
| 2 | 三域占比落在目标 ±15%；`domain_source` 分布可解释；人工抽检 50 条准确率 ≥ 95% |
| 3 | **去污 verdict = PASS**，且报告里能看到近重复命中明细 |
| 4 | 路由集与专家训练集 uid 交集 = **0**（脚本断言） |
| 5–8 | 每档消融都有独立 config + 独立输出目录，**不许共用目录覆盖** |
| 9 | ★ checkpoint 只能由内部验证集选定；CLaw 只跑一次终评 |

---

## 8. 验收标准

- 每域有效样本 ≥ 目标值，总 10 万 ±5%
- 去污零泄漏（verdict = PASS）
- 三域域内验证集准确率均有提升，且**无任一域下降**
- 通用能力不显著退化（用通用回放集做对照）
- 路由：准确率 ≥ 0.90，跨域 top-2 召回 ≥ 0.95（**待人工评测集建成后校准阈值**）
- 所有产物带 SHA-256，可复现

---

## 9. 已知风险

| 风险 | 影响 | 处置 |
|---|---|---|
| 程序法公开语料稀薄 | 程序法专家欠拟合 | 合成补量（40%），**论文如实披露** |
| 多个优质源 license 不明（`Kuugo/*`、`Dusker/chinese-laws-pretrain`、`Brench/*`、CAIL2018） | 论文可发表性 | 降为 B 级，**先用 A 级源把 10 万条凑出来**；B 级只做内部增强 |
| LexRubric 司考 split 与 `Skepsun` 司考数据同源风险 | 评测污染 | 阶段 3 去污强制卡口（第 6 节第 3 条） |
| 域标签全靠机器打标 | 标签噪声 | 双路交叉验证 + 人工抽检 + `domain_source` 留痕 |
| MNBVC-judgment 124 GB | 磁盘/时间 | **只抽样子集**，禁止整下 |
| 民法域一枝独大（40%） | 路由器偏置 | 训练时按域加权采样 / 损失重加权 |

---

## 10. 现在的状态与直接下一步

**已就位**：Qwen3-8B + Qwen3-Embedding-0.6B、`envs/main` 全栈、Neo4j、两个基准已登记、判分层已写

**完全没开始**：`indexes/` 空、无 `corpus/`、无 `data/raw`

**第一步（阶段 1）建议先只拉 A 级源**，量小、协议干净、当天可完成：

```
ShengbinYue/DISC-Law-SFT          579.2 MB   apache-2.0   主力
Skepsun/lawyer_llama_data          24.0 MB   apache-2.0   程序法
Dusker/lawyer-llama               446.3 MB   MIT          犯罪知识图谱
twang2218/chinese-law-and-regulations 160.3 MB Apache-2.0 法条原文（可发表版）
pandalla/chinese_law_examples       0.5 MB   apache-2.0
                                    合计 ≈ 1.21 GB
```

B 级源（法条原文结构更好的 `Kuugo/Chinese_Law`、`Dusker/chinese-laws-pretrain`、CAIL2018）
**先拉但标记为「仅内部」**，等授权口径确认后再决定是否进论文实验集。

---

*本文件的候选源清单同时以机器可读形式维护在 `configs/corpus_sources.yaml`，
采集脚本应直接读该文件，避免脚本里硬编码数据集名。*
