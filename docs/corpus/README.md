# 训练语料登记

> **本目录只登记「训练语料」，与另外两处登记完全分离，任何情况下不得互相混排：**
>
> | 登记处 | 管什么 | 能否进训练 |
> |---|---|---|
> | `docs/data_manifest.json` | CLaw 语料固化清单（306 部 / 64,849 条 / 254 案） | ❌ 绝不可 |
> | `docs/benchmarks/` | 评测基准（LexRubric / LexEval） | ❌ 绝不可 |
> | **`docs/corpus/`（本目录）** | **训练语料（原始源 + 归一化产物 + 域标签）** | ✅ 是训练来源（仍须过阶段 3 去污） |
>
> 登记时间：2026-09-17　登记机器：`192.168.195.61`（2×A800 80GB）
> 服务器落盘位置：`/mnt/data/lidian/law-agent/data/corpus/`

## 目录导航

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件：采集与归一化的完整登记 |
| `MANIFEST.json` | 阶段 1 总清单（逐文件 SHA-256 + 每数据集链路溯源） |
| `sources/<slug>.source.json` | 每个数据集的采集溯源（commit / 体积 / 命中文件） |
| [`DEDUP_REPORT.md`](DEDUP_REPORT.md) | **跨源判重报告**（三指纹交叉证据 + 丢弃/保留判定） |

---

## 一、本次采集结果（阶段 1）

### 阶段 1 采集结果（已完成）

**只采 `license_grade = A`（协议明确：Apache-2.0 / MIT）的源。** 合计 **1.2 GB / 23 文件**。

| 数据集 | 文件 | 体积 | 协议 | commit | 服务器落点 |
|---|---|---|---|---|---|
| `ShengbinYue/DISC-Law-SFT` | 6 | 552.4 MiB | Apache-2.0 | `fb12cf02` | `raw/ShengbinYue__DISC-Law-SFT/` |
| `Dusker/lawyer-llama` | 7 | 425.7 MiB | MIT | `bd021100` | `raw/Dusker__lawyer-llama/` |
| `twang2218/chinese-law-and-regulations` | 4 | 152.8 MiB | Apache-2.0 | `58db8b54` | `raw/twang2218__chinese-law-and-regulations/` |
| `Skepsun/lawyer_llama_data` | 3 | 22.9 MiB | Apache-2.0 | `10ca3119` | `raw/Skepsun__lawyer_llama_data/` |
| `pandalla/chinese_law_examples` | 3 | 519.8 KiB | Apache-2.0 | `cdd5f044` | `raw/pandalla__chinese_law_examples/` |
| **合计** | **23** | **1.2 GiB** | | | |

**验收结论：PASS** —— 逐文件 SHA-256 全部登记（缺失 0 条），磁盘实测与 `MANIFEST.json` 逐文件比对
**无缺失、无大小不符**；所有 LFS 大文件的 SHA-256 与 HF 侧 `lfs.oid` **校验通过**。

### 未采 / 跳过（都有明确理由）

| 数据集 | 状态 | 理由 |
|---|---|---|
| `Aiiluo/Chinese-Law-SFT-Dataset` | `SKIPPED_GATED_NO_TOKEN` | `gated=auto`，服务器未配置 `HF_TOKEN`。仅 2.6 MB，已在 `configs/corpus_sources.yaml` 登记，配好 token 后可单独补采 |
| `wormtooth/MNBVC-judgment` | 体积护栏拒绝 | **121.5 GB**，超过 `--max-gb`（默认 3 GB）。A 级但**只允许按需抽样单文件，禁止整下** |
| `china-ai-law-challenge/cail2018` | 未采 | `license_grade = B`（license=unknown），**仅内部探索**，进论文实验集前须确认授权 |
| `Kuugo/Chinese_Law` | 未采 | B 级（license 未声明） |
| `Dusker/chinese-laws-pretrain` | 未采 | B 级（license 未声明） |
| `Brench/chinese_law_data_rag_ft` | 未采 | B 级（license 未声明） |

---

## 二、可复现流程

**采集合一（幂等，已下载且 SHA-256 校验通过的文件会跳过）**：

```bash
# 服务器上
bash scripts/corpus/prepare_corpus.sh --list     # 先看计划，不下载
bash scripts/corpus/prepare_corpus.sh            # 采 A 级（默认）
bash scripts/corpus/prepare_corpus.sh --threads 12
bash scripts/corpus/prepare_corpus.sh --only ShengbinYue/DISC-Law-SFT
```

**数据集清单只在 [`configs/corpus_sources.yaml`](../../configs/corpus_sources.yaml) 维护**，
脚本不硬编码任何数据集名。改清单 → 重跑即可，无需改代码。

**关键参数**：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--grade` | `A` | 只采该 license_grade |
| `--max-gb` | `3.0` | **单数据集体积上限**（MNBVC 121.5GB 靠这条拦下） |
| `--only` / `--skip` | — | 指定 / 排除数据集 |
| `--list` | — | 干跑，只列计划 |

**把登记拉回仓库**（`_remote/dl_corpus.txt` 维护文件清单）：

```
docs/corpus/MANIFEST.json                                    ← 总清单（逐文件 SHA-256）
docs/corpus/sources/<slug>.source.json                        ← 每个数据集一份溯源
```

---

## 三、文件命名规范

```
docs/corpus/
├── README.md                       # 本文件
├── DEDUP_REPORT.md                 # 跨源判重报告（三指纹证据）
├── MANIFEST.json                   # 阶段 1 总清单：逐文件 SHA-256 + 每数据集链路溯源
├── normalized_STATS.json           # 阶段 2 统计（逐来源/逐域/逐任务 + 与 target_mix 差距）
├── normalized_DEDUP_REPORT.json    # 阶段 2 全局精确去重明细
└── sources/
    ├── ShengbinYue__DISC-Law-SFT.source.json          # 数据集 id 里的 / 替换为 __
    ├── Dusker__lawyer-llama.source.json
    └── ...
```

新增数据集沿用同一命名 `<id 中的 / 换成 __>.source.json`，**不合并进任何既有文件**；
阶段产出的统计文件用 `阶段前缀_` 区分（如 `normalized_*`），后期去污阶段将是 `decontamination_*`。

---

## 四、采集阶段已踩过的坑（可直接复用）

1. **hf-mirror 对「不存在的仓库」返回 `HTTP 401` 而不是 `404`** —— 按记忆猜 ID 探测会把不存在的
   仓库误判成「无权访问」。必须**先用搜索接口反查真实 ID**。
2. **HF 搜索接口对中文关键词完全无效**（`法律`/`刑法`/`判决`/`法条` 均返回 0 条）。
3. **不要按 `usage` 里的 `case_corpus` 标签一刀切排除** —— `Dusker/lawyer-llama` 的 usage 是
   `[expert_train, case_corpus]` 但只有 446 MB，是合法要采的。真正的超大语料交给**体积护栏**拦。
4. **`Dusker/lawyer-llama` 内含 `DISC-Law-SFT-Pair.json` / `-Triplet.json`，与主力源重复**
   —— 已在阶段 2 处置完毕，**但结论与"全丢"不同**：`Pair` 与主源 100% 重合（丢弃），
   `Triplet` 只有 20.5% 重合（**保留**，另有 12,684 条独有）。完整证据见 [`DEDUP_REPORT.md`](DEDUP_REPORT.md)。
5. **单连接慢、分块快**：小文件（< 32 MB）走单连接只有几百 KB/s；大文件分块并行可达
   6–14 MB/s。个别分块会卡住重试（本次有一个 89.7 MB 文件在 64.3% 停了约 2.7 分钟后完成），
   属正常重试行为，**不要中途杀进程**。
6. **gated 仓库需要 `HF_TOKEN`**，否则直接跳过并记为 `SKIPPED_GATED_NO_TOKEN`（不是失败，但会让退出码非 0）。

---

## 五、阶段 2：归一化 + 域打标（已完成）

> 配置分两层，**职责不重叠**：
> - [`configs/corpus_sources.yaml`](../../configs/corpus_sources.yaml) —— **合规层**（能不能采、license、体积、等级）
> - [`configs/corpus_adapters.yaml`](../../configs/corpus_adapters.yaml) —— **解析层**（怎么读、进哪条流、`drop_files`、过滤阈值、打分权重）

```bash
bash scripts/corpus/prepare_normalize.sh --list                              # 干跑
bash scripts/corpus/prepare_normalize.sh --limit-per-file 300 --suffix .sample  # 抽样验证打标
bash scripts/corpus/prepare_normalize.sh                                     # 全量
```

**产出**（服务器 `data/corpus/normalized/`，3.8 GB）：

| 路径 | 内容 |
|---|---|
| `qa/<slug>.jsonl` × 3 | 训练语料，统一 schema（含域标签 + `messages` 规范 chat 渲染） |
| `qa/_all.jsonl` | 合并流 |
| `statutes/<slug>.jsonl` × 2 | **法条库**，保留 `effective_from` / `effective_period` / `status`（支撑历史版本硬约定） |
| `STATS.json` | 逐来源 / 逐域 / 逐任务统计 + 与 `target_mix` 的差距 |
| `DEDUP_REPORT.json` | 全局精确去重明细 |

### 5.1 数据量与配比核算（★ 结论：**每域都超额，总缺口 = 0**）

raw **349,665 条** → qa 流 **285,257 条** + 法条库 **23,510 条**。

| 域 | 实得 | 目标 | 占比 | 达成率 |
|---|---|---|---|---|
| criminal 刑法 | **97,414** | 30,000 | 34.1% | **325%** |
| civil 民法 | **104,691** | 40,000 | 36.7% | **262%** |
| procedural 程序法 | **25,445** | 20,000 | 8.9% | **127%** |
| general 通用/其他 | 57,707 | 10,000 | 20.2% | 577% |
| **合计** | **285,257** | 100,000 | | **总缺口 0** |

**两个必须记住的结论**：

1. **程序法只占 8.9%，但绝对量够（25,445 > 20,000）** —— 8,000 条合成数据可由真实数据替代，
   也可以保留「12,000 真实 + 8,000 合成」的原方案。**不存在"凑不齐"的问题，只需要下采样。**
2. `general`（宪法 / 行政法 / 法治理论 / 职业道德等**域不明确但仍属法律**的样本）占 20.2%，
   已超过「通用回放 1 万」的需求 → **不需要再单独找通用数据集**。

### 5.2 域打标方法：**加权打分**，不是优先级短路

实测两种短路顺序都会系统性错标：程序法优先 → 判决预测被附带引用带偏；实体法优先 → 刑诉考题判成刑法。
最终实现六路证据加权求和，权重全部写在 `configs/corpus_adapters.yaml`：

| 信号 | 权重 | 说明 |
|---|---|---|
| 问句里引用的法条 | ×2.0 | 最可靠 |
| 答案/解析里引用的法条 | ×1.0 | 最可靠 |
| 文书类型（刑事/民事判决书） | ×2.5 | 判决类主力 |
| 任务型强指示（刑期预测→刑法） | ×1.5 | 少数任务 |
| 罪名释义类 | ×3.0 | — |
| **议题关键词**（只在问句里找） | 程序0.8/个 刑民0.5/个 | **必须在，否则 fallback 35%** |

- **判决类任务给程序法引用降权 ×0.35**（判决问的是实体结论）
- 每条记 `domain_source` + 完整 `scores` / `evidence` / `contrib_primary`，可事后审计
- **`fallback` 占 11.7%**，落在可接受区间（这些多为宪法/法治理论/职业道德，归 `general` 是正确的）

**⚠️ 一个已修的审计 bug（留痕）**：`domain_source` 最初取「**所有域加总**最大的信号」，
导致 `topic_keyword` 因为同时对三个域加分、总和虚高而虚占 **47.5%**，把更可靠的 `statute_ref` / `doc_type` 盖掉。
改为「对**胜出域**贡献最大的信号」后：`topic_keyword` 47.5% → 40.3%，`doc_type` 2.8% → 5.3%，
`task_hint` 2.2% → 3.0%，`statute_ref_in_question` 7.5% → 9.6% —— 更符合直觉。

### 5.3 质量过滤与"被砍掉的 2.6 万条"是什么

`min_output_chars: 8` 砍掉 25,895 条，逐条查过**都是该砍的**：

| 任务 | 砍掉 | 实际内容 |
|---|---|---|
| `leg_case_cls` | 14,052 | 输出是「盗用风险」「欺诈风险」——**金融风控标签**，输入是投诉/报案描述，**不是法律问答** |
| `legal_question_answering` | 7,755 | 1–7 字的极短答案（如「是」「不行」） |
| `leg_ele_extra` | 3,772 | 输出是占位符 **「O」** —— 数据本身退化的行 |
| `jud_read_compre` | 316 | 「否」「杜1」「四口人」类阅读理解极短答案 |

如需找回，`--min-output-chars 4` 即可（不建议）。全局精确重复另丢弃 **14,997 条**。

### 5.4 跨源判重

**完整证据见 [`DEDUP_REPORT.md`](DEDUP_REPORT.md)。核心铁律：
`文件名相同 + 条数相同` 完全不能作为判重依据，必须用「严格 / 宽松 / 案件正文」三种指纹交叉。**

| 文件 | 判定 | 依据 |
|---|---|---|
| `Dusker/.../DISC-Law-SFT-Pair.json`（166,758 条） | **丢弃** | 案件正文指纹 100% 覆盖（153,231 / 153,231，两侧独有 0） |
| `Dusker/.../DISC-Law-SFT-Triplet.json`（16,000 条） | **保留** | 仅 20.5% 重合，另有 **12,684 条独有** |

丢弃动作落在配置层（`drop_files` + `keep_files`），**不物理删 raw**（遵守只读约定），可审计可回滚。

---

## 六、下一步（阶段 3）：双向去污（硬门禁）

- 方法：精确哈希 + simhash(64bit, 汉明距离 ≤ 3)
- **`verdict` 必须 PASS 才允许进入训练**
- **特别注意同源风险**：`Skepsun/lawyer_llama_data` 的司考题（7,000 条）与
  LexRubric 的 `sifakaoshi` split（176 条）**可能取自同一批公开真题**，必须核查
- 黑名单：CLaw 254 案、LexRubric 649 题 + 12,335 rubric、LexEval 23 任务 / 14,150 题

之后阶段 4 需在**已有超额数据上做分层下采样**（按 `task` × `domain` × `source` 分层，
避免某一任务型垄断某个专家），并在此之前先切出 val/test 与路由集（与专家训练集 disjoint）。
