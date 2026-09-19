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
| `normalized_STATS.json` / `normalized_DEDUP_REPORT.json` | 阶段 2 统计与全局精确去重明细 |
| `DECONTAMINATION_REPORT.md` / `.json` | 阶段 3 **首扫**证据（`verdict = FAIL`，含 18,587 处命中明细） |
| `DECONTAMINATION_REPORT_PASS.md` / `.json` | 阶段 3 **复扫 PASS** 证据（门禁用） |
| `DECONTAM_REMOVED_UIDS.txt` / `DECONTAM_REMOVED_UIDS_PASS.txt` | 剔除清单 15,007 行 / 复扫清单**空文件**（门禁证据） |
| `STATUTE_SPLIT_REPORT.md` / `statute_items_STATS.json` / `statute_items_VERIFY.md` / `.json` | 阶段 2b 切条报告 + 质检 **PASS** |
| `DERIVE_REPORT.md` / `DERIVE_STATS.json` | 阶段 2c 程序法条文任务派生报告 + 自检 |
| `SPLIT_REPORT.md` / `SPLIT_STATS.json` / `SPLIT_VERIFY.md` / `.json` | 阶段 4 切分报告 + 质检门禁 **PASS** |
| `VECTOR_DB_PROBE.json` | 阶段 4b 前置探查：向量库 / 知识图谱设计的**可行性实测**（节点可派生性、时间字段覆盖、交叉引用规模、文本长度分布） |
| `CASE_PROBE.md` | 案例侧实测：案件类样本规模、Case 属性可得性、论文 17 类实体的可实例化判定（5 可建 / 2 弱 / 10 无料） |
| `LONG_TEXT_PROBE.md` | 长文本隐患结论：item 最长 26,980 字**不是切条遗漏**（行首多条约号 = 0），而是条文自带附表；影响面 372/65,037 |
| [`RETRIEVAL_SCOPE.md`](RETRIEVAL_SCOPE.md) | **阶段 4b-0 / 4b-1 实测**：检索库范围门禁（PASS，评测集 22,000 条全在 qa 流、案件类独占 11,811）+ 法名**两级**规范化（1,807→1,727→**1,579**，跨源交集 0→80→**228**） |
| `RETRIEVAL_POOL_PROBE.json` | 4b-0 证据：四份 split 交集复核、`task_kind` 分布、三种检索库范围的实得量 |
| `LAW_TITLE_NORM.json` | 4b-1 证据：逐文件 `raw / L1 / L2` 法名数、跨源交集、合并明细 |
| `docs/retrieval/`（另一目录） | **阶段 4b-2 ~ 4b-5 产物**：`EDGE_EXTRACT`（图谱边）/ `EMBEDDING_REPORT`（向量）/ `NEO4J_IMPORT`（图谱导入）/ `SMOKE_RETRIEVAL_{VECTOR,HYBRID}`（检索基线） |

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

## 六、阶段 3：双向去污（✅ 已完成，复扫 verdict = PASS，2026-09-18）

> ★ 本节数字是 **2026-09-18 修复阶段 2 `uid` 撞号后全链路重跑**的结果（旧数字见第十节对照）。
> 重跑后两条硬断言当场成立，**不再有「整组连坐」误删**。

**两遍式流程与结果**：

1. **首扫**（308,767 行 / 耗时 ~1.6 h）：黑名单 = LexEval 14,150 + LexRubric 649
   （CLaw 已退出论文，不在黑名单）→ **精确命中 0 / 近似命中 18,587 处 → 剔除 15,007 个唯一 uid**。
   近似命中汉明距离分布 `{0:493, 1:986, 2:2,524, 3:14,584}`；命中集中在 `DISC-Law-SFT`（−12,598 行）
   与 LexEval 案件类任务的**同案近文**样本。
2. **应用剔除**：生成清洗镜像 `data/corpus/decontaminated/`（原 `normalized/` 保持只读，
   逐文件 SHA-256 已登记，见镜像内 `APPLY_SUMMARY.json`）。**两条硬断言**：
   - `uid_unique_assert = true` —— 输入 308,767 行 / 308,767 个唯一 uid / **重复 0**；
   - `removal_identity_assert = true` —— **实际剔除 15,007 条 == 命中 15,007 个 uid**（恒等式成立）。
   - 另有 `banned_missing_uids = 0`、`stale_files_in_dst = []`（镜像目录无未登记残留）。
3. **复扫**（清洗镜像 293,760 行 / 耗时 5,716s）：**精确 0 / 近似 0 → verdict = PASS**，
   `DECONTAM_REMOVED_UIDS_PASS.txt` 为**空文件**（0 行，门禁证据）。
   **恒等式闭环**：首扫 308,767 − 复扫 293,760 = **15,007 = 剔除 uid 数**；
   逐域差额 criminal 6,038 / civil 5,096 / procedural 1,389 / general 2,484 合计亦为 15,007。

**逐文件剔除明细**（`APPLY_SUMMARY.json` 的 `per_file`，合计 removed = 15,007）：

| 文件 | kept | removed |
|---|---|---|
| `qa/Dusker__lawyer-llama.jsonl` | 17,318 | 1,117 |
| `qa/ShengbinYue__DISC-Law-SFT.jsonl` | 239,757 | **12,598** |
| `qa/Skepsun__lawyer_llama_data.jsonl` | 13,914 | 553 |
| `statutes/pandalla__chinese_law_examples.jsonl` | 984 | 16 |
| `statutes/twang2218__chinese-law-and-regulations.jsonl` | 21,787 | 723 |

**同源风险专项**：Skepsun 司考（首扫 normalized **14,467** 条 → 剔除后 **13,914** 条）
vs LexRubric `sifakaoshi`（**176** 条）——
首扫 **精确 0 / 近似 1**（汉明 3，uid `Skepsun__lawyer_llama_data__all:12431`，**已落在 15,007 剔除集内**）；
剔除后复扫 **精确 0 / 近似 0** → **风险排除**（两者虽都源自公开司考真题，但题目集不相交）。

**清洗后分域（唯一记录，排除 `_all.jsonl` 合并副本双计）**：

| 流 | 刑法 | 民法 | 程序法 | 通用 | 合计 |
|---|---|---|---|---|---|
| qa | **91,400** | **99,705** | **24,088** | **55,796** | **270,989** |
| statutes | 382 | 2,921 | 522 | 18,946 | 22,771 |

逐域剔除量：criminal 6,038 / civil 5,096 / procedural 1,389 / general 2,484（合计 15,007）。
每域仍超额（目标 3万/4万/2万/1万），10 万配比不受去污影响。

报告：`DECONTAMINATION_REPORT.md`（首扫 FAIL 证据）+ `DECONTAMINATION_REPORT_PASS.md/.json`（复扫 PASS）
+ `DECONTAM_REMOVED_UIDS.txt`（15,007 uid 剔除清单）+ `DECONTAM_REMOVED_UIDS_PASS.txt`（空，门禁证据）。

**阶段 3 的两个加固（2026-09-18，防「错误的 PASS」）**：

1. **`_all.jsonl` 新鲜度守卫**：首扫优先读合并副本（只扫一遍避免双计），但若它的 mtime
   **早于任何正式分文件**即判定陈旧、改用分文件扫描。旧实现无条件信任 `_all.jsonl`，
   复扫会读到旧产物 → **直接给出错误的 PASS**。
2. **跳过 `_` 前缀与 `*.sample.jsonl`**：合并副本与调试抽样残留一律不参与扫描/剔除，
   否则会双计（详见第十节 10.2）。

> **实证：假 PASS 不是假想。** 本轮修复前，2026-09-18 **13:18:37** 曾产出过一份同名
> `DECONTAMINATION_REPORT_PASS.json` —— 它也确实写着 `verdict = PASS`，但 `checked.total_rows = 294,937`、
> `by_file` 里**只有 `_all.jsonl`(272,166) + pandalla + twang2218**：它信任了**陈旧的合并副本**。
> 修复后复扫（**20:41:35**）改走五份正式分文件，`checked = 293,760`，与 `apply` 阶段登记的
> `kept = 293,760` **逐个数字吻合**。**凡是不能与"被查总量"对上的 PASS，都不是证据。**
> 这也是"新鲜度守卫"必须存在的直接原因（详见第十节 10.2）。

之后阶段 4 需在**已有超额数据上做分层下采样**（按 `task` × `domain` × `source` 分层，
避免某一任务型垄断某个专家），并在此之前先切出 val/test 与路由集（与专家训练集 disjoint）。

---

## 七、阶段 2b：法条切条（✅ 已完成，质检 verdict = PASS，2026-09-18）

脚本 `scripts/corpus/split_statutes.py`（幂等，只读输入），质检 `scripts/corpus/verify_statute_items.py`，
包装 `scripts/corpus/prepare_split_statutes.sh`。报告：`STATUTE_SPLIT_REPORT.md` + `statute_items_STATS.json`
（切条）+ `statute_items_VERIFY.md/.json`（质检）。

### 7.1 规模（唯一的数字口径）

**源 22,771 条整部法规 → 产物 65,037 条检索单元**（本地 `data/corpus/statute_items/`）：

| 项 | 数量 | 说明 |
|---|---|---|
| 源文档（文档级） | 21,787 | 其中白名单内 **1,810 部**（法律 382 / 司法解释 740 / 行政法规 663 / 法律解释 25） |
| 逐条切出 | **64,055** | `level="item"`，条号锚点来自行首「第X条」 |
| 整篇兜底入库 | **337** | `level="doc"`，无条号结构的批复/决定整篇入库 |
| 条级源规范化 | **982** | pandalla 样例（已剥掉正文里自带的条号） |
| **合计** | **65,037** | 质检 C1–C7 全部通过 |

**分域**：general 42,834（65.9%）／civil 12,356／procedural 6,601／criminal 3,246。
`general` 占比高是**正确的**：国家级法律里行政法/经济法/社会法（立法法、海关法、食品安全法…）
本就占大头，按「民法/刑法/程序法 + 通用」的三法域约定归 general。域标签走**法规名关键词规则**
（`domain_source="statute_title_keyword"`），与阶段 2 QA 侧的六路加权打分**是两套东西，不得混用**。

### 7.2 过滤口径（`--types`，默认白名单）

只留**国家级规范**：`法律 / 法律解释 / 司法解释 / 行政法规`。处置明细：

| 丢弃类型 | 部数 | 理由 |
|---|---|---|
| 地方性法规 | 19,152 | 占源 87%，是「某地的规则」，会污染全国性问答检索 |
| 修改、废止的决定 | 649 | 修改指令，非规范条文 |
| 有关法律问题和重大问题的决定 | 169 | 同上（且 status 字段错位，见 7.3） |
| **宪法** | **6** | ⚠️ **决策点**：宪法条文极少被作为裁判依据引用，且修正案文本是「将…修改为…」的**修改指令**。如要入库：`--types 法律,法律解释,司法解释,行政法规,宪法` |
| 监察法规 | 1 | 非三法域对象 |

`AMENDMENT_INSTRUCTION`（刑法修正案(九)(十) 等 9 部）**一律不进库**：正文用「一、二、」编号，
是修改指令而非规范条文，硬拆会造出「假法条」；合并后的正式法本已在库内。

### 7.3 status 脏值（`status="7"`）

全量来源 **812 部**带 `status="7"`，交叉分布：修改废止的决定 635 / 重大问题决定 169 / 地方性法规 8。
→ **落在白名单类型上的 = 0**：脏值被 7.2 的类型过滤**顺带清干净**，向量库里 status 100% 合法
（有效 1,385 / 已修改 256 / 已废止 168 / 尚未生效 1）。脚本仍带规范化兜底（未知值 → 「未知」+ 留 `status_raw`）。

### 7.4 三个必须留痕的实测坑

1. **锚点必须盯行首**：正文行首统一带一个半角空格（表意空格 0 处）。全文正则命中 970,278 次 vs
   行首唯一锚点 871,906 次 → 差额约 9.8 万是**交叉引用**（「依据本法第三十二条」）。
   若在文中间找「第X条」，切条会碎成一地。
2. **目录块会吞掉正文章节**：318 部带「目 录」。目录列完章节后**正文重新从「第一章」开始**，
   若不识别这个「重启点」，正文首章会被目录模式吞掉 —— 实测导致全法章节归属丢失
   （修复后带章标记占比 41.7% → **46.0%**，`行政处罚法` 第一条 章节从空变为「第一章 总则」）。
   实现：目录内记录标题序列，出现**重复标题**即判定正文重启。
3. **无条号文书不能硬拆**：337 部（司法解释批复 279 + 行政法规 24 + 法律解释 11 + …）用
   「一、二、」或纯叙述组织，没有条号锚点。**整篇入库**（`level="doc"`）比硬拆更正确 ——
   这类文书通常仅数百字，本身就是自洽的检索单元。入库时剥掉开头标题重复与发布信息块
   （累计 12,849 字，只剥元信息，不改正文）。

### 7.5 质检门槛（`verify_statute_items.py`）

| 检查 | 结果 |
|---|---|
| C1 必需字段 / uid 唯一 | 无缺失；重复 0（切条阶段已按 uid 去重并留痕 75 条样例） |
| C2 `text_full` 与 sha1 一致性 | 0 处不一致 |
| C3 正文残留行首条号 / 目录块 | **0 / 0** |
| C4 过短碎片（<12 字） | 47 条（0.07%），全部是「本法自公布之日起施行。」这类合法的附则条文 |
| C5 章节归属 | 46.0% 条目带章标记 |
| C6 三法域非空 | civil/criminal/procedural 均有量，抽样可读 |
| **verdict** | **PASS** |

### 7.6 字段口径（对接阶段 4 / 向量库)

`uid` = `<dataset>__<file_stem>:<source_index>#<条号>`（整篇入库为 `…#__doc__`）；
**与阶段 2 的 `uid`、阶段 4 的 `uid_g` 完全同口径** —— 唯一性由构造成立，质检 C1 断言。
（旧实现为 `<source_id>#<条号>`，`source_id` 只是**文档在文件内的序号**，漏了「源文件」这一维；
当前 twang2218 只有 1 个 parquet 故未暴露，但换分片 parquet 或跨数据集同号会**静默撞号**。）
`text` = 条文正文（**不含条号**，与文档级源对齐）；`text_full` = 「法规名 + 条号 + 正文」（**用于 embedding**）；
`level` ∈ {item, doc}；`split_mode` 说明来源；`status` / `effective_from` / `effective_period` 承载
**历史版本链**（同一条号的多版本各自保留，靠 uid 区分，不删）。

**后续（均已完成）**：阶段 2c 用程序法条文派生题型补量（见第八节）；
阶段 4 用 qc **270,989** + 派生 **13,484** 做分层下采样与切分（见第九节）；
法条 **65,037** 条不进 SFT，做向量库主料（见 9.5）。

---

## 八、阶段 2c：程序法条文任务派生（✅ 已完成，verdict = PASS，2026-09-18）

脚本 `scripts/corpus/derive_statute_tasks.py`（只读输入、无随机数；时间戳用 `--stamp` 固定，见 8.4），
包装 `scripts/corpus/prepare_derive_statute_tasks.sh`。
报告：`DERIVE_REPORT.md` + `DERIVE_STATS.json`。

**为什么要它**：阶段 2b 切出的 6,601 条程序法条文**只是检索语料**（向量库主料），不进 SFT；
而程序法 SFT 侧的题型**高度单一** —— 不做派生时 `legal_question_answering` 独占 **49.3%**，
迫使阶段 4 放宽 35% 软上限（见 9.3 与 10.3）。README 3.1 已把「诉讼法条文任务」列为程序法来源，
本节即把那批条文**逐字改写成任务型样本**。

### 8.1 规模（唯一的数字口径）

| 项 | 数量 | 说明 |
|---|---|---|
| 输入检索单元 | 65,037 | 阶段 2b 产物（只读） |
| 其中程序法 `level=item` 候选 | **6,529** | 来自 **116 部**程序法 |
| `statute_recall`（法名+条号 → 正文） | **5,605** | 答案是条文正文，逐字取自原文 |
| `statute_locate`（正文 → 法名+条号） | **5,605** | 答案是「《法名》第X条」 |
| `statute_structure`（法名+条号 → 编/章/节） | **2,274** | 该条无编章节结构时不可用（丢弃 4,176） |
| **合计** | **13,484** | `unique_content_sha1` / `unique_uid` 均 13,484 |
| 被丢弃 | 4,176 / 862 / 862 / 79 / 62 / 62 | 结构不可用 / 内容重复 / 标题过短 |

### 8.2 三条硬原则
1. **答案逐字取自原文**：`verbatim_check` 逐条回查 —— 13,484 / 13,484 通过，**不符 0**；
2. **只派生程序法**：`domain` 一律 `procedural`，不污染其他域（不引入跨域假标签）；
3. **一律 `synthetic=true` + `derived=true`**：阶段 4 断言它们**只进 train**，
   val/test/router 一律只用真实数据（防评测集被合成样本污染）。

### 8.3 两个实测坑（都已写成硬断言）
- **`law_title` 自带书名号 → 生成 `《《…》》`**：源数据的 `law_title` 已是
  `《中华人民共和国刑事诉讼法(2018修正)》`，模板再写 `"《%s》" % law` 就套成双层。
  双层书名号会当作**格式噪声**学进 SFT。修复：渲染处统一 `unwrap_title()` 剥壳
  （长度过滤仍用原始标题，不改既有 drop 口径），并新增 `title_wrap_check` 硬断言
  —— 实测 **0 条**异常，异常即 `verdict = FAIL`。
- **跨法规逐字相同的条文**：`dup_content` 丢弃 862 / 862 / 79 条（附则模板句一类）。

### 8.4 确定性的**边界**（必须说清，否则「可复现」是句空话）

脚本**不含随机数**：输入扫描顺序（文件名排序）、派生顺序、丢弃顺序全部确定 →
相同输入必然得到**相同的样本集合、相同的 uid、相同的 `content_sha1`、相同的行序**。

但产物的 **SHA-256 不是天然稳定的**：每条记录写 `normalized_at`（= 运行时刻），
报告写 `generated_at`，二者同源。**这是唯一的非确定性来源。**
→ 要逐字节复现产物：`--stamp 2026-09-18T19:27:41`（传与上次相同的值）。
实测：无 `--stamp` 重跑得 `cf7dafe0…`，带同一 `--stamp` 重跑精确回到 `3b11ae6b…`。
`--stamp` 的值也会写进 `DERIVE_STATS.json` 的 `config.stamp`，便于审计。

**为什么这不会让阶段 4 失效**：`content_sha1 = sha1(instruction, input, output)`，不含时间戳
→ 阶段 4 的样本键与配额分配完全不受时间戳影响。实测对照：train 里的 3,000 条派生记录
与新 2c 产物**逐字段一致**，仅 `normalized_at` 与阶段 4 追加的 `split`/`split_stage`/`uid_g` 不同。

### 8.5 与阶段 4 的接口
- 阶段 4 读 `data/corpus/derived/`，把本流视为独立 source（`derived/statute-items`）；
- **派生份额上限 `--derived-max-share`（默认 0.15）按「域内配额」生效**：
  程序法域 20,000 × 15% = **3,000 条**上限，实测正好取满 3,000（其余 10,484 条不启用）。
  该上限的目的是防「程序法专家只会背法条」；
- 派生样本**不进** `data/corpus/statute_items/` 检索语料，也不参与评测。

---

## 九、阶段 4：分层下采样 + 切分（✅ 已完成，质检 verdict = PASS，2026-09-18）

脚本 `scripts/corpus/downsample_split.py`（确定性、幂等、输入只读），质检门禁 `scripts/corpus/verify_split.py`，
包装 `scripts/corpus/prepare_downsample_split.sh`。
报告：`SPLIT_REPORT.md` + `SPLIT_STATS.json`（切分）+ `SPLIT_VERIFY.md/.json`（质检）。

### 9.1 规模（唯一的数字口径）

| split | 文件 | 条数 |
|---|---|---|
| 训练集（A0 全域） | `data/train/train.jsonl` | **100,000** |
| 训练集（分域视图） | `data/train/{civil,criminal,procedure,general}.jsonl` | 40,000 / 30,000 / 20,000 / 10,000 |
| 验证集 | `data/dev/dev.jsonl`（+ 逐域视图） | **1,000** |
| 测试集 | `data/test/test.jsonl`（+ 逐域视图） | **1,000** |
| 路由集 | `data/router/router_train.jsonl` | **20,000** |
| **合计占用** | | **122,000**（`uid_g` 唯一 122,000；池剩余 162,473，可回溯） |

输入池 **284,473 条** = 真实 QA **270,989** + 派生 **13,484**（阶段 2c）；真实部分 `content_sha1`
**全局唯一、0 重复**，且 `uid_collision_records = 0`（阶段 2 的 uid 修复已生效，见第十节）。
逐域达成率 **100%**（30,000 / 40,000 / 20,000 / 10,000 **精确命中，不是「接近」**）；
val/test 按目标配比分层（criminal 300 / civil 400 / procedural 200 / general 100），保证程序法有足够评测样本。

**产物直接落在 `configs/` 写死的路径上，配置零改动**：
`qlora_unified.yaml` 的 `data/train/train.jsonl` + `data/dev/dev.jsonl`、
`adapters_router.yaml` 的 `data/train/{civil,criminal,procedural→procedure}.jsonl`。

### 9.2 切分口径（写死）
1. 样本唯一键 = `content_sha1`（实测 284,473 条**全局唯一**，0 重复）
2. 四份 split 全部用 **sha1 排序确定性抽取，不使用随机数**，且**不向记录写入任何时间戳**
   → **实测验证（2026-09-18）**：同一份输入重跑阶段 4，**16 个产物文件 SHA-256 全部不变**
   （`REPRO_RESULT = IDENTICAL`，含 train/val/test/router 主文件与全部逐域视图）
3. 顺序：先切 router（20,000）→ 再切 val/test（按域分层）→ 余下才是训练池 → 下采样到 100,000
4. **硬卡口**：四份两两不相交，`uid_g` 与 `content_sha1` 双口径断言 = 0（质检 C3 —— 六对组合**全部为 0**）
5. 报告侧由 **C11** 把 16 个产物的 SHA-256 与 `SPLIT_STATS.json` 的声明逐一对账（`checked = 16`）

**为什么锚点是「先切 router 再下采样」而不是「从 10 万里抽 20%」**：
后者会把训练集从 10 万削到 8 万，与论文声称的 10 万训练量不符。池子有 28.4 万，
「保证 10 万训练集」和「路由集与训练集 disjoint」两件事不必二选一 —— 所以先预留再下采样。

### 9.3 分层下采样口径（三层约束，逐级放宽 + 每级留痕）
- 分层：`domain × task × source_dataset`
- 域内任务配额：`weight = count^alpha`（默认 alpha=1，标准等比分层）+ 地板（每任务 ≥100 条）
- **任务份额软上限 35%**，防单一任务型垄断某个专家
- **派生组份额上限 15%**（按**域内配额**生效），防合成样本喧宾夺主
- 约束不可行时**逐级放宽并留痕**（`cap_relaxed` / `derived_cap_relaxed`），
  **绝不静默改口径、绝不静默丢数据**

**实测：四个域全部 `cap_relaxed = False`、`derived_cap_relaxed = False`** —— 三层约束全部满足：

| 域 | 目标 | 达成 | 训练池 | 池利用率 | 头号任务份额 | 派生 | `cap_relaxed` |
|---|---|---|---|---|---|---|---|
| criminal | 30,000 | **30,000** | 84,013 | 35.7% | 31.5%（judgement_predit） | 0 | False |
| civil | 40,000 | **40,000** | 91,504 | 43.7% | 35.0%（legal_question_answering） | 0 | False |
| procedural | 20,000 | **20,000** | 35,464 | 56.4% | **35.0%**（legal_question_answering） | **3,000（15.0%）** | False |
| general | 10,000 | **10,000** | 51,492 | 19.4% | 35.0%（legal_question_answering） | 0 | False |

★ **把程序法派生做掉之后，该域的 35% 软上限不再需要放宽**（对照实测）：

| | 不做派生 | 接入阶段 2c 派生 |
|---|---|---|
| 程序法训练池 | 21,980（=目标的 1.10 倍） | **35,464** |
| 池利用率 | 91.0% | **56.4%** |
| 头号任务份额 | `legal_question_answering` **49.3%** | **35.0%** |
| `cap_relaxed` | **true**（数学上不可达） | **false** |

→ **派生不只是「补量」，它把 35% 软上限从「数学上不可达」变回「成立」。**
程序法新增三个题型进 train：`statute_recall` 1,227 / `statute_locate` 1,228 / `statute_structure` 545。

- 训练集来源分布：DISC-Law-SFT 85,095（85.10%）/ Skepsun 6,048（6.05%）/ Dusker 5,857（5.86%）
  / **derived/statute-items 3,000（3.00%）**

### 9.4 路由集
- 20,000 条，每条带 `router_label`（多标签 `domains`）与 `router_query`（user 侧问题文本）
- 域分布 civil 7,401 / criminal 6,787 / general 4,104 / procedural 1,708；跨域多标签 **5.43%**
- ⚠️ 阶段 7 所需的「贴近 CLaw 254 案分布的 **200–500 条人工标注**路由评测集」**必须新建**，
  不在本阶段产物内，也不能从任何已有数据集借。

### 9.5 法条流的去向（明确不进 SFT）
65,037 条法条条目**不做下采样、不进 SFT 训练集** —— 它是**检索语料（向量库主料）**，
检索侧语料越多越好，砍它没有收益只会掉召回。
跨流检查：QA 四份 split 的 `content_sha1` ∩ 法条条目 `content_sha1` = **0**（`cross_flow_overlap = 0`）。
要「法条任务」样本时**另派生**（阶段 2c，见第八节）并保持与 val/test disjoint —— 已实现。

### 9.6 三个实测坑（已写进 legal-corpus-curation skill）
1. **`uid` 必须全局唯一**：阶段 2 的 uid 构造曾漏 `source_file` → 20,947 组撞号 →
   阶段 3 按 uid 剔除时**整组连坐**。阶段 4 曾用 `uid_g` 止血，**现已从源头修复并全链路重跑**（见第十节）。
   质检 **C14** 常驻交叉验证 `uid == uid_g`（实测 **0 处不一致，122,000/122,000 行全覆盖**；
   `checked != expected` 同样判 FAIL —— 空结果不能冒充「都比过且都相等」）。
2. **任务池可用量必须按「训练池」算**，不能按全池算 —— 否则会把「池子本来就不够」误报成
   「分配不足」（实测误报程序法 9 个尾任务「样本偏少」）。
3. **域池大小是配额可行性的前提**：程序法池只有目标的 1.08–1.10 倍时，35% 软上限必然被打破
   —— 只能扩源或派生（见 9.3）。

### 9.7 一个待决策项（不属阶段 4 范围，避免静默处理）
| 项 | 现状 | 建议 |
|---|---|---|
| **通用回放（10%）** | `general` 桶 10,000 条**实质是「法律领域但域不明确的样本」**（宪法/行政法/法治理论），不等于「非法律通用指令数据」；实测 `replay=false` **全部 0 条** —— DISC-Law-SFT 内置的 Alpaca-GPT4/Firefly 通用回放**并不在已下载的 4 个文件里**（raw 目录只有 Pair / Pair-QA / Triplet / Triplet-QA） | 要真正满足「防灾难性遗忘」的设计意图，须**另采一份非法律中文通用指令数据**（阶段 0/1 动作，需先过 license A 级审查）；或保持现状并在论文如实说明「回放桶为法律领域内的通用样本」 |

> 原 8.7 的第一项「**程序法任务多样性**」（头号任务占 49.3%、软上限被迫放宽）
> **已由阶段 2c 解决**，故此处只剩回放一项 —— 决策已执行，结果见 9.3。

---

## 十、阶段 2 遗留缺陷：**已修复** + 前后对照（2026-09-18 全链路重跑验证）

> 本节原为「缺陷登记（未修）」。缺陷已按要求从源头修好，**阶段 2 → 3 → 2b → 2c → 4 全部重跑**，
> 下面的数字都是重跑后的实测值，旧值并列存档以便审计。

### 10.1 `uid` 撞号 —— 根因与修复

`scripts/corpus/normalize_corpus.py` 旧实现（阶段 2）：

```python
rec["uid"] = "%s:%s" % (raw["source_dataset"].replace("/", "__"), raw["source_id"])
```

`source_id` 是**文件内**行号（如 `legal_question_answering_1`），**不含 `source_file`** →
`DISC-Law-SFT-Pair-QA-released.jsonl` 与 `DISC-Law-SFT-Triplet-QA-released.jsonl` 撞号。

**修复**：新增 `uid_of(dataset, source_file, source_index)`，格式
`<dataset>__<file_stem>:<source_index>`（与阶段 4 的 `uid_g`、阶段 2b 的条目 uid **完全同口径**）；
并在阶段 2 收尾**当场断言唯一性**（撞号即 `return 3`，拒绝产出）。
同时登记 `uid_legacy`（旧格式）便于回溯，`content_sha1` 不变。

### 10.2 残留抽样文件 —— 根因与修复

`normalized/qa/*.sample.jsonl`（调试冒烟产物，共 3,598 行）混在正式目录里，
旧合并逻辑按「目录下所有非 `_` 开头的 `*.jsonl`」合并 → `_all.jsonl` = 288,855 行
= 正式 3 文件 285,257 + **3,598 残留**；阶段 3 首扫因此多扫了 3,598 行。

**修复**：
1. `normalize_corpus.py` 合并流**只合并本轮真正产出的文件**（以 `stats["sources"]` 登记的
   `output` 为准），其它 `.jsonl` 一律排除并**打印 stray 警告**；
2. `decontaminate.py` / `apply_decontam.py` 一律跳过 `_` 前缀与 `*.sample.jsonl`；
3. `decontaminate.py` 新增 **`_all.jsonl` 新鲜度守卫**：合并副本 mtime 早于任何正式分文件即判陈旧、
   改用分文件扫描 —— 旧实现无条件信任 `_all.jsonl`，复扫会读到旧产物**直接给出错误的 PASS**；
4. 服务器侧把过期残留**移动**（非删除）到 `data/corpus/_obsolete_20260918/`
   （6 个 `.sample.jsonl` + 1 个过期 `decontaminated/qa/_all.jsonl` + 2 个 `*.sample.json`），
   并生成 `MANIFEST.md5` 供回溯。

### 10.3 前后对照（重跑实测）

| 指标 | 旧（撞号 / 有残留） | 新（已修） |
|---|---|---|
| `normalized/qa` uid 撞号 | 23,218 组（每组 2 条） | **0**（`uid_unique.duplicates = 0`） |
| `normalized/qa/_all.jsonl` 行数 | 288,855（含 3,598 残留） | **285,257**（仅 3 个正式文件，`stray_not_merged = []`） |
| 阶段 3 首扫行数 | 312,365 | **308,767** |
| 近似命中**处数** | 18,778 | 18,587（**−191，恰好对应不再扫的 3,598 行残留**） |
| 其中 `_all.jsonl` | 18,028 | 17,837 |
| 汉明分布 `hamming=0` | 493 | **493（完全一致 → 检出核心未被改动）** |
| 剔除清单 uid | 14,957 | **15,007**（+50：原先被撞号掩盖的条数被正确识别） |
| **实际剔除行数** | **17,228**（= 14,957 + **2,271 连坐**） | **15,007**（恒等式成立） |
| 清洗镜像 qa | 268,768 | **270,989（+2,221）** |
| ↳ 其中 DISC-Law-SFT | 237,536 | **239,757（+2,221）** |
| ↳ Dusker / Skepsun | 17,318 / 13,914 | 17,318 / 13,914（**不变**） |
| 清洗镜像 statutes | 22,771 | 22,771（**不变**） |
| 可断言的硬不变量 | —（撞号下恒等式成了伪命题） | `uid_unique_assert = true`、`removal_identity_assert = true` |

**根因自证**：多删的 2,221 条**全部集中在 `DISC-Law-SFT` 一个文件** —— 正是发生 uid 撞号的
那个数据集；Dusker 与 Skepsun 一条不差，法条流一条不差。这与根因完全自洽。

**内容侧零变化（关键对照）**：`qa_domain_total`（civil 104,691 / criminal 97,414 /
general 57,707 / procedural 25,445）与各源 `written`（252,355 / 18,435 / 14,467 / 22,510 / 1,000）
**与旧版逐项相同**，`normalized_DEDUP_REPORT.json` 的去重样例也逐条相同
→ 证明本次改动**只影响 uid 生成，不影响语料内容**。

**连带加固（同源缺陷，主动预防）**：阶段 2b 的条目 uid 旧实现是 `<source_id>#<条号>`，
`source_id` 同样只是文件内文档序号 —— 当前 twang2218 只有 1 个 parquet 所以**未暴露**，
但换分片 parquet 或跨数据集同号会**静默撞号**。已改为
`<dataset>__<file_stem>:<source_index>#<条号>`，唯一性由构造成立并由质检 C1 断言。

### 10.4 旧结论该怎么改

旧登记的结论是「方向是多删（保守），不可能漏删 → **无污染风险**，阶段 3 的 PASS 依然成立」。
**前半句成立**（多删确实不会造成评测集泄漏），**但结论不充分**：

1. 17,228 条里 **2,271 条是误删**（真实训练样本被连带删除）——不是「无影响」，只是恰好
   对 10 万配比无影响（LQA 本就超额）；
2. 更要紧的是**当轮无法自证**：`records_under_removed_uids == actual_removed_records` 这个恒等式
   在撞号下会「自动成立」（因为一条 uid 对应多条记录），**它并没有校验到任何东西**。
   修复后 `removed == len(hit_uids)` 才成为一条**真正有鉴别力的不变量**，并被常驻断言。

### 10.5 修复的净收益

- **找回 2,221 条被误删的真实样本**（全部在 DISC-Law-SFT）；
- 剔除清单**更完整**（+50 个原本被撞号掩盖的 uid）；
- 两条恒等式断言成为**常驻不变量**，此后任何 uid 退化都会当场失败而非静默污染；
- 去污证据**可审计**：不再出现「扫描了 3,598 行调试残留」这类口径污染；
- 连带效应：重跑暴露并验证了程序法 35% 软上限的恢复（见 9.3）。

