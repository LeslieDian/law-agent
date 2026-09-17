# 训练语料登记（阶段 1）

> **本目录只登记「训练语料」，与另外两处登记完全分离，任何情况下不得互相混排：**
>
> | 登记处 | 管什么 | 能否进训练 |
> |---|---|---|
> | `docs/data_manifest.json` | CLaw 语料固化清单（306 部 / 64,849 条 / 254 案） | ❌ 绝不可 |
> | `docs/benchmarks/` | 评测基准（LexRubric / LexEval） | ❌ 绝不可 |
> | **`docs/corpus/`（本目录）** | **训练语料（阶段 1 采集的原始源）** | ✅ 是训练来源（仍须过阶段 3 去污） |
>
> 登记时间：2026-09-17　登记机器：`192.168.195.61`（2×A800 80GB）
> 服务器落盘位置：`/mnt/data/lidian/law-agent/data/corpus/raw/`

---

## 一、本次采集结果（阶段 1）

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
├── README.md                  # 本文件
├── MANIFEST.json              # 阶段 1 总清单：逐文件 SHA-256 + 每数据集链路溯源
└── sources/
    ├── ShengbinYue__DISC-Law-SFT.source.json          # 数据集 id 里的 / 替换为 __
    ├── Dusker__lawyer-llama.source.json
    └── ...
```

新增数据集沿用同一命名 `<id 中的 / 换成 __>.source.json`，**不合并进任何既有文件**。

---

## 四、采集阶段已踩过的坑（可直接复用）

1. **hf-mirror 对「不存在的仓库」返回 `HTTP 401` 而不是 `404`** —— 按记忆猜 ID 探测会把不存在的
   仓库误判成「无权访问」。必须**先用搜索接口反查真实 ID**。
2. **HF 搜索接口对中文关键词完全无效**（`法律`/`刑法`/`判决`/`法条` 均返回 0 条）。
3. **不要按 `usage` 里的 `case_corpus` 标签一刀切排除** —— `Dusker/lawyer-llama` 的 usage 是
   `[expert_train, case_corpus]` 但只有 446 MB，是合法要采的。真正的超大语料交给**体积护栏**拦。
4. **`Dusker/lawyer-llama` 内含 `DISC-Law-SFT-Pair.json` / `-Triplet.json`，与主力源重复**
   —— 阶段 2 归一化时必须按内容哈希去重，否则民法/刑法域会被同一批样本注水。
5. **单连接慢、分块快**：小文件（< 32 MB）走单连接只有几百 KB/s；大文件分块并行可达
   6–14 MB/s。个别分块会卡住重试（本次有一个 89.7 MB 文件在 64.3% 停了约 2.7 分钟后完成），
   属正常重试行为，**不要中途杀进程**。
6. **gated 仓库需要 `HF_TOKEN`**，否则直接跳过并记为 `SKIPPED_GATED_NO_TOKEN`（不是失败，但会让退出码非 0）。

---

## 五、下一步（阶段 2）

按 [`docs/finetune_data_plan.md`](../finetune_data_plan.md) 第 7 节：

- 阶段 2：归一化到统一 schema + **域打标**（优先级链：答案引用法条全名 > 文书类型+案由 > id 前缀 > 分类器兜底）
- 阶段 3：**双向去污**（硬门禁，`verdict` 必须 PASS）—— 特别注意
  `Skepsun/lawyer_llama_data` 的司考题与 LexRubric 的 `sifakaoshi` split **存在同源风险**
