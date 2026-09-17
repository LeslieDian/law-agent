# 评测基准登记（LexRubric / LexEval）

> **本目录只登记"外部评测基准"，与 `docs/data_manifest.json` 中的 CLaw 语料固化清单完全分离。**
> 两个基准各自独立成一份 MANIFEST / VERIFY_REPORT，**任何情况下都不得合并哈希、合并题目、合并评分口径**。
> 登记时间：2026-09-17，登记机器：`192.168.195.61`（2×A800 80GB），落盘位置 `/mnt/data/lidian/law-agent/data/benchmark/`。

> ⚠️ **服务器路径大小写提醒**：权威目录是 **`data/benchmark/`（单数）**。
> 早期用 `git clone` 那条慢路试拉时曾产生 `data/benchmarks/`（复数），已停止该脚本
> 并于 2026-09-17 移入 `data/_obsolete_20260917_benchmarks_plural/`（逐文件 md5 已核对一致，可随时删）。
> **读取数据一律走 `data/benchmark/`，不要用复数路径。**

## 为什么引入这两个基准

原选用的 LawBench 已老化（对比模型停留在 GPT-3 / GPT-4 时代），缺乏当代可比基线；
CLaw 又无法自行测评（无官方评测脚本开放）。因此按三条标准重筛：

1. **足够新** —— 2024 年以后发布，含当代模型成绩；
2. **可自评** —— 评测脚本公开、数据可复现、需要 API key 的环节可控；
3. **已有公开基线** —— 论文/仓库给出多家模型的公布成绩，可直接横向对照。

最终选定 **LexRubric**（论文级 rubric 开放式评测）与 **LexEval**（任务级多认知层次评测），
两者定位互补，但**互为独立实验，结果不得混排**。

---

## 一、LexRubric（开放式 rubric 评测）

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/foggpoy/LexRubric` |
| commit | `9141ee4b3715834ac7e1507cbfc225cf858f7328`（2026-08-29） |
| 许可 | MIT（`repo/LICENSE`） |
| 论文 | arXiv:2606.09389，EMNLP 2026 Main |
| 服务器路径 | `/mnt/data/lidian/law-agent/data/benchmark/lexrubric/` |
| 规模 | 2 个 split：法律咨询 `falvzixun` 473 条 + 司法考试 `sifakaoshi` 176 条 = **649 条** |
| rubric 总数 | **12,335** 条（论文自述 12,337，差 −2，见下） |
| 每条 rubric 字段 | `criterion` / `point` / `dimension` |
| 校验结论 | **PASS**（0 失败，2 警告） |
| 清单 | `lexrubric.MANIFEST.json`（9 个文件，12.542 MiB） |
| 校验报告 | `lexrubric.VERIFY_REPORT.json` |

### 使用要点与已知坑

- **题目字段名是大写 `ID`**；README 示例里写的 `id` 不可信。rubric 分值是单数 `point`，README 写的 `points` 同样不可信。**一律以实测为准。**
- **两个 split 存在跨 split ID 撞号**：`111`、`195` 在法律咨询与司法考试中同时出现。
  → **评测键必须写成 `(split, ID)`**，任何"按裸 ID 建索引"的实现都会静默丢数据。
- **评分方向规则**（沿用上游 `eval.py`）：`point >= 0 → direction="不是"`，`point < 0 → direction="是"`；
  `total_score = Σ(point if criteria_met)`。负分项共 **612** 条，正分项 11,723 条，无 0 分项。
- **偏差 −2 的定位**：仅出现在「法律准确性」维度（实测 3945 vs 论文 3947），其余五个维度（逻辑推理与分析 4151、全面性 1505、表达与结构 593、指令/题意遵从 1641、伦理安全 500）**完全一致**。
  推断论文统计含未公开的私有 split，公开仓缺 2 条。**论文中需如实说明该差异，不得声称"与论文完全一致"。**
- `.DS_Store` 按规则从清单中排除（已记录于 `summary.excluded_by_rule`）。
- 双盲判分改造见 `src/evaluation/lexrubric_doubleblind.py`（保留上游 `prompt.txt` 原文与 `criteria_met` 判定，仅把单一 judge 换成 Gemini + DeepSeek 双盲）。

---

## 二、LexEval（多任务 / 多认知层次评测）

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/CSHaitao/LexEval` |
| commit | `044c695f62894deef41ad9a30797e1da0404945c`（2024-10-30） |
| 许可 | MIT |
| 论文 | arXiv:2409.20288，NeurIPS 2024（Datasets & Benchmarks） |
| 服务器路径 | `/mnt/data/lidian/law-agent/data/benchmark/lexeval/` |
| 规模 | 23 个任务 / **14,150 题**，覆盖 6 个认知层次 |
| 题目字段 | `instruction` / `input` / `answer` |
| 校验结论 | **PASS**（0 失败，0 警告，14,150 与论文**完全一致**） |
| 清单 | `lexeval.MANIFEST.json`（65 个文件，37.137 MiB） |
| 校验报告 | `lexeval.VERIFY_REPORT.json` |

### 使用要点与已知坑

- **`data/*.json` 实际是 JSONL**（每行一个对象），扩展名具有误导性。读取必须"先试 JSON，失败回退逐行解析"。
- **刻意排除 `model_output/`**（占整仓 3.1 GB 的绝大部分）。本次仅取 `data/` + `code/`；如需对齐论文给出的各家模型成绩，需另行按需下载，**不得默认拉全量**。
- 23 个任务编号与题量（`1_1`…`6_3`）已逐文件核对，明细见 `lexeval.VERIFY_REPORT.json`。
- LexEval 任务类型包含客观题与生成题；**生成类题目的自动评分口径与 LexRubric 的 rubric 口径不同，两者成绩不可相互换算或并列汇报**。

---

## 三、隔离约束（硬性）

1. 两个基准**均属评测集，禁止进入训练集 / 验证集**（与 CLaw 254 案例同等级别的隔离要求）。
2. 内部验证集只用于选 checkpoint；**选 checkpoint 一律不得使用 LexRubric / LexEval 的任何分割**。
3. 论文中两个基准的成绩必须**分节、分表**呈现，各自注明：评测集版本（commit）、题量口径、判分方式（单 judge / 双盲）。
4. 任何"把两者合起来算一个平均分"的做法都是错误的，禁止。

## 四、文件命名规范

本目录下每个基准一份清单 + 一份报告，文件名前缀即数据集名：

```
docs/benchmarks/
├── README.md                       # 本文件
├── lexrubric.MANIFEST.json         # LexRubric 逐文件 SHA-256 清单
├── lexrubric.VERIFY_REPORT.json    # LexRubric 结构与完整性校验
├── lexeval.MANIFEST.json           # LexEval 逐文件 SHA-256 清单
└── lexeval.VERIFY_REPORT.json      # LexEval 结构与完整性校验
```

新增基准时沿用同一命名：`<dataset>.MANIFEST.json` + `<dataset>.VERIFY_REPORT.json`，
**不合并进任何既有文件**。

## 五、评测里的两个角色（务必分清，别混为一谈）

一次基准评测有**两个完全不同的角色**，运行位置也不同：

| 角色 | 是谁 | 跑在哪 | 说明 |
|---|---|---|---|
| **被评方** | 我们微调/部署的模型，即 `Qwen3-8B`（+ 后续 LoRA 适配器） | **服务器本地**（`models/Qwen3-8B`，16 GB，4bit 常驻 5.66 GB） | 无需任何外部 API；以本地 OpenAI 兼容端点形式暴露给评测脚本 |
| **判分器** | `Gemini-2.5-Pro` + `DeepSeek-R1` | **外部 API** | 权重不公开 / 规模远超本地可部署范围，只能调 API |

- 上游 LexRubric `eval.py` 里的 "API 或本地端点" 分支，指的是**被评方**怎么接进来 —— 我们走本地端点。
- 判分这一侧是我们自己写的双盲层（`src/evaluation/judge/`），与上游的单 judge 实现不同。
- **判分器无法本地部署**，这是「必须走 API」的唯一原因，**与"服务器上有没有部署模型"无关**。
- 若坚持全本地判分，只能用本地模型充当裁判（例如本地起 `DeepSeek-R1-Distill-32B`），
  但**同族模型当裁判存在系统性偏好**，会削弱双盲设计的意义 → 需在论文的局限性一节显式说明。
