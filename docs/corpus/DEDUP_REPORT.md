# 训练语料去重报告（阶段 2 前置）

> 机器可读版：服务器 `data/corpus/normalized/DEDUP_REPORT.json`（由 `normalize_corpus.py` 产出）
> 核查脚本：`scripts/corpus/probe_dup.py` / `probe_dup2.py` / `probe_dup3.py`（只读，可重跑复核）
> 问题来源：`Dusker/lawyer-llama` 内含 `DISC-Law-SFT-Pair.json` / `DISC-Law-SFT-Triplet.json`，
> 文件名与主力源 `ShengbinYue/DISC-Law-SFT` 高度相似，怀疑是同一份数据的两种发布。
> 核查日期：2026-09-17，机器 `192.168.195.61`

## 一、为什么要用三种指纹

两个来源**内容相同但字段布局不同**，只看一种指纹会得出错误结论：

| 指纹 | 定义 | 对什么免疫 |
|---|---|---|
| **严格** | `sha1(归一化 input + output)` | 什么都不免疫（字段布局一变就判不同） |
| **宽松** | `sha1(input 尾 300 字 + output 头 160 字)` | 指令前缀被塞进 `input` vs 放在 `system` 的差异 |
| **案件正文** | `sha1(input 尾 400 字)` | 同上，且完全不受答案改写影响 |

**实测到的布局差异**：`Dusker/lawyer-llama` 把指令放在独立的 `system` / `instruction` 字段，
`ShengbinYue/DISC-Law-SFT` 把同一句话**拼进 `input` 头部**（`基于下列案件，推测可能的判决结果。\n<案件原文>`）。
→ 严格指纹会把同一份数据判成"完全不同"（交集 0），必须用宽松/案件正文指纹才能看到真相。

## 二、实测结果

### 2.1 严格指纹（首次核查，**结论不可用**）

| | records | unique |
|---|---|---|
| Dusker-Pair | 166,758 | 161,854 |
| ShengbinYue-Pair | 166,758 | 161,854 |
| Dusker-Triplet | 16,000 | 16,000 |
| ShengbinYue-Triplet | 16,000 | 16,000 |

- `Dusker-Pair ∩ SB-Pair = 161,854`（100% / 100%，两侧独有均为 0）→ 完全重复
- `Dusker-Triplet ∩ SB-Triplet = 0`（0%）→ **与宽松指纹的结论矛盾，说明指纹选错了**

### 2.2 宽松指纹

| 对 | 交集 | 仅左 | 仅右 | 左覆盖率 |
|---|---|---|---|---|
| Dusker-Pair vs SB-Pair | 155,401 | 0 | 0 | **100.0%** |
| Dusker-Triplet vs SB-Triplet | 5,443 | 10,557 | 10,557 | 34.0% |
| Dusker-Triplet vs SB-TripletQA | 0 | 16,000 | 23,331 | 0% |

### 2.3 案件正文指纹（决定性）

| | records | unique 案件正文 |
|---|---|---|
| Dusker-Pair | 166,758 | 153,231 |
| SB-Pair | 166,758 | 153,231 |
| Dusker-Triplet | 16,000 | 15,954 |
| SB-Triplet | 16,000 | 15,993 |
| SB-TripletQA | 23,331 | 23,328 |

| 对 | 交集 | 仅左 | 仅右 | 左覆盖率 |
|---|---|---|---|---|
| **Dusker-Pair vs SB-Pair** | **153,231** | **0** | **0** | **100.0%** |
| **Dusker-Triplet vs SB-Triplet** | **3,270** | **12,684** | **12,723** | **20.5%** |
| Dusker-Triplet vs SB-TripletQA | 0 | 15,954 | 23,328 | 0.0% |

> 注：严格指纹下 161,854 个唯一值 > 案件正文指纹下 153,231 个，差值 8,623 是
> **同一案件不同任务/不同提问方式**的样本（例如同一份判决书既做摘要又做要素抽取），属于合法重复而非数据错误。

## 三、判定

| 文件 | 判定 | 依据 |
|---|---|---|
| `Dusker/lawyer-llama/DISC-Law-SFT-Pair.json`（360.8 MiB / 166,758 条） | **丢弃** | 案件正文指纹 100% 覆盖（153,231/153,231，两侧独有 0），主力源有完整副本 |
| `Dusker/lawyer-llama/DISC-Law-SFT-Triplet.json`（52.5 MiB / 16,000 条） | **保留** | 仅 20.5% 重合，另有 12,684 条案件正文为本来源独有 —— 丢掉会白损约 1.27 万条 |
| `Dusker/lawyer-llama/fakao_gpt4.json`（1,000） | 保留 | 与任何来源交集均为 0 |
| `Dusker/lawyer-llama/zixun_gpt4.json`（1,000） | 保留 | 与任何来源交集均为 0 |
| `Dusker/lawyer-llama/kg_crime_llama.json`（856） | 保留 | **唯一来源**（罪名释义知识），内部自身有重复（856 → 435 唯一宽松指纹） |

**⚠️ 一个反直觉的结论必须留痕**：Dusker 与主力源**同名不同内容**的两份 Triplet 各自 16,000 条、
任务名都叫判决预测，但只有 20.5% 是同一批案件。**"文件名相同 / 条数相同"完全不能作为判重的依据。**

## 四、连带发现（主源内部）

| 对 | 交集 | 说明 |
|---|---|---|
| SB-Pair ∩ SB-PairQA | 431 | 两条流之间有 431 条完全同题 |
| SB-Triplet ∩ SB-TripletQA | 0 | — |
| SB-Pair ∩ SB-Triplet | 0 | — |

→ 431 条重叠由 `normalize_corpus.py` 的**全局内容哈希去重**在归一化阶段自动消除
（先处理的来源保留，后处理的丢弃并计入 `DEDUP_REPORT.json`）。

## 五、落地方式

- **不物理删除 raw 文件**：遵守「数据只读 + 落盘即登记」硬约定，`raw/` 是登记过的只读区，
  360 MB 在 5.1 TB 可用空间里可忽略。丢弃动作在**管线层**声明（`configs/corpus_adapters.yaml`
  的 `drop_files`），可审计、可回滚。
- 脚本每次运行都会打印 `[DROP] <file> ← DUPLICATE_CONFIRMED`，并把 `drop_files` 与
  `keep_files` 的**理由**一并写进 `STATS.json`，避免以后有人看到 raw 里有文件却不在管线里而困惑。
