# 阶段 3：双向去污报告（DECONTAMINATION REPORT）
> 生成时间：2026-09-19T15:00:10　耗时 5747.0s
> **verdict = `PASS`**
## 0. 结论
- **SFT 面向的流（qa）与已就位评测集之间未发现精确复制或近似改写。**
- ⚠️ 但法条流（检索语料）存在 **750** 个近似命中，按硬约定 4 **只记录不剔除**（见第 4b 节）—— 这是设计选择，不是漏检。

## 1. 方法
- `exact`：sha1(normalize(instruction+input)) 与 sha1(normalize(instruction+input+output))
- `near`：simhash64(char-3gram, crc32), 汉明距离 <= 3
- `lsh`：4 x 16bit 分段索引取候选，再算精确汉明距离
- `normalize`：NFKC + 去所有空白
- `max_chars_for_fingerprint`：800

## 2. 黑名单（评测集）
- 合计 **14799** 条：{'lexrubric_zixun': 473, 'lexrubric_sikao': 176, 'lexeval': 14150}
- CLaw 254 案：**缺失**（官方未公开发布，走自建路线）

## 3. 被查训练语料
- 合计 **294499** 行；按域 {'criminal': 91806, 'civil': 102736, 'procedural': 24642, 'general': 75315}
  - `Dusker__lawyer-llama.jsonl`：17318
  - `ShengbinYue__DISC-Law-SFT.jsonl`：239757
  - `Skepsun__lawyer_llama_data.jsonl`：13914
  - `pandalla__chinese_law_examples.jsonl`：1000
  - `twang2218__chinese-law-and-regulations.jsonl`：22510

## 4. 命中
- 精确命中：**0** 条
- 近似命中（汉明 <= 3）：**0** 个事件（同一行 question/full 两侧各算一次），涉及唯一训练样本 **0** 条
- 剔除清单（唯一 uid）：`DECONTAM_REMOVED_UIDS_PASS.txt`（0 条）

## 4b. 法条流近似命中（★ 只记录、不剔除）
- 策略 `--statutes-near = audit`（`audit` = 只记录不剔除，`remove` = 历史行为，`ignore` = 不算）
- 法条流近似命中事件：**750** 个，涉及整部法规 **739** 部
  - 按基准：{'lexeval': 716, 'lexrubric_zixun': 30, 'lexrubric_sikao': 4}
  - 按文件：{'pandalla__chinese_law_examples.jsonl': 16, 'twang2218__chinese-law-and-regulations.jsonl': 734}
  - 汉明距离分布：{1: 5, 2: 79, 3: 666}
- **为什么不剔除**：法条流是检索语料（向量库主料），README 3.1.3 明确「法条流不进 SFT」；其 uid 粒度为整部法规，simhash 只看正文前 800 字 → 一处近似即整部法典连坐。评测集必然引用法条原文，误报是结构性必然 ⇒ 近似命中只记录、不剔除。精确 sha1 命中仍照常剔除。
- 影响：法条流的近似命中**不进入剔除清单**，因此 `statutes/` 镜像保持完整（检索层需要完整法条库）。

## 5. 同源风险专项：Skepsun 司考 vs LexRubric sifakaoshi
- 检查 Skepsun 训练样本 **13914** 条
- 精确命中 **0** 条 / 近似命中 **0** 条

## 6. 处置
1. 命中的训练样本一律**从训练集剔除**（脚本按 `train_uid` 生成黑名单）。
2. 剔除后**重跑本脚本**，verdict 必须变为 `PASS` 才允许进入阶段 4。
3. CLaw 254 案自建完成后**必须再跑一次**，否则论文里不能声称已做完整去污。
4. ★ 法条流近似命中不剔除（硬约定 4）—— 论文里如实披露：「法条向量库作为 RAG 开卷依据保留完整法条，不与评测集做近似去污；仅做精确复制扫描」。
