# 阶段 3：双向去污报告（DECONTAMINATION REPORT）
> 生成时间：2026-09-18T20:41:35　耗时 5716.4s
> **verdict = `PASS`**
## 0. 结论
- 训练语料与已就位评测集之间**未发现精确复制或近似改写**。

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
- 合计 **293760** 行；按域 {'criminal': 91782, 'civil': 102626, 'procedural': 24610, 'general': 74742}
  - `Dusker__lawyer-llama.jsonl`：17318
  - `ShengbinYue__DISC-Law-SFT.jsonl`：239757
  - `Skepsun__lawyer_llama_data.jsonl`：13914
  - `pandalla__chinese_law_examples.jsonl`：984
  - `twang2218__chinese-law-and-regulations.jsonl`：21787

## 4. 命中
- 精确命中：**0** 条
- 近似命中（汉明 <= 3）：**0** 个事件（同一行 question/full 两侧各算一次），涉及唯一训练样本 **0** 条
- 剔除清单（唯一 uid）：`DECONTAM_REMOVED_UIDS_PASS.txt`（0 条）

## 5. 同源风险专项：Skepsun 司考 vs LexRubric sifakaoshi
- 检查 Skepsun 训练样本 **13914** 条
- 精确命中 **0** 条 / 近似命中 **0** 条

## 6. 处置
1. 命中的训练样本一律**从训练集剔除**（脚本按 `train_uid` 生成黑名单）。
2. 剔除后**重跑本脚本**，verdict 必须变为 `PASS` 才允许进入阶段 4。
3. CLaw 254 案自建完成后**必须再跑一次**，否则论文里不能声称已做完整去污。
