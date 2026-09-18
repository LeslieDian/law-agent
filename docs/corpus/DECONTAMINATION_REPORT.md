# 阶段 3：双向去污报告（DECONTAMINATION REPORT）
> 生成时间：2026-09-18T19:05:38　耗时 6041.6s
> **verdict = `FAIL`**
## 0. 结论
- ⚠️ 发现 **精确命中 0 条 / 近似命中 18587 条**，必须剔除后才能进训练。

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
- 合计 **308767** 行；按域 {'criminal': 97820, 'civil': 107722, 'procedural': 25999, 'general': 77226}
  - `_all.jsonl`：285257
  - `pandalla__chinese_law_examples.jsonl`：1000
  - `twang2218__chinese-law-and-regulations.jsonl`：22510

## 4. 命中
- 精确命中：**0** 条
- 近似命中（汉明 <= 3）：**18587** 个事件（同一行 question/full 两侧各算一次），涉及唯一训练样本 **15007** 条
  - 按基准：{'lexeval': 18059, 'lexrubric_zixun': 458, 'lexrubric_sikao': 70}
  - 按训练文件：{'_all.jsonl': 17837, 'pandalla__chinese_law_examples.jsonl': 16, 'twang2218__chinese-law-and-regulations.jsonl': 734}
  - 按域：{'criminal': 7049, 'civil': 7105, 'procedural': 1654, 'general': 2779}
  - 汉明距离分布：{0: 493, 1: 986, 2: 2524, 3: 14584}
- 剔除清单（唯一 uid）：`DECONTAM_REMOVED_UIDS.txt`（15007 条）

## 5. 同源风险专项：Skepsun 司考 vs LexRubric sifakaoshi
- 检查 Skepsun 训练样本 **14467** 条
- 精确命中 **0** 条 / 近似命中 **1** 条
- 样例（前 20）：
  - `{"uid": "Skepsun__lawyer_llama_data__all:12431", "type": "near", "hamming": 3, "bench_idx": 143, "train_head": "您好 我2016年5月注册了公司,2017年11月才办理国税,地税一"}`

## 6. 处置
1. 命中的训练样本一律**从训练集剔除**（脚本按 `train_uid` 生成黑名单）。
2. 剔除后**重跑本脚本**，verdict 必须变为 `PASS` 才允许进入阶段 4。
3. CLaw 254 案自建完成后**必须再跑一次**，否则论文里不能声称已做完整去污。
