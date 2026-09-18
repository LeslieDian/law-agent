# 阶段 4 切分质检报告

**verdict = PASS**

| 检查 | 结果 |
|---|---|
| C1 行数与声明一致 | ✅ |
| C2 `uid_g` 全局唯一 | ✅ |
| C3 **四份 split 两两不相交（硬卡口）** | ✅ |
| C4 训练集逐域配比 = 目标 | ✅ |
| C5 任务覆盖 / 份额上限 | ✅ |
| C6 val/test 无合成数据 | ✅ |
| C7 messages 结构完整 | ✅ |
| C8 字段合法 | ✅ |
| C9 逐域视图一致 | ✅ |
| C10 路由集字段 | ✅ |
| C11 SHA-256 登记比对 | ✅ |

## 规模

| split | 行数 |
|---|---|
| train | 100000 |
| val | 1000 |
| test | 1000 |
| router | 20000 |

训练集逐域实测：{'criminal': 30000, 'civil': 40000, 'procedural': 20000, 'general': 10000}

未使用池剩余：146768

## 不相交性明细（硬卡口）

| 组合 | uid_g 交集 | content_sha1 交集 |
|---|---|---|
| train∩val | **0** | **0** |
| train∩test | **0** | **0** |
| train∩router | **0** | **0** |
| val∩test | **0** | **0** |
| val∩router | **0** | **0** |
| test∩router | **0** | **0** |

## 份额上限留痕

- 域 **procedural**：`legal_question_answering` 占 49.4%（上限 35%），已留痕 cap_relaxed=True

## 警告

- C5 域 procedural 单任务份额 49.4% > 上限 35%，已在 SPLIT_STATS 留痕（cap_relaxed=true，原因是池可用量不足）

