# 混合专家（MoE）：两级路由与层内软混合

> 对应论文题目《基于混合专家模型和知识图谱的法律领域智能体研究与构建》里的
> **"混合专家模型"** 那一半。本文说明为什么需要两级、每一级是什么、用了哪些参数、
> **每个参数为什么这么选**，以及怎么证明实现没写错。
>
> 相关代码：`src/moe/gated_lora.py`（L2 核心）、`scripts/train/train_router.py`（L1）、
> `scripts/train/train_moe_gate.py`（L2 训练）、`scripts/moe/verify_mixture.py`（自检）。

---

## 0. 起点：为什么"只在路由上"不够

项目原状是：4 个领域适配器（`civil` / `criminal` / `procedure` / `unified`）
＋ 一个 `rule_then_llm` 的路由配置（关键词规则 → LLM 分类 → 置信度低则回退 unified）。

这个形态有两个问题：

1. **它不是"混合"，是"选择"。** 请求级 top-1 硬选意味着**专家之间从不同时参与**——
   一句话进来，最终只有一个专家的权重在起作用。文献里这叫 expert selection
   （专家选择），与 mixture of experts（专家混合）不是一回事。
2. **路由本身还没实现。** `configs/adapters_router.yaml` 只是一份配置；
   `src/` 与 `scripts/` 里 MoE / gate / mixture 的实现是零。

所以要补两级，分工如下：

| 级 | 名称 | 机制 | 解决什么 |
|---|---|---|---|
| **L1** | 请求级硬路由 | 一个问题 → 一个专家（top-1） | 让"分发"可学习、可量化（准确率/宏 F1） |
| **L2** | 层内软混合 | 同一层 k 个专家**同时参与**，逐层加权求和 | 让题目里"**混合**"二字字面成立 |

**L1 是 L2 的退化情形**（门控输出 one-hot 且只在单层生效）。这个关系不是修辞，
它被写成了可执行的自检（见第 3 节）。

---

## 1. 核心思想：层内软混合（MoLE 正统）

### 1.1 一行公式

$$
y \;=\; W_0\,x \;+\; \sum_{i=1}^{K} g_i(x)\cdot \frac{\alpha_i}{r_i}\cdot B_i A_i\,x
\qquad\text{其中}\quad
g(x)=\mathrm{softmax}\!\left(W_g\cdot \mathrm{pool}(x)\right)
$$

| 符号 | 含义 | 在本项目里是什么 |
|---|---|---|
| $W_0$ | 冻结的底座权重 | Qwen3-8B（nf4 4bit 量化） |
| $A_i, B_i$ | 第 $i$ 个专家的低秩矩阵 | 第 $i$ 个 LoRA 适配器，**已训好、冻结** |
| $\alpha_i/r_i$ | LoRA 缩放 | 逐个专家从各自 `adapter_config.json` 读，不假定相同 |
| $g_i(x)$ | 门控权重，$\sum_i g_i=1$ | 由该层输入决定，**每层一套** |
| $K$ | 专家数 | 4（`civil` / `criminal` / `procedure` / `unified`） |

### 1.2 "同一层挂 k 个 LoRA 专家，逐层加权求和"具体是什么意思

底座 Qwen3-8B 有 **36 层**，每层有 7 个线性模块被 LoRA 命中
（`q/k/v/o_proj`、`gate/up/down_proj`）。

L2 做的事是：**把这 7 个模块全部换成一个"混合模块"**，每个混合模块里并排躺着
4 个专家的低秩矩阵 $A_i, B_i$：

```
第 l 层：  输入 x
             │
             ├─► W0·x                          ← 冻结底座，照常算
             │
             ├─► pool(x) ─► W_g ─► softmax ─► [g1, g2, g3, g4]   ← 这一层的门控
             │
             └─► Σ_i  g_i · (α_i/r_i) · B_i·A_i·x                ← 4 个专家同时出力
             │
             └────────────────────────────► y = W0·x + Σ ...      ← 加权求和后回主干
```

三个"逐"字是关键：

- **逐层**：36 层各有自己的 $W_g$（默认 72 个门控，见 2.2），
  所以"这句话属于哪个法域"在第 3 层和第 30 层可以有不同答案；
- **逐（模块）输入维度**：门控读的是**该模块自己的输入**，不是全局表示；
- **逐样本**：$g$ 是输入 $x$ 的函数，不同问题得到不同权重——不是固定的静态混合比例。

### 1.3 为什么是"软"加权而不是"硬"选择

| | 硬选（L1） | 软混合（L2） |
|---|---|---|
| 参与专家数 | 1 | K（全部，只是权重不同） |
| 跨法域问题 | 只能回退或硬塞 | 权重天然摊到两个专家上 |
| 可微 | 否（不可端到端训门控） | 是 |
| 与文献对应 | InstructMoLE 式 sequence-level gating | MoLE 原文式 per-layer gating |

法律语料的法域边界本来就糊（民事案件大量涉及管辖/举证，刑事附带民事诉讼是民刑交叉；
路由集里天生多标签的样本占 5.4%）。软加权是处理这种模糊边界的自然方式——
**权重 0.6/0.4 比"二选一"更诚实地表达了输入的性质**。

### 1.4 文献位置

| 工作 | 关系 |
|---|---|
| **MoLE**（Hu et al., 2022；Feng et al., 2024） | 本实现的直接来源：把 LoRA 当专家、可学习门控、冻结底座 |
| **PHATGOOSE**（Muqeeth et al., 2024） | 本实现的两阶段策略来源：先独立训专家，再只学"怎么组合" |
| **LoRAHub**（Huang et al., 2023） | 动态 LoRA 组合 |
| **LoRAMoE**（Dou et al., 2023） | LoRA 专家作为插件、保留世界知识 |
| **InstructMoLE** | 指令级全局路由（L1 在文献里的位置） |
| Switch Transformer（Fedus et al., 2022） | 负载均衡损失的出处 |

---

## 2. 参数总表与选型理由

### 2.1 专家侧：**全部继承 A0，一个都不改**

| 参数 | 取值 | 为什么是这个值 |
|---|---|---|
| 专家集合 | `civil`, `criminal`, `procedure`, `unified` | 把**兜底专家也作为一个专家**放进混合体：回退行为由门控自然学出来，而不是靠 `if-else`。`unified` 就是 A0，本身是在四域混合数据上训的"通才"。 |
| $K$ | 4 | = 可用适配器数。少于 3 个谈不上"混合" |
| $r$（LoRA 秩） | 16 | 继承 A0。改了就不能与 A0 对照 |
| $\alpha$ | 32 | 继承 A0，故缩放 $\alpha/r = 2.0$ |
| `target_modules` | `q,k,v,o,gate,up,down_proj`（7 个） | 继承 A0。覆盖注意力与 FFN 两处，是 LoRA 的标准选择 |
| 专家是否可训练 | **否（冻结）** | ①已训好；②重训要十几小时且无理论收益；③参数从 4×43.6M 降到 1–5M；④天然不遗忘 |

> ⚠️ 专家侧的 $r$ / $\alpha$ / `target_modules` **必须与 A0 逐字一致**，
> 否则"统一基线 vs 混合专家"的对照不成立。脚本从各适配器的
> `adapter_config.json` **读**这些值，而不是硬编码——避免将来漂移。

### 2.2 门控侧：这一级才是新增的设计

| 参数 | 取值 | 为什么是这个值 |
|---|---|---|
| 门控结构 | `Linear(in_features, K)` | 最小可用门控：一个线性层 + softmax。门控要学的是"这段话里的法域信号如何映射到专家权重"，这是**线性可分**的工作（L1 用线性分类器就能到不错的准确率，见 `ROUTER_L1.md`），不需要 MLP |
| `--gate-share` | `in_features`（默认） | 同一层里**输入维度相同**的模块共用一个门控（`q/k/v/o/gate/up_proj` 都是 4096 维 → 1 个；`down_proj` 是 12288 → 1 个），全模型 72 个门控、**约 2.36M 参数**。理由：①"这段话属于哪个法域"是**序列级**属性，不该让 `q_proj` 和 `gate_proj` 给出互相矛盾的判断；②20k 训练样本撑不住 `module` 模式下的 5.31M 参数（会过拟合）。备选 `--gate-share module` 更贴近 MoLE 原文，作为消融项保留 |
| 门控输入 | `masked_mean(x)` | 门控要判断的是**整段文本的法域**，是序列级属性 → 把所有有效 token 池化成一个向量。**必须用掩码**：不过滤 padding 会让短句的门控输入被 pad token 稀释，是那种"不报错、只静默掉点"的坑。掩码由挂在模型顶层的 pre-hook 捕获（模型内部的模块拿不到 `attention_mask` 参数） |
| 池化方式 | 均值（而非末位 token） | 末位 token 在右 padding 时是 pad、在左 padding 时是真实 token——训练与推理的 padding 侧不一致，用末位会引入静默偏差。均值 + 掩码两侧都稳 |
| 门控精度 | **fp32**（强制，外层 autocast 关掉） | bf16 下 softmax 的 logit 分辨率太粗，多个专家概率会挤在 0.25 附近、梯度信号被抹平，症状是"门控学不动"。这一项是实测踩出来的 |
| `--gate-init` | `unified`（默认） | 把 `unified` 专家的 logit 初始化为 +4.0 → 起点时 softmax 给它 ≈0.98 的权重，**训练起点 ≈ A0 的水平**（只走通才）。于是门控训练只可能变好、不会一开始就掉点。备选 `uniform`（各专家均等 1/K）作为消融 |
| `--balance-alpha` | 0.01 | Switch Transformer 的标准值。损失项 $L_{bal}=\alpha K\sum_i f_i P_i$，$f_i$ 是硬分配比例（detach）、$P_i$ 是可导的软概率均值。**不加这一项门控会坍缩**到只用一个专家——那等于退化成 L1，"混合"名存实亡，论文里专家使用率那张图也会很难看 |

### 2.3 训练侧

| 参数 | 取值 | 为什么是这个值 |
|---|---|---|
| 训练数据 | `data/router/router_train.jsonl`（20,000 条） | 见下方"为什么不用 train.jsonl" |
| train / dev | 90% / 10%，`seed=42` | **与 L1 调用同一个 `stratified_split`**（`train_moe_gate.py` 直接 `from train_router import stratified_split`）。复制一份早晚会漂移，而"两级用了不同切分"会让 L1/L2 对照失效 |
| 分层依据 | 标签组合（多标签样本也算一个独立桶） | 跨法域样本占 5.4%，按标签组合分层才能保证 dev 里也有它们 |
| `--lr` | **1e-3** | 比 LoRA 的 1e-4 **高一个数量级**。门控是在**已冻结的表示**上学的薄线性层（线性探针）：这里的问题不是过拟合，而是学不动；且它参数少、正则够 |
| `--batch-size` × `--grad-accum` | 8 × 2 = 16 | 与 A0 的等效 batch 16 对齐，让训练动态可比 |
| `--epochs` | 1 | 20k 样本训一个 2.36M 的线性门控，1 个 epoch 足够；多训只会过拟合到路由集的措辞上 |
| `--max-seq-length` | 2048 | 与 A0 一致。实测路由集前 500 条 token 超限率 **0%**（字符 p50=406 / max=1404） |
| `--chat-template` | `models/adapters/A0_unified_qwen3_8b/chat_template.jinja` | 显式指定，保证与 A0 训练**同一套模板**。模板一变，损失信号与专家训练就不同源 |
| `enable_thinking` | `False` | 与 A0 训练、与推理脚本一致 |
| `assistant_only_loss` | 是（`render_and_mask` 逐字复制自 `train_qlora.py`） | 只对 assistant 段计损失。掩码方式一变，门控学到的信号就与专家受训时不同 |
| padding 侧 | **右** | 与训练一致（生成时才用左 padding） |
| `gradient_checkpointing` | 开（`use_reentrant=False`） | 4bit 底座 + 36 层全反向图，必须开以控显存 |
| 优化器 | `paged_adamw_8bit` | 与 A0 一致；门控是 fp32，paged 版本避免优化器状态爆显存 |
| `warmup_ratio` | 0.03 | 与 A0 一致 |

**为什么用 router 集（20k）而不是 train.jsonl（100k）？** 三条：

1. **与 L1 公平对照**：L1 路由器就训在这份数据上。L2 若改用 100k，两者训练数据量差
   5 倍，"L2 比 L1 好"就无法归因到机制上。
2. **与终评集隔离**：阶段 4 的 C1–C14 质检已保证 train / val / test / router 四份
   **两两 disjoint** → 在 router 上训门控、在 test 1k 上报数，不存在泄漏。
3. **性价比**：20k×1 epoch 约 1–1.5 小时；100k×2 epoch 是十几小时，而门控只有
   2.36M 参数，多喂 5 倍数据几乎不带来收益。

### 2.4 三个关键工程决策（写错了不会报错，只会静默掉点）

| 决策 | 做法 | 为什么 |
|---|---|---|
| **专家前向不算梯度** | 专家增量在 `torch.no_grad()` 里算，再与带梯度的 $g_i$ 相乘 | 损失对 $g_i$ 的梯度**只需要专家输出的数值**，不需要对 $A_i/B_i$ 求导。反向图从"整个模型所有 LoRA 路径"缩到"72 个门控线性层"，显存回到纯推理量级 |
| **门控记录开关** | 读完负载均衡损失后立刻 `ctx.recording = False` | 梯度检查点在 backward 阶段会**重算前向**，若那时还记录，会把整张重算图挂进 `_probs` 里、**跨 step 泄漏显存** |
| **掩码捕获 hook** | 装在模型顶层的 `forward_pre_hook` | 模型内部的混合模块拿不到 `attention_mask` 参数；不捕获就只能对含 padding 的序列做全位置平均 |

---

## 3. 怎么证明实现没写错：等价性自检

层内软混合的数学只有一行，但有四处**极易写错且不会报错**：
A/B 的转置方向、缩放 $\alpha/r$、键名解析（有没有对错模块）、加权求和有没有漏乘 $g_i$。

`scripts/moe/verify_mixture.py` 用**退化的 L1 情形**反查这四项：

> 把门控冻结成 one-hot（第 $i$ 个专家权重 = 1），
> 混合模型的输出**必须**与 peft 单独加载第 $i$ 个适配器的输出逐元素一致。

逐个专家跑一遍 → 每个专家的 A/B/缩放/挂载全部被验证。另外还检查
"门控参数张量彼此独立、同一输入下不同层的门控输出不同"（证明真的是**逐层**）。

```bash
# 快跑：CPU、只留 4 层、fp32 严格容差（相对误差 ≤ 1e-3）
python scripts/moe/verify_mixture.py \
    --experts criminal,A0_unified_qwen3_8b \
    --device cpu --max-layers 4 --dtype fp32

# 真实口径：全部 36 层、nf4 4bit（容差自动放宽到 5e-1 相对）
python scripts/moe/verify_mixture.py \
    --experts civil,criminal,procedure,unified \
    --device cuda:0 --4bit --max-layers 0
```

> **若自检失败，第一件事是回去查 Qwen3 的建模源码用的是不是模块调用。**
> transformers 5.17 的 `modeling_qwen3.py` 是 `self.q_proj(hidden_states)`、
> `self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))` —— 模块调用，
> 所以我们替换模块有效。若将来升级到某个用 `F.linear(x, self.q_proj.weight)`
> 直接吃权重的版本，替换会**静默失效**（模型照跑，但门控完全不生效）。

---

## 4. 实验矩阵：L1 / L2 怎么进论文

| 编号 | 配置 | 说明 | 对应代码 |
|---|---|---|---|
| **E5-a** | `router.enabled=false`，只加载 `unified` | 论文原 A4 消融：不做任何专家分工 | 现成 |
| **E5-b** | L1 请求级硬路由 | 训分类器 → 出 Accuracy / 宏 F1 / 混淆矩阵 / 回退率 | `train_router.py` |
| **E5-c** | L2 层内软混合 | 训门控 → 出专家使用率 / 与 E5-b 的同口径对比 | `train_moe_gate.py` |
| **A7**（新增） | L2 换 `--gate-share module` | 门控粒度消融（逐模块 vs 逐输入维度） | 同脚本 |
| **A8**（新增） | L2 换 `--gate-init uniform` | 初始化消融（是否从 A0 起点出发） | 同脚本 |
| **A9**（新增） | L2 换 `--balance-alpha 0` | 负载均衡消融（验证门控是否坍缩） | 同脚本 |

**主结论句**：在相同专家集合、相同训练数据与切分下，层内软混合（L2）相比
请求级硬选（L1）在跨法域样本上的准确率提升 X 个百分点，且专家使用率分布更均衡
（最大/最小权重比从 a 降到 b）。

---

## 5. 复现命令

```bash
# 0) 前置：四个专家适配器必须已训好
#    models/adapters/{civil,criminal,procedure}  +  models/adapters/A0_unified_qwen3_8b

# 1) 自检（必须先过）
python scripts/moe/verify_mixture.py \
    --experts civil,criminal,procedure,unified --device cuda:0 --4bit --max-layers 0 \
    --report-json docs/moe/VERIFY_MIXTURE.json

# 2) L2 门控训练
python scripts/train/train_moe_gate.py \
    --experts civil,criminal,procedure,unified \
    --data data/router/router_train.jsonl \
    --gpu 1 --gate-share in_features --gate-init unified \
    --out-dir models/moe/L2_gate \
    --report-json docs/moe/L2_GATE.json --report-md docs/moe/L2_GATE.md

# 3) 拿门控做推理，与 L1 在同一评测集上对比
#    （推理侧加载：底座 + 4 个适配器 + models/moe/L2_gate/gate_weights.pt）
```

---

## 6. 已知限制（写进论文的"局限"章节）

1. **这是适配器级 MoE，不是原生稀疏 MoE。** 底座仍是稠密的 Qwen3-8B，
   专家是 LoRA 适配器、**全部同时参与计算**（软加权，不是稀疏激活）。
   因此**推理成本随 $K$ 线性增长**，没有稀疏 MoE 的"参数量大、算力小"特性。
   论文里必须明确界定这一点并引用 MoLE 一系，否则"混合专家"的成色会被质疑。
   （若需要原生稀疏 MoE 的对照，`docs/experiment_matrix.md` 的 A6 里已有
   `Qwen3-30B-A3B` 这一格。）
2. **专家是"独立训好再组合"，不是联合训练。** 这是 PHATGOOSE 式的两阶段，
   好处是便宜、可增量加专家；代价是专家之间可能存在冗余（同理，
   `civil` 与 `procedure` 的边界本来就很模糊）。
3. **消融还不够。** 门控共享粒度、初始化方式、负载均衡系数都只列了设计，
   还没跑数（见第 4 节 A7–A9）。
4. **L2 只训门控、不动专家**，所以它的上限受制于专家本身的质量。
   若 `procedure` 专家本身没训好，门控再准也无济于事。
