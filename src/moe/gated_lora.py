#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""层内软混合 LoRA 专家（L2）—— MoLE 正统实现。

一句话
======
把 k 个**已训好的 LoRA 适配器当作 k 个专家**，让它们**挂在同一层的同一批模块上**，
逐层计算门控权重 g_i(x)，再把各专家的低秩增量加权求和：

    y = W0·x + Σ_i  g_i(x) · (α_i / r_i) · B_i · A_i · x

其中 W0 是冻结的底座权重，A_i / B_i 是第 i 个专家的低秩矩阵，
g_i(x) = softmax_i( W_g · pool(x) ) 是**逐层**算出来的门控权重。

为什么题目里"混合"二字需要它
============================
L1（`scripts/train/train_router.py`）是 **请求级 top-1 硬选**：一个问题只激活一个
专家，**专家之间从不同时参与** —— 严格说这叫"专家选择（expert selection）"。
L2 是 **层内软混合**：同一层 k 个专家**同时参与**，权重由输入决定、且**逐层不同**。

数学上 L1 是 L2 的退化情形（门控输出 one-hot、且只在第 0 层生效）。
这一点没有停留在口头上 —— `scripts/moe/verify_mixture.py` 把它做成了
**等价性自检**：把门控冻结成 one-hot 后，混合模型的输出必须与 peft
单适配器加载的输出逐元素一致。不通过就说明门控-专家耦合写错了。

为什么只训门控、专家冻结
========================
1. 四个专家已经（正在）各自训好，重训一遍要十几小时且无理论收益；
2. 这正是 **PHATGOOSE**（Muqeeth et al., 2024）的两阶段思路：先独立训专家，
   再只学"如何把它们组合起来"；
3. 可训练参数从 4×43.6M 降到 1–5M，20k 路由样本足够，且天然不遗忘。

★ 关键工程优化：专家前向不需要梯度
==================================
损失对门控参数 g_i 的梯度**只需要专家输出的数值**，不需要对 A_i/B_i 求导。
所以模块在计算专家增量时放进 `torch.no_grad()`，再与（带梯度的）g_i 相乘 ——
反向图从"整个模型所有 LoRA 路径"缩小到"几十个门控线性层"，
显存占用回到**纯推理**量级。开关是 `MixtureContext.detach_experts`。

★ 门控初始化：起点等价于 A0
============================
门控权重默认零初始化 → 各专家权重均等（1/K）。但更稳的默认是 `--gate-init unified`：
把兜底专家 `unified` 的 logit 初始化为正值，使训练**起点 ≈ A0**（只走 unified），
于是门控训练只可能变好、不会一开始就掉点。这同时说明一件事：
**L2 把 L1 里那个 if-else 回退分支，换成了可学习的门控权重。**
"""
from __future__ import annotations

import glob
import json
import os
import re

import torch
import torch.nn as nn

# peft 0.21 实测键名（504 个，已核对）：
#   base_model.model.model.layers.{l}.{module}.lora_A.weight
# 注意 **有两个 model.**：外层是 peft 的 `base_model.model`，内层是 Qwen3 自己的
# 结构路径 `model.layers.{l}`。写成 `model\.layers\.` 会全部失配 —— 症状不是报错，
# 而是 bank 为空、静默不挂载任何专家（实测踩过）。故这里用 `(?:model\.)+` 吃掉
# 任意层数的 model. 前缀，兼容裸 `model.layers.{l}...` 与 `base_model.model...`。
_ADAPTER_KEY_RE = re.compile(
    r"^(?:base_model\.)?(?:model\.)+layers\.(\d+)\.(.+?)\.(lora_A|lora_B)"
    r"(?:\.default)?(?:\.weight)$")
_MODULE_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


# ---------------------------------------------------------------------------
# 1. 读专家权重
# ---------------------------------------------------------------------------
def _safetensors_file(adapter_dir):
    fs = sorted(glob.glob(os.path.join(adapter_dir, "*.safetensors")))
    if not fs:
        raise FileNotFoundError("适配器目录里没有 safetensors：%s" % adapter_dir)
    return fs[0]


def load_expert_bank(adapter_dirs, names=None, dtype=torch.float32):
    """读 k 个适配器 → (bank, meta)。

    bank[(layer, module)] = ([A_0..A_{k-1}], [B_0..B_{k-1}])
      A_i: [r, in_features]（peft 存法，**不需要**转置）
      B_i: [out_features, r]
    前向用 x @ A.T @ B.T 还原 (B·A)·x。

    缩放 α/r **逐个专家从各自的 adapter_config.json 读**，不假定相同 ——
    不同专家若用了不同的 r 或 alpha，硬编码会把增量算错。
    """
    from safetensors.torch import load_file

    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in adapter_dirs]
    if len(names) != len(adapter_dirs):
        raise ValueError("names 与 adapter_dirs 数量不一致")

    per_expert, unmatched = [], []
    for d, nm in zip(adapter_dirs, names):
        cfg_path = os.path.join(d, "adapter_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError("缺 adapter_config.json：%s" % d)
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        r = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha", r))
        sd = load_file(_safetensors_file(d), device="cpu")
        w = {}
        for k, t in sd.items():
            m = _ADAPTER_KEY_RE.match(k)
            if not m:
                unmatched.append((nm, k))
                continue
            l, mod, which = int(m.group(1)), m.group(2), m.group(3)
            w.setdefault((l, mod), {})[which] = t.to(dtype)
        per_expert.append({"name": nm, "dir": d, "r": r, "alpha": alpha,
                           "scale": alpha / r, "w": w})

    # ★ 硬门禁：键名一个都没匹配上，说明正则/peft 版本对不上。
    #   这时 bank 会是空的，install_mixture 只是"什么都没挂"——不报错、静默掉点，
    #   比崩溃危险得多。所以这里直接拦死。
    if unmatched and not any(e["w"] for e in per_expert):
        raise ValueError(
            "适配器键名全部失配（%d 个），bank 会是空的。样例：%s\n"
            "多半是 peft 版本改了键名前缀 —— 检查 _ADAPTER_KEY_RE。"
            % (len(unmatched), [k for _, k in unmatched[:3]]))

    # 以第一个专家的键集合为基准；缺键的模块整体跳过（并记录，不静默）
    bank, missing = {}, []
    for key in per_expert[0]["w"]:
        As, Bs, ok = [], [], True
        for e in per_expert:
            got = e["w"].get(key, {})
            if "lora_A" not in got or "lora_B" not in got:
                ok = False
                break
            As.append(got["lora_A"])
            Bs.append(got["lora_B"])
        if ok:
            bank[key] = (As, Bs)
        else:
            missing.append(key)

    meta = {
        "expert_names": [e["name"] for e in per_expert],
        "expert_dirs": [e["dir"] for e in per_expert],
        "r": [e["r"] for e in per_expert],
        "alpha": [e["alpha"] for e in per_expert],
        "scaling": [e["scale"] for e in per_expert],
        "n_modules_in_bank": len(bank),
        "modules_missing_in_some_expert": sorted("%d.%s" % k for k in missing),
        "unmatched_keys": unmatched[:10],
        "n_unmatched_keys": len(unmatched),
    }
    return bank, meta


# ---------------------------------------------------------------------------
# 2. 门控上下文（掩码池化 + 负载均衡 + 使用率统计）
# ---------------------------------------------------------------------------
class MixtureContext:
    """一次 forward 内的共享状态。

    - 捕获 attention_mask（由装在模型顶层的 pre-hook 塞进来），用于**掩码均值池化**：
      门控输入是"整段文本属于哪个法域"，是序列级属性，所以把所有有效 token 池化。
      不过滤 padding 会让短句的门控输入被 pad token 稀释 —— 这是会静默掉点的坑。
    - `detach_experts=True`：专家增量走 no_grad（默认，见模块 docstring）
    - `force_onehot=i`：把门控强制成 one-hot，供等价性自检用
    """

    def __init__(self, detach_experts=True, use_mask=True):
        self.detach_experts = detach_experts
        self.use_mask = use_mask
        self.mask = None
        self.force_onehot = None
        # ★ recording 开关：梯度检查点会在 backward 阶段**重算前向**，
        #   若那时还继续记录，会把整张重算图挂进 _probs 里、跨 step 泄漏显存。
        #   训练循环在读完负载均衡损失后立刻置 False。
        self.recording = True
        self._probs = []       # 本批：带梯度，供负载均衡损失
        self._all_probs = []   # 累计：detach，供使用率报告
        self._hard = []

    # --- 池化 -------------------------------------------------------------
    def pool(self, x):
        m = self.mask
        if self.use_mask and m is not None and m.dim() == 2 and m.shape[1] == x.shape[1]:
            w = m.to(x.dtype).unsqueeze(-1)
            return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return x.mean(dim=1)

    # --- 记录 -------------------------------------------------------------
    def record(self, g, tag=""):
        if not self.recording:
            return
        self._probs.append(g.mean(dim=0))
        self._all_probs.append(g.detach().mean(dim=0))
        self._hard.append(g.detach().argmax(dim=-1))

    def reset(self, hard=False):
        self._probs.clear()
        self._hard.clear()
        self.recording = True
        if hard:
            self._all_probs.clear()
            self.mask = None

    # --- 负载均衡（Switch/MoE 标准式）-------------------------------------
    def balance_loss(self, alpha):
        """L = α · K · Σ_i  f_i · P_i

        f_i = 硬分配比例（detach），P_i = 软概率均值（可导）。
        不加这一项时门控会"坍缩"到只用一个专家 —— 那等于退化成 L1，
        「混合」就没了，而且论文里那张专家使用率图会很难看。
        """
        if not self._probs or alpha <= 0:
            return None
        P = torch.stack(self._probs).mean(dim=0)
        K = P.shape[0]
        hard = torch.cat(self._hard)
        f = torch.bincount(hard, minlength=K).to(P.dtype) / max(hard.numel(), 1)
        return alpha * K * (f.to(P.device) * P).sum()

    def utilization(self):
        if not self._all_probs:
            return None
        return torch.stack(self._all_probs).mean(dim=0).tolist()


def install_mask_capture(model, ctx):
    """在模型顶层挂 pre-hook，把 attention_mask 塞进 ctx。

    为什么必须这样：混合模块在模型内部，拿不到 forward 的 attention_mask 参数。
    不捕获它就只能对含 padding 的序列做全位置平均，门控输入被污染。
    """
    def _fn(module, args, kwargs):
        m = None
        if kwargs:
            m = kwargs.get("attention_mask")
        if m is None and len(args) >= 2 and torch.is_tensor(args[1]) and args[1].dim() == 2:
            m = args[1]
        ctx.mask = m
    return model.register_forward_pre_hook(_fn, with_kwargs=True)


# ---------------------------------------------------------------------------
# 3. 混合模块本体
# ---------------------------------------------------------------------------
class GatedLoRAMixture(nn.Module):
    """包住一个冻结的 nn.Linear（或 bnb Linear4bit），挂 k 个 LoRA 专家。

    前向：
        base_out = W0·x
        pooled   = masked_mean(x)                      # [B, in]
        g        = softmax(W_g · pooled)               # [B, K]，fp32 算，逐层不同
        y        = base_out + Σ_i g_i · scale_i · B_i·A_i·x
    """

    def __init__(self, base, As, Bs, scales, gate, ctx, tag, expert_names):
        super().__init__()
        self.base = base
        self.gate = gate
        self.ctx = ctx
        self.tag = tag
        self.expert_names = list(expert_names)
        self.K = len(As)
        # ★ 专家张量必须与底座同设备：load_expert_bank 在 CPU 上读 safetensors
        #   （故意的，省搬显存峰值），不在这里搬设备就会在前向 `xin @ A.T` 处
        #   直接 device mismatch —— 4bit 底座在 cuda 时必崩。
        #   CPU 自检（--device cpu）恰好掩盖了这一点，只有上 GPU 才暴露。
        _w = getattr(base, "weight", None)
        _dev = _w.device if _w is not None else next(base.parameters()).device
        # persistent=False：门控权重才是要保存的；专家权重随适配器目录走，
        # 塞进 checkpoint 会让每个 ckpt 白胖 700MB。
        self.register_buffer("A", torch.stack(As).to(_dev), persistent=False)  # [K, r, in]
        self.register_buffer("B", torch.stack(Bs).to(_dev), persistent=False)  # [K, out, r]
        self.register_buffer("scaling",
                             torch.tensor(scales, dtype=torch.float32, device=_dev),
                             persistent=False)                                # [K]

    def forward(self, x):
        out = self.base(x)                       # 冻结底座，fp/4bit 由底座自己决定

        pooled = self.ctx.pool(x)
        # ★ 门控强制 fp32：bf16 下 softmax 的 logit 分辨率太粗，会让两个专家
        #   概率几乎相等、梯度信号被抹平（实测会出现门控"学不动"）。
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.gate(pooled.float())
            if self.ctx.force_onehot is not None:
                g = torch.zeros_like(logits)
                g[:, int(self.ctx.force_onehot)] = 1.0
            else:
                g = torch.softmax(logits, dim=-1)          # [B, K]
        self.ctx.record(g, self.tag)

        detached = self.ctx.detach_experts or self.ctx.force_onehot is not None
        acc = None
        with (torch.no_grad() if detached else _Null()):
            xin = x.detach() if detached else x
            # ★ dtype 口径必须与 peft 一致：peft 的 LoraLayer.forward 在 4bit 底座上
            #   会先 `x = x.to(self.lora_A[...].weight.dtype)`，即把激活 **upcast 到
            #   LoRA 权重自身的 fp32** 再算增量。我们若反过来把专家降到激活的 bf16
            #   去算，36 层累积后与 peft 差 **2.0–2.4% 相对**（实测四项一致），
            #   会被等价性自检按 1% 容差判 FAIL —— 是精度口径问题，不是装配错误
            #   （装配错会导致量级失控，而非稳定的 2%）。
            #   CPU/fp32 自检里两者同为 fp32，**永远掩盖这一点**（与设备 bug 同类）。
            _wd = self.A.dtype
            xin = xin.to(_wd)
            for i in range(self.K):
                d = (xin @ self.A[i].t()) @ self.B[i].t()   # [B, T, out]
                d = (d * self.scaling[i]).to(out.dtype)
                wi = g[:, i].to(d.dtype).view(-1, 1, 1)
                term = d * wi
                acc = term if acc is None else acc + term
        if acc is not None:
            out = out + acc
        return out


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# 4. 装配
# ---------------------------------------------------------------------------
def index_linear_modules(model):
    """扫出模型里所有 (layer_idx, module_path) → 完整模块名。

    bnb 的 Linear4bit 继承 nn.Linear，但这里用属性探测而不是 isinstance，
    避免将来换了量化后端就整体失配。
    """
    idx = {}
    for name, mod in model.named_modules():
        if not (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
            continue
        m = _MODULE_NAME_RE.search(name)
        if m:
            idx[(int(m.group(1)), m.group(2))] = name
    return idx


def install_mixture(model, bank, ctx, gate_share="in_features", gate_init="unified",
                    gate_init_logit=4.0, num_layers=None, expert_meta=None):
    """把 bank 里的模块逐个替换成 GatedLoRAMixture，并建门控池。

    gate_share:
      - "in_features"（默认）：同一层里 *输入维度相同* 的模块共用一个门控。
        q/k/v/o/gate/up_proj 都是 4096 维输入 → 共用 1 个；down_proj 是 12288 → 单独 1 个。
        每层 2 个门控，全模型 72 个，可训练参数 ≈ 2.36M。
        理由：**"这段话属于哪个法域"是序列级属性**，不该在 q_proj 和 gate_proj 之间
        给出互相矛盾的判断；而且 20k 样本撑不住 5.3M 个门控参数（会过拟合）。
      - "module"：每 (层, 模块) 一个门控（更贴近 MoLE 原文），72→252 个，参数 ≈ 5.31M。
    gate_init:
      - "unified"（默认）：把兜底专家 unified 的 logit 初始化为 +gate_init_logit，
        起点 ≈ A0；训练只会从这里往上走。
      - "uniform"：零初始化，各专家均等 1/K。
    """
    if expert_meta is None:
        expert_meta = {}
    if not bank:
        raise ValueError("bank 为空：没有可挂载的专家模块")
    # 默认专家名/缩放从 bank 自身推，避免调用方漏传时静默用错值
    K_default = len(next(iter(bank.values()))[0])
    ex_names = expert_meta.get("expert_names") or ["e%d" % i for i in range(K_default)]
    scales_default = expert_meta.get("scaling") or [1.0] * K_default

    idx = index_linear_modules(model)
    # 门控建在模型所在设备上 —— 建在 CPU 上会让前向直接报 device mismatch
    dev = next(model.parameters()).device
    gates = {}
    stat = {"n_replaced": 0, "n_gates": 0, "skipped_no_target": [],
            "shape_mismatch": [], "gate_keys": [], "K": K_default}

    for (l, mod), (As, Bs) in sorted(bank.items()):
        name = idx.get((l, mod))
        if name is None:
            stat["skipped_no_target"].append("%d.%s" % (l, mod))
            continue
        target = model.get_submodule(name)
        # 形状必须对齐，否则是键名对错了模块（比"跑起来才发现 NaN"早一步拦住）
        if As[0].shape[1] != int(target.in_features):
            stat["shape_mismatch"].append("%d.%s A=%s in=%d"
                                          % (l, mod, tuple(As[0].shape),
                                             int(target.in_features)))
            continue
        if Bs[0].shape[0] != int(target.out_features):
            stat["shape_mismatch"].append("%d.%s B=%s out=%d"
                                          % (l, mod, tuple(Bs[0].shape),
                                             int(target.out_features)))
            continue

        K = len(As)
        if gate_share == "in_features":
            gkey = "L%02d_in%d" % (l, int(target.in_features))
        else:
            gkey = "L%02d_%s" % (l, mod.replace(".", "_"))
        gate = gates.get(gkey)
        if gate is None:
            gate = nn.Linear(int(target.in_features), K, device=dev).to(torch.float32)
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
            if gate_init == "unified" and "unified" in ex_names:
                with torch.no_grad():
                    gate.bias[int(ex_names.index("unified"))] = float(gate_init_logit)
            gates[gkey] = gate

        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, GatedLoRAMixture(
            base=target, As=As, Bs=Bs,
            scales=[float(s) for s in scales_default],
            gate=gate, ctx=ctx, tag="%d.%s" % (l, mod), expert_names=ex_names))
        stat["n_replaced"] += 1

    # 门控挂到模型上，才能被 .parameters() / .state_dict() 看见
    model._gate_pool = nn.ModuleDict(gates)
    stat["n_gates"] = len(gates)
    stat["gate_keys"] = sorted(gates.keys())[:6] + (["..."] if len(gates) > 6 else [])
    stat["K"] = K
    return stat


def freeze_all_but_gates(model):
    """只留门控可训练。返回 (可训练参数数, 总参数数)。"""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for g in model._gate_pool.values():
        for p in g.parameters():
            p.requires_grad_(True)
            n += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return n, total


def gate_state_dict(model):
    return {"gate_pool": {k: v.detach().cpu() for k, v in model._gate_pool.state_dict().items()}}


def save_gates(model, out_dir, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    payload = gate_state_dict(model)
    payload["meta"] = extra or {}
    torch.save(payload, os.path.join(out_dir, "gate_weights.pt"))
    with open(os.path.join(out_dir, "gate_meta.json"), "w", encoding="utf-8") as f:
        json.dump(extra or {}, f, ensure_ascii=False, indent=1)
    return os.path.join(out_dir, "gate_weights.pt")


def load_gates(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model._gate_pool.load_state_dict(payload["gate_pool"], strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected),
            "meta": payload.get("meta", {})}
