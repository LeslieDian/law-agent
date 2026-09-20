#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 层内软混合门控训练 —— 专家冻结，只训门控（PHATGOOSE 式两阶段第二阶段）。

做什么
======
把 k 个已训好的 LoRA 适配器（专家）**同时挂在每一层**，只训练门控线性层：

    y = W0·x + Σ_i g_i(x)·(α_i/r_i)·B_i·A_i·x
    g(x) = softmax( W_g · masked_mean(x) )        ← 逐层各有一个 W_g

训练完成后产出一份 `gate_weights.pt`（几 MB），推理时与专家适配器一起加载即可。
**专家权重一个字节都不动。**

数据：为什么用 router 集而不是 train.jsonl
==========================================
默认 `data/router/router_train.jsonl`（20,000 条，本身带 `messages` 可算语言建模损失）。
三条理由：
1. **与 L1 公平对照**：L1 路由器也是在这份数据上训的，L2 用同一份、同一个
   分层切分函数与 seed（`from train_router import stratified_split`，见下）。
   若 L2 改用 100k 的 train.jsonl，两级的训练数据量差 5 倍，"L2 比 L1 好"就无法归因。
2. **与终评集隔离**：阶段 4 的 C1–C14 已保证 train/val/test/router 四份两两 disjoint，
   所以 router 集上训门控、test 1k 上报数，不存在泄漏。
3. **便宜**：20k×1 epoch 约 1–1.5 小时，而 100k×2 epoch 是十几小时，收益不成比例。

★ 与 L1 共用切分函数：脚本直接 `from train_router import stratified_split`，
  不复制一份 —— 复制早晚会漂移，而"两级用了不同切分"会让对照失效。

关键参数与理由见 `docs/moe/README.md`。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # train_router
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))   # moe.*

from train_router import stratified_split  # noqa: E402  ★ 与 L1 共用同一切分
from moe.gated_lora import (  # noqa: E402
    MixtureContext, freeze_all_but_gates,
    install_mask_capture, install_mixture, load_expert_bank, save_gates,
)

DEFAULT_EXPERTS = {"civil": "models/adapters/civil",
                   "criminal": "models/adapters/criminal",
                   "procedure": "models/adapters/procedure",
                   "unified": "models/adapters/A0_unified_qwen3_8b"}

DEFAULT_TEMPLATE = "models/adapters/A0_unified_qwen3_8b/chat_template.jinja"


def jprint(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 数据：render_and_mask 与 train_qlora.py **逐字一致**（复制而非 import ——
# train_qlora.py 在模块级执行 ARGS = parse_args()，import 会污染本脚本的 argv）
# ---------------------------------------------------------------------------
def load_rows(path, limit=0):
    """读 jsonl → [(messages, router_label_raw)]，并统计丢弃原因。"""
    rows, drop = [], collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            msgs = o.get("messages")
            if not isinstance(msgs, list) or not msgs:
                drop["no_messages"] += 1
                continue
            if not any(m.get("role") == "assistant" for m in msgs):
                drop["no_assistant"] += 1
                continue
            t = (o.get("router_query") or "").strip()
            lab = o.get("router_label") or []
            if not t or not lab:
                drop["no_query_or_label"] += 1      # ★ 与 L1 的过滤条件保持一致
                continue
            rows.append((msgs, list(lab)))
            if limit and len(rows) >= limit:
                break
    return rows, dict(drop)


def render_and_mask(tok, msgs, max_len, enable_thinking=None):
    """渲染 messages → (input_ids, labels)（labels 只在最后一个 assistant 段有值）。

    ★ 这段与 `train_qlora.py` 的 `render_and_mask` **逐字相同**。
      门控是在专家训出来的表示上学组合，掩码方式一变，损失信号就与专家训练不同源。
    """
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not idx:
        return None, None, {"dropped": "no_assistant"}
    i = idx[-1]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False, **kw)
        pref = tok.apply_chat_template(msgs[:i], tokenize=False,
                                       add_generation_prompt=True, **kw)
    except TypeError:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        pref = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        kw = {}
    if not full.startswith(pref):
        return None, None, {"dropped": "prefix_not_prefix"}
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    pref_len = len(tok(pref, add_special_tokens=False)["input_ids"])
    info = {"pref_len": pref_len, "total_len": len(full_ids), "truncated": False}
    if pref_len >= len(full_ids):
        return None, None, {"dropped": "empty_assistant_span"}
    if max_len and len(full_ids) > max_len:
        drop = len(full_ids) - max_len
        full_ids = full_ids[drop:]
        pref_len = max(0, pref_len - drop)     # 左截断：保尾部 = 保答案
        info["truncated"] = True
    n = min(pref_len, len(full_ids))
    labels = [-100] * n + full_ids[n:]
    info["assistant_tokens"] = len([x for x in labels if x != -100])
    if info["assistant_tokens"] < 2:
        return None, None, {"dropped": "assistant_span_too_short"}
    return full_ids, labels, info


class ChatDataset(torch.utils.data.Dataset):
    """惰性渲染。故意不用 HF datasets.Dataset（见 train_qlora.py 的说明）。"""

    def __init__(self, msgs_list, tok, max_len, enable_thinking):
        self.msgs_list = msgs_list
        self.tok = tok
        self.max_len = max_len
        self.enable_thinking = enable_thinking
        self.stats = collections.Counter()
        self._cache = {}

    def __len__(self):
        return len(self.msgs_list)

    def __getitem__(self, i):
        if i in self._cache:
            return self._cache[i]
        ids, labels, info = render_and_mask(self.tok, self.msgs_list[i],
                                            self.max_len, self.enable_thinking)
        if ids is None:
            self.stats[info["dropped"]] += 1
            return None
        if info.get("truncated"):
            self.stats["truncated"] += 1
        self.stats["ok"] += 1
        item = {"input_ids": ids, "labels": labels}
        self._cache[i] = item
        return item


class PadCollator:
    """右 padding（与训练一致；生成时才用左 padding）。"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        mx = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            p = mx - n
            input_ids.append(b["input_ids"] + [self.pad_id] * p)
            labels.append(b["labels"] + [-100] * p)
            attn.append([1] * n + [0] * p)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long)}


# ---------------------------------------------------------------------------
# Trainer：在语言建模损失上叠加负载均衡损失
# ---------------------------------------------------------------------------
class GatedTrainer:
    """用组合而非继承，避免不同 transformers 版本 Trainer 签名差异带来的脆弱性。"""

    @staticmethod
    def make(ctx, balance_alpha):
        from transformers import Trainer

        class _T(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                ctx.reset()                          # 每步清空 → 只统计本批
                out = model(**inputs)
                loss = out.loss
                if model.training and balance_alpha > 0:
                    aux = ctx.balance_loss(balance_alpha)
                    if aux is not None:
                        loss = loss + aux
                # ★ 立刻停止记录：梯度检查点在 backward 阶段会重算前向，
                #   继续记录会把重算图挂住、跨 step 泄漏显存
                ctx.recording = False
                return (loss, out) if return_outputs else loss

        return _T


class PeakVRAM:
    @staticmethod
    def make():
        from transformers import TrainerCallback

        class _C(TrainerCallback):
            def __init__(self):
                self.peak = 0

            def on_step_end(self, args, state, control, **kw):
                if torch.cuda.is_available():
                    self.peak = max(self.peak,
                                    torch.cuda.max_memory_allocated() / 1024 ** 3)

        return _C


@torch.no_grad()
def collect_utilization(model, ctx, dataset, collator, device, max_batches=40):
    """跑若干批前向，统计各专家的平均门控权重。

    这一项是论文里的必需图：**专家使用率**。若某个专家权重长期 ≈ 0，
    说明门控坍缩、"混合"名存实亡 —— 也是负载均衡系数是否起作用的最直接证据。
    """
    model.eval()
    ctx.reset(hard=True)
    ctx.recording = True
    idxs = list(range(min(len(dataset), max_batches * 8)))
    n = 0
    for s in range(0, len(idxs), 8):
        batch = collator([dataset[i] for i in idxs[s:s + 8]])
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        n += 1
        if n >= max_batches:
            break
    u = ctx.utilization()
    ctx.recording = False
    model.train()
    return u


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="L2 层内软混合门控训练（专家冻结）")
    ap.add_argument("--base", default="models/Qwen3-8B")
    ap.add_argument("--experts", default="civil,criminal,procedure,unified",
                    help="逗号分隔；取值见 DEFAULT_EXPERTS")
    ap.add_argument("--data", default="data/router/router_train.jsonl")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--out-dir", default="models/moe/L2_gate")
    ap.add_argument("--report-json", default="docs/moe/L2_GATE.json")
    ap.add_argument("--report-md", default="docs/moe/L2_GATE.md")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    # ★ 门控 lr 比 LoRA 的 1e-4 高一个量级：门控是在**已冻结的表示**上学的
    #   一个薄线性层（线性探针），学不动才是问题；且它参数少、不容易过拟合。
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--balance-alpha", type=float, default=0.01,
                    help="负载均衡损失系数（Switch Transformer 默认 0.01）")
    ap.add_argument("--gate-share", default="in_features",
                    choices=["in_features", "module"])
    ap.add_argument("--gate-init", default="unified", choices=["unified", "uniform"])
    ap.add_argument("--gate-init-logit", type=float, default=4.0)
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-template-only", action="store_true",
                    help="只做数据与装配，不训练（秒级冒烟）")
    a = ap.parse_args()

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (a.report_json, a.report_md):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    names = [x.strip() for x in a.experts.split(",") if x.strip()]
    dirs = []
    for nm in names:
        d = DEFAULT_EXPERTS.get(nm, nm)
        if not os.path.isdir(d):
            jprint("[FATAL] 适配器目录不存在：%s（专家 %s）" % (d, nm))
            return 2
        dirs.append(d)

    # ---- 数据 ------------------------------------------------------------
    rows, drop = load_rows(a.data, a.max_samples)
    jprint("[数据] %s → %d 行（丢弃 %s）" % (a.data, len(rows), drop))
    if drop.get("no_messages"):
        jprint("  [WARN] 有 %d 行没有 messages —— 这会使本脚本的切分与 L1 不完全一致"
               % drop["no_messages"])
    labels_raw = [l for _m, l in rows]
    tr_idx, dv_idx, n_buckets = stratified_split(labels_raw, a.holdout, a.seed)
    jprint("[切分] 与 L1 共用 stratified_split(seed=%d)：%d 桶 → train=%d dev=%d"
           % (a.seed, n_buckets, len(tr_idx), len(dv_idx)))

    # ---- 模型 ------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    dev = "cuda:%d" % a.gpu if a.device.startswith("cuda") else a.device
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if os.path.isfile(a.chat_template):
        tok.chat_template = open(a.chat_template, encoding="utf-8").read()
        jprint("[模板] 使用 %s（与 A0 训练同源）" % a.chat_template)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from transformers import BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, quantization_config=qcfg, device_map={"": dev},
        trust_remote_code=True, attn_implementation="sdpa")
    model.config.use_cache = False

    ctx = MixtureContext(detach_experts=True, use_mask=True)
    bank, meta = load_expert_bank(dirs, names=names, dtype=torch.float32)
    jprint("[专家] K=%d  %s" % (len(names), list(zip(names, meta["scaling"]))))
    jprint("       bank 覆盖模块 %d 个；某专家缺键 %d 个；未匹配键 %d 个"
           % (meta["n_modules_in_bank"], len(meta["modules_missing_in_some_expert"]),
              meta["n_unmatched_keys"]))

    stat = install_mixture(model, bank, ctx, gate_share=a.gate_share,
                           gate_init=a.gate_init, gate_init_logit=a.gate_init_logit,
                           expert_meta={"expert_names": names, "scaling": meta["scaling"]})
    install_mask_capture(model, ctx)
    n_tr, n_all = freeze_all_but_gates(model)
    jprint("[装配] 替换模块 %d；门控 %d 个；未匹配目标 %d；形状不符 %d"
           % (stat["n_replaced"], stat["n_gates"], len(stat["skipped_no_target"]),
              len(stat["shape_mismatch"])))
    jprint("       可训练参数 = %d (%.4f%%)，其余 %d 个参数全部冻结"
           % (n_tr, 100.0 * n_tr / n_all, n_all - n_tr))
    if stat["shape_mismatch"]:
        jprint("  [FATAL] 形状不符说明键名对错了模块：%s" % stat["shape_mismatch"][:5])
        return 3
    if n_tr == 0:
        jprint("  [FATAL] 可训练参数为 0，装配失败")
        return 3

    ds_tr = ChatDataset([rows[i][0] for i in tr_idx], tok, a.max_seq_length, False)
    ds_dv = ChatDataset([rows[i][0] for i in dv_idx], tok, a.max_seq_length, False)
    jprint("[数据] 渲染：train=%d dev=%d（未渲染样本会在 batch 里被丢弃）"
           % (len(ds_tr), len(ds_dv)))

    if a.threshold_template_only:
        jprint("[DRY] 只做装配检查，不训练。")
        jprint("L2_GATE_DRYRUN_DONE")
        return 0

    collator = PadCollator(tok.pad_token_id)
    # ★ transformers 5.17 已删除 `TrainingArguments.warmup_ratio`（只剩 `warmup_steps`），
    #   硬传 ratio 会直接 TypeError。train_qlora.py 早就适配并留了注释，**本脚本漏了** ——
    #   后果是门控训练在这个环境里从未真正跑起来过（一执行到 TrainingArguments 就崩，
    #   而且前面的装配/渲染全部正常，很容易被误判成"已经在跑"）。
    #   这里按 train_qlora 同口径换算（ratio × 估计总步数），并对未知键过滤 + 留痕。
    steps_per_epoch = int(math.ceil(len(ds_tr) / max(1, a.batch_size * a.grad_accum)))
    total_steps_est = max(1, int(round(steps_per_epoch * a.epochs)))
    warmup_steps = int(round(a.warmup_ratio * total_steps_est))
    jprint("[环境适配] warmup_ratio=%s x est_total_steps=%d -> warmup_steps=%d"
           % (a.warmup_ratio, total_steps_est, warmup_steps))
    raw = dict(
        output_dir=os.path.join(a.out_dir, "run"),
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        weight_decay=a.weight_decay,
        warmup_steps=warmup_steps,            # ← 由 warmup_ratio 换算，不用已删除的字段
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        gradient_checkpointing=True,          # 4bit 底座 + 全层反向图，必须开
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=[],
        seed=a.seed,
        dataloader_num_workers=2,
        label_names=["labels"],
    )
    _fields = set(getattr(TrainingArguments, "__dataclass_fields__", {}).keys())
    _dropped = sorted(k for k in raw if k not in _fields)
    if _dropped:
        jprint("⚠️ TrainingArguments 不认识的键（已丢弃并留痕）:", _dropped)
    targs = TrainingArguments(**{k: v for k, v in raw.items() if k in _fields})
    vram = PeakVRAM.make()()
    trainer = GatedTrainer.make(ctx, a.balance_alpha)(
        model=model, args=targs, train_dataset=ds_tr,
        data_collator=collator, callbacks=[vram])

    t0 = time.time()
    out = trainer.train()
    train_s = time.time() - t0
    jprint("[训练] 完成：%.1fs，train_loss=%.4f，峰值显存=%.2fGB"
           % (train_s, out.training_loss, vram.peak))
    jprint("[渲染统计] %s" % dict(ds_tr.stats))

    util = collect_utilization(model, ctx, ds_dv, collator, dev)
    if util:
        jprint("[专家使用率] " + "  ".join("%s=%.4f" % (n, u)
                                          for n, u in zip(names, util)))

    gate_path = save_gates(model, a.out_dir, extra={
        "base": a.base, "experts": names, "expert_dirs": dirs,
        "expert_scaling": meta["scaling"],
        "gate_share": a.gate_share, "gate_init": a.gate_init,
        "K": len(names), "n_gates": stat["n_gates"],
        "trainable_params": n_tr, "balance_alpha": a.balance_alpha,
        # 训练口径（论文"训练配置"要用实际值，不是 CLI 默认值）
        "data": a.data,
        "train_samples": len(ds_tr), "dev_samples": len(ds_dv),
        "batch_size": a.batch_size, "grad_accum": a.grad_accum,
        "eff_batch": a.batch_size * a.grad_accum,
        "steps_per_epoch": steps_per_epoch, "total_steps_est": total_steps_est,
        "epochs": a.epochs, "lr": a.lr, "weight_decay": a.weight_decay,
        "warmup_ratio_cfg": a.warmup_ratio, "warmup_steps_actual": warmup_steps,
        "max_seq_length": a.max_seq_length, "seed": a.seed,
        "dropped_training_args": _dropped,
    })
    jprint("[落盘] %s" % gate_path)

    cfg = {
        "level": "L2_intra_layer_soft_mixture",
        "formula": "y = W0·x + Σ_i g_i(x)·(α_i/r_i)·B_i·A_i·x",
        "base": a.base, "experts": names, "expert_dirs": dirs,
        "expert_scaling": meta["scaling"],
        "data": a.data, "n_rows": len(rows), "drop": drop,
        "n_train": len(tr_idx), "n_dev": len(dv_idx),
        "holdout": a.holdout, "seed": a.seed,
        "gate_share": a.gate_share, "gate_init": a.gate_init,
        "gate_init_logit": a.gate_init_logit,
        "n_gates": stat["n_gates"], "n_modules_replaced": stat["n_replaced"],
        "modules_missing_in_some_expert": len(meta["modules_missing_in_some_expert"]),
        "trainable_params": n_tr, "total_params": n_all,
        "trainable_ratio": round(100.0 * n_tr / n_all, 4),
        "batch_size": a.batch_size, "grad_accum": a.grad_accum,
        "effective_batch": a.batch_size * a.grad_accum,
        "epochs": a.epochs, "lr": a.lr, "weight_decay": a.weight_decay,
        "warmup_ratio": a.warmup_ratio, "balance_alpha": a.balance_alpha,
        "max_seq_length": a.max_seq_length, "chat_template": a.chat_template,
        "train_seconds": round(train_s, 1),
        "train_loss": round(float(out.training_loss), 4),
        "peak_vram_gb": round(vram.peak, 2),
        "expert_utilization": ({n: round(u, 4) for n, u in zip(names, util)}
                               if util else None),
        "gate_weights": gate_path,
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(a.report_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    write_md(a.report_md, cfg)
    jprint("[落盘] %s / %s" % (a.report_json, a.report_md))
    jprint("L2_GATE_DONE")
    return 0


def write_md(path, d):
    L = ["# L2 层内软混合 —— 门控训练报告\n"]
    L.append("> %s\n" % d["formula"])
    L.append("## 1. 配置\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 底座 | `%s` |" % d["base"])
    L.append("| 专家（K=%d） | %s |" % (len(d["experts"]), ", ".join(d["experts"])))
    L.append("| 专家缩放 α/r | %s |" % d["expert_scaling"])
    L.append("| 训练数据 | `%s`（%d 行） |" % (d["data"], d["n_rows"]))
    L.append("| train / dev | %d / %d（holdout=%.2f, seed=%d） |"
             % (d["n_train"], d["n_dev"], d["holdout"], d["seed"]))
    L.append("| 门控共享方式 | `%s` |" % d["gate_share"])
    L.append("| 门控初始化 | `%s`（logit=%.1f） |" % (d["gate_init"], d["gate_init_logit"]))
    L.append("| 门控个数 | %d（替换模块 %d 个） |" % (d["n_gates"], d["n_modules_replaced"]))
    L.append("| **可训练参数** | **%d（%.4f%%）**，其余全部冻结 |"
             % (d["trainable_params"], d["trainable_ratio"]))
    L.append("| batch × grad_accum | %d × %d = %d |"
             % (d["batch_size"], d["grad_accum"], d["effective_batch"]))
    L.append("| epochs / lr / wd | %s / %s / %s |"
             % (d["epochs"], d["lr"], d["weight_decay"]))
    L.append("| 负载均衡系数 α | %s |" % d["balance_alpha"])
    L.append("| max_seq_length | %d |" % d["max_seq_length"])
    L.append("| chat template | `%s` |" % d["chat_template"])
    L.append("")
    L.append("## 2. 结果\n")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append("| train_loss | %s |" % d["train_loss"])
    L.append("| 训练耗时 | %.1fs |" % d["train_seconds"])
    L.append("| 峰值显存 | %.2f GB |" % d["peak_vram_gb"])
    L.append("| 门控权重 | `%s` |" % d["gate_weights"])
    L.append("")
    if d.get("expert_utilization"):
        L.append("### 专家使用率（平均门控权重）\n")
        L.append("| 专家 | 平均权重 |")
        L.append("|---|---|")
        for k, v in d["expert_utilization"].items():
            L.append("| %s | %.4f |" % (k, v))
        L.append("")
        L.append("> 若某专家权重长期 ≈ 0，说明门控坍缩、混合退化 —— "
                 "调大 `--balance-alpha` 重跑。\n")
    L.append("## 3. 复现\n")
    L.append("```bash")
    L.append("python scripts/train/train_moe_gate.py \\")
    L.append("  --experts %s \\" % ",".join(d["experts"]))
    L.append("  --data %s \\" % d["data"])
    L.append("  --gpu 1 --gate-share %s --gate-init %s"
             % (d["gate_share"], d["gate_init"]))
    L.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
