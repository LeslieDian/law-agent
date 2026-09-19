#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QLoRA 法律适配器训练（阶段 5，A0 统一适配器基线）。

设计要点（每条都对应一次实测，不是凭空选的）
--------------------------------------------------------------------------
1. ★ 不用 TRL 的 `assistant_only_loss`
   实测（2026-09-19，服务器）：`trl 1.13.0` 的 `SFTConfig` 确实有
   `assistant_only_loss` 字段，但 **Qwen3-8B 自带 chat_template 不含
   `{% generation %}` 标记**，TRL 的 assistant-only / completion-only 掩码
   依赖该标记 → 直接报错。改用**前缀边界**做等价掩码：

       full = apply_chat_template(msgs,                    add_generation_prompt=False)
       pref = apply_chat_template(msgs[:-1],               add_generation_prompt=True)

   实测 Qwen3 模板满足 `full.startswith(pref) == True`
   （`...<|im_start|>assistant\n` 是前缀，其后接 assistant 正文）
   → `len(pref)` 即 assistant 段起点，之前全部置 -100。**不依赖任何模板标记**。

2. ★ 用 `transformers.Trainer` 而不是 `SFTTrainer`
   TRL 1.x 的 API 在 1.x 内部多次改名（`max_seq_length` → `max_length`、
   `tokenizer` → `processing_class`…）。本脚本用 `TrainingArguments` +
   `Trainer`，并用 `_accepted_kwargs()` **按 dataclass 实际字段过滤**传入参数，
   不认识的键自动丢弃并**打印留痕** —— 换版本不会因为一个未知参数而崩。

3. ★ 长样本截断方向 = **左截断（保尾部）**
   法律问答里 assistant 段是要学的目标；右截断会把答案切掉，样本变噪声。
   左截断保留答案，只丢上下文开头。截断后若 assistant 段被切空 → **跳过并计数**。

4. ★ 只训练**最后一个** assistant 轮次
   本语料 100% 是 `system/user/assistant` 单轮（实测抽样 2 万行 key 组合一致），
   但按通式实现，多轮时只为最后一轮计 loss —— 行为明确、可解释。

5. ★ `enable_thinking=False`
   Qwen3 模板默认会在 assistant 段前插 `<think>\\n\\n</think>\\n\\n`（实测）。
   语料里没有思维链，若默认渲染，等于教模型输出空 think 块 → 显式关掉，
   模板不支持该 kwarg 时自动回退并留痕。

用法
----
冒烟（500 条、20 步、不评测，用于验证模板/掩码/显存）::

    python scripts/train/train_qlora.py --config configs/qlora_unified.yaml \
        --smoke 500 --max-steps 20 --gpu 1 \
        --output-dir models/adapters/_smoke_qwen3_8b \
        --report-json docs/train/SMOKE_TRAIN.json --report-md docs/train/SMOKE_TRAIN.md

全量::

    python scripts/train/train_qlora.py --config configs/qlora_unified.yaml \
        --gpu 1 --report-json docs/train/TRAIN_A0.json --report-md docs/train/TRAIN_A0.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ----------------------------------------------------------------------------
# 先解析参数（要在 import torch 之前设 CUDA_VISIBLE_DEVICES，故 import 延后）
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="QLoRA 法律适配器训练（A0 基线）")
    ap.add_argument("--config", default="configs/qlora_unified.yaml")
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--train-file", default=None, help="覆盖 config.data.train_file")
    ap.add_argument("--dev-file", default=None, help="覆盖 config.data.dev_file")
    ap.add_argument("--output-dir", default=None, help="覆盖 config.output.output_dir")
    ap.add_argument("--logging-dir", default=None, help="覆盖 config.output.logging_dir")
    ap.add_argument("--smoke", type=int, default=0,
                    help=">0 则只取前 N 条训练样本（冒烟）")
    ap.add_argument("--preview", type=int, default=2000,
                    help="在父进程预渲染 N 条做统计/断言（不受 DataLoader worker 影响）")
    ap.add_argument("--micro-batch", type=int, default=None, help="覆盖 micro_batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="覆盖 gradient_accumulation_steps")
    ap.add_argument("--dev-limit", type=int, default=400, help="dev 评测取样条数")
    ap.add_argument("--no-bucket", action="store_true",
                    help="关掉长度分桶采样（用于做「padding 浪费」的对照实验）")
    ap.add_argument("--max-batch-tokens", type=int, default=None,
                    help="每批 token 数上限（动态批大小；0=不限制，退回定长批）。"
                         "★ 分桶后同批长度一致，峰值显存由每批 token 数决定，"
                         "不设上限会把 80GB 打爆（实测 32768 tok/batch → OOM）")
    ap.add_argument("--max-steps", type=int, default=-1, help="覆盖训练步数（冒烟用）")
    ap.add_argument("--max-seq-length", type=int, default=None, help="覆盖 max_seq_length")
    ap.add_argument("--num-epochs", type=float, default=None, help="覆盖 num_train_epochs")
    ap.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate")
    ap.add_argument("--gpu", default=None, help="指定单卡，如 1；不填则不限制")
    ap.add_argument("--no-4bit", action="store_true", help="关掉 4bit 量化（调试用）")
    ap.add_argument("--no-eval", action="store_true", help="跳过 dev 评测")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--report-md", default=None)
    return ap.parse_args(argv)


ARGS = parse_args()
if ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.path.abspath(ARGS.root)
os.chdir(ROOT)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

import yaml  # noqa: E402


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def jprint(*a):
    print(*a, flush=True)


def accepted_kwargs(cls, kwargs):
    """按 dataclass 实际字段过滤 kwargs，返回 (accepted, dropped)。

    ★ 目的：transformers 5.x 删改了不少 TrainingArguments 字段。硬传未知键会
      TypeError；直接少传又会静默丢配置。过滤 + 打印留痕是唯一两头都不亏的做法。
    """
    fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
    accepted = {k: v for k, v in kwargs.items() if k in fields}
    dropped = sorted(k for k in kwargs if k not in fields)
    return accepted, dropped


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_messages(path, limit=0):
    """读 jsonl，抽出 messages；同时统计样本形态。"""
    rows = []
    stats = {"rows": 0, "has_messages": 0, "single_turn": 0, "multi_turn": 0,
             "no_assistant": 0, "skipped_no_messages": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            stats["rows"] += 1
            msgs = o.get("messages")
            if not isinstance(msgs, list) or not msgs:
                stats["skipped_no_messages"] += 1
                continue
            stats["has_messages"] += 1
            n_asst = sum(1 for m in msgs if m.get("role") == "assistant")
            if n_asst == 0:
                stats["no_assistant"] += 1
            elif n_asst == 1:
                stats["single_turn"] += 1
            else:
                stats["multi_turn"] += 1
            rows.append(msgs)
            if limit and len(rows) >= limit:
                break
    return rows, stats


def render_and_mask(tok, msgs, max_len, enable_thinking=None):
    """渲染 messages → (input_ids, labels)（labels 只在最后一个 assistant 段有值）。

    返回 (ids, labels, info)。info 含 pref_len / total_len / truncated / dropped。
    """
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking

    # 找最后一个 assistant 轮次
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
        # 模板不支持 enable_thinking → 回退
        full = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        pref = tok.apply_chat_template(msgs[:i], tokenize=False,
                                       add_generation_prompt=True)
        kw = {}

    if not full.startswith(pref):
        # 模板不满足前缀性质 → 该样本无法安全掩码，丢弃并计数（不给错误结果）
        return None, None, {"dropped": "prefix_not_prefix"}

    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    pref_len = len(tok(pref, add_special_tokens=False)["input_ids"])

    info = {"pref_len": pref_len, "total_len": len(full_ids), "truncated": False,
            "thinking_kw": bool(kw)}
    if pref_len >= len(full_ids):
        return None, None, {"dropped": "empty_assistant_span"}

    # 左截断（保尾部 = 保答案）
    if max_len and len(full_ids) > max_len:
        drop = len(full_ids) - max_len
        full_ids = full_ids[drop:]
        pref_len = max(0, pref_len - drop)
        info["truncated"] = True
        info["dropped_tokens"] = drop

    labels = [-100] * min(pref_len, len(full_ids)) + full_ids[min(pref_len, len(full_ids)):]
    labels = [-100 if l == -100 else l for l in labels]
    info["assistant_tokens"] = len([x for x in labels if x != -100])
    if info["assistant_tokens"] < 2:
        return None, None, {"dropped": "assistant_span_too_short"}
    return full_ids, labels, info


class ChatSFTDataset(torch.utils.data.Dataset):
    """惰性渲染 + 掩码。避免一次性把 10 万条渲染进内存。

    ★ 基类用 `torch.utils.data.Dataset` 而**不是** `datasets.Dataset`：
      HF `Dataset` 是 pyarrow 支撑的、有自己的构造协议，裸子类化会踩
      `column_names` / `with_format` 等内部假设。我们自带 collator，
      只需要 `__len__` + `__getitem__`，用 torch 的抽象最短也最稳。
    """

    def __init__(self, msgs_list, tok, max_len, enable_thinking):
        self.msgs_list = msgs_list
        self.tok = tok
        self.max_len = max_len
        self.enable_thinking = enable_thinking
        self.stats = {"rendered": 0, "dropped": 0, "drop_reasons": {},
                      "truncated": 0, "asst_tok": 0, "tot_tok": 0}

    def __len__(self):
        return len(self.msgs_list)

    def preview(self, n=2000):
        """在**父进程**里跑一遍前 n 条，拿真实统计（不依赖 DataLoader worker）。

        ★ 为什么必须单独做：`dataloader_num_workers>0` 时 `__getitem__` 在
          **子进程**里执行，`self.stats` 的累加**不会回传父进程** →
          父进程读到 `rendered=0`，把「渲染全成功」误判成失败。
          （2026-09-19 首次冒烟实测踩到：checks.rendered_gt_0 = false，
           而 loss 正常从 1.142 降到 0.513。）
        """
        acc = {"scanned": 0, "rendered": 0, "dropped": 0, "drop_reasons": {},
               "truncated": 0, "asst_tok": 0, "tot_tok": 0}
        for i in range(min(n, len(self.msgs_list))):
            ids, labels, info = render_and_mask(self.tok, self.msgs_list[i],
                                                self.max_len, self.enable_thinking)
            acc["scanned"] += 1
            if ids is None:
                r = info.get("dropped", "unknown")
                acc["dropped"] += 1
                acc["drop_reasons"][r] = acc["drop_reasons"].get(r, 0) + 1
                continue
            acc["rendered"] += 1
            if info.get("truncated"):
                acc["truncated"] += 1
            acc["asst_tok"] += info["assistant_tokens"]
            acc["tot_tok"] += len(ids)
        return acc

    def __getitem__(self, i):
        ids, labels, info = render_and_mask(self.tok, self.msgs_list[i],
                                            self.max_len, self.enable_thinking)
        if ids is None:
            r = info.get("dropped", "unknown")
            self.stats["dropped"] += 1
            self.stats["drop_reasons"][r] = self.stats["drop_reasons"].get(r, 0) + 1
            # 绝不返回坏样本；返回第 0 条的形式空样本会被 collator 忽略 ——
            # 更安全的做法是返回 None，由 collator 过滤掉。
            return {"input_ids": [], "labels": [], "attention_mask": []}
        self.stats["rendered"] += 1
        if info.get("truncated"):
            self.stats["truncated"] += 1
        self.stats["asst_tok"] += info["assistant_tokens"]
        self.stats["tot_tok"] += len(ids)
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}


class PadCollator:
    """右填充 + 丢弃空样本；pad 位置 attention_mask=0、labels=-100。"""

    def __init__(self, pad_id, max_len):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, feats):
        feats = [f for f in feats if f["input_ids"]]
        if not feats:
            return None
        if self.max_len:
            feats = [f for f in feats if len(f["input_ids"]) <= self.max_len]
            if not feats:
                return None
        n = max(len(f["input_ids"]) for f in feats)
        ids, labs, am = [], [], []
        for f in feats:
            pad = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            am.append(f["attention_mask"] + [0] * pad)
            labs.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
        }


class LengthBucketedBatchSampler(torch.utils.data.Sampler):
    """按 token 长度分桶 + 桶内打乱 + 按 **token 预算** 切批的 BatchSampler。

    ★ 为什么必须自己做分桶（实测驱动）
      transformers 5.17 的 `TrainingArguments` **已删除 `group_by_length`**
      （字段表里只剩 `length_column_name`），Trainer 只能退回 RandomSampler。
      后果：一个 4000 token 的长样本会把同批 15 个 ~700 token 的样本一起
      padding 到 4000 → 实测 batch=16 时 13.1 s/step（≈944 tok/s），
      远低于 A800 80GB 跑 4bit-8B LoRA 应有的水平。**padding 是主要浪费源。**

    ★ 为什么必须是 BatchSampler 而不是 Sampler（2026-09-19 血泪）
      分桶后同批长度高度一致 → 峰值显存由 **每批 token 数** 决定：
      每批 16 条 × 2048 token = 32,768 token 时，仅 LM head 的 logits
      就是 32768×151936×2B ≈ 9.9 GB（再算上梯度/上采样 ~20 GB），
      叠加 36 层 block 输入激活 → **实测把 80GB 打爆**
      （`torch.OutOfMemoryError: Tried to allocate 18.55 GiB`，step 44）。
      修法有二：① 缩小 micro-batch —— 但短样本也一起变小，白白浪费算力；
      ② **按 token 预算动态定批大小** —— 短样本照样 16 条一批，
      长样本自动降到 8 条。选 ②。
      而 DataLoader 在 `batch_size` 已给定时会**无视 sampler 的分组**、
      自行每 N 个切一批，所以只能交 **batch_sampler** 给 DataLoader
      （见 `BucketedTrainer.get_train_dataloader`）。

    做法（不改变任何优化语义）
      1. 先算每条样本的 token 长度（`compute_lengths`，带 json 缓存）；
      2. 按长度分桶（桶宽 32 token），桶内随机打乱；
      3. 桶内按顺序累积成批：条数满 `batch_size` **或** 再加一条会超
         `max_batch_tokens` 就封批（同桶长度相近，用桶内最大值估）；
      4. **打乱批的顺序**；
      5. 长度 = 0 的样本（渲染/截断后没有 assistant 段）**直接排除** ——
         它们无论如何都不该参与训练，顺带消除「空批导致 collator 返回 None」的隐患。

    注：批顺序在构造时打乱一次，跨 epoch 顺序固定（确定性、可复现）——
    与旧 Sampler 行为一致，避免「同一样本每 epoch 换批」带来的不可复现。
    """

    def __init__(self, lengths, batch_size, seed=42, world_size=1, rank=0,
                 bucket_width=32, max_batch_tokens=0):
        self.lengths = list(lengths)
        self.batch_size = max(1, int(batch_size))
        self.max_batch_tokens = max(0, int(max_batch_tokens))
        self.seed = int(seed)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.bucket_width = max(1, int(bucket_width))
        self.usable = [i for i, L in enumerate(self.lengths) if L > 0]
        self.dropped_zero = len(self.lengths) - len(self.usable)
        self._batches = self._build()

    def _build(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        buckets = {}
        for i in self.usable:
            buckets.setdefault(self.lengths[i] // self.bucket_width, []).append(i)
        batches = []
        for key in sorted(buckets):
            idx = buckets[key]
            perm = torch.randperm(len(idx), generator=g).tolist()
            cur, cur_max = [], 0
            for p in perm:
                i = idx[p]
                L = self.lengths[i]
                nxt = cur_max if cur_max >= L else L
                over_n = len(cur) >= self.batch_size
                over_tok = bool(self.max_batch_tokens) and \
                    nxt * (len(cur) + 1) > self.max_batch_tokens
                if cur and (over_n or over_tok):
                    batches.append(cur)
                    cur, cur_max, nxt = [], 0, L
                cur.append(i)
                cur_max = nxt
            if cur:
                batches.append(cur)
        perm = torch.randperm(len(batches), generator=g).tolist()
        return [batches[p] for p in perm]

    def stats(self):
        sizes = [len(b) for b in self._batches]
        toks = [sum(self.lengths[i] for i in b) for b in self._batches]
        return {
            "n_batches": len(self._batches),
            "batch_size_min": min(sizes) if sizes else 0,
            "batch_size_max": max(sizes) if sizes else 0,
            "batch_size_mean": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
            "tokens_per_batch_max": max(toks) if toks else 0,
            "tokens_per_batch_mean": round(sum(toks) / len(toks), 1) if toks else 0.0,
            "max_batch_tokens_cap": self.max_batch_tokens,
            "dropped_zero_len": self.dropped_zero,
        }

    def __iter__(self):
        batches = self._batches
        if self.world_size > 1:
            batches = batches[self.rank::self.world_size]
        return iter(batches)

    def __len__(self):
        n = len(self._batches)
        return n // self.world_size if self.world_size > 1 else n


# 旧名保留为别名：报告/文档里提到过 LengthBucketedSampler
LengthBucketedSampler = LengthBucketedBatchSampler


class BucketedTrainer(Trainer):
    """只在「有长度表」时接管 train dataloader；其余完全走父类。"""

    def __init__(self, *a, train_lengths=None, sampler_batch=None, sampler_cb=None,
                 max_batch_tokens=0, **kw):
        super().__init__(*a, **kw)
        self._lengths = train_lengths
        self._sampler_batch = sampler_batch
        self._sampler_cb = sampler_cb
        self._max_batch_tokens = int(max_batch_tokens or 0)
        self._batch_sampler = None
        if self._lengths:
            ws = int(getattr(self.args, "world_size", 1) or 1)
            rk = int(getattr(self.args, "process_index", 0) or 0)
            # bucket_width=32 实测够细：桶内长度差 ≤31 token，padding 浪费可忽略
            self._batch_sampler = LengthBucketedBatchSampler(
                self._lengths,
                self._sampler_batch or self.args.per_device_train_batch_size,
                seed=self.args.seed, world_size=ws, rank=rk,
                max_batch_tokens=self._max_batch_tokens)
            if self._sampler_cb is not None:
                self._sampler_cb.append(self._batch_sampler)

    def get_train_dataloader(self):
        """★ 必须整个接管：按 token 预算定的「每批几条」只有 batch_sampler 能表达。"""
        if self._batch_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader
        a = self.args
        nw = int(getattr(a, "dataloader_num_workers", 0) or 0)
        kw = {}
        if nw > 0:
            kw["persistent_workers"] = bool(
                getattr(a, "dataloader_persistent_workers", False))
            pf = getattr(a, "dataloader_prefetch_factor", None)
            if pf:
                kw["prefetch_factor"] = int(pf)
        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=bool(getattr(a, "dataloader_pin_memory", True)),
            **kw)
        return self.accelerator.prepare(dl)


def compute_lengths(msgs_list, tok, max_len, enable_thinking, cache_path=None,
                    log_every=25000):
    """渲染全量拿 token 长度（长度分桶用）。带 json 缓存，重跑不重算。"""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("n") == len(msgs_list) and d.get("max_len") == max_len:
                jprint("      长度表命中缓存：%s" % cache_path)
                return d["lengths"]
        except Exception as e:      # 缓存坏了不该阻塞训练
            jprint("      长度表缓存不可用（%s），重算" % e)
    lens = []
    for i, msgs in enumerate(msgs_list):
        ids, _l, _i = render_and_mask(tok, msgs, max_len, enable_thinking)
        lens.append(len(ids) if ids else 0)
        if log_every and (i + 1) % log_every == 0:
            jprint("      长度统计 %d/%d ..." % (i + 1, len(msgs_list)))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"n": len(msgs_list), "max_len": max_len, "lengths": lens}, f)
        jprint("      长度表已缓存：%s" % cache_path)
    return lens


class PeakVRAM(TrainerCallback):
    def __init__(self):
        self.peak = 0.0
        self.losses = []

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available():
            self.peak = max(self.peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
        return control

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses.append((state.global_step, float(logs["loss"])))
        return control


def gpu_info():
    if not torch.cuda.is_available():
        return {}
    out = {}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out[i] = {"name": p.name, "total_gb": round(p.total_memory / 1024 ** 3, 1)}
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    cfg = load_cfg(os.path.join(ROOT, ARGS.config) if not os.path.isabs(ARGS.config)
                   else ARGS.config)
    m, q, l, tr, dt, outp = (cfg["model"], cfg["quantization"], cfg["lora"],
                             cfg["training"], cfg["data"], cfg["output"])

    base_model = m["base_model"]
    train_file = ARGS.train_file or dt["train_file"]
    dev_file = ARGS.dev_file or dt.get("dev_file")
    out_dir = ARGS.output_dir or outp["output_dir"]
    log_dir = ARGS.logging_dir or outp["logging_dir"]
    max_len = ARGS.max_seq_length or tr["max_seq_length"]
    enable_thinking = False    # 见文档说明 5

    jprint("=" * 78)
    jprint("GPU 可见:", os.environ.get("CUDA_VISIBLE_DEVICES", "(未限制)"))
    jprint("CUDA 设备:", gpu_info())
    jprint("底座:", base_model)
    jprint("训练集:", train_file, " 冒烟条数:", ARGS.smoke or "全量")
    jprint("输出:", out_dir)
    jprint("=" * 78)

    checks = {}
    notes = []

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    jprint("tokenizer pad_token =", repr(tok.pad_token), " eos =", repr(tok.eos_token))

    # ---- 数据 ----
    msgs_train, st_tr = load_messages(os.path.join(ROOT, train_file), ARGS.smoke)
    jprint("训练样本形态:", st_tr)
    if not msgs_train:
        jprint("[FATAL] 训练样本为空")
        return 2

    # ---- 模板/掩码自检（★ 冒烟核心检查）----
    ids, labels, info = render_and_mask(tok, msgs_train[0], max_len, enable_thinking)
    checks["chat_template_renders"] = ids is not None
    if ids is None:
        jprint("[FATAL] 首条样本渲染失败:", info)
        return 3
    n_asst = sum(1 for x in labels if x != -100)
    checks["assistant_only_loss"] = (n_asst == len(ids) - info["pref_len"] and n_asst > 0)
    jprint("-" * 78)
    jprint("★ 掩码自检  总 token=%d  assistant token=%d  前缀(被掩)=%d"
           % (len(ids), n_asst, info["pref_len"]))
    jprint("  渲染文本尾部 =", repr(tok.decode(ids[-60:])))
    jprint("  是否左截断 =", info["truncated"], " enable_thinking kwarg 生效 =", info["thinking_kw"])
    jprint("-" * 78)

    dev_msgs = []
    if dev_file and not ARGS.no_eval:
        p = os.path.join(ROOT, dev_file)
        if os.path.exists(p):
            dev_msgs, st_dev = load_messages(p, ARGS.dev_limit)
            jprint("dev 样本:", st_dev)

    # ---- 量化 + 模型 ----
    model_kw = {"trust_remote_code": m.get("trust_remote_code", True),
                "attn_implementation": m.get("attn_implementation", "sdpa")}
    # transformers 5.x 用 dtype；旧版用 torch_dtype —— 两版都兼容
    try:
        import transformers as _tf
        _major = int(_tf.__version__.split(".")[0])
    except Exception:
        _major = 5
    if not ARGS.no_4bit:
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16)
    if _major >= 5:
        model_kw["dtype"] = torch.bfloat16
    else:
        model_kw["torch_dtype"] = torch.bfloat16

    jprint("加载底座中 ...")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kw)
    model.config.use_cache = False
    if not ARGS.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tr.get("gradient_checkpointing", True))
    if tr.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=l["r"], lora_alpha=l["lora_alpha"], lora_dropout=l["lora_dropout"],
        bias=l.get("bias", "none"), task_type=l.get("task_type", "CAUSAL_LM"),
        target_modules=l["target_modules"])
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    jprint("LoRA 可训练参数 = %d / %d (%.4f%%)"
           % (trainable, total, 100.0 * trainable / max(total, 1)))
    checks["lora_trainable_gt_0"] = trainable > 0

    ds_train = ChatSFTDataset(msgs_train, tok, max_len, enable_thinking)
    ds_dev = ChatSFTDataset(dev_msgs, tok, max_len, enable_thinking) if dev_msgs else None

    # ★ 在父进程预渲染一小批，拿**真实**统计（不受 DataLoader worker 影响）
    prev = ds_train.preview(ARGS.preview)
    jprint("★ 预渲染统计（父进程，n=%d）：%s"
           % (prev["scanned"], json.dumps(prev, ensure_ascii=False)))
    if prev["rendered"]:
        jprint("  assistant token 占比 = %.3f"
               % (prev["asst_tok"] / max(prev["tot_tok"], 1)))

    # ★ token 长度表（长度分桶用）—— 消除 padding 浪费
    lens = None
    lens_pct = {}
    if not ARGS.no_bucket:
        jprint("★ 计算 token 长度表（长度分桶，消除 padding 浪费）...")
        tl0 = time.time()
        lens = compute_lengths(
            msgs_train, tok, max_len, enable_thinking,
            cache_path=os.path.join(ROOT, log_dir,
                                    "lengths_%s_%d.json"
                                    % (os.path.basename(train_file), max_len)))
        nz = sorted(x for x in lens if x > 0)
        if nz:
            lens_pct = {"p10": nz[int(0.10 * (len(nz) - 1))],
                        "p50": nz[int(0.50 * (len(nz) - 1))],
                        "p90": nz[int(0.90 * (len(nz) - 1))],
                        "p99": nz[int(0.99 * (len(nz) - 1))],
                        "max": nz[-1], "mean": round(sum(nz) / len(nz), 1)}
        jprint("      长度表耗时 %.1fs  非零样本 %d / %d  %s"
               % (time.time() - tl0, len(nz), len(lens),
                  json.dumps(lens_pct, ensure_ascii=False)))

    # ---- 训练参数（★ 按 dataclass 字段过滤，未知键留痕）----
    #
    # ★ transformers 5.17 实测已删除两个我们配置里在用的字段（字段表共 112 个）：
    #     `warmup_ratio`  → 只剩 `warmup_steps`（无 ratio 版）
    #     `logging_dir`   → 直接消失（日志目录跟 output_dir 走）
    #   若不处理会**静默丢弃** → warmup 完全不生效、实验配置与报告不一致。
    #   处置：自己按「每 epoch 步数 × epoch 数」换算 warmup_steps（与 accelerate 同口径），
    #   日志目录改为手动建目录 + 写进报告。换算过程写进报告 `env_adaptations` 留痕。
    import math
    raw_epochs = ARGS.num_epochs if ARGS.num_epochs is not None else tr["num_train_epochs"]
    if ARGS.smoke:
        raw_epochs = 1
    micro = ARGS.micro_batch or tr["micro_batch_size"]
    ga = ARGS.grad_accum or tr["gradient_accumulation_steps"]
    # ★ 每批 token 预算（动态批大小）：默认 1024 token/条 × micro 条。
    #   分桶后同批长度一致 → 峰值显存 ∝ 每批 token 数；不封顶实测会 OOM。
    max_batch_tokens = ARGS.max_batch_tokens
    if max_batch_tokens is None:
        max_batch_tokens = int(tr.get("max_batch_tokens", max(4096, micro * 1024)))
    steps_per_epoch = max(1, math.ceil(len(ds_train) / max(1, micro * ga)))
    if ARGS.max_steps and ARGS.max_steps > 0:
        total_steps_est = ARGS.max_steps
    else:
        total_steps_est = int(steps_per_epoch * raw_epochs)
    warmup_ratio = float(tr.get("warmup_ratio", 0.0))
    warmup_steps = int(round(warmup_ratio * total_steps_est))
    os.makedirs(os.path.join(ROOT, log_dir), exist_ok=True)
    env_adaptations = {
        "warmup_ratio -> warmup_steps": "%s(ratio) x %d(est_total_steps) -> %d steps"
                                        % (warmup_ratio, total_steps_est, warmup_steps),
        "logging_dir": "字段已被 transformers 5.x 删除；已手动建 %s（日志随 output_dir 落盘）"
                       % log_dir,
        "steps_per_epoch_est": steps_per_epoch,
    }

    raw = dict(
        output_dir=os.path.join(ROOT, out_dir),
        run_name=outp.get("run_name") if isinstance(outp, dict) else None,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=ga,
        learning_rate=ARGS.lr or tr["learning_rate"],
        num_train_epochs=raw_epochs,
        warmup_steps=warmup_steps,
        weight_decay=tr["weight_decay"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        max_grad_norm=tr["max_grad_norm"],
        optim=tr["optim"],
        bf16=tr.get("bf16", True),
        fp16=tr.get("fp16", False),
        logging_steps=tr["logging_steps"],
        save_strategy=tr["save_strategy"],
        save_total_limit=tr["save_total_limit"],
        seed=ARGS.seed if ARGS.seed is not None else tr["seed"],
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        report_to=[],
        remove_unused_columns=False,      # ★ 我们用自有 dataset/collator
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )
    raw = {k: v for k, v in raw.items() if v is not None}
    if ARGS.smoke:
        raw["save_strategy"] = "no"
        raw["logging_steps"] = 1
    if ARGS.max_steps and ARGS.max_steps > 0:
        raw["max_steps"] = ARGS.max_steps
    if ds_dev is not None:
        raw["eval_strategy"] = "epoch"
        raw["per_device_eval_batch_size"] = max(1, tr["micro_batch_size"])
    else:
        raw["eval_strategy"] = "no"

    ta_kw, dropped = accepted_kwargs(TrainingArguments, raw)
    if dropped:
        jprint("⚠️ TrainingArguments 不认识的键（已丢弃并留痕）:", dropped)
        notes.append("TrainingArguments 丢弃的键：%s" % ", ".join(dropped))
    targs = TrainingArguments(**ta_kw)

    cb = PeakVRAM()
    collator = PadCollator(tok.pad_token_id, max_len)
    sampler_holder = []
    trainer = BucketedTrainer(model=model, args=targs, train_dataset=ds_train,
                              eval_dataset=ds_dev, data_collator=collator,
                              callbacks=[cb],
                              train_lengths=lens,
                              sampler_batch=ta_kw.get("per_device_train_batch_size"),
                              sampler_cb=sampler_holder,
                              max_batch_tokens=max_batch_tokens)
    checks["bucketed_sampler_active"] = bool(sampler_holder) or bool(ARGS.no_bucket)
    if sampler_holder:
        sst = sampler_holder[0].stats()
        jprint("★ 采样器（token 预算批）:", json.dumps(sst, ensure_ascii=False))
        notes.append("动态批大小统计：%s" % json.dumps(sst, ensure_ascii=False))
        checks["batch_token_budget_respected"] = (
            max_batch_tokens <= 0
            or sst["tokens_per_batch_max"] <= max_batch_tokens)

    jprint("开始训练 ...")
    t1 = time.time()
    result = trainer.train()
    train_sec = time.time() - t1

    # ---- 结论 ----
    ds_stats = ds_train.stats          # ★ worker 里累加的，父进程可能读到 0，仅作旁证
    checks["rendered_gt_0"] = prev["rendered"] > 0
    checks["drop_rate_ok"] = (prev["dropped"] / max(prev["scanned"], 1)) < 0.05
    asst_share = prev["asst_tok"] / max(prev["tot_tok"], 1)
    checks["assistant_token_share_plausible"] = 0.005 < asst_share < 0.9
    checks["loss_no_nan"] = all(x == x and abs(x) != float("inf") for _, x in cb.losses)
    checks["loss_decreased"] = (len(cb.losses) >= 2 and cb.losses[-1][1] < cb.losses[0][1])
    peak = max(cb.peak, (torch.cuda.max_memory_allocated() / 1024 ** 3
                         if torch.cuda.is_available() else 0.0))
    checks["vram_peak_recorded"] = peak > 0

    verdict = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "stage": "5-train",
        "run": "A0_unified_qlora" + ("_SMOKE" if ARGS.smoke else ""),
        "generated_at": time.strftime("%F %T"),
        "verdict": verdict,
        "checks": checks,
        "notes": notes,
        "config": {
            "base_model": base_model, "train_file": train_file, "dev_file": dev_file,
            "smoke": ARGS.smoke, "max_seq_length": max_len,
            "micro_batch_size": tr["micro_batch_size"],
            "micro_batch_size_effective": micro,
            "gradient_accumulation_steps": tr["gradient_accumulation_steps"],
            "max_batch_tokens": max_batch_tokens,
            "lr": raw["learning_rate"], "epochs": raw["num_train_epochs"],
            "lora_r": l["r"], "lora_alpha": l["lora_alpha"],
            "target_modules": l["target_modules"],
            "quantization": "none" if ARGS.no_4bit else q,
            "enable_thinking": enable_thinking,
            "warmup_ratio_cfg": warmup_ratio,
            "warmup_steps_actual": warmup_steps,
            "total_steps_est": total_steps_est,
        },
        "env_adaptations": env_adaptations,
        "data": {"train_scan": st_tr, "dataset_stats": ds_stats,
                 "dataset_preview": prev,
                 "dataset_stats_note": "dataset_stats 由 DataLoader 子进程累加，"
                                       "num_workers>0 时父进程可能读到 0；**以 dataset_preview 为准**",
                 "n_train_samples": len(msgs_train)},
        "trainable_params": {"lora": trainable, "total": total,
                             "ratio_pct": round(100.0 * trainable / max(total, 1), 4)},
        "train_result": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in dict(result.metrics).items()},
        "loss_curve": cb.losses,
        "length_stats": {"bucketing": not ARGS.no_bucket, "percentiles": lens_pct,
                         "n_nonzero": (len([x for x in lens if x > 0]) if lens else None),
                         "n_zero_dropped": (len([x for x in lens if x == 0]) if lens else None)},
        "sampler": ("LengthBucketedBatchSampler(bucket_width=32, batch<=%s, "
                    "max_batch_tokens=%s)"
                    % (ta_kw.get("per_device_train_batch_size"), max_batch_tokens))
                   if lens else "default",
        "peak_vram_gb": round(peak, 2),
        "train_seconds": round(train_sec, 1),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if torch.cuda.is_available():
        report["gpu"] = gpu_info()

    # ---- 落盘 ----
    adapter_dir = os.path.join(ROOT, out_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    jprint("适配器已保存:", adapter_dir)

    if ARGS.report_json:
        p = os.path.join(ROOT, ARGS.report_json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        jprint("报告:", p)
    if ARGS.report_md:
        p = os.path.join(ROOT, ARGS.report_md)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(render_md(report))
        jprint("报告:", p)

    jprint("=" * 78)
    jprint("VERDICT =", verdict)
    for k, v in checks.items():
        jprint("  %-26s %s" % (k, v))
    jprint("峰值显存 = %.2f GB  训练耗时 = %.1fs" % (peak, train_sec))
    jprint("=" * 78)
    print("TRAIN_%s" % ("SMOKE" if ARGS.smoke else "FULL") + "_DONE")
    return 0 if verdict == "PASS" else 1


def render_md(r):
    L = []
    W = L.append
    W("# QLoRA 训练报告 —— %s" % r["run"])
    W("")
    W("> 生成时间：%s　词表：`Qwen3-8B` + LoRA" % r["generated_at"])
    W("> 结论：**%s**" % r["verdict"])
    W("")
    W("## 1. 关键检查")
    W("")
    W("| 检查 | 结果 |")
    W("|---|---|")
    for k, v in r["checks"].items():
        W("| `%s` | %s |" % (k, "✅" if v else "❌"))
    W("")
    W("## 2. 训练配置")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    for k, v in r["config"].items():
        W("| `%s` | %s |" % (k, v))
    W("| LoRA 可训练参数 | **%d** / %d（%.4f%%） |"
      % (r["trainable_params"]["lora"], r["trainable_params"]["total"],
         r["trainable_params"]["ratio_pct"]))
    W("")
    W("## 3. 数据")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    W("| 训练样本 | %d |" % r["data"]["n_train_samples"])
    for k, v in r["data"]["train_scan"].items():
        W("| 扫描-%s | %s |" % (k, v))
    for k, v in r["data"]["dataset_preview"].items():
        W("| 预渲染-%s | %s |" % (k, v))
    W("")
    W("> %s" % r["data"].get("dataset_stats_note", ""))
    W("")
    W("### 运行环境适配（transformers 5.x 字段变更，已自动换算并留痕）")
    W("")
    W("| 项 | 处置 |")
    W("|---|---|")
    for k, v in r.get("env_adaptations", {}).items():
        W("| `%s` | %s |" % (k, v))
    W("")
    W("## 4. 训练结果")
    W("")
    W("| 指标 | 值 |")
    W("|---|---|")
    for k, v in r["train_result"].items():
        W("| `%s` | %s |" % (k, v))
    W("| 峰值显存 (GB) | %s |" % r["peak_vram_gb"])
    W("| 训练耗时 (s) | %s |" % r["train_seconds"])
    W("")
    W("### 长度分布与批内 padding（探针用于解释吞吐）")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    W("| 长度分桶 | %s |" % r["length_stats"]["bucketing"])
    W("| 采样器 | %s |" % r["sampler"])
    W("| 非零样本 | %s |" % r["length_stats"]["n_nonzero"])
    W("| 零长丢弃 | %s |" % r["length_stats"]["n_zero_dropped"])
    for k, v in (r["length_stats"]["percentiles"] or {}).items():
        W("| token 长度 %s | %s |" % (k, v))
    W("")
    W("## 5. loss 曲线（前 40 个记录点）")
    W("")
    W("| step | loss |")
    W("|---|---|")
    for s, v in r["loss_curve"][:40]:
        W("| %d | %.4f |" % (s, v))
    W("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
