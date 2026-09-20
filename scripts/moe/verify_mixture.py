#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 混合模块的**等价性自检** —— 不通过就不许开始训练。

为什么必须有这个脚本
====================
层内软混合的数学只有一行：
    y = W0·x + Σ_i g_i(x)·(α_i/r_i)·B_i·A_i·x
但它有四处极易写错、而且**写错了不会报错、只会静默掉点**：
  1. A、B 的转置方向（peft 存的是 A:[r,in]、B:[out,r]，前向要 x@Aᵀ@Bᵀ）
  2. 缩放系数 α/r 有没有乘、乘的是哪个专家的（不同专家可能不同）
  3. 键名 `base_model.model.model.layers.{l}.{mod}` 解析后有没有对错模块
  4. 加权求和有没有把 g_i 乘进去（漏乘就变成"各专家简单相加"）

本脚本用**退化的 L1 情形**反查这四项：把门控冻结成 one-hot（第 i 个专家权重=1），
此时混合模型的输出**必须**与 peft 单独加载第 i 个适配器的输出一致。
逐个专家都跑一遍 → 每个专家的 A/B/缩放/挂载全部被验证。

前提已核实
==========
transformers 5.17 的 `modeling_qwen3.py` 用的是**模块调用**
（`self.q_proj(hidden_states)`、`self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`），
没有 `F.linear(x, self.q_proj.weight)` 这种绕过模块的写法 —— 所以替换模块是有效的。
若将来升级 transformers 后本自检失败，第一件事就是回去查这一条。

用法
----
    # 快跑（CPU、只留 4 层、fp32 严格容差）—— 验证数学正确性
    python scripts/moe/verify_mixture.py \
        --experts criminal,A0_unified_qwen3_8b \
        --device cpu --max-layers 4 --dtype fp32

    # 真实口径（GPU、全部 36 层、nf4 4bit）—— 验证工程装配
    python scripts/moe/verify_mixture.py \
        --experts civil,criminal,procedure,unified \
        --device cuda:0 --4bit --max-layers 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from moe.gated_lora import (  # noqa: E402
    GatedLoRAMixture, MixtureContext, freeze_all_but_gates,
    install_mask_capture, install_mixture, load_expert_bank,
)

PROMPT = [{"role": "user",
           "content": "甲以非法占有为目的，虚构工程项目骗取乙人民币五十万元，"
                      "用于个人挥霍。甲的行为构成何罪？请说明法律依据。"}]

DTYPE_TABLE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
# 容差：fp32 走同一套算子应到 1e-3 内；bf16/4bit 有舍入，放宽到 5e-2（相对值）
TOL = {"fp32": 1e-3, "bf16": 5e-2, "fp16": 5e-2}


def resolve_expert(tok):
    """专家名/路径 → 适配器目录。unified 指向 A0 那个目录。"""
    if "/" in tok or os.path.isdir(tok):
        return tok
    cand = [os.path.join("models", "adapters", tok)]
    if tok in ("unified", "A0", "a0"):
        cand.insert(0, os.path.join("models", "adapters", "A0_unified_qwen3_8b"))
    for c in cand:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("找不到适配器目录：%s（试过 %s）" % (tok, cand))


def adapter_scales(adapter_dirs):
    """逐个适配器读 α/r —— 不假定所有专家相同。"""
    out = []
    for d in adapter_dirs:
        cfg = json.load(open(os.path.join(d, "adapter_config.json"), encoding="utf-8"))
        r = float(cfg["r"])
        out.append(float(cfg.get("lora_alpha", r)) / r)
    return out


def build_inputs(tok, device):
    text = tok.apply_chat_template(PROMPT, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def load_base(base_dir, device, dtype, four_bit, max_layers):
    from transformers import AutoModelForCausalLM
    kw = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": device}
    else:
        kw["dtype"] = dtype
    m = AutoModelForCausalLM.from_pretrained(base_dir, **kw)
    if not four_bit:
        m = m.to(device)
    if max_layers and 0 < max_layers < len(m.model.layers):
        m.model.layers = m.model.layers[:max_layers]
        m.config.num_hidden_layers = max_layers
    m.eval()
    m.config.use_cache = False
    return m


def logits_peft(base_dir, adapter_dir, device, dtype, four_bit, max_layers, inputs):
    """基准路径：底座 + peft 单适配器。"""
    from peft import PeftModel
    base = load_base(base_dir, device, dtype, four_bit, max_layers)
    if four_bit:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    with torch.no_grad():
        out = model(**inputs).logits.detach().float().cpu()
    del model, base
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def logits_mixture(base_dir, adapter_dirs, names, scales, device, dtype, four_bit,
                   max_layers, inputs, gate_share, onehot):
    """被测路径：底座 + 自建层内软混合，门控强制 one-hot 到第 onehot 个专家。"""
    base = load_base(base_dir, device, dtype, four_bit, max_layers)
    ctx = MixtureContext(detach_experts=True, use_mask=True)
    ctx.force_onehot = int(onehot)
    bank, meta = load_expert_bank(adapter_dirs, names=names, dtype=torch.float32)
    stat = install_mixture(base, bank, ctx, gate_share=gate_share, gate_init="uniform",
                           expert_meta={"expert_names": names, "scaling": scales})
    install_mask_capture(base, ctx)
    n_tr, n_all = freeze_all_but_gates(base)
    with torch.no_grad():
        out = base(**inputs).logits.detach().float().cpu()
    del base
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out, {"stat": stat, "meta": meta, "scales": scales,
                 "trainable_gate_params": n_tr, "total_params": n_all}


def main():
    ap = argparse.ArgumentParser(description="L2 层内软混合等价性自检")
    ap.add_argument("--base", default="models/Qwen3-8B")
    ap.add_argument("--experts", default="criminal,A0_unified_qwen3_8b",
                    help="逗号分隔的适配器名（unified 自动解析到 A0 目录）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPE_TABLE))
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="用训练同口径的 nf4 量化底座（容差自动放宽）")
    ap.add_argument("--max-layers", type=int, default=4,
                    help="只保留前 N 层；0=全部。默认 4 层即可验证数学与装配")
    ap.add_argument("--gate-share", default="in_features",
                    choices=["in_features", "module"])
    ap.add_argument("--report-json", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(0)
    dirs = [resolve_expert(t) for t in a.experts.split(",")]
    names = [os.path.basename(os.path.normpath(d)) for d in dirs]
    scales = adapter_scales(dirs)
    print("[配置] base=%s  device=%s  dtype=%s  4bit=%s  max_layers=%s  gate_share=%s"
          % (a.base, a.device, a.dtype, a.four_bit, a.max_layers, a.gate_share))
    for nm, d, s in zip(names, dirs, scales):
        print("       专家 %-28s %s  (α/r = %g)" % (nm, d, s))

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    inputs = build_inputs(tok, "cpu" if a.device == "cpu" else a.device)
    print("[输入] prompt tokens = %d" % inputs["input_ids"].shape[1])

    dtype = DTYPE_TABLE[a.dtype]
    tol = TOL[a.dtype] * (10 if a.four_bit else 1)
    results = {"base": a.base, "experts": names, "expert_dirs": dirs,
               "dtype": a.dtype, "four_bit": a.four_bit, "max_layers": a.max_layers,
               "gate_share": a.gate_share, "scales": scales,
               "tolerance_relative": tol, "per_expert": []}
    all_ok = True

    for i, nm in enumerate(names):
        print("\n=== 专家 %d/%d : %s ===" % (i + 1, len(names), nm))
        A = logits_peft(a.base, dirs[i], a.device, dtype, a.four_bit, a.max_layers, inputs)
        B, info = logits_mixture(a.base, dirs, names, scales, a.device, dtype,
                                 a.four_bit, a.max_layers, inputs, a.gate_share, onehot=i)
        mx = float((A - B).abs().max())
        scale = float(A.abs().max().clamp(min=1e-6))
        rel = mx / scale
        ok = rel <= tol
        all_ok = all_ok and ok
        print("  peft     |logit|max = %.4f" % scale)
        print("  mixture  max|Δ| = %.6g   相对 = %.3e  -> %s"
              % (mx, rel, "PASS" if ok else "FAIL"))
        if i == 0:
            print("  装配：替换模块 %d 个，建门控 %d 个，K = %s"
                  % (info["stat"]["n_replaced"], info["stat"]["n_gates"],
                     info["stat"].get("K")))
            print("        未匹配目标模块 %d 个（max_layers 截断属正常）／形状不符 %d 个"
                  % (len(info["stat"]["skipped_no_target"]),
                     len(info["stat"]["shape_mismatch"])))
            print("        门控可训练参数 = %d  (%.4f%%)"
                  % (info["trainable_gate_params"],
                     100.0 * info["trainable_gate_params"] / info["total_params"]))
            results["install"] = dict(info["stat"])
            results["trainable_gate_params"] = info["trainable_gate_params"]
            results["total_params"] = info["total_params"]
        results["per_expert"].append({"expert": nm, "max_abs_diff": mx,
                                      "relative": rel, "pass": bool(ok)})

    # --- 附加检查：门控确实是"逐层"独立的 ---------------------------------
    print("\n=== 逐层门控检查 ===")
    base = load_base(a.base, a.device, dtype, a.four_bit, a.max_layers)
    ctx = MixtureContext()
    bank, _ = load_expert_bank(dirs, names=names, dtype=torch.float32)
    install_mixture(base, bank, ctx, gate_share=a.gate_share, gate_init="uniform",
                    expert_meta={"expert_names": names, "scaling": scales})
    n_gates = len(base._gate_pool)

    # (1) 参数独立性：每个门控必须有自己的 weight/bias 张量（不共享 → 才叫"逐层"）
    pids = {id(p) for g in base._gate_pool.values() for p in g.parameters()}
    n_p = sum(p.numel() for g in base._gate_pool.values() for p in g.parameters())
    indep_ok = len(pids) == 2 * n_gates
    print("  门控总数 = %d，独立参数张量 = %d（应 = %d）→ %s"
          % (n_gates, len(pids), 2 * n_gates, "独立 ✓" if indep_ok else "**共享了参数**"))
    print("  门控参数合计 = %d" % n_p)

    # (2) 功能差异：同一输入喂给所有"同输入维度"的门控，输出不应全部相同
    torch.manual_seed(1)
    with torch.no_grad():
        for g in base._gate_pool.values():
            g.weight.normal_(0, 1.0)          # 全零初值时各层门控必然相同，测不出"逐层"
    dev = next(base.parameters()).device
    dims = [g.in_features for g in base._gate_pool.values()]
    common = max(set(dims), key=dims.count)
    probe = torch.randn(1, common, device=dev)
    outs = {}
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        for k, g in base._gate_pool.items():
            if g.in_features != common:
                continue
            outs[k] = torch.softmax(g(probe.float()), dim=-1)[0].tolist()
    uniq = {json.dumps([round(x, 4) for x in v]) for v in outs.values()}
    diff_ok = len(uniq) > 1
    print("  同维（%d）门控 %d 个，输出各不相同的有 %d 种 → %s"
          % (common, len(outs), len(uniq),
             "逐层独立 ✓" if diff_ok else "**所有门控输出相同（装配有误）**"))
    for k in list(outs)[:3]:
        print("    %-16s -> %s" % (k, [round(x, 3) for x in outs[k]]))

    results["n_gates"] = n_gates
    results["gate_param_tensors_independent"] = bool(indep_ok)
    results["gate_params_total"] = int(n_p)
    results["n_distinct_gate_outputs"] = len(uniq)
    all_ok = all_ok and indep_ok and diff_ok
    del base
    if str(a.device).startswith("cuda"):
        torch.cuda.empty_cache()

    results["all_pass"] = bool(all_ok)
    print("\n" + "=" * 70)
    print("VERDICT = %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 70)
    print("MIXTURE_VERIFY_DONE")

    if a.report_json:
        os.makedirs(os.path.dirname(a.report_json) or ".", exist_ok=True)
        with open(a.report_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print("[落盘] %s" % a.report_json)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
