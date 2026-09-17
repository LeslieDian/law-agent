#!/usr/bin/env python
"""law-agent 环境端到端冒烟测试。

验证「能加载模型 → 能用 GPU → 能 4bit 量化 → 能挂 LoRA 并反向传播」全链路，
这是真正决定「环境能不能做实验」的判据，单一 import 成功不算数。

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # 权重已在本地，禁止联网
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(os.environ.get("LAW_ROOT", "/mnt/data/lidian/law-agent"))
BASE_MODEL = ROOT / "models" / "Qwen3-8B"
EMB_MODEL = ROOT / "models" / "Qwen3-Embedding-0.6B"

results: list[tuple[str, bool, str]] = []


def step(name: str):
    def deco(fn):
        def wrapper() -> bool:
            t0 = time.time()
            print("\n--- {} ---".format(name))
            try:
                detail = fn() or ""
                ok = True
            except Exception as exc:  # noqa: BLE001
                detail = "{}: {}".format(type(exc).__name__, str(exc)[:220])
                ok = False
            print("  {} {} ({:.1f}s) {}".format("[OK]" if ok else "[!!]", name,
                                                time.time() - t0, detail))
            results.append((name, ok, detail))
            return ok
        return wrapper
    return deco


@step("1. GPU 与 PyTorch 版本匹配")
def t_torch():
    import subprocess

    import torch

    drv_line = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    drv_cuda = drv_line.split("CUDA Version:")[1].split()[0].strip() if "CUDA Version:" in drv_line else "?"
    tc = torch.version.cuda or "?"
    print("     torch={} torch.cuda={} 驱动CUDA={} 可用={} 卡数={}".format(
        torch.__version__, tc, drv_cuda, torch.cuda.is_available(), torch.cuda.device_count()))
    assert torch.cuda.is_available(), "torch.cuda 不可用"
    assert drv_cuda.split(".")[0] == tc.split(".")[0], \
        "CUDA 主版本不一致：驱动 {} vs torch {}".format(drv_cuda, tc)
    return "torch {} / cu{}".format(torch.__version__, tc)


@step("2. 加载 Qwen3-Embedding-0.6B 并编码中文")
def t_embedding():
    import torch
    from transformers import AutoModel, AutoTokenizer

    assert EMB_MODEL.is_dir(), "模型目录不存在: {}".format(EMB_MODEL)
    tok = AutoTokenizer.from_pretrained(str(EMB_MODEL), padding_side="left")
    # 注意：transformers >= 5.0 已弃用 torch_dtype，改用 dtype
    model = AutoModel.from_pretrained(str(EMB_MODEL), dtype=torch.bfloat16).cuda().eval()

    # Qwen3-Embedding 是 instruction-aware：查询侧用 Instruct 前缀
    texts = [
        "Instruct: 检索相关法律条文\nQuery: 诉讼时效期间届满后债务人自愿履行的法律后果",
        "向人民法院请求保护民事权利的诉讼时效期间为三年。",
    ]
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**batch)
        # Qwen3-Embedding 用 last non-pad token 池化
        last = batch["attention_mask"].sum(dim=1) - 1
        emb = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0)), last]
        emb = torch.nn.functional.normalize(emb, dim=-1)
    cos = float((emb[0] @ emb[1]).item())
    assert emb.shape[-1] == 1024, "维度异常: {}".format(tuple(emb.shape))
    return "dim={} 两段文本余弦={:.4f}".format(emb.shape[-1], cos)


@step("3. 4bit NF4 量化加载 Qwen3-8B")
def t_4bit():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert BASE_MODEL.is_dir(), "模型目录不存在: {}".format(BASE_MODEL)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
    )
    used = torch.cuda.memory_allocated() / 1024 ** 3
    globals()["_model"], globals()["_tok"] = model, tok
    return "加载 {:.0f}s，显存 {:.2f} GB".format(time.time() - t0, used)


@step("4. 生成（非思考模式 /no_think）")
def t_generate():
    import torch

    model, tok = globals()["_model"], globals()["_tok"]
    msgs = [{"role": "user", "content": "/no_think 用一句话说明诉讼时效的含义。"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, do_sample=False, temperature=None, top_p=None,
                             max_new_tokens=64, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    assert len(gen.strip()) > 0, "生成为空"
    return "{:.1f}s | {}".format(time.time() - t0, gen.strip().replace("\n", " ")[:60])


@step("5. LoRA 挂载 + 单步反向传播（QLoRA 训练链路）")
def t_qlora_step():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, tok = globals()["_model"], globals()["_tok"]
    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    globals()["_model"] = model

    batch = tok(["诉讼时效期间为三年。"], return_tensors="pt", padding=True).to("cuda")
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    grads = sum(1 for _, p in model.named_parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    assert grads > 0, "没有任何 LoRA 参数拿到梯度"
    assert loss.item() == loss.item(), "loss 为 NaN"
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return "可训练 {:.1f}M/{:.1f}M ({:.3f}%) loss={:.3f} 有梯度层 {} 峰值显存 {:.2f}GB".format(
        trainable / 1e6, total / 1e6, 100 * trainable / total, loss.item(), grads, peak)


@step("6. 释放显存")
def t_free():
    import gc

    import torch

    globals().pop("_model", None), globals().pop("_tok", None)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return "已释放，当前占用 {:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3)


print("=" * 70)
print("law-agent 环境端到端冒烟测试")
print("LAW_ROOT = {}".format(ROOT))
print("=" * 70)

for fn in (t_torch, t_embedding, t_4bit, t_generate, t_qlora_step, t_free):
    fn()

print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
print("通过 {}/{}".format(passed, len(results)))
for name, ok, detail in results:
    if not ok:
        print("  !! {} -> {}".format(name, detail))
print("=" * 70)
sys.exit(0 if passed == len(results) else 1)
