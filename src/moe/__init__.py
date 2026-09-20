# -*- coding: utf-8 -*-
"""层内软混合 LoRA 专家（L2 MoLE）。

对外接口
--------
    from moe.gated_lora import (
        load_expert_bank,        # 读 k 个适配器 → 专家权重
        MixtureContext,          # 门控上下文（掩码池化 / 负载均衡 / 使用率）
        install_mixture,         # 把模块替换成 GatedLoRAMixture
        install_mask_capture,    # 捕获 attention_mask
        freeze_all_but_gates,    # 只留门控可训练
        save_gates, load_gates,  # 门控权重存取
    )

数学形式：y = W0·x + Σ_i g_i(x)·(α_i/r_i)·B_i·A_i·x，同一层 k 个专家逐层加权求和。
详见 `gated_lora.py` 模块 docstring 与 `docs/moe/README.md`。
"""
from .gated_lora import (  # noqa: F401
    GatedLoRAMixture,
    MixtureContext,
    freeze_all_but_gates,
    gate_state_dict,
    index_linear_modules,
    install_mask_capture,
    install_mixture,
    load_expert_bank,
    load_gates,
    save_gates,
)

__all__ = [
    "GatedLoRAMixture", "MixtureContext", "load_expert_bank",
    "install_mixture", "install_mask_capture", "freeze_all_but_gates",
    "save_gates", "load_gates", "gate_state_dict", "index_linear_modules",
]
