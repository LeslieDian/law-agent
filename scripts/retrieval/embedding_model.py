#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Embedding 封装（阶段 4b-3 建库 / 4b-5 检索共用）。

为什么单独抽一个模块
--------------------
建库（4b-3）与检索（4b-5）**必须用同一个模型、同一套归一化与同一个 max_length**，
否则查询向量与库向量不在同一空间，recall 会莫名偏低。把它固定在一处，杜绝两边写岔。

★ Qwen3-Embedding 是 instruction-aware 模型
-------------------------------------------
官方用法：**查询侧**加前缀 `Instruct: {task}\nQuery: {q}`，**文档侧不加**（加了反而掉点）。
`retrieval.yaml` 里也写明 `instruction_aware: true`，查询侧约 +15% 召回。
"""
import os

DEFAULT_MODEL_DIR = "models/Qwen3-Embedding-0.6B"
DEFAULT_MAX_SEQ_LENGTH = 2048

# 检索任务的指令描述（查询侧前缀用；换任务要换这句，并同步写进论文实现细节）
QUERY_INSTRUCTION = (
    "Given a Chinese legal question, case description or issue, "
    "retrieve the most relevant statute provisions and legal documents")


def load_model(model_dir=None, device="cuda:0", max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
               fp16=True):
    """加载 SentenceTransformer。返回 (model, info)。"""
    from sentence_transformers import SentenceTransformer
    import torch

    model_dir = model_dir or DEFAULT_MODEL_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("模型目录不存在：%s" % model_dir)

    kwargs = {}
    if fp16 and str(device).startswith("cuda"):
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    model = SentenceTransformer(model_dir, device=device, **kwargs)
    model.max_seq_length = max_seq_length
    # sentence-transformers 6.x 把 get_sentence_embedding_dimension 改名为
    # get_embedding_dimension，两个都探测，避免将来直接报 AttributeError
    dim = None
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        if hasattr(model, attr):
            dim = int(getattr(model, attr)())
            break
    if dim is None:
        raise RuntimeError("无法取得模型输出维度")
    return model, {
        "model_dir": model_dir,
        "max_seq_length": max_seq_length,
        "dim": dim,
        "device": str(device),
        "fp16": bool(fp16),
        "instruction_aware": True,
    }


def encode_docs(model, texts, batch_size=64):
    """文档侧编码（**不加** instruction 前缀），返回 L2 归一化后的 float32 数组。"""
    import numpy as np
    v = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def encode_queries(model, queries, batch_size=64, instruction=None):
    """查询侧编码（**加** `Instruct: …\\nQuery: …` 前缀）。"""
    import numpy as np
    ins = instruction or QUERY_INSTRUCTION
    q = ["Instruct: %s\nQuery: %s" % (ins, x) for x in queries]
    v = model.encode(q, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")
