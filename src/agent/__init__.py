# -*- coding: utf-8 -*-
"""src/agent —— 法律领域智能体的轻量多步循环。

设计动机（对应论文题目里的「智能体」）：
    检索 → 生成 → 引用核验 → （不足则）补检 → 终答
    每一步都是显式 tool 调用、有 trace 留痕，而不是一条 RAG 管线跑到底。

复用关系：检索侧直接复用 scripts/retrieval/smoke_retrieval.py 的加载器与
融合函数（同一份索引、同一套 RRF 权重），保证「智能体看到的候选」与
检索消融报告可对照；生成侧复用 run_inference 的加载口径（nf4 / adapter
目录 tokenizer / enable_thinking=False），保证与训练同源。
"""
from .law_agent import CitationVerifier, GenerationTool, LawAgent, RetrievalTool  # noqa: F401
