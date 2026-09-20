# -*- coding: utf-8 -*-
"""轻量法律智能体：检索 → 生成 → 引用核验 → 补检 的多步循环。

为什么不是一条 RAG 管线（写论文时这是「智能体」的落点）：
  1. **工具化**：检索、生成、核验是三个显式工具，循环器决定调用顺序与次数；
  2. **自我核验**：模型答案里引用的《法名》第 X 条会被逐条对照知识库——
     是否存在、是否在本次检索候选内、是否与条文原文一致；
  3. **按需补检**：核验发现引用缺失/编造时，用引用到的法名条号重构查询
     再检索一轮（而不是盲目多轮），步数有上限、trace 全留痕。

显存口径：生成 4bit nf4 ≈ 9GB（与 run_inference 实测一致）；检索的
embedding 编码每次只编码 1 条 query，可与其共存于同一张卡。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_CITE_RE = re.compile(r"《([^》]{2,40})》\s*第\s*([一二三四五六七八九十百千零〇0-9]{1,12})\s*条")


def _uniq_cites(answer: str) -> list:
    """答案里的《法名》第X条 → 去重后的 (法名, 条号原文) 列表（保序）。

    ★ 去重是必须的：A0 的答案会把《刑法》第 358 条连写十来次，
      不去重的话「引用条数」「引用落实率」全被重复计数污染。
    """
    out, seen = [], set()
    for law, art_s in _CITE_RE.findall(answer):
        if (law, art_s) in seen:
            continue
        seen.add((law, art_s))
        out.append((law, art_s))
    return out


def _make_cn2int():
    """复用 scripts/retrieval/extract_edges.py 的 cn2int（与 gold 解析同一实现）；
    找不到时退回内置简版，保证本模块可独立做 dry-run。"""
    try:
        sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "scripts", "retrieval")
        if sdir not in sys.path:
            sys.path.insert(0, sdir)
        from extract_edges import cn2int  # noqa: E402
        return cn2int
    except Exception:
        return _cn2int_fallback


_CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000}


def _cn2int_fallback(s: str):
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        v = _CN_NUM.get(ch)
        if v is None:
            return None
        if v in (10, 100, 1000):
            total += (cur or 1) * v
            cur = 0
        else:
            cur = cur * 10 + v
    return total + cur or None


cn2int = _make_cn2int()


@dataclass
class AgentTrace:
    """一次回答的完整轨迹（论文里「智能体行为分析」的数据源）。"""
    question: str = ""
    steps: list = field(default_factory=list)
    n_retrievals: int = 0
    n_generations: int = 0
    citations_first: list = field(default_factory=list)
    citations_final: list = field(default_factory=list)
    verified_final: dict = field(default_factory=dict)
    requery_triggered: bool = False
    requery_reason: str = ""
    answer: str = ""
    latency_sec: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RetrievalTool:
    """法条检索工具：dense + BM25 + 图谱扩展 + 加权 RRF + 热度（hybrid 口径）。

    所有重索引在首次调用时懒加载（约 2–3 分钟），之后单次检索 < 1s。
    """

    def __init__(self, root: str, device: str = "cuda:0", top_k: int = 50,
                 w_dense: float = 1.0, w_bm25: float = 0.7, w_graph: float = 0.25,
                 rrf_k: int = 10, graph_seed: int = 10, graph_cap: int = 60,
                 w_hot: float = 0.05, final_k: int = 8):
        self.root = root.rstrip("/")
        self.device = device
        self.top_k = top_k
        self.w = (w_dense, w_bm25, w_graph)
        self.rrf_k = rrf_k
        self.graph_seed, self.graph_cap = graph_seed, graph_cap
        self.w_hot = w_hot
        self.final_k = final_k
        self._loaded = False

    # ---- 懒加载 ---------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        t0 = time.time()
        sdir = os.path.join(self.root, "scripts", "retrieval")
        if sdir not in sys.path:
            sys.path.insert(0, sdir)
        import smoke_retrieval as sr  # noqa: E402  （仓库内单一事实源）
        self.sr = sr
        emb_root = os.path.join(self.root, "indexes/retrieval/embeddings")
        self.ids, self.mat = sr.load_item_vectors(
            os.path.join(emb_root, sr.PROVISION_COLLECTION))
        self.provision_index = sr.build_provision_index(self.root)
        # ★ 引用核验必需：模型写的是简称（《刑法》），索引键是 L2 的 law_id
        #   （中华人民共和国刑法）。解析器与 gold 构造共用同一个实现（单一事实源）。
        self.law_ids, self.resolve_law, self.law_resolve_stat = sr.build_law_resolver(
            os.path.join(self.root, "indexes/retrieval/provision_nodes.jsonl"))
        self.item_texts, _ = sr.build_item_texts(self.root, self.ids)
        self.D, self.vocab, _ = sr.build_bm25(self.item_texts)
        self.adj_next = sr.load_edges(os.path.join(self.root, "indexes/retrieval/edges_next.jsonl"))
        self.adj_cites = sr.load_edges(os.path.join(self.root, "indexes/retrieval/edges_cites.jsonl"))
        self.hot, _ = sr.load_hotness(os.path.join(self.root, sr.DEFAULT_HOTNESS))
        from embedding_model import encode_queries, load_model  # noqa: E402
        self.emb_model, _ = load_model(os.path.join(self.root, "models/Qwen3-Embedding-0.6B"),
                                       device=self.device, max_seq_length=2048)
        self._encode_queries = encode_queries
        import numpy as np  # noqa: E402
        self.np = np
        self.pid_index = {p: i for i, p in enumerate(self.ids)}
        self._loaded = True
        print("[RetrievalTool] 索引就绪：%.1fs，%d 条法条 item"
              % (time.time() - t0, len(self.ids)), flush=True)

    # ---- 工具调用 -------------------------------------------------------
    def search(self, query: str) -> list[dict]:
        """返回 [{pid, text, score}]，按融合分排序取 final_k。"""
        self._load()
        sr, np = self.sr, self.np
        qvec = self._encode_queries(self.emb_model, [query], batch_size=1)[0]
        d_scores = self.mat @ qvec
        bm_scores, _ = sr.bm25_search(self.D, self.vocab, [query], self.top_k)
        bm_row = bm_scores[0]

        def top(scores, k):
            order = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            order = order[np.argsort(-scores[order])]
            return [self.ids[i] for i in order]

        dense_list = top(d_scores, self.top_k)
        bm_list = top(bm_row, self.top_k)
        graph_list = sr.graph_expand(dense_list, self.adj_next, self.adj_cites,
                                     top_seed=self.graph_seed, cap=self.graph_cap)
        fused = sr.rrf_fuse([dense_list, bm_list, graph_list], k=self.rrf_k,
                            weights=list(self.w))
        fused = sr.apply_hotness(fused, self.hot, self.w_hot)
        hits = []
        for pid, score in fused[:self.final_k]:
            idx = self.pid_index.get(pid)
            hits.append({
                "pid": pid, "score": float(score),
                "text": self.item_texts[idx] if idx is not None else "",
            })
        return hits

    def lookup(self, law: str, article: int) -> dict:
        """按《法名》+ 条号直查条文（引用核验的兜底通道）。

        ★ 这里有两处曾导致**每条引用都被误判成「疑似编造」**的坑，勿回退：
          1) 索引键是 L2 的 `law_id`（中华人民共和国刑法），而模型引用写的是
             **简称**（《刑法》）——必须先过 `build_law_resolver` 解析；
          2) `build_provision_index` 的值是 **list**（同一 (法名,条号) 可能有多版本，
             如 `...a358` / `...a358-v2`），当标量用会永远 miss。

        返回三态之一：
          - `found`          : 命中，带 pid/text（是否在本次候选内由核验器判断）
          - `not_in_library` : 法名解析成功，但库里确实没有该条 → 疑似编造
          - `law_unresolved` : 法名无法解析（不在库/歧义）→ **不足以判定编造**，
                               诚实起见单列，不并入「库中无」。
        """
        self._load()
        lids = self.resolve_law(law)
        if not lids:
            return {"status": "law_unresolved", "law": law, "article": article}
        for lid in lids:
            for pid in self.provision_index.get((lid, article)) or []:
                idx = self.pid_index.get(pid)
                if idx is not None:
                    return {"status": "found", "pid": pid, "law_id": lid,
                            "law": law, "article": article,
                            "text": self.item_texts[idx]}
        return {"status": "not_in_library", "law": law, "article": article,
                "law_id": lids[0], "law_ids": list(lids)}


class GenerationTool:
    """生成工具：底座 + LoRA 适配器，口径与 scripts/train/run_inference.py 一致。"""

    def __init__(self, root: str, adapter: str, device: str = "cuda:0",
                 max_new_tokens: int = 1024):
        self.root, self.adapter, self.device = root.rstrip("/"), adapter, device
        self.max_new_tokens = max_new_tokens
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        base = os.path.join(self.root, "models/Qwen3-8B")
        adp = self.adapter if os.path.isabs(self.adapter) else os.path.join(self.root, self.adapter)
        tok = AutoTokenizer.from_pretrained(adp)          # ← 模板来自 adapter 目录
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adp)
        model.eval()
        self.torch, self.tok, self.model = torch, tok, model
        self._loaded = True
        print("[GenerationTool] 生成器就绪（nf4，adapter=%s）" % adp, flush=True)

    def __call__(self, question: str, contexts: list[dict]) -> str:
        self._load()
        ctx = "\n\n".join("[%d] %s" % (i + 1, h["text"][:1200])
                          for i, h in enumerate(contexts)) or "（无检索结果）"
        sys_p = ("你是一名中国法律助手。仅依据给出的检索片段作答；"
                 "引用法条时必须使用《法名》第X条的格式。")
        user = "检索片段：\n%s\n\n问题：%s" % (ctx, question)
        messages = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        # 剥离可能的空 think 块（与训练渲染一致时会保留 <think>\n\n</think>）
        return re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S).strip()


class CitationVerifier:
    """引用核验工具：答案里的《法名》第X条逐条对库核验。"""

    def __init__(self, retriever: RetrievalTool):
        self.retriever = retriever

    def __call__(self, answer: str, retrieved: list[dict]) -> dict:
        self.retriever._load()
        pids_in_ctx = {h["pid"] for h in retrieved}
        found, missing, out_of_ctx, unresolved = [], [], [], []
        seen = set()
        n_mentions = 0
        for law, art_s in _CITE_RE.findall(answer):
            n_mentions += 1
            art = cn2int(art_s)
            if art is None or art < 0:
                continue
            if (law, art) in seen:
                continue      # ★ 同一法条在答案里被重复引用多次，只判一次
            seen.add((law, art))
            hit = self.retriever.lookup(law, art)
            rec = {"law": law, "article": art}
            st = hit["status"]
            if st == "law_unresolved":
                unresolved.append(rec)       # 法名不在库/歧义 → 不足以判定编造
            elif st == "not_in_library":
                missing.append(rec)          # 法名已解析、库里确实没有 → 疑似编造
            elif hit["pid"] not in pids_in_ctx:
                out_of_ctx.append(rec)       # 库里有、但不在本次候选 → 应触发补检
            else:
                found.append(rec)
        return {"found": found, "missing": missing, "out_of_ctx": out_of_ctx,
                "law_unresolved": unresolved,
                "n_citations": len(found) + len(missing) + len(out_of_ctx)
                + len(unresolved),
                "n_mentions": n_mentions}


class LawAgent:
    """多步循环器。max_steps=2：首轮 + 至多一轮补检（步数可配）。"""

    def __init__(self, retriever: RetrievalTool, generator: GenerationTool,
                 max_steps: int = 2):
        self.retriever = retriever
        self.generator = generator
        self.verifier = CitationVerifier(retriever)
        self.max_steps = max_steps

    def answer(self, question: str) -> AgentTrace:
        t0 = time.time()
        tr = AgentTrace(question=question)
        contexts = self.retriever.search(question)
        tr.n_retrievals += 1
        tr.steps.append({"step": 1, "tool": "retrieve", "query": question,
                         "n_hits": len(contexts),
                         "pids": [h["pid"] for h in contexts]})
        answer = self.generator(question, contexts)
        tr.n_generations += 1
        tr.citations_first = _uniq_cites(answer)

        for step_i in range(2, self.max_steps + 1):
            verdict = self.verifier(answer, contexts)
            tr.steps.append({"step": step_i, "tool": "verify", "verdict": verdict})
            if not verdict["missing"] and not verdict["out_of_ctx"]:
                break                          # 全部引用已落实，无需补检
            # ---- 补检：用缺失/出上下文的法名重构查询 ----
            tr.requery_triggered = True
            tr.requery_reason = ("missing" if verdict["missing"] else "") + \
                                ("+out_of_ctx" if verdict["out_of_ctx"] else "")
            extra_q = " ".join("%s 第%d条" % (c["law"], c["article"])
                               for c in (verdict["missing"] + verdict["out_of_ctx"]))
            refine_q = question + " " + extra_q   # 实际送进检索的查询（trace 记它）
            more = self.retriever.search(refine_q)
            tr.n_retrievals += 1
            tr.steps.append({"step": step_i, "tool": "retrieve(refine)",
                             "query": refine_q, "extra_terms": extra_q,
                             "n_hits": len(more),
                             "pids": [h["pid"] for h in more]})
            seen = {h["pid"] for h in contexts}
            contexts = contexts + [h for h in more if h["pid"] not in seen]
            answer = self.generator(question, contexts)
            tr.n_generations += 1

        tr.verified_final = self.verifier(answer, contexts)
        tr.citations_final = _uniq_cites(answer)
        tr.answer = answer
        tr.latency_sec = round(time.time() - t0, 2)
        return tr


def build_agent(root: str, adapter: str, device: str = "cuda:0",
                max_steps: int = 2, final_k: int = 8) -> LawAgent:
    ret = RetrievalTool(root=root, device=device, final_k=final_k)
    gen = GenerationTool(root=root, adapter=adapter, device=device)
    return LawAgent(ret, gen, max_steps=max_steps)


def main():  # 通路自检：python -m src.agent.law_agent --dry-run
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--adapter", default="models/adapters/A0_unified_qwen3_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="只测引用解析与条号换算")
    a = ap.parse_args()
    if a.dry_run:
        demo = "依据《中华人民共和国刑法》第一百七十五条与《民法典》第188条……"
        print("citations:", _CITE_RE.findall(demo))
        print("cn2int(一百七十五) =", cn2int("一百七十五"))
        print("cn2int(188) =", cn2int("188"))
        return 0
    agent = build_agent(a.root, a.adapter, a.device, a.max_steps)
    print(json.dumps(agent.answer("盗窃罪的量刑标准是什么？").to_dict(),
                     ensure_ascii=False, indent=1)[:2000])
    return 0

if __name__ == "__main__":
    sys.exit(main())
