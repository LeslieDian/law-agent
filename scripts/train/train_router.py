#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 路由分类器训练 —— 请求级门控（request-level gating）。

论文里的位置
============
「混合专家」两级路由方案的第一级：给定一个法律问题，先判断它属于哪个法域，
再交给对应的领域专家适配器（adapter）。

★ 它做不到什么（也就是 L2 要补的那一半）
----------------------------------------
请求级路由是 **top-1 硬选**：一个问题只激活一个专家，**专家之间从不同时参与**。
严格说这叫「专家选择」，不叫「混合」。MoLE 一系文献把这类 sequence /
instruction-level gating 也算作 MoLE 的一支（如 InstructMoLE），所以它站得住；
但要让论文题目里「混合专家」四个字字面成立，需要 L2 —— **层内软混合**：

    y = W0·x + Σ_i g_i(x) · B_i·A_i·x

同一层挂 k 个 LoRA 专家，逐层算门控权重、加权求和。见 `src/moe/gated_lora.py`
与 `scripts/train/train_moe_gate.py`。

设计要点（每条下方都有对应实现）
================================
1. **标签名必须映射**：路由数据用 `procedural`，而适配器目录与配置用 `procedure`；
   另有 `general`（4,133 条）对应兜底适配器 `unified`。不映射的话程序法 / 通用法
   样本全部对不上标签 —— 这是最容易被忽略、且会直接毁掉训练的一处。
2. **多标签不丢样本**：20,000 条里 **1,087 条**是跨法域（如 ['criminal','civil']）。
   用多标签（每类一个二分类器 + sigmoid）而不是单标签 softmax，把这 1,087 条
   完整保留成监督信号；推理时再按 `cross_domain_policy` 决定回退 unified 还是
   召唤多个适配器聚合。
3. **编码器复用检索侧**：Qwen3-Embedding-0.6B，与 4b-3 建库是同一个模型。
   理由：系统里不必带第二个 embedding 模型；路由输入与检索 query 同分布。
4. **但不加 instruction 前缀**：`embedding_model.encode_queries()` 会加
   `Instruct: Given a Chinese legal question ... retrieve the most relevant statute ...`，
   那是**检索任务**的指令，会把表示往「和法条像」的方向拉，对「判法域」是干扰。
   所以这里调 `encode_docs()`（无前缀）—— 对分类任务才是中性选择。
5. **分类头用逻辑回归**：20k×1024 的规模，CPU 上秒级训练、毫秒级推理；
   `predict_proba` 直接给出可用于阈值回退的置信度；完全可复现（无采样）。
   对比配置现状 `rule_then_llm`：让 Qwen3-8B 判域要 1–2 秒/次，本方案 <10ms，
   **快两个数量级** —— 这是论文里一个能写的小贡献点。
6. **dev 集自建**：`data/router/` 下**只有** `router_train.jsonl`，没有官方 dev/test。
   所以按「标签组合」分层切 10% 作内部 dev（seed=42 固定）；
   另可用 `--extra-eval data/test/test.jsonl`（1,000 条，带 `domain` 字段）
   做**独立集交叉验证** —— 该集合与 router 集在阶段 4 已做过集合级隔离，可安全评估。

无 GPU 依赖（默认 `--device cpu`），可与训练任务并行而不抢卡。

用法
----
    # 全量：训 + 内部 dev 评估 + 独立集交叉验证
    python scripts/train/train_router.py \
        --data data/router/router_train.jsonl \
        --extra-eval data/test/test.jsonl \
        --out-dir outputs/router/L1_linear \
        --report-json docs/moe/ROUTER_L1.json \
        --report-md  docs/moe/ROUTER_L1.md

    # 冒烟（只取 500 条，验证通路）
    python scripts/train/train_router.py --limit 500 --out-dir /tmp/router_smoke
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time

# ---- 复用检索侧编码器：单一真源，保证 tokenizer / 归一化与建库一致 -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "retrieval"))

# ★ 适配器名 ↔ 路由标签的映射。左边是路由数据里的标签，右边是适配器目录名。
#   procedural->procedure 是**必须**的修正；general->unified 是语义对应（兜底专家）。
DEFAULT_LABEL_MAP = {
    "civil": "civil",
    "criminal": "criminal",
    "procedural": "procedure",
    "general": "unified",
}
# 内部 dev 的比例与随机种子（固定 seed 保证可复现）
DEFAULT_HOLDOUT = 0.10
DEFAULT_SEED = 42


def jprint(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def canon_labels(raw, label_map):
    """路由标签 → 适配器名。未知标签保留原样并在报告里亮出来（不静默吞）。"""
    out, unknown = [], []
    for x in raw or []:
        y = label_map.get(str(x))
        if y is None:
            unknown.append(str(x))
            y = str(x)
        if y not in out:
            out.append(y)
    return out, unknown


def load_router_data(path, label_map, limit=0):
    """返回 (texts, label_lists, unknown_counter)。"""
    texts, labels = [], []
    unknown = collections.Counter()
    for r in iter_jsonl(path):
        t = (r.get("router_query") or "").strip()
        lab, unk = canon_labels(r.get("router_label") or [], label_map)
        if not t or not lab:
            continue
        for u in unk:
            unknown[u] += 1
        texts.append(t)
        labels.append(lab)
        if limit and len(texts) >= limit:
            break
    return texts, labels, unknown


def build_eval_text(r):
    """独立集（data/test/test.jsonl）上的路由输入构造。

    ★ 必须与 router_query 的构造方式一致：`instruction + "\\n" + input`，
      **不含 system（法条文本）**。因为真实系统里路由发生在检索之前，
      那一刻还没有法条可喂 —— 训练与推理的条件必须一致，否则数字不可比。
    """
    ins = (r.get("instruction") or "").strip()
    inp = (r.get("input") or "").strip()
    if ins and inp:
        return ins + "\n" + inp
    return (r.get("ask") or ins or inp).strip()


def load_eval_data(path, label_map):
    texts, labels = [], []
    for r in iter_jsonl(path):
        t = build_eval_text(r)
        dom = r.get("domain") or (r.get("domains") or [None])[0]
        if not t or not dom:
            continue
        lab, _ = canon_labels([dom], label_map)
        texts.append(t)
        labels.append(lab)
    return texts, labels


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------
def encode(texts, model_dir, device, max_seq_length, batch_size, tag):
    from embedding_model import load_model, encode_docs
    import numpy as np

    jprint("[编码] %s：%d 条文本，device=%s，max_seq_length=%d"
           % (tag, len(texts), device, max_seq_length))
    model, info = load_model(model_dir, device=device,
                             max_seq_length=max_seq_length, fp16=False)
    jprint("       编码器 %s，维度 %d" % (info["model_dir"], info["dim"]))

    # 字符长度分布 —— 用于解释有多少长文本会被 tokenizer 截断
    cl = sorted(len(t) for t in texts)
    n = len(cl)
    jprint("       字符长度 p50=%d p90=%d p99=%d max=%d"
           % (cl[n // 2], cl[int(n * 0.9)], cl[int(n * 0.99)], cl[-1]))

    # token 长度（能拿到 tokenizer 就算；拿不到只报字符数）
    n_trunc = None
    try:
        tk = getattr(model, "tokenizer", None)
        if tk is not None:
            n_trunc = 0
            for i in range(0, n, 256):
                for ids in tk(texts[i:i + 256], add_special_tokens=False)["input_ids"]:
                    if len(ids) > max_seq_length:
                        n_trunc += 1
            jprint("       超过 max_seq_length(%d) 会被截断的条数 = %d (%.2f%%)"
                   % (max_seq_length, n_trunc, 100.0 * n_trunc / n))
    except Exception as e:
        jprint("       [WARN] token 长度统计失败：%s" % e)

    t0 = time.time()
    X = encode_docs(model, texts, batch_size=batch_size)   # ★ 无 instruction 前缀
    dt = time.time() - t0
    jprint("       完成：shape=%s，耗时 %.1fs（%.1f 条/s）"
           % (tuple(X.shape), dt, n / max(dt, 1e-9)))
    del model
    try:
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return X, {"dim": int(X.shape[1]), "n": n, "encode_seconds": round(dt, 1),
               "n_truncated_est": n_trunc}


# ---------------------------------------------------------------------------
# 分层切分（按标签组合，保证多标签样本也均衡）
# ---------------------------------------------------------------------------
def stratified_split(labels, holdout, seed):
    """按标签组合分层切分。组合过于稀有（<2 条）时并入 'other' 桶。"""
    import random
    keys = ["+".join(sorted(set(l))) for l in labels]
    cnt = collections.Counter(keys)
    # 出现 1 次的组合无法同时在 train/dev 出现，合并进 other，避免 stratify 报错
    merged = [k if cnt[k] >= 2 else "__other__" for k in keys]
    buckets = collections.defaultdict(list)
    for i, k in enumerate(merged):
        buckets[k].append(i)
    rng = random.Random(seed)
    tr, dv = [], []
    for k, idxs in sorted(buckets.items()):
        rng.shuffle(idxs)
        n_dev = max(1, int(round(len(idxs) * holdout))) if len(idxs) >= 2 else 0
        dv.extend(idxs[:n_dev])
        tr.extend(idxs[n_dev:])
    return sorted(tr), sorted(dv), len(buckets)


# ---------------------------------------------------------------------------
# 分类头：每类一个二分类逻辑回归（多标签）
# ---------------------------------------------------------------------------
def train_head(Xtr, Ytr, C, max_iter):
    from sklearn.linear_model import LogisticRegression

    heads = []
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]
        if yj.min() == yj.max():          # 该类全 0 或全 1 —— 退化为常数预测
            heads.append(("const", float(yj.mean())))
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter, solver="liblinear")
        clf.fit(Xtr, yj)
        heads.append(("lr", clf))
    return heads


def predict_proba(heads, X):
    import numpy as np
    cols = []
    for kind, obj in heads:
        if kind == "const":
            cols.append(np.full(X.shape[0], obj, dtype="float64"))
        else:
            cols.append(obj.predict_proba(X)[:, 1])
    import numpy as np2
    return np2.vstack(cols).T


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def evaluate(Y_true, P, labels, threshold):
    """返回指标 dict。

    ★ 报两套指标，因为它们的用途不同：
      - 标签集指标（exact_match / micro_f1 / macro_f1）：衡量"分类器本身"
      - 决策指标（top1_acc / fallback_rate / routed_acc）：衡量"系统实际会发生什么"
        系统一旦套上阈值回退策略，分类器的高置信错误会变成回退，低置信正确会变成
        回退损失 —— 只看 F1 看不到这一层，必须单独报。
    """
    import numpy as np
    from sklearn.metrics import (f1_score, hamming_loss, accuracy_score,
                                 precision_recall_fscore_support)
    K = len(labels)
    Yp = (P >= threshold).astype(int)
    # 至少给一个标签：若全部低于阈值，取 argmax（否则该样本无预测，指标会失真）
    empty = Yp.sum(axis=1) == 0
    if empty.any():
        Yp[empty, np.argmax(P[empty], axis=1)] = 1

    out = {
        "n": int(Y_true.shape[0]),
        "threshold": threshold,
        "exact_match": round(float(accuracy_score(Y_true, Yp)), 4),
        "hamming_loss": round(float(hamming_loss(Y_true, Yp)), 4),
        "micro_f1": round(float(f1_score(Y_true, Yp, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(Y_true, Yp, average="macro", zero_division=0)), 4),
    }
    p, r, f, s = precision_recall_fscore_support(Y_true, Yp, average=None,
                                                zero_division=0, labels=list(range(K)))
    out["per_class"] = {}
    for i, name in enumerate(labels):
        out["per_class"][name] = {"n_true": int(Y_true[:, i].sum()),
                                 "precision": round(float(p[i]), 4),
                                 "recall": round(float(r[i]), 4),
                                 "f1": round(float(f[i]), 4)}

    # ---- 决策层指标：模拟 adapters_router.yaml 的跨域策略 -------------------
    top1 = np.argmax(P, axis=1)
    # top-1 命中正确标签集
    top1_hit = Y_true[np.arange(len(top1)), top1].astype(bool)
    # 跨域判定：有 >=2 个类超过阈值 → 按 cross_domain_policy 走多适配器 / unified
    n_over = (P >= threshold).sum(axis=1)
    cross = n_over >= 2
    low_conf = P.max(axis=1) < threshold
    out["decision"] = {
        "top1_acc": round(float(top1_hit.mean()), 4),
        "fallback_rate": round(float((cross | low_conf).mean()), 4),
        "cross_domain_rate": round(float(cross.mean()), 4),
        "low_conf_rate": round(float(low_conf.mean()), 4),
        # 回退是"有意的保守"，不计入错误：统计接受单路由那部分样本的准确率
        "routed_acc": round(float(top1_hit[~(cross | low_conf)].mean())
                            if (~(cross | low_conf)).any() else 0.0, 4),
        "routed_cov": round(float((~(cross | low_conf)).mean()), 4),
    }
    return out


def confusion(Y_true, P, labels, threshold):
    """单标签视角的混淆矩阵（top-1 vs 真实主标签）。

    仅供人看：把"每条样本主标签"与 top-1 预测对照，能一眼看出哪两个法域最易混。
    多标签样本的真值取第一个标签（与路由数据的书写顺序一致）。
    """
    import numpy as np
    top1 = np.argmax(P, axis=1)
    true1 = np.argmax(Y_true, axis=1)     # 真值里第一个为 1 的类
    K = len(labels)
    M = np.zeros((K, K), dtype=int)
    for t, q in zip(true1, top1):
        M[t, q] += 1
    return M.tolist()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="L1 请求级路由分类器训练")
    ap.add_argument("--data", default="data/router/router_train.jsonl")
    ap.add_argument("--extra-eval", default=None,
                    help="额外评估集（如 data/test/test.jsonl，用 domain 字段）")
    ap.add_argument("--out-dir", default="outputs/router/L1_linear")
    ap.add_argument("--report-json", default="docs/moe/ROUTER_L1.json")
    ap.add_argument("--report-md", default="docs/moe/ROUTER_L1.md")
    ap.add_argument("--encoder", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cpu", help="默认 cpu：不占 GPU，可与训练并行")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--C", type=float, default=1.0, help="逻辑回归 L2 强度的倒数")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="与 configs/adapters_router.yaml 的 confidence_threshold 对齐")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条（冒烟）")
    ap.add_argument("--dump-preds", action="store_true", help="落 dev 集逐条预测")
    a = ap.parse_args()

    import numpy as np

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (a.report_json, a.report_md):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    # ---- 1. 读数据 --------------------------------------------------------
    texts, labels_raw, unknown = load_router_data(a.data, DEFAULT_LABEL_MAP, a.limit)
    if not texts:
        jprint("[FATAL] 没有读到任何路由样本")
        return 2
    labels = sorted({y for ls in labels_raw for y in ls},
                    key=lambda x: -sum(1 for ls in labels_raw if x in ls))
    jprint("[数据] %d 条；标签空间 %s" % (len(texts), labels))
    if unknown:
        jprint("  [WARN] 未在 label_map 里的标签：%s —— 请确认是否要补映射"
               % dict(unknown))
    flat = collections.Counter(y for ls in labels_raw for y in ls)
    jprint("  扁平计数（多标签重复计）：%s" % dict(flat.most_common()))
    n_multi = sum(1 for ls in labels_raw if len(ls) > 1)
    jprint("  跨法域（多标签）样本 = %d (%.2f%%)"
           % (n_multi, 100.0 * n_multi / len(labels_raw)))

    # ---- 2. 编码 ----------------------------------------------------------
    X, enc_info = encode(texts, a.encoder, a.device, a.max_seq_length,
                         a.batch_size, "路由训练集")
    lidx = {y: i for i, y in enumerate(labels)}
    Y = np.zeros((len(texts), len(labels)), dtype=int)
    for i, ls in enumerate(labels_raw):
        for y in ls:
            Y[i, lidx[y]] = 1

    # ---- 3. 切分 ----------------------------------------------------------
    tr, dv, n_buckets = stratified_split(labels_raw, a.holdout, a.seed)
    jprint("[切分] 按标签组合分层（%d 个桶，seed=%d）：train=%d dev=%d"
           % (n_buckets, a.seed, len(tr), len(dv)))

    # ---- 4. 训练 ----------------------------------------------------------
    t0 = time.time()
    heads = train_head(X[tr], Y[tr], a.C, a.max_iter)
    train_s = time.time() - t0
    jprint("[训练] 逻辑回归 ×%d 完成，耗时 %.2fs" % (len(labels), train_s))

    # ---- 5. 评估 ----------------------------------------------------------
    P_dev = predict_proba(heads, X[dv])
    res_dev = evaluate(Y[dv], P_dev, labels, a.threshold)
    res_dev["confusion_top1"] = confusion(Y[dv], P_dev, labels, a.threshold)
    jprint("[内置 dev] exact_match=%.4f micro_f1=%.4f macro_f1=%.4f "
           "top1_acc=%.4f fallback=%.4f"
           % (res_dev["exact_match"], res_dev["micro_f1"], res_dev["macro_f1"],
              res_dev["decision"]["top1_acc"], res_dev["decision"]["fallback_rate"]))

    res_extra = None
    if a.extra_eval and os.path.isfile(a.extra_eval):
        et, el_raw = load_eval_data(a.extra_eval, DEFAULT_LABEL_MAP)
        # 独立集可能含训练时没见过的标签（如 domain=general 之外的域）
        unseen = sorted({y for ls in el_raw for y in ls if y not in lidx})
        if unseen:
            jprint("[独立集] [WARN] 出现标签空间外的标签 %s —— 这些标签按全错计"
                   % unseen)
        EX, _ = encode(et, a.encoder, a.device, a.max_seq_length,
                       a.batch_size, "独立评估集")
        EY = np.zeros((len(et), len(labels)), dtype=int)
        for i, ls in enumerate(el_raw):
            for y in ls:
                if y in lidx:
                    EY[i, lidx[y]] = 1
        P_ex = predict_proba(heads, EX)
        res_extra = evaluate(EY, P_ex, labels, a.threshold)
        res_extra["path"] = a.extra_eval
        res_extra["unseen_labels"] = unseen
        jprint("[独立集] exact_match=%.4f micro_f1=%.4f macro_f1=%.4f "
               "top1_acc=%.4f fallback=%.4f"
               % (res_extra["exact_match"], res_extra["micro_f1"],
                  res_extra["macro_f1"], res_extra["decision"]["top1_acc"],
                  res_extra["decision"]["fallback_rate"]))

    # ---- 6. 落模型与报告 --------------------------------------------------
    with open(os.path.join(a.out_dir, "router_head.pkl"), "wb") as f:
        pickle.dump({"heads": heads, "labels": labels,
                     "label_map": DEFAULT_LABEL_MAP,
                     "threshold": a.threshold}, f)
    jprint("[落盘] %s/router_head.pkl" % a.out_dir)

    if a.dump_preds:
        pth = os.path.join(a.out_dir, "dev_preds.jsonl")
        with open(pth, "w", encoding="utf-8") as f:
            for i, idx in enumerate(dv):
                f.write(json.dumps({
                    "text_head": texts[idx][:160],
                    "true": labels_raw[idx],
                    "proba": {labels[j]: round(float(P_dev[i, j]), 4)
                              for j in range(len(labels))},
                    "top1": labels[int(np.argmax(P_dev[i]))],
                }, ensure_ascii=False) + "\n")
        jprint("[落盘] %s" % pth)

    cfg = {
        "level": "L1_request_level_hard_routing",
        "data": a.data, "n_total": len(texts), "n_train": len(tr), "n_dev": len(dv),
        "holdout": a.holdout, "seed": a.seed,
        "encoder": a.encoder, "encode": enc_info,
        "head": "LogisticRegression(OvR, one binary per class)",
        "C": a.C, "max_iter": a.max_iter, "threshold": a.threshold,
        "label_map": DEFAULT_LABEL_MAP, "labels": labels,
        "n_multi_label": n_multi, "unknown_labels_in_data": dict(unknown),
        "head_train_seconds": round(train_s, 3),
        "eval_internal_dev": res_dev,
        "eval_extra": res_extra,
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(a.report_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    jprint("[落盘] %s" % a.report_json)

    write_md(a.report_md, cfg)
    jprint("[落盘] %s" % a.report_md)
    jprint("ROUTER_L1_DONE total=%.1fs" % (time.time() - t_start))
    return 0


def write_md(path, cfg):
    L = []
    L.append("# L1 路由分类器 —— 训练报告（请求级门控）\n")
    L.append("> 层级：**L1 请求级硬路由**（top-1）。专家不同时参与 → 仍属"
             "「专家选择」；真正的「混合」由 L2 层内软混合承担。\n")
    L.append("## 1. 数据\n")
    d = cfg
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 训练文件 | `%s` |" % d["data"])
    L.append("| 总样本 | %d |" % d["n_total"])
    L.append("| train / 内部 dev | %d / %d（holdout=%.2f, seed=%d） |"
             % (d["n_train"], d["n_dev"], d["holdout"], d["seed"]))
    L.append("| 跨法域（多标签）样本 | %d |" % d["n_multi_label"])
    L.append("| 标签空间（映射后） | %s |" % ", ".join(d["labels"]))
    L.append("")
    L.append("### 标签名映射（**必须做**）\n")
    L.append("| 路由数据标签 | 适配器名 | 说明 |")
    L.append("|---|---|---|")
    for k, v in d["label_map"].items():
        note = "原样" if k == v else "**改名**"
        L.append("| `%s` | `%s` | %s |" % (k, v, note))
    L.append("")
    L.append("## 2. 编码与分类头\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 编码器 | `%s`（= 4b-3 建库同款） |" % d["encoder"])
    L.append("| 前缀 | **不加** retrieval instruction（见脚本头注释第 4 条） |")
    L.append("| max_seq_length | %d |" % d["encode"].get("max_seq_length", 2048))
    L.append("| 向量维度 | %d |" % d["encode"]["dim"])
    L.append("| 编码耗时 | %.1fs（%d 条） |"
             % (d["encode"]["encode_seconds"], d["encode"]["n"]))
    L.append("| 分类头 | %s |" % d["head"])
    L.append("| C / max_iter | %s / %s |" % (d["C"], d["max_iter"]))
    L.append("| 置信度阈值 | %s（与 `adapters_router.yaml` 对齐） |" % d["threshold"])
    L.append("| 头训练耗时 | %.2fs |" % d["head_train_seconds"])
    L.append("")
    for key, title in (("eval_internal_dev", "3. 内部 dev"),
                       ("eval_extra", "4. 独立集交叉验证")):
        r = d.get(key)
        if not r:
            continue
        L.append("## %s\n" % title)
        if key == "eval_extra":
            L.append("来源 `%s`（该集合与 router 集在阶段 4 已做集合级隔离）\n" % r["path"])
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append("| n | %d |" % r["n"])
        L.append("| exact_match（标签集完全一致） | %.4f |" % r["exact_match"])
        L.append("| micro_f1 / macro_f1 | %.4f / %.4f |" % (r["micro_f1"], r["macro_f1"]))
        L.append("| hamming_loss | %.4f |" % r["hamming_loss"])
        L.append("")
        L.append("### 逐类\n")
        L.append("| 法域 | 真值条数 | Precision | Recall | F1 |")
        L.append("|---|---|---|---|---|")
        for name, m in r["per_class"].items():
            L.append("| %s | %d | %.4f | %.4f | %.4f |"
                     % (name, m["n_true"], m["precision"], m["recall"], m["f1"]))
        L.append("")
        L.append("### 决策层（套用阈值回退策略后的系统行为）\n")
        L.append("| 指标 | 值 | 含义 |")
        L.append("|---|---|---|")
        dd = r["decision"]
        L.append("| top1_acc | %.4f | top-1 落在真实标签集内的比例 |" % dd["top1_acc"])
        L.append("| fallback_rate | %.4f | 触发回退（低置信 或 跨域）的比例 |"
                 % dd["fallback_rate"])
        L.append("| low_conf_rate | %.4f | 最大概率 < 阈值的比例 |" % dd["low_conf_rate"])
        L.append("| cross_domain_rate | %.4f | ≥2 类超阈值的比例 |" % dd["cross_domain_rate"])
        L.append("| routed_acc | %.4f | **接受单路由那部分**的准确率 |" % dd["routed_acc"])
        L.append("| routed_cov | %.4f | 接受单路由的覆盖率 |" % dd["routed_cov"])
        L.append("")
        L.append("混淆矩阵（top-1 vs 真实主标签），行=真值 列=预测：\n")
        M = r.get("confusion_top1")
        if M:
            L.append("| 真值\\预测 | " + " | ".join(d["labels"]) + " |")
            L.append("|" + "---|" * (len(d["labels"]) + 1))
            for i, name in enumerate(d["labels"]):
                L.append("| **%s** | %s |"
                         % (name, " | ".join(str(x) for x in M[i])))
        L.append("")
    L.append("## 5. 复现\n")
    L.append("```bash")
    L.append("python scripts/train/train_router.py \\")
    L.append("  --data %s \\" % d["data"])
    if d.get("eval_extra"):
        L.append("  --extra-eval %s \\" % d["eval_extra"]["path"])
    L.append("  --out-dir outputs/router/L1_linear \\")
    L.append("  --report-json docs/moe/ROUTER_L1.json \\")
    L.append("  --report-md docs/moe/ROUTER_L1.md")
    L.append("```")
    L.append("")
    L.append("总耗时 %.1fs。" % d["total_seconds"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
