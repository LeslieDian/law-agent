#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 5c：生成答案的客观打分（不依赖外部 API、不占 GPU）。

配套 `scripts/train/run_inference.py` 产出的 answers.jsonl 使用。

三类指标
--------------------------------------------------------------------------
1. **选择题 Accuracy**（LexEval 1_x–4_x / 6_x；对齐官方 `--metrics_choice Accuracy`）
   从生成文本里抽取选项字母集合，与 gold 比对：
     `exact`   预测集合 == gold 集合（严格，官方口径）
     `subset`  gold ⊆ 预测（宽松：多选了也算命中）
2. **ROUGE-L**（LexEval 5_x 生成题 + 内部 test）
   按**字符**做 LCS，beta=1.2（与 `rouge_score` 库同口径）；长文本只取前 N 字符
   参与计算并在报告里标注，避免 O(n·m) 爆炸。
3. **法条引用命中率**（内部 test 专有；法律领域最有解释力的一个指标）
   从 gold 抽 `《法名》第N条` 二元组，看生成文本是否含同名同条。

为什么不用 `rouge_score` 包
--------------------------------------------------------------------------
远程 envs/main 里只有 torch/transformers/peft/trl/datasets/accelerate/bitsandbytes
（实测），为免引入新依赖；且该库按空格分词，中文会被切成整段，ROUGE 会失真。
自己按字符实现更准也更可控。

用法
--------------------------------------------------------------------------
    python scripts/eval/score_answers.py --task lexeval \
        --answers outputs/infer/lexeval_A0/answers.jsonl \
        --out outputs/score/lexeval_A0.json

    python scripts/eval/score_answers.py --task internal_test \
        --answers outputs/infer/A0_internal_test/answers.jsonl \
        --out outputs/score/A0_internal_test.json --metrics all

★ 分域结果（判断"要不要训单域专家"的关键依据）：加上 `--group-by-domain`，
  报告的 by_split 就会变成 civil / criminal / procedural / general 四域分层：

    python scripts/eval/score_answers.py --task internal_test \
        --answers outputs/infer/A0_internal_test/answers.jsonl \
        --group-by-domain data/test/test.jsonl \
        --out outputs/score/A0_internal_test_bydomain.json --metrics all
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def jprint(*a):
    print(*a, flush=True)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_uid_domain_map(path: str) -> dict:
    """从 test.jsonl 建 uid -> domain 映射。

    ★ 为什么需要它：run_inference.py 对 internal_test 把 `split` 一律写成 "test"，
    所以 by_split 拿不到分域结果；真正的域标签只存在于原始 test.jsonl 的 `domain` 字段。
    case_id 形如 "test::<uid>"，其中 uid 取的是 `uid or uid_g or source_id`，
    与 run_inference.load_records 的取值顺序必须保持一致。
    """
    m = {}
    for r in iter_jsonl(path):
        key = r.get("uid") or r.get("uid_g") or r.get("source_id")
        dom = r.get("domain")
        if key and dom:
            m[str(key)] = dom
    return m


def domain_of(case_id: str, uid2dom: dict) -> str:
    """case_id 'test::<uid>' -> domain；查不到返回 'unknown'。"""
    cid = str(case_id or "")
    key = cid.split("::", 1)[1] if "::" in cid else cid
    return uid2dom.get(key, "unknown")


def regroup_by_domain(recs: list, uid2dom: dict) -> dict:
    """把每条记录的 `split` 覆写成它的域标签（就地），返回域计数。

    只在 internal_test 上调用 —— 其它任务的 split 本身有意义（lexeval 的类别、
    lexrubric 的 legal_consultation/judicial_exam），不能被覆写。
    """
    cnt = {}
    for r in recs:
        d = domain_of(r.get("case_id"), uid2dom)
        r["split"] = d
        cnt[d] = cnt.get(d, 0) + 1
    return cnt


# ---- 选择题答案抽取 -------------------------------------------------------
# ★ 实测（2026-09-20，A0 冒烟 20 条）：模型输出形如
#     "C: 犯罪行为尚未实行完毕的情况下\n解析:..."
#   即它**先复述选项字母+选项原文，再解释**，而不是裸字母。
#   首版只认「整行纯字母」→ 15/20 抽不出来（unparsed），Accuracy 假 0。
#   这里按三种形态依次尝试：前缀式 → 关键词式 → 短文本兜底。
_CHOICE_HEAD_RE = re.compile(
    r"^\s*[\*#\->\s]*([A-Fa-f]{1,6})\s*(?=[:：、,，.。\s]|$)")
_CHOICE_KEY_RE = re.compile(
    r"(?:答案|正确选项|应选|选项|选择|正确答案)[^\nA-Fa-f]{0,8}([A-Fa-f]{1,6})")
_CHOICE_LOOSE_RE = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")


def extract_choices(text: str) -> set:
    """从自由文本里抽出选项字母集合。抽不出返回空集（按错处理）。"""
    t = (text or "").strip()
    if not t:
        return set()
    m = _CHOICE_HEAD_RE.match(t)
    if m and len(m.group(1)) <= 6:
        return set(m.group(1).upper())
    m = _CHOICE_KEY_RE.search(t[:400])
    if m:
        return set(m.group(1).upper())
    # 兜底：只在短文本里做孤立大写字母抽取 —— 长文本里句首的 "A" 会误伤
    if len(t) <= 60:
        return set(_CHOICE_LOOSE_RE.findall(t))
    return set()


def is_choice_gold(gold: str) -> bool:
    g = (gold or "").strip()
    return bool(g) and len(g) <= 5 and bool(re.fullmatch(r"[A-Fa-f]+", g))


# ---- ROUGE-L（字符级 LCS，beta=1.2） --------------------------------------
def lcs_len(a: str, b: str) -> int:
    """最长公共子序列长度（滚动数组，O(min) 空间）。"""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else (prev[j] if prev[j] >= cur[j - 1]
                                                     else cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str, beta: float = 1.2, cap: int = 800) -> dict:
    p = (pred or "").strip()[:cap]
    g = (gold or "").strip()[:cap]
    if not p or not g:
        return {"f": 0.0, "p": 0.0, "r": 0.0, "truncated": False}
    n = lcs_len(p, g)
    prec = n / len(p)
    rec = n / len(g)
    b2 = beta * beta
    f = 0.0 if (prec + rec) == 0 else (1 + b2) * prec * rec / (rec + b2 * prec)
    return {"f": round(f, 4), "p": round(prec, 4), "r": round(rec, 4),
            "truncated": len(pred or "") > cap or len(gold or "") > cap}


# ---- 法条引用命中 ---------------------------------------------------------
_STAT_RE = re.compile(r"《([^》\n]{1,40}?)》\s*第\s*([零一二三四五六七八九十百千0-9]+)\s*条")


def statute_set(text: str) -> set:
    return {(m.group(1).strip(), m.group(2).strip()) for m in _STAT_RE.finditer(text or "")}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="生成答案客观打分")
    ap.add_argument("--answers", required=True, help="run_inference.py 产出的 answers.jsonl")
    ap.add_argument("--task", required=True,
                    choices=["lexeval", "internal_test", "lexrubric"])
    ap.add_argument("--out", required=True, help="结果 json 路径")
    ap.add_argument("--metrics", default="auto",
                    help="auto | all | accuracy | rouge_l | statute_hit")
    ap.add_argument("--rouge-cap", type=int, default=800)
    ap.add_argument("--group-by-domain", default=None,
                    help="internal_test 专用：传 data/test/test.jsonl，"
                         "把 by_split 换成按 domain 分层（civil/criminal/"
                         "procedural/general）")
    a = ap.parse_args()

    recs = [r for r in iter_jsonl(a.answers) if not r.get("error")]
    n_err = sum(1 for r in iter_jsonl(a.answers) if r.get("error"))
    jprint("读到 %d 条有效答案（另有 %d 条 error 被排除）" % (len(recs), n_err))
    if not recs:
        jprint("[FATAL] 没有有效答案")
        return 2

    if a.group_by_domain:
        uid2dom = build_uid_domain_map(a.group_by_domain)
        cnt = regroup_by_domain(recs, uid2dom)
        jprint("按 domain 分层：%s（映射表 %d 条）" % (cnt, len(uid2dom)))
        if cnt.get("unknown"):
            jprint("  [WARN] %d 条 case_id 在 %s 里查不到 domain —— 检查 uid 取值口径"
                   % (cnt["unknown"], a.group_by_domain))

    want = set()
    if a.metrics == "auto":
        want = {"accuracy", "rouge_l"} if a.task == "lexeval" else \
               {"rouge_l", "statute_hit"}
    elif a.metrics == "all":
        want = {"accuracy", "rouge_l", "statute_hit"}
    else:
        want = {m.strip() for m in a.metrics.split(",")}

    result = {"task": a.task, "answers": os.path.abspath(a.answers),
              "n_valid": len(recs), "n_error": n_err, "metrics": {}}

    # ---- 1) 选择题 Accuracy ----
    if "accuracy" in want:
        per_task = collections.defaultdict(lambda: {"n": 0, "exact": 0, "subset": 0,
                                                    "unparsed": 0})
        n_exact = n_subset = n_choice = n_unparsed = 0
        for r in recs:
            gold = str(r.get("reference", "")).strip().upper()
            if not is_choice_gold(gold):
                continue                     # 生成类题不参与选择题指标
            n_choice += 1
            pred = extract_choices(r.get("answer", ""))
            key = r.get("split", "?")
            per_task[key]["n"] += 1
            if not pred:
                n_unparsed += 1
                per_task[key]["unparsed"] += 1
                continue
            gset = set(gold)
            if pred == gset:
                n_exact += 1
                n_subset += 1
                per_task[key]["exact"] += 1
                per_task[key]["subset"] += 1
            elif gset <= pred:
                n_subset += 1
                per_task[key]["subset"] += 1
        result["metrics"]["accuracy"] = {
            "n_choice_questions": n_choice,
            "exact_match": round(n_exact / n_choice, 4) if n_choice else None,
            "subset_match": round(n_subset / n_choice, 4) if n_choice else None,
            "unparsed": n_unparsed,
            "unparsed_rate": round(n_unparsed / n_choice, 4) if n_choice else None,
            "by_task": {k: {"n": v["n"],
                            "exact": round(v["exact"] / v["n"], 4) if v["n"] else None,
                            "subset": round(v["subset"] / v["n"], 4) if v["n"] else None,
                            "unparsed": v["unparsed"]}
                        for k, v in sorted(per_task.items())},
        }
        jprint("选择题 Accuracy: n=%d  exact=%.4f  subset=%.4f  unparsed=%d"
               % (n_choice,
                  (n_exact / n_choice) if n_choice else 0,
                  (n_subset / n_choice) if n_choice else 0,
                  n_unparsed))

    # ---- 2) ROUGE-L ----
    if "rouge_l" in want:
        fs = []
        by_split = collections.defaultdict(list)
        n_trunc = 0
        for r in recs:
            gold = str(r.get("reference", ""))
            if not gold.strip():
                continue
            s = rouge_l(r.get("answer", ""), gold, cap=a.rouge_cap)
            fs.append(s["f"])
            by_split[r.get("split", "?")].append(s["f"])
            n_trunc += int(s["truncated"])

        def _mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        result["metrics"]["rouge_l"] = {
            "n": len(fs),
            "f_mean": _mean(fs),
            "f_p50": round(sorted(fs)[len(fs) // 2], 4) if fs else None,
            "n_truncated": n_trunc,
            "cap": a.rouge_cap,
            "by_split": {k: {"n": len(v), "f_mean": _mean(v)}
                         for k, v in sorted(by_split.items())},
        }
        jprint("ROUGE-L: n=%d  f_mean=%s" % (len(fs), _mean(fs)))

    # ---- 3) 法条引用命中 ----
    if "statute_hit" in want:
        hits, goldsizes, by_split = [], [], collections.defaultdict(list)
        for r in recs:
            g = statute_set(r.get("reference", ""))
            if not g:
                continue
            p = statute_set(r.get("answer", ""))
            h = len(g & p) / len(g)
            hits.append(h)
            goldsizes.append(len(g))
            by_split[r.get("split", "?")].append(h)
        if hits:
            result["metrics"]["statute_hit"] = {
                "n": len(hits),
                "hit_rate_mean": round(sum(hits) / len(hits), 4),
                "avg_gold_statutes": round(sum(goldsizes) / len(goldsizes), 2),
                "by_split": {k: {"n": len(v),
                                 "hit_rate": round(sum(v) / len(v), 4)}
                             for k, v in sorted(by_split.items())},
            }
            jprint("法条引用命中率: n=%d  mean=%.4f  平均 gold 法条数=%.2f"
                   % (len(hits), sum(hits) / len(hits),
                      sum(goldsizes) / len(goldsizes)))
        else:
            result["metrics"]["statute_hit"] = {"n": 0, "note": "gold 里没有可解析的法条引用"}
            jprint("法条引用命中率: gold 里没有可解析的法条引用")

    p = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    jprint("结果:", p)
    print("MARKER_SCORE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
