#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2c：从**程序法条文**派生「程序法条文任务」，补程序法域的任务多样性。

输入  <root>/data/corpus/statute_items/*.jsonl    （阶段 2b 产物，只读）
输出  <root>/data/corpus/derived/statute_tasks.jsonl
报告  --out-json / --out-md

============================ 为什么需要它（实测驱动） ============================
阶段 4 实测：程序法真实训练池只有 **21,980** 条（目标 20,000 的 **1.10 倍**），且头号任务
`legal_question_answering` 独占池子 **49.3%**。这导致「单任务份额 ≤35%」的软上限
在程序法上**数学上不可达**（Σ min(cap, avail) < target），只能放宽上限并留痕
（实测 49.3%，cap_relaxed=true）。根因不是配额算法，是**程序法池子太窄、题型单一**。
接入本脚本产物后：池 21,980 → **35,464**、头号任务 49.3% → **35.0%**、cap_relaxed → **false**。

README 3.1 已把「诉讼法条文任务」列为程序法来源之一，本脚本实现该来源：
用阶段 2b 已切出的程序法条文（逐字权威文本）派生出**题型不同**的训练样本。

========================= 确定性与可复现性（说清边界） ==========================
本脚本**不含随机数**：输入扫描顺序（文件名排序）、派生顺序、丢弃顺序全部确定；
相同输入必然得到**相同的样本集合、相同的 uid、相同的 content_sha1、相同的行序**。

但产物的 **SHA-256 不是天然逐字节稳定的** —— 每条记录写
`normalized_at`（= 运行时刻），报告写 `generated_at`。这是唯一的非确定性来源。
→ 要得到逐字节一致的产物：`--stamp 2026-09-18T19:27:41`（传与上次相同的值）。
→ 生产运行留空即可；**复现实验/校验产物哈希时必须传 `--stamp`**。
（`content_sha1 = sha1(instruction, input, output)`，不含时间戳 —— 所以下游阶段 4 的
 选择与配比不受时间戳影响，这也是「重跑 2c 不会让阶段 4 失效」的原因。）

====================== 派生原则（违反任一条即产物不可信） ======================
1. **答案逐字取自原文，不做任何生成式改写** → 正确率 100%，不引入 LLM 幻觉。
   D1/D2 的答案就是条文正文或条号本身；D3 的答案是原文里的编/章/节标题字段。
   脚本自带 `verbatim` 自检（见报告第 3 节），任何一条对不上即 verdict=FAIL。
2. **只派生程序法**（`domain == "procedural"` 且 `level == "item"`）。民法/刑法/通用
   三域池子充足（超额），不引入合成数据。
3. **一律 `synthetic=true` + `derived=true`**。阶段 4 据此：
   * 只允许派生样本进 **train**；val/test 必须真实数据（评测公平性）；
   * 对派生样本设**组份额上限**并留痕（防「程序法专家只会背法条」）。

============================ 三种任务型（D1–D3） ============================
| 代号 | task | 输入 → 输出 | 不可用条件 |
|---|---|---|---|
| D1 | `statute_recall` | 法名+条号 → 条文正文 | 正文过短 |
| D2 | `statute_locate` | 条文正文 → 法名+条号（反向定位） | 正文过短 |
| D3 | `statute_structure` | 法名+条号 → 所在编/章/节 | 无章节字段 |

schema 与阶段 2 **完全一致**（同一套字段、同一套 `content_sha1` 算法、同一套
`messages` 渲染），便于阶段 4 直接并入同一池子；额外字段：
`derived` / `derivation` / `derived_from_uid` / `derived_from_law_title` /
`derived_from_source_dataset`。

只读约定：不修改任何输入文件。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

ROOT_DEFAULT = "/mnt/data/lidian/law-agent"
ITEMS_REL = "data/corpus/statute_items"
DERIVED_REL = "data/corpus/derived"
OUT_NAME = "statute_tasks.jsonl"

# 派生流在语料里的身份（stage 4 会把它当作一个独立 source 统计）
DERIVED_DATASET = "derived/statute-items"
DERIVATION_TAG = "statute_item_procedural:v1"
TARGET_DOMAIN = "procedural"
TARGET_LEVEL = "item"

MIN_TEXT_CHARS = 20      # 条文正文短于此不值得做成任务
MIN_LAW_CHARS = 4        # 法名过短（脏数据）跳过

TASK_KINDS = ("statute_recall", "statute_locate", "statute_structure")


# --------------------------------------------------------------- 基础工具
def norm_ws(s):
    """与 normalize_corpus.norm_ws 完全一致（NFKC + 空白折叠，保留换行）。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[ \t\r\f\v]+", " ", s.replace("\u3000", " ")).strip()


def sha1(*parts):
    """与 normalize_corpus.sha1 完全一致（各段 norm_ws 后以 0x1e 分隔）。"""
    h = hashlib.sha1()
    for p in parts:
        h.update(norm_ws(p).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def uid_of(dataset, source_file, source_index):
    """与 normalize_corpus.uid_of / 阶段 4 的 uid_g 同口径（唯一性由构造成立）。"""
    return "%s__%s:%s" % ((dataset or "?").replace("/", "__"),
                          os.path.splitext(os.path.basename(source_file or "?"))[0],
                          source_index)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                yield lineno, json.loads(line)


# --------------------------------------------------------------- 派生规则
def unwrap_title(t):
    """剥掉法规名自带的书名号，避免模板再包一层变成 `《《…》》`。

    ⚠️ 实测坑（2026-09-18 重跑时发现）：`law_title` 在源数据里**已经带 `《》`**
    （如 `《中华人民共和国刑事诉讼法(2018修正)》`），而模板又写 `"《%s》" % law`
    → 生成 `请说明《《…》》第四十八条…`。双层书名号会作为**格式噪声**进 SFT 训练集。
    这里统一在**渲染处**剥壳（`derive_one` / `check_verbatim`），长度过滤仍用原始标题，
    以免改变既有 drop 口径。
    """
    t = norm_ws(t)
    while len(t) >= 2 and t[0] == "《" and t[-1] == "》":
        t = t[1:-1].strip()
    return t


def derive_one(item, kind):
    """单条条文 → 单条派生样本（不含 uid/index，由调用方填充）。

    返回 (instruction, input, output, refs_q, refs_a) 或 None（不可用）。
    答案一律来自 item 的原文/元数据字段，绝不改写。
    """
    law = unwrap_title(item.get("law_title") or "")
    label = norm_ws(item.get("article_label") or "")
    text = norm_ws(item.get("text") or "")
    ref = "《%s》" % law

    if kind == "statute_recall":
        if len(law) < MIN_LAW_CHARS or not label or len(text) < MIN_TEXT_CHARS:
            return None
        ins = "请说明%s%s规定了什么。" % (ref, label)
        return ins, "", text, [ref], []

    if kind == "statute_locate":
        if len(law) < MIN_LAW_CHARS or not label or len(text) < MIN_TEXT_CHARS:
            return None
        ins = "请判断下列规定出自哪一部法律、哪一条，并给出法律名称与条号。"
        ans = "%s%s" % (ref, label)
        return ins, text, ans, [], [ref]

    if kind == "statute_structure":
        if len(law) < MIN_LAW_CHARS or not label:
            return None
        levels = [norm_ws(item.get(k) or "") for k in ("part", "chapter", "section")]
        levels = [x for x in levels if x]
        if not levels:
            return None
        ins = "%s%s位于该法的哪一编、哪一章、哪一节？请给出完整的层级名称。" % (ref, label)
        ans = "、".join(levels)
        return ins, "", ans, [ref], []

    raise ValueError("unknown kind: %s" % kind)


def build_record(item, kind, ins, inp, out, refs_q, refs_a, idx, now):
    """组装成与阶段 2 完全同构的记录（含 messages 渲染）。"""
    task_kind = "qa"
    system = ""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    user = (ins + "\n" + inp).strip() if ins else inp
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": out})

    rec = {
        "source_dataset": DERIVED_DATASET,
        "source_file": OUT_NAME,
        "source_index": idx,
        "source_id": item.get("uid") or "",
        "source_license": item.get("source_license") or "",
        "source_grade": item.get("source_grade") or "",
        "task": kind,
        "task_kind": task_kind,
        "system": system,
        "instruction": ins,
        "input": inp,
        "output": out,
        "references": [],
        "refs_question": refs_q,
        "refs_answer": refs_a,
        "ask": ins if ins else inp[:300],
        "task_domain_hint": None,
        "lang": "zh",
        # ★ 合成标记：阶段 4 只允许其进 train（val/test 必须真实数据）
        "synthetic": True,
        "replay": False,
        "derived": True,
        "derivation": DERIVATION_TAG,
        "derived_from_uid": item.get("uid") or "",
        "derived_from_law_title": norm_ws(item.get("law_title") or ""),
        "derived_from_source_dataset": item.get("source_dataset") or "",
        "char_len": {"instruction": len(ins), "input": len(inp), "output": len(out)},
        "normalized_at": now,
        # 域标签：来源即程序法条文，无需启发式判定
        "domain": TARGET_DOMAIN,
        "domains": [TARGET_DOMAIN],
        "domain_source": "derived_from_statute_item",
        "domain_evidence": {"rule": "source item domain==procedural",
                            "law_title": norm_ws(item.get("law_title") or ""),
                            "source_uid": item.get("uid") or ""},
    }
    rec["uid"] = uid_of(DERIVED_DATASET, OUT_NAME, idx)
    rec["content_sha1"] = sha1(ins, inp, out)
    rec["case_sha1"] = sha1(inp[-400:]) if inp else ""
    rec["messages"] = msgs
    return rec


# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--items-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--types", default=",".join(TASK_KINDS),
                    help="要派生的任务型（逗号分隔）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--stamp", default="",
                    help="固定时间戳（YYYY-MM-DDTHH:MM:SS）。留空 = 用当前时间。"
                         "每条记录写 normalized_at、报告写 generated_at，二者同源 ——"
                         "**这是本脚本唯一的非确定性来源**。要逐字节复现产物就传与上次相同的值。")
    a = ap.parse_args()

    kinds = [x.strip() for x in a.types.split(",") if x.strip()]
    bad = [k for k in kinds if k not in TASK_KINDS]
    if bad:
        sys.exit("未知任务型：%s（可选 %s）" % (bad, list(TASK_KINDS)))

    items_dir = a.items_dir or os.path.join(a.root, ITEMS_REL)
    out_dir = a.out_dir or os.path.join(a.root, DERIVED_REL)
    if not os.path.isdir(items_dir):
        sys.exit("法条条目目录不存在：%s" % items_dir)
    os.makedirs(out_dir, exist_ok=True)
    now = a.stamp.strip() or time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()

    files = sorted(f for f in os.listdir(items_dir)
                   if f.endswith(".jsonl") and not f.startswith("_"))

    # ---------------- 输入画像 ----------------
    in_by_domain = collections.Counter()
    in_by_level = collections.Counter()
    in_by_task = collections.Counter()
    in_files = {}
    n_items = 0
    for fn in files:
        n = 0
        for _, it in iter_jsonl(os.path.join(items_dir, fn)):
            n += 1
            n_items += 1
            in_by_domain[it.get("domain") or "<empty>"] += 1
            in_by_level[it.get("level") or "<empty>"] += 1
            in_by_task[it.get("task") or "<empty>"] += 1
        in_files[fn] = n

    # ---------------- 派生 ----------------
    out_path = os.path.join(out_dir, a.out_name)
    produced = collections.Counter()
    dropped = collections.Counter()          # "kind|reason"
    per_law = collections.Counter()
    per_type_sample = collections.defaultdict(list)
    seen_sha = {}
    seen_uid = {}
    verbatim_bad = []
    idx = 0
    n_cand = 0                              # 进入候选的程序法条级条目数
    n_proc_laws = set()

    with open(out_path, "w", encoding="utf-8") as fo:
        for fn in files:
            for _, it in iter_jsonl(os.path.join(items_dir, fn)):
                if (it.get("domain") or "") != TARGET_DOMAIN:
                    continue
                if (it.get("level") or "item") != TARGET_LEVEL:
                    continue
                n_cand += 1
                law = norm_ws(it.get("law_title") or "")
                if law:
                    n_proc_laws.add(law)
                if len(law) < MIN_LAW_CHARS:
                    dropped["*|short_law_title"] += 1
                    continue
                for kind in kinds:
                    got = derive_one(it, kind)
                    if got is None:
                        dropped["%s|not_applicable" % kind] += 1
                        continue
                    ins, inp, out, rq, ra = got
                    if not out:
                        dropped["%s|empty_output" % kind] += 1
                        continue
                    rec = build_record(it, kind, ins, inp, out, rq, ra, idx, now)

                    # 唯一性：content_sha1 全局唯一是阶段 4 的样本键前提。
                    # 实测确有必要：同一部法不同历史版本可能条号+正文完全相同。
                    if rec["content_sha1"] in seen_sha:
                        dropped["%s|dup_content" % kind] += 1
                        continue
                    seen_sha[rec["content_sha1"]] = rec["uid"]
                    seen_uid[rec["uid"]] = 1

                    # 逐字自检：答案必须能在原文/元数据里原样找到
                    ok = check_verbatim(it, kind, out)
                    if not ok:
                        verbatim_bad.append({"uid": rec["uid"], "task": kind,
                                             "output": out[:120],
                                             "law": law, "label": it.get("article_label")})

                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    produced[kind] += 1
                    per_law[law] += 1
                    if len(per_type_sample[kind]) < 4:
                        per_type_sample[kind].append(rec)
                    idx += 1

    # ---------------- 汇总（重读一遍产物做独立校验，不复用生成期的中间状态） ----------------
    dom_dist = collections.Counter()
    law_dist = collections.Counter()
    synth_bad = 0
    wrap_bad = []            # 书名号套层（`《《…》》`）—— 格式噪声，不允许多于 0 条
    out_sha1 = set()
    out_uids = set()
    n_out = 0
    for _, r in iter_jsonl(out_path):
        n_out += 1
        dom_dist[r.get("domain")] += 1
        law_dist[r.get("derived_from_law_title")] += 1
        out_sha1.add(r.get("content_sha1") or "")
        out_uids.add(r.get("uid") or "")
        if not r.get("synthetic") or not r.get("derived"):
            synth_bad += 1
        blob = "%s %s %s" % (r.get("instruction") or "", r.get("input") or "",
                             r.get("output") or "")
        if "《《" in blob or "》》" in blob:
            wrap_bad.append({"uid": r.get("uid"), "task": r.get("task")})

    total_out = n_out
    stats = {
        "generated_by": "scripts/corpus/derive_statute_tasks.py",
        "generated_at": now,
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "elapsed_sec": round(time.time() - t0, 1),
        "config": {
            "items_dir": items_dir, "out_path": out_path, "types": kinds,
            "target_domain": TARGET_DOMAIN, "target_level": TARGET_LEVEL,
            "min_text_chars": MIN_TEXT_CHARS, "derivation": DERIVATION_TAG,
            "stamp": a.stamp.strip() or "(now)",
        },
        "input": {
            "files": in_files, "total_items": n_items,
            "by_domain": dict(in_by_domain.most_common()),
            "by_level": dict(in_by_level.most_common()),
            "by_task": dict(in_by_task.most_common()),
            "procedural_item_candidates": n_cand,
            "procedural_distinct_laws": len(n_proc_laws),
        },
        "output": {
            "path": out_path,
            "total": total_out,
            "rows_reread": n_out,
            "generated": sum(produced.values()),
            "by_task": dict(produced.most_common()),
            "by_domain": dict(dom_dist.most_common()),
            "distinct_laws": len(law_dist),
            "unique_content_sha1": len(out_sha1),
            "unique_uid": len(out_uids),
            "synthetic_flag_bad": synth_bad,
            "sha256": sha256_file(out_path),
            "bytes": os.path.getsize(out_path),
        },
        "drop_reasons": dict(dropped.most_common()),
        "verbatim_check": {
            "checked": total_out,
            "mismatch": len(verbatim_bad),
            "samples": verbatim_bad[:10],
        },
        "title_wrap_check": {
            "checked": total_out,
            "nested_brackets": len(wrap_bad),
            "samples": wrap_bad[:10],
        },
        "samples": {k: [{kk: vv for kk, vv in r.items() if kk != "messages"}
                        for r in v] for k, v in per_type_sample.items()},
    }
    stats["verdict"] = (
        "PASS" if (total_out > 0 and not verbatim_bad and synth_bad == 0
                   and not wrap_bad
                   and len(out_sha1) == total_out
                   and len(out_uids) == total_out
                   and n_out == sum(produced.values())) else "FAIL")

    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    if a.out_md:
        render_md(stats, a.out_md)

    print("VERDICT=%s" % stats["verdict"])
    print("IN items=%d (procedural level=item: %d, laws=%d)"
          % (n_items, n_cand, len(n_proc_laws)))
    print("OUT total=%d  by_task=%s" % (total_out, dict(produced.most_common())))
    print("unique_sha1=%d unique_uid=%d verbatim_mismatch=%d synthetic_bad=%d"
          % (stats["output"]["unique_content_sha1"], stats["output"]["unique_uid"],
             len(verbatim_bad), synth_bad))
    print("dropped=%s" % dict(dropped.most_common(8)))
    print("OK -> %s" % out_path)
    return 0 if stats["verdict"] == "PASS" else 4


def check_verbatim(item, kind, out):
    """答案逐字自检：派生输出必须能在原条目字段里原样找到。"""
    text = norm_ws(item.get("text") or "")
    law = unwrap_title(item.get("law_title") or "")
    label = norm_ws(item.get("article_label") or "")
    if kind == "statute_recall":
        return out == text
    if kind == "statute_locate":
        return out == "%s%s" % ("《%s》" % law, label)
    if kind == "statute_structure":
        levels = [norm_ws(item.get(k) or "") for k in ("part", "chapter", "section")]
        return out == "、".join(x for x in levels if x)
    return False


def render_md(S, path):
    with open(path, "w", encoding="utf-8") as fh:
        W = lambda s="": fh.write(s + "\n")
        W("# 阶段 2c：程序法条文任务派生报告")
        W()
        W("**verdict = %s**" % S["verdict"])
        W()
        W("> 生成脚本 `scripts/corpus/derive_statute_tasks.py`（只读输入、无随机数）")
        W("> 生成时间 %s（服务器 %s，耗时 %.1fs）"
          % (S["generated_at"], S["host"], S["elapsed_sec"]))
        W("> **`--stamp` = `%s`** —— 这是产物里唯一的非确定性来源（每条 `normalized_at`）；"
          % S["config"]["stamp"])
        W("> 传固定值即可逐字节复现产物（样本集合/顺序/uid/content_sha1 本来就确定）。")
        W()
        W("## 0. 口径与动机")
        W()
        W("- 输入：`%s/*.jsonl`（阶段 2b 法条条目，只读）" % S["config"]["items_dir"])
        W("- 输出：`%s`" % S["config"]["out_path"])
        W("- **动机**：阶段 4 实测程序法真实池仅目标的 1.10 倍（21,980 / 20,000）、"
          "头号任务独占 49.3%，"
          "「单任务份额 ≤35%」数学上不可达 → 只能放宽并留痕。根因是**池子太窄、题型单一**，"
          "本脚本按 README 3.1 地把「诉讼法条文任务」这一来源补齐。")
        W("- 只派生 `domain=%s` 且 `level=%s` 的条目；**答案逐字取自原文，不做生成式改写**。"
          % (S["config"]["target_domain"], S["config"]["target_level"]))
        W("- 全部样本 `synthetic=true` + `derived=true`：阶段 4 只允许其进 **train**，"
          "val/test 保持真实数据。")
        W()
        W("## 1. 输入画像")
        W()
        W("| 项 | 值 |")
        W("|---|---|")
        W("| 条目总数 | %d |" % S["input"]["total_items"])
        W("| 程序法条级候选（可派生） | **%d** |" % S["input"]["procedural_item_candidates"])
        W("| 覆盖程序法法规数 | %d |" % S["input"]["procedural_distinct_laws"])
        W("| 逐文件 | %s |" % "、".join("%s %d" % (k, v)
                                      for k, v in S["input"]["files"].items()))
        W("| 输入逐域 | %s |" % "、".join("%s %d" % (k, v)
                                        for k, v in S["input"]["by_domain"].items()))
        W()
        W("## 2. 派生结果")
        W()
        W("| 任务型 | 产出条数 |")
        W("|---|---|")
        for k in TASK_KINDS:
            W("| `%s` | **%d** |" % (k, S["output"]["by_task"].get(k, 0)))
        W("| **合计** | **%d** |" % S["output"]["total"])
        W()
        W("覆盖程序法法规 **%d** 部。" % S["output"]["distinct_laws"])
        W()
        if S["drop_reasons"]:
            W("未派生（按原因计数，不静默丢）：")
            W()
            W("| 任务型\\|原因 | 条数 |")
            W("|---|---|")
            for k, v in S["drop_reasons"].items():
                W("| %s | %d |" % (k, v))
            W()
        W("## 3. 正确性与唯一性自检")
        W()
        W("| 检查 | 结果 |")
        W("|---|---|")
        W("| 逐字自检（答案能在原文找到） | %d / %d 通过，不符 **%d** |"
          % (S["verbatim_check"]["checked"] - S["verbatim_check"]["mismatch"],
             S["verbatim_check"]["checked"], S["verbatim_check"]["mismatch"]))
        W("| `content_sha1` 唯一 | %d / %d |"
          % (S["output"]["unique_content_sha1"], S["output"]["total"]))
        W("| `uid` 唯一 | %d / %d |" % (S["output"]["unique_uid"], S["output"]["total"]))
        W("| `synthetic`/`derived` 标记完整 | 异常 %d 条 |" % S["output"]["synthetic_flag_bad"])
        W("| 输出 SHA-256 | `%s` |" % S["output"]["sha256"])
        W()
        if S["verbatim_check"]["samples"]:
            W("⚠️ 逐字自检不符样例：")
            W()
            for s in S["verbatim_check"]["samples"]:
                W("- `%s`（%s）：%s" % (s["uid"], s["task"], s["output"]))
            W()
        W("## 4. 与阶段 4 的接口")
        W()
        W("- 阶段 4 直接读 `%s/`，把本流视为独立 source（`%s`）。"
          % (os.path.dirname(S["config"]["out_path"]), DERIVED_DATASET))
        W("- **只进 train**：router / val / test 一律只用真实数据（脚本断言）。")
        W("- **派生份额上限**：`--derived-max-share`（默认 0.15）限制派生样本占该域"
          "训练配额的比重，溢出部分自动放宽并留痕 —— 目的是防「程序法专家只会背法条」。")
        W("- 派生样本不参与评测，也不进入 `data/corpus/statute_items/` 的检索语料。")
        W()
        W("## 5. 样例（每型最多 4 条）")
        W()
        for k in TASK_KINDS:
            ss = S["samples"].get(k) or []
            if not ss:
                continue
            W("### `%s`" % k)
            W()
            for r in ss:
                W("- **Q**：%s" % r["instruction"])
                if r["input"]:
                    W("  - input：%s" % r["input"][:110].replace("\n", " "))
                W("  - **A**：%s" % r["output"][:160].replace("\n", " "))
            W()


if __name__ == "__main__":
    sys.exit(main())
