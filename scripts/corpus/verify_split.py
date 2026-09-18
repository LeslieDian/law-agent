#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4 切分质检（硬门禁）：verdict PASS 才允许进入 GPU 阶段。

检查项（C 系列，全部对**全量**算，不用样例代表总体）：
  C1 输出文件存在、行数与 `SPLIT_STATS.json` 声明一致
  C2 `uid_g` 全局唯一（修复阶段 2 uid 撞号后的唯一键）
  C3 **四份 split 两两不相交**（uid_g 与 content_sha1 双重口径）★ 阶段 4 硬卡口
  C4 训练集逐域配比 = 目标配比（不是「接近」，是相等）
  C5 任务覆盖：池里有的每个 (domain, task) 在训练集里都有；且单任务份额不超软上限
     （若 `SPLIT_STATS.json` 已留痕 `cap_relaxed`，则不判 FAIL 但计入警告）
  C6 val / test **不含合成数据**（synthetic=false）
  C7 样本可用性：messages 非空且含 user/assistant 两段
  C8 字段合法性：domain ∈ 四域、task 非空
  C9 逐域视图文件与主文件的行数/内容一致
  C10 路由集自带 `router_label` 与 `router_query`
  C11 逐文件 SHA-256 与 `SPLIT_STATS.json` 登记值一致

用法：
  python scripts/corpus/verify_split.py --root /mnt/data/lidian/law-agent \
      --expect docs/corpus/SPLIT_STATS.json \
      --out-json docs/corpus/SPLIT_VERIFY.json --out-md docs/corpus/SPLIT_VERIFY.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

DOMAINS = ("criminal", "civil", "procedural", "general")
DOMAIN_FILE = {"criminal": "criminal", "civil": "civil",
               "procedural": "procedure", "general": "general"}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--expect", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--train-dir", default="")
    ap.add_argument("--dev-dir", default="")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--router-dir", default="")
    args = ap.parse_args()

    exp = json.load(open(args.expect, encoding="utf-8"))
    train_dir = args.train_dir or os.path.join(args.root, "data/train")
    dev_dir = args.dev_dir or os.path.join(args.root, "data/dev")
    test_dir = args.test_dir or os.path.join(args.root, "data/test")
    router_dir = args.router_dir or os.path.join(args.root, "data/router")

    files = {
        "train": os.path.join(train_dir, "train.jsonl"),
        "val": os.path.join(dev_dir, "dev.jsonl"),
        "test": os.path.join(test_dir, "test.jsonl"),
        "router": os.path.join(router_dir, "router_train.jsonl"),
    }
    res = {"verdict": "PASS", "failures": [], "warnings": [], "checks": {}}
    uids = {}
    shas = {}
    counts = {}
    domains = {}
    tasks = {}
    domain_tasks = {}
    msg_bad = collections.Counter()
    synth_bad = collections.Counter()
    field_bad = collections.Counter()
    router_bad = collections.Counter()

    # ---------------- C1 / C2 / C6 / C7 / C8 / C10
    for split, path in files.items():
        if not os.path.isfile(path):
            res["failures"].append("C1 缺少输出文件：%s" % path)
            continue
        uids[split] = set()
        shas[split] = set()
        domains[split] = collections.Counter()
        tasks[split] = collections.Counter()
        domain_tasks[split] = collections.Counter()
        n = 0
        for rec in iter_jsonl(path):
            n += 1
            ug = rec.get("uid_g") or ""
            sh = rec.get("content_sha1") or ""
            if not ug:
                field_bad["%s:missing_uid_g" % split] += 1
            uids[split].add(ug)
            shas[split].add(sh)
            d = rec.get("domain") or ""
            tk = rec.get("task") or ""
            domains[split][d] += 1
            tasks[split][tk] += 1
            domain_tasks[split]["%s|%s" % (d, tk)] += 1
            if d not in DOMAINS:
                field_bad["%s:domain_not_in_4" % split] += 1
            if not tk:
                field_bad["%s:empty_task" % split] += 1
            if rec.get("split") != split:
                field_bad["%s:split_field_mismatch" % split] += 1
            if split in ("val", "test") and rec.get("synthetic"):
                synth_bad[split] += 1
            msgs = rec.get("messages") or []
            roles = [m.get("role") for m in msgs]
            if "user" not in roles or "assistant" not in roles:
                msg_bad[split] += 1
            else:
                for m in msgs:
                    if m.get("role") in ("user", "assistant") and not (m.get("content") or "").strip():
                        msg_bad[split] += 1
                        break
            if split == "router":
                if not rec.get("router_label") or not (rec.get("router_query") or "").strip():
                    router_bad["missing_label_or_query"] += 1
        counts[split] = n
        uids[split].discard("")
        shas[split].discard("")

    res["checks"]["C1_counts"] = {
        "declared": {k: exp["splits"][k] for k in ("train", "val", "test", "router")},
        "actual": counts,
        "match": all(counts.get(k) == exp["splits"][k]
                     for k in ("train", "val", "test", "router")),
    }
    if not res["checks"]["C1_counts"]["match"]:
        res["failures"].append("C1 行数与 SPLIT_STATS.json 不一致：%s"
                               % res["checks"]["C1_counts"])

    dup = {s: counts[s] - len(uids[s]) for s in uids}
    res["checks"]["C2_uid_g_unique"] = dup
    if any(v > 0 for v in dup.values()):
        res["failures"].append("C2 uid_g 有重复：%s" % dup)

    # ---------------- C3 两两不相交（硬卡口）
    pairs = {}
    names = ["train", "val", "test", "router"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            iu = uids.get(a, set()) & uids.get(b, set())
            ish = shas.get(a, set()) & shas.get(b, set())
            pairs["%s∩%s" % (a, b)] = {"uid_g": len(iu), "content_sha1": len(ish),
                                       "samples": sorted(iu)[:3]}
    res["checks"]["C3_disjoint"] = pairs
    bad = {k: v for k, v in pairs.items() if v["uid_g"] or v["content_sha1"]}
    if bad:
        res["failures"].append("C3 切分相交（硬卡口失败）：%s" % bad)

    # ---------------- C4 逐域配比
    mix = exp["config"]["mix"]
    got = {d: domains.get("train", collections.Counter()).get(d, 0) for d in DOMAINS}
    res["checks"]["C4_train_mix"] = {
        "target": {d: mix[d] for d in DOMAINS},
        "actual": got,
        "match": got == {d: mix[d] for d in DOMAINS},
        "total": sum(got.values()),
    }
    if not res["checks"]["C4_train_mix"]["match"]:
        res["failures"].append("C4 训练集逐域配比与目标不符：%s"
                               % res["checks"]["C4_train_mix"])

    # ---------------- C5 任务覆盖 + 份额上限
    train_dt = domain_tasks.get("train", collections.Counter())
    missing_tasks = []
    weak_tasks = []
    for d in DOMAINS:
        avail = {k.split("|", 1)[1]: v for k, v in exp["task_avail"].items()
                 if k.startswith(d + "|")}
        for t in avail:
            got_n = train_dt.get("%s|%s" % (d, t), 0)
            if got_n == 0:
                missing_tasks.append("%s|%s" % (d, t))
            elif got_n < min(exp["config"]["min_task_samples"], avail[t]):
                weak_tasks.append({"stratum": "%s|%s" % (d, t), "got": got_n,
                                   "avail": avail[t]})
    cap = exp["config"]["task_max_share"]
    over = []
    for d in DOMAINS:
        sub = {k.split("|", 1)[1]: v for k, v in train_dt.items()
               if k.startswith(d + "|")}
        tot = sum(sub.values())
        if not tot:
            continue
        t, v = max(sub.items(), key=lambda x: x[1])
        if v / tot > cap + 1e-9:
            over.append({"domain": d, "task": t, "share": round(v / tot, 4),
                         "cap": cap,
                         "cap_relaxed_documented":
                             bool(exp["train_domain"][d].get("cap_relaxed"))})
    res["checks"]["C5_task_coverage"] = {
        "missing_tasks": missing_tasks, "weak_tasks": weak_tasks,
        "over_cap": over, "cap": cap,
    }
    if missing_tasks:
        res["failures"].append("C5 有任务型在训练集里完全缺席：%s" % missing_tasks)
    for o in over:
        if o["cap_relaxed_documented"]:
            res["warnings"].append(
                "C5 域 %s 单任务份额 %.1f%% > 上限 %.0f%%，已在 SPLIT_STATS 留痕"
                "（cap_relaxed=true，原因是池可用量不足）"
                % (o["domain"], 100 * o["share"], 100 * cap))
        else:
            res["failures"].append("C5 域 %s 单任务份额 %.1f%% 超上限且无留痕"
                                   % (o["domain"], 100 * o["share"]))

    # ---------------- C6 / C7 / C8
    res["checks"]["C6_no_synthetic_in_dev"] = {"val": synth_bad.get("val", 0),
                                               "test": synth_bad.get("test", 0)}
    if synth_bad:
        res["failures"].append("C6 val/test 含 synthetic 样本：%s" % dict(synth_bad))
    res["checks"]["C7_messages"] = dict(msg_bad)
    if msg_bad:
        res["failures"].append("C7 messages 结构异常：%s" % dict(msg_bad))
    res["checks"]["C8_fields"] = dict(field_bad)
    if field_bad:
        res["failures"].append("C8 字段异常：%s" % dict(field_bad))

    # ---------------- C9 逐域视图一致性
    view = {}
    for split, base in (("train", train_dir), ("val", dev_dir), ("test", test_dir)):
        want = {d: domains.get(split, collections.Counter()).get(d, 0) for d in DOMAINS}
        for d in DOMAINS:
            p = os.path.join(base, DOMAIN_FILE[d] + ".jsonl")
            if not os.path.isfile(p):
                view["%s/%s" % (split, DOMAIN_FILE[d])] = {"error": "missing"}
                continue
            n = sum(1 for _ in iter_jsonl(p))
            dom_ok = all(r.get("domain") == d for r in iter_jsonl(p))
            view["%s/%s" % (split, DOMAIN_FILE[d])] = {
                "lines": n, "expected": want[d],
                "match": n == want[d], "domain_consistent": dom_ok}
    res["checks"]["C9_domain_views"] = view
    for k, v in view.items():
        if v.get("error") or not v.get("match") or not v.get("domain_consistent"):
            res["failures"].append("C9 逐域视图不一致：%s => %s" % (k, v))

    # ---------------- C10 路由集
    res["checks"]["C10_router"] = dict(router_bad)
    if router_bad:
        res["failures"].append("C10 路由集缺 router_label/router_query：%s"
                               % dict(router_bad))

    # ---------------- C11 SHA-256 登记比对
    sha_mismatch = []
    for p, meta in exp.get("outputs", {}).items():
        if not os.path.isfile(p):
            sha_mismatch.append({"path": p, "error": "missing"})
            continue
        cur = sha256_file(p)
        if cur != meta["sha256"]:
            sha_mismatch.append({"path": p, "declared": meta["sha256"][:16],
                                 "actual": cur[:16]})
    res["checks"]["C11_sha256"] = {"mismatch": sha_mismatch,
                                   "checked": len(exp.get("outputs", {}))}
    if sha_mismatch:
        res["warnings"].append("C11 SHA-256 与登记不一致（若为切分后再次改动则正常）：%s"
                               % sha_mismatch)

    # ---------------- 汇总
    res["summary"] = {
        "files": {k: v for k, v in counts.items()},
        "splits_total": sum(counts.values()),
        "train_mix": got,
        "router_multilabel_pct": exp.get("router_multilabel_pct"),
        "leftover_pool": exp["splits"]["leftover"],
        "task_strata": len(train_dt),
    }
    if res["failures"]:
        res["verdict"] = "FAIL"

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            W = lambda s="": fh.write(s + "\n")
            W("# 阶段 4 切分质检报告")
            W()
            W("**verdict = %s**" % res["verdict"])
            W()
            W("| 检查 | 结果 |")
            W("|---|---|")
            W("| C1 行数与声明一致 | %s |" % _ok(res["checks"]["C1_counts"]["match"]))
            W("| C2 `uid_g` 全局唯一 | %s |" % _ok(not any(dup.values())))
            W("| C3 **四份 split 两两不相交（硬卡口）** | %s |"
              % _ok(not bad))
            W("| C4 训练集逐域配比 = 目标 | %s |"
              % _ok(res["checks"]["C4_train_mix"]["match"]))
            W("| C5 任务覆盖 / 份额上限 | %s |"
              % _ok(not missing_tasks and not [o for o in over
                                               if not o["cap_relaxed_documented"]]))
            W("| C6 val/test 无合成数据 | %s |" % _ok(not synth_bad))
            W("| C7 messages 结构完整 | %s |" % _ok(not msg_bad))
            W("| C8 字段合法 | %s |" % _ok(not field_bad))
            W("| C9 逐域视图一致 | %s |" % _ok(not [v for v in view.values()
                                                    if v.get("error") or not v.get("match")
                                                    or not v.get("domain_consistent")]))
            W("| C10 路由集字段 | %s |" % _ok(not router_bad))
            W("| C11 SHA-256 登记比对 | %s |" % _ok(not sha_mismatch))
            W()
            W("## 规模")
            W()
            W("| split | 行数 |")
            W("|---|---|")
            for k in ("train", "val", "test", "router"):
                W("| %s | %d |" % (k, counts.get(k, 0)))
            W()
            W("训练集逐域实测：%s" % got)
            W()
            W("未使用池剩余：%d" % exp["splits"]["leftover"])
            W()
            W("## 不相交性明细（硬卡口）")
            W()
            W("| 组合 | uid_g 交集 | content_sha1 交集 |")
            W("|---|---|---|")
            for k, v in pairs.items():
                W("| %s | **%d** | **%d** |" % (k, v["uid_g"], v["content_sha1"]))
            W()
            if over:
                W("## 份额上限留痕")
                W()
                for o in over:
                    W("- 域 **%s**：`%s` 占 %.1f%%（上限 %.0f%%），"
                      "已留痕 cap_relaxed=%s"
                      % (o["domain"], o["task"], 100 * o["share"], 100 * cap,
                         o["cap_relaxed_documented"]))
                W()
            if weak_tasks:
                W("## 样本偏少的任务型（未缺席，仅提示）")
                W()
                for w in weak_tasks[:20]:
                    W("- `%s`：训练集 %d 条 / 池 %d 条" % (w["stratum"], w["got"], w["avail"]))
                W()
            if res["warnings"]:
                W("## 警告")
                W()
                for w in res["warnings"]:
                    W("- %s" % w)
                W()
            if res["failures"]:
                W("## 失败项")
                W()
                for f in res["failures"]:
                    W("- %s" % f)
                W()

    print("VERDICT=%s" % res["verdict"])
    print("counts=%s" % counts)
    print("disjoint=%s" % {k: (v["uid_g"], v["content_sha1"]) for k, v in pairs.items()})
    print("train_mix=%s (target %s)" % (got, {d: mix[d] for d in DOMAINS}))
    print("missing_tasks=%s weak=%d over_cap=%d"
          % (missing_tasks, len(weak_tasks), len(over)))
    print("failures=%s" % res["failures"])
    print("warnings=%s" % res["warnings"])
    print("OK -> %s" % args.out_json)
    if res["verdict"] != "PASS":
        sys.exit(1)


def _ok(flag):
    return "✅" if flag else "❌"


if __name__ == "__main__":
    main()
