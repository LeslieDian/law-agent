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
  C12 **派生(合成)样本只出现在 train** —— router/val/test 必须 100% 真实样本
  C13 派生样本份额不超过 `--derived-max-share`（超限须在 SPLIT_STATS 留痕）
  C14 `uid` 与 `uid_g` 必须逐条相等（证明阶段 2 的 uid 唯一化修复已生效），
      且**必须覆盖全部行**（`checked == expected`）—— 空 Counter 不能冒充 PASS

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
    derived_by_split = collections.Counter()
    derived_train_domain = collections.Counter()
    uid_eq_bad = collections.Counter()
    uid_eq_checked = collections.Counter()

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
            if rec.get("derived"):
                derived_by_split[split] += 1
                if split == "train":
                    derived_train_domain[d] += 1
            if ug:
                # ★ 必须计数「实际比较了多少行」：空 Counter 无法区分
                #   「全部相等」与「一行都没比过」—— 后者是永远为真的假断言。
                uid_eq_checked[split] += 1
                if (rec.get("uid") or "") != ug:
                    uid_eq_bad[split] += 1
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

    # ---------------- C12 派生样本只许进 train
    leak = {s: derived_by_split.get(s, 0)
            for s in ("val", "test", "router") if derived_by_split.get(s, 0) > 0}
    res["checks"]["C12_derived_only_in_train"] = {
        "derived_by_split": dict(derived_by_split),
        "train_derived": derived_by_split.get("train", 0),
        "leaked": leak,
    }
    if leak:
        res["failures"].append(
            "C12 派生(合成)样本泄漏到 router/val/test：%s —— 评测公平性被破坏" % leak)

    # ---------------- C13 派生份额上限
    dcap = exp["config"].get("derived_max_share")
    dcap_relaxed = {d: bool(exp["train_domain"][d].get("derived_cap_relaxed"))
                    for d in DOMAINS}
    d_share = {d: (derived_train_domain.get(d, 0) / got[d]) if got.get(d) else 0.0
               for d in DOMAINS}
    over_d = []
    if dcap is not None:
        for d in DOMAINS:
            if got.get(d) and d_share[d] > dcap + 1e-9:
                over_d.append({"domain": d, "share": round(d_share[d], 4),
                               "cap": dcap,
                               "cap_relaxed_documented": dcap_relaxed.get(d, False)})
    res["checks"]["C13_derived_share"] = {
        "cap": dcap,
        "derived_by_domain": {d: derived_train_domain.get(d, 0) for d in DOMAINS},
        "share": {d: round(d_share[d], 4) for d in DOMAINS},
        "over_cap": over_d,
    }
    for o in over_d:
        if o["cap_relaxed_documented"]:
            res["warnings"].append(
                "C13 域 %s 派生份额 %.1f%% 超上限 %.0f%%，已在 SPLIT_STATS 留痕"
                "（derived_cap_relaxed=true）" % (o["domain"], 100 * o["share"],
                                                   100 * dcap))
        else:
            res["failures"].append(
                "C13 域 %s 派生份额 %.1f%% 超上限 %.0f%% 且无留痕"
                % (o["domain"], 100 * o["share"], 100 * dcap))

    # ---------------- C14 uid == uid_g（阶段 2 唯一化修复生效的交叉验证）
    uid_eq_total = sum(uid_eq_checked.values())
    uid_eq_expected = sum(counts.values())
    res["checks"]["C14_uid_equals_uid_g"] = {
        "mismatch": dict(uid_eq_bad),
        "checked": uid_eq_total,
        "expected": uid_eq_expected,
    }
    if uid_eq_bad:
        res["failures"].append(
            "C14 存在 uid != uid_g 的记录：%s —— 阶段 2 的 uid 唯一化修复未生效"
            "或该记录来自旧产物" % dict(uid_eq_bad))
    elif uid_eq_total != uid_eq_expected:
        # 覆盖不全同样是失败：不能让「没比过」冒充「都比过且都相等」
        res["failures"].append(
            "C14 覆盖不全：只比较了 %d 行 / 应有 %d 行 —— 存在缺 uid_g 的记录"
            % (uid_eq_total, uid_eq_expected))

    # ---------------- 汇总
    res["summary"] = {
        "files": {k: v for k, v in counts.items()},
        "splits_total": sum(counts.values()),
        "train_mix": got,
        "derived_by_split": dict(derived_by_split),
        "derived_train_by_domain": {d: derived_train_domain.get(d, 0) for d in DOMAINS},
        "derived_share_by_domain": {d: round(d_share[d], 4) for d in DOMAINS},
        "derived_cap": dcap,
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
            W("| C12 **派生(合成)样本只出现在 train** | %s |" % _ok(not leak))
            W("| C13 派生份额不超上限 | %s |"
              % _ok(not [o for o in over_d
                         if not o["cap_relaxed_documented"]]))
            W("| C14 `uid` == `uid_g`（阶段 2 修复生效） | %s（%d/%d 行已比对） |"
              % (_ok(not uid_eq_bad and uid_eq_total == uid_eq_expected),
                 uid_eq_total, uid_eq_expected))
            W()
            W("## 规模")
            W()
            W("| split | 行数 | 其中派生(合成) |")
            W("|---|---|---|")
            for k in ("train", "val", "test", "router"):
                W("| %s | %d | %d |" % (k, counts.get(k, 0),
                                        derived_by_split.get(k, 0)))
            W()
            W("训练集逐域实测：%s" % got)
            W()
            if dcap is not None:
                W("训练集逐域派生份额（上限 %.0f%%）：%s"
                  % (100 * dcap,
                     "、".join("%s %.1f%%" % (d, 100 * d_share[d]) for d in DOMAINS)))
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
                W("## 份额上限留痕（单任务）")
                W()
                for o in over:
                    W("- 域 **%s**：`%s` 占 %.1f%%（上限 %.0f%%），"
                      "已留痕 cap_relaxed=%s"
                      % (o["domain"], o["task"], 100 * o["share"], 100 * cap,
                         o["cap_relaxed_documented"]))
                W()
            if over_d:
                W("## 份额上限留痕（派生组）")
                W()
                for o in over_d:
                    W("- 域 **%s**：派生样本占 %.1f%%（上限 %.0f%%），"
                      "已留痕 derived_cap_relaxed=%s"
                      % (o["domain"], 100 * o["share"], 100 * o["cap"],
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
    print("derived_by_split=%s" % dict(derived_by_split))
    print("derived_train_by_domain=%s (cap %s)"
          % ({d: derived_train_domain.get(d, 0) for d in DOMAINS}, dcap))
    print("derived_share=%s" % {d: round(d_share[d], 4) for d in DOMAINS})
    print("uid_uidg_mismatch=%s" % dict(uid_eq_bad))
    print("failures=%s" % res["failures"])
    print("warnings=%s" % res["warnings"])
    print("OK -> %s" % args.out_json)
    if res["verdict"] != "PASS":
        sys.exit(1)


def _ok(flag):
    return "✅" if flag else "❌"


if __name__ == "__main__":
    main()
