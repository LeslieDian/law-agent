#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4：分层下采样 + train/val/test + 路由集切分（确定性、幂等、输入只读）。

============================ 硬约定（改动必须同步 docs 与本文件顶部） ============================
1. 输入 = 阶段 3 去污后的 QA 流（`data/corpus/decontaminated/qa/*.jsonl`）
   **+ 阶段 2c 派生流**（`data/corpus/derived/*.jsonl`），
   **显式排除 `_all.jsonl`**（合并副本，读它必然双计）。输入只读，一个字节都不改。
2. 样本唯一键 = `content_sha1`。重跑实测：真实 QA **270,989** 条 + 派生 **13,484** 条（合计池
   **284,473**），`content_sha1` **全局唯一（0 重复）**。2026-09-18 起阶段 2 的 `uid` 已修为
   `<dataset>__<file_stem>:<source_index>`（与下方 `uid_g` 同口径），不再撞号。
   `uid_g` 保留为下游唯一键，并额外保留 `uid_legacy` 供血缘回溯。
3. 切分**全部**基于 sha1 排序的确定性抽取（不使用随机数、不依赖字典序以外的隐式状态），
   同一份输入重跑结果逐字节一致。
4. 规模：router 20,000（= 训练目标的 20%）/ val 1,000 / test 1,000 / train 100,000。
   四者**两两不相交**，由「每个样本只被指派一次」的构造保证，并由 verify 脚本断言。
5. **派生样本（`derived=true` / `synthetic=true`）只能进 train**：
   router / val / test 一律**只从真实样本**抽（评测公平性 + verify C6 硬卡口）。
6. train 按 `domain × task × source_dataset` 三层分层下采样。
   * 域内任务配额：`weight = count^alpha`（alpha=1 → 等比，标准分层抽样）；alpha<1 抹平。
   * **任务份额软上限** `--task-max-share`（默认 0.35）：单任务不得超过域配额的 35%。
   * **派生组份额上限** `--derived-max-share`（默认 0.15）：派生样本（合成）合计不得超过
     域配额的该比例 —— 下限是「把单任务份额压回上限所需的最小补量」（实测约 4%），
     上限是「防止合成样本喧宾夺主」，0.15 取两者的保守中间值。
   * 约束**逐级放宽且必须留痕**：① 单任务上限 → ② 派生组上限 → ③ 报告里写清
     `cap_relaxed` / `derived_cap_relaxed` + 实测份额，**绝不静默改口径，绝不静默丢数据**。
7. val / test 按**目标配比**分层（每域 300/400/200/100），保证程序法有足够评测样本。
8. 法条条目流（`data/corpus/statute_items/`，65,037 条）**不进 SFT 训练集**：
   它是检索语料（向量库主料）。本脚本只做登记 + 跨流 `content_sha1` 重叠检查。

用法：
  # 全量
  python scripts/corpus/downsample_split.py --root /mnt/data/lidian/law-agent \
      --out-json docs/corpus/SPLIT_STATS.json --out-md docs/corpus/SPLIT_REPORT.md

  # 干跑（只算配额不落盘）
  python scripts/corpus/downsample_split.py --root ... --dry-run --out-json /tmp/p.json

  # 冒烟（小目标 + 落到 /tmp，全量扫描但产物很小）
  python scripts/corpus/downsample_split.py --root ... \
      --train-target 2000 --mix criminal=600,civil=800,procedural=400,general=200 \
      --router-rate 0.2 --val-size 40 --test-size 40 \
      --train-dir /tmp/split_smoke/train --dev-dir /tmp/split_smoke/dev \
      --test-dir /tmp/split_smoke/test --router-dir /tmp/split_smoke/router \
      --out-json /tmp/split_smoke/STATS.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

QA_REL = "data/corpus/decontaminated/qa"
DERIVED_REL = "data/corpus/derived"
ST_REL = "data/corpus/statute_items"

DERIVED_GROUP = "__derived_group__"

DOMAINS = ("criminal", "civil", "procedural", "general")
# 域 → 落盘文件名（adapters_router.yaml 用的是 procedure，不是 procedural，必须对齐）
DOMAIN_FILE = {"criminal": "criminal", "civil": "civil",
               "procedural": "procedure", "general": "general"}

DEFAULT_MIX = {"criminal": 30000, "civil": 40000, "procedural": 20000,
               "general": 10000}


# ============================================================ 基础工具

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def hkey(salt: str, key: str) -> int:
    """确定性排序键：sha1(salt NUL key) 的前 16 个 hex 位 → 整数。"""
    return int(sha1_hex(salt + "\x00" + key)[:16], 16)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def largest_remainder(weights: dict, total: int) -> dict:
    """按权重把 total 拆成整数，最大余数法（和恰为 total）。"""
    out = {k: 0 for k in weights}
    if total <= 0:
        return out
    s = sum(weights.values())
    if s <= 0:
        return out
    raw = {k: total * v / s for k, v in weights.items()}
    for k, v in raw.items():
        out[k] = int(v)
    rem = total - sum(out.values())
    if rem > 0:
        order = sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), str(k)))
        for k in order[:rem]:
            out[k] += 1
    return out


def waterfill(weights: dict, room: dict, total: int, max_iter: int = 200):
    """按权重分配 total，但每项不超过 room[k]；返回 (alloc, 未分完的量)。"""
    alloc = {k: 0 for k in weights}
    budget = total
    for _ in range(max_iter):
        active = [k for k in weights if alloc[k] < room.get(k, 0) and weights[k] > 0]
        if budget <= 0 or not active:
            break
        share = largest_remainder({k: weights[k] for k in active}, budget)
        moved = 0
        for k in active:
            give = min(share[k], room[k] - alloc[k])
            if give > 0:
                alloc[k] += give
                moved += give
        budget -= moved
        if moved == 0:
            break
    return alloc, budget


# ============================================================ 配额计算

def _split_group(dkeys, counts, group_alloc, min_samples):
    """把派生组的配额拆到各派生任务（先满足每个任务的地板，再按可用量等比）。

    返回 (alloc, lost)；lost = 组内拆不完的量（由调用方回填给非派生任务）。
    """
    if not dkeys:
        return {}, max(0, group_alloc)
    if group_alloc <= 0:
        return {k: 0 for k in dkeys}, max(0, group_alloc)
    floors = {k: min(min_samples, counts[k]) for k in dkeys}
    F = sum(floors.values())
    if F >= group_alloc:
        alloc = largest_remainder(floors, group_alloc)
    else:
        w = {k: max(counts[k] - floors[k], 1) for k in dkeys}
        room = {k: counts[k] - floors[k] for k in dkeys}
        extra, _ = waterfill(w, room, group_alloc - F)
        alloc = {k: floors[k] + extra[k] for k in dkeys}
    alloc = {k: min(v, counts[k]) for k, v in alloc.items()}
    return alloc, group_alloc - sum(alloc.values())


def allocate_tasks(counts, total, alpha, max_share, min_samples,
                   derived_keys=(), derived_max_share=None):
    """域内任务配额：地板 + alpha 比例 + 单任务软上限 + **派生组上限**。

    三层约束**逐级放宽、每级留痕**（绝不静默改口径）：
      第 1 轮：单任务上限 `max_share` 且派生组上限 `derived_max_share` 同时生效；
      第 2 轮：只剩分不完的余量时，放宽**单任务上限**（`cap_relaxed=true`）；
      第 3 轮：仍分不完，再放宽**派生组上限**（`derived_cap_relaxed=true`）。

    注意判据是 `Σ min(cap, avail) ≥ target`，不是「有没有任务超过上限」。
    返回 (alloc, info)。
    """
    info = {"cap_binding": False, "cap_relaxed": False,
            "derived_cap_relaxed": False, "derived_cap": None,
            "derived_alloc": 0, "unallocated": 0}
    keys = [k for k, v in counts.items() if v > 0]
    if not keys:
        info["empty"] = True
        return {}, info

    dset = set(derived_keys or ())
    dkeys = sorted(k for k in keys if k in dset)
    ndkeys = sorted(k for k in keys if k not in dset)

    floors = {k: min(min_samples, counts[k]) for k in keys}
    F = sum(floors.values())
    info["floors"] = floors
    if F >= total:
        info["floors_scaled"] = True
        return largest_remainder(floors, total), info

    budget = total - F
    cap_total = int(max_share * total)
    info["cap_total"] = cap_total
    info["cap_binding"] = any(counts[k] > cap_total for k in keys)

    dsum = sum(counts[k] for k in dkeys)
    dfloor = sum(floors[k] for k in dkeys)
    info["derived_available"] = dsum
    info["derived_floor"] = dfloor
    dcap = None
    if dkeys:
        dcap = dsum if derived_max_share is None else int(derived_max_share * total)
        dcap = max(dcap, dfloor)
    info["derived_cap"] = dcap

    w = {k: max(counts[k] - floors[k], 1) ** alpha for k in ndkeys}
    if dkeys:
        w[DERIVED_GROUP] = max(dsum - dfloor, 1) ** alpha

    alloc = {k: floors[k] for k in ndkeys}
    group_alloc = dfloor

    def room_of(group_room):
        r = {k: max(0, min(counts[k], cap_total) - floors[k]) for k in ndkeys}
        if dkeys:
            r[DERIVED_GROUP] = max(0, group_room - dfloor)
        return r

    # ---- 第 1 轮：单任务上限 + 派生组上限
    extra, leftover = waterfill(w, room_of(min(dsum, dcap) if dkeys else 0), budget)
    for k in ndkeys:
        alloc[k] += extra.get(k, 0)
    group_alloc += extra.get(DERIVED_GROUP, 0)

    # ---- 第 2 轮：放宽单任务上限（派生组上限不动）
    if leftover > 0:
        room2 = {k: max(0, counts[k] - alloc[k]) for k in ndkeys}
        if dkeys:
            room2[DERIVED_GROUP] = max(0, min(dsum, dcap) - group_alloc)
        e2, leftover = waterfill(w, room2, leftover)
        for k in ndkeys:
            alloc[k] += e2.get(k, 0)
        group_alloc += e2.get(DERIVED_GROUP, 0)
        info["cap_relaxed"] = True

    # ---- 第 3 轮：放宽派生组上限
    if leftover > 0 and dkeys:
        room3 = {k: max(0, counts[k] - alloc[k]) for k in ndkeys}
        room3[DERIVED_GROUP] = max(0, dsum - group_alloc)
        e3, leftover = waterfill(w, room3, leftover)
        for k in ndkeys:
            alloc[k] += e3.get(k, 0)
        group_alloc += e3.get(DERIVED_GROUP, 0)
        info["derived_cap_relaxed"] = True

    # ---- 拆分派生组配额
    sub, lost = _split_group(dkeys, counts, group_alloc, min_samples)
    alloc.update(sub)
    info["derived_alloc"] = sum(sub.values())

    # ---- 组内拆不完（可用量不足）→ 回填给非派生任务
    if lost > 0 and ndkeys:
        room4 = {k: max(0, counts[k] - alloc[k]) for k in ndkeys}
        e4, lost = waterfill({k: max(counts[k], 1) for k in ndkeys}, room4, lost)
        for k in ndkeys:
            alloc[k] += e4.get(k, 0)
    info["unallocated"] = lost
    return alloc, info


def allocate_sources(counts: dict, total: int):
    """任务内按来源等比分配（clamp 到各来源可用量）。"""
    keys = [k for k, v in counts.items() if v > 0]
    if not keys:
        return {}, 0
    alloc, leftover = waterfill({k: counts[k] for k in keys},
                                {k: counts[k] for k in keys}, total)
    return alloc, leftover


def parse_mix(text: str) -> dict:
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = int(v)
    return out


# ============================================================ 主流程

def input_files(root, derived_dir=""):
    """返回 `[(path, is_derived)]`：正式 QA 流 + 派生流。

    两个目录都**显式排除 `_` 开头的文件**（`_all.jsonl` 合并副本 → 读了必然双计）。
    """
    out, skipped = [], []
    qa_dir = os.path.join(root, QA_REL)
    if not os.path.isdir(qa_dir):
        raise SystemExit("QA 目录不存在：%s" % qa_dir)
    for fn in sorted(os.listdir(qa_dir)):
        if fn.endswith(".jsonl") and not fn.startswith("_"):
            out.append((os.path.join(qa_dir, fn), False))
        elif fn.endswith(".jsonl"):
            skipped.append("qa/" + fn)
    dd = derived_dir or os.path.join(root, DERIVED_REL)
    if os.path.isdir(dd):
        for fn in sorted(os.listdir(dd)):
            if fn.endswith(".jsonl") and not fn.startswith("_"):
                out.append((os.path.join(dd, fn), True))
            elif fn.endswith(".jsonl"):
                skipped.append("derived/" + fn)
    return out, skipped


def load_light(files, max_records):
    """第一遍：只读轻字段，建立样本表。"""
    recs = []
    per_file = collections.Counter()
    bad = collections.Counter()
    for path, is_derived in files:
        fn = ("derived/" if is_derived else "") + os.path.basename(path)
        for lineno, rec in iter_jsonl(path):
            key = rec.get("content_sha1") or ""
            if not key:
                bad["missing_content_sha1"] += 1
                continue
            ds = rec.get("source_dataset") or "?"
            sf = os.path.basename(rec.get("source_file") or "?")
            recs.append({
                "key": key,
                "legacy_uid": rec.get("uid") or "",
                "uid_g": "%s__%s:%s" % (ds.replace("/", "__"),
                                        os.path.splitext(sf)[0],
                                        rec.get("source_index")),
                "domain": rec.get("domain") or "",
                "domains": rec.get("domains") or [],
                "task": rec.get("task") or "",
                "task_kind": rec.get("task_kind") or "",
                "source": ds,
                "source_file": sf,
                "synthetic": bool(rec.get("synthetic")),
                "derived": bool(rec.get("derived")) or is_derived,
                "replay": bool(rec.get("replay")),
            })
            per_file[fn] += 1
            if max_records and len(recs) >= max_records:
                break
        if max_records and len(recs) >= max_records:
            break
    return recs, per_file, bad


def build_splits(recs, mix, router_size, val_size, test_size,
                 task_alpha, task_max_share, min_task_samples,
                 derived_max_share=None):
    """构造四份互不相交的样本键集合 + 分层明细。

    ★ 路由集 / val / test **只从真实样本抽**：派生样本 `synthetic=true`，仅能进 train
      （评测公平性；verify C6 会硬卡口 val/test 无合成数据）。
    """
    by_key = {r["key"]: r for r in recs}
    if len(by_key) != len(recs):
        raise SystemExit("输入 content_sha1 不唯一，无法作为样本键（%d vs %d）"
                         % (len(by_key), len(recs)))

    real = [r for r in recs if not r["derived"]]
    derived = [r for r in recs if r["derived"]]
    need = router_size + val_size + test_size + sum(mix.values())
    if len(real) < need:
        raise SystemExit("真实样本不足以支撑 router/val/test + 训练目标：%d < %d"
                         % (len(real), need))

    # ---- 1) 路由集：取 router 哈希键最小的 router_size 个（真实样本，均匀子集）
    order = sorted(real, key=lambda r: hkey("stage4/router", r["key"]))
    router_keys = {r["key"] for r in order[:router_size]}
    rest = order[router_size:]

    # ---- 2) val / test：每个域按目标配比取，先按 dev 键排序，再按 vt 键二分
    mix_sum = sum(mix.values())
    quotas = largest_remainder(
        {d: mix[d] for d in mix}, val_size + test_size)
    val_keys, test_keys = set(), set()
    dev_detail = {}
    for d in DOMAINS:
        need = quotas.get(d, 0)
        cand = sorted((r for r in rest if r["domain"] == d),
                      key=lambda r: hkey("stage4/dev", r["key"]))
        pick = cand[:need]
        if len(pick) < need:
            raise SystemExit("域 %s 可用样本不足 val/test 配额：%d < %d"
                             % (d, len(pick), need))
        v_quota = largest_remainder({d: mix[d] for d in mix}, val_size).get(d, 0)
        v_quota = min(v_quota, len(pick))
        pick.sort(key=lambda r: hkey("stage4/vt", r["key"]))
        val_keys |= {r["key"] for r in pick[:v_quota]}
        test_keys |= {r["key"] for r in pick[v_quota:]}
        dev_detail[d] = {"val": v_quota, "test": len(pick) - v_quota}

    # ---- 3) 训练池 = 真实剩余 + 全部派生样本
    train_pool = [r for r in rest
                  if r["key"] not in val_keys and r["key"] not in test_keys] + derived

    # ---- 4) 分层下采样
    real_tasks = {r["task"] for r in real}
    d_by_dom = {d: {r["task"] for r in train_pool
                    if r["derived"] and r["domain"] == d} for d in DOMAINS}
    clash = set().union(*d_by_dom.values()) & real_tasks
    if clash:
        raise SystemExit("派生任务名与真实任务名撞名，配额无法区分：%s" % sorted(clash))

    train_keys = set()
    detail = {"domain": {}, "task_alloc_info": {}, "source_alloc": {}}
    for d in DOMAINS:
        target = mix[d]
        sub = [r for r in train_pool if r["domain"] == d]
        tc = collections.Counter(r["task"] for r in sub)
        alloc_t, info = allocate_tasks(tc, target, task_alpha, task_max_share,
                                       min_task_samples, d_by_dom[d],
                                       derived_max_share)
        detail["task_alloc_info"][d] = info
        got_total = 0
        got_derived = 0
        for t in sorted(alloc_t):
            q = alloc_t[t]
            if q <= 0:
                continue
            pool_t = [r for r in sub if r["task"] == t]
            sc = collections.Counter(r["source"] for r in pool_t)
            alloc_s, lost = allocate_sources(sc, q)
            if lost:
                raise SystemExit("域 %s 任务 %s 来源配额未分完（剩 %d）" % (d, t, lost))
            detail["source_alloc"]["%s|%s" % (d, t)] = alloc_s
            for s in sorted(alloc_s):
                qs = alloc_s[s]
                if qs <= 0:
                    continue
                strat = [r for r in pool_t if r["source"] == s]
                strat.sort(key=lambda r: hkey("stage4/train", r["key"]))
                for r in strat[:qs]:
                    train_keys.add(r["key"])
                    if r["derived"]:
                        got_derived += 1
                got_total += min(qs, len(strat))
        detail["domain"][d] = {
            "target": target,
            "achieved": got_total,
            "derived_achieved": got_derived,
            "pool_available": len(sub),
            "pool_derived_available": sum(1 for r in sub if r["derived"]),
            "pool_usage": round(100.0 * got_total / max(1, len(sub)), 2),
            "requested": sum(alloc_t.values()),
        }
        if got_total != target:
            raise SystemExit("域 %s 训练配额未达成：%d != %d（池 %d）"
                             % (d, got_total, target, len(sub)))

    # ---- 5) 不相交性构造断言
    groups = {"router": router_keys, "val": val_keys, "test": test_keys,
              "train": train_keys}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = groups[names[i]] & groups[names[j]]
            if inter:
                raise SystemExit("切分不相交断言失败：%s ∩ %s = %d"
                                 % (names[i], names[j], len(inter)))

    # ---- 6) 派生样本不得出现在 router/val/test（构造上已保证，显式断言以防回退）
    non_train = router_keys | val_keys | test_keys
    leak = [k for k in non_train if by_key[k]["derived"]]
    if leak:
        raise SystemExit("派生（合成）样本泄漏到 router/val/test：%d 条，"
                         "评测公平性被破坏" % len(leak))
    return by_key, groups, dev_detail, detail, files_and_skipped(recs)


def files_and_skipped(recs):
    per = collections.Counter()
    for r in recs:
        per["%s|%s" % (r["source"], r["source_file"])] += 1
    return dict(per)


# ============================================================ 落盘

def write_splits(args, by_key, groups, files):
    """第二遍：重读输入（QA 流 + 派生流），按已定好的键集写四个 split（含逐域视图）。"""
    paths = {
        "train_all": os.path.join(args.train_dir, "train.jsonl"),
        "dev_all": os.path.join(args.dev_dir, "dev.jsonl"),
        "test_all": os.path.join(args.test_dir, "test.jsonl"),
        "router_all": os.path.join(args.router_dir, "router_train.jsonl"),
    }
    for d in DOMAINS:
        f = DOMAIN_FILE[d]
        paths["train_" + d] = os.path.join(args.train_dir, f + ".jsonl")
        paths["dev_" + d] = os.path.join(args.dev_dir, f + ".jsonl")
        paths["test_" + d] = os.path.join(args.test_dir, f + ".jsonl")

    if args.dry_run:
        return {}, {}

    for k, p in paths.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    handles = {k: open(p, "w", encoding="utf-8") for k, p in paths.items()}
    domain_of = {}
    for g, keys in groups.items():
        for k in keys:
            domain_of[k] = by_key[k]["domain"]

    cnt = collections.Counter()
    seen_uidg = collections.Counter()
    try:
        for path, is_derived in files:
            cnt["input_files"] += 1
            for _, rec in iter_jsonl(path):
                key = rec.get("content_sha1") or ""
                split = None
                for g, keys in groups.items():
                    if key in keys:
                        split = g
                        break
                if split is None:
                    cnt["unassigned"] += 1
                    continue
                dom = rec.get("domain") or ""
                out = dict(rec)
                out["uid_g"] = "%s__%s:%s" % (
                    (rec.get("source_dataset") or "?").replace("/", "__"),
                    os.path.splitext(os.path.basename(
                        rec.get("source_file") or "?"))[0],
                    rec.get("source_index"))
                out["split"] = split
                out["split_stage"] = "stage4"
                if is_derived:
                    cnt["derived_%s" % split] += 1
                if split == "router":
                    out["router_label"] = rec.get("domains") or [dom]
                    out["router_query"] = _router_query(rec)
                line = json.dumps(out, ensure_ascii=False) + "\n"
                seen_uidg[out["uid_g"]] += 1
                cnt[split] += 1
                if split == "train":
                    handles["train_all"].write(line)
                    if dom in DOMAINS:
                        handles["train_" + dom].write(line)
                    else:
                        cnt["train_unknown_domain"] += 1
                elif split == "val":
                    handles["dev_all"].write(line)
                    if dom in DOMAINS:
                        handles["dev_" + dom].write(line)
                elif split == "test":
                    handles["test_all"].write(line)
                    if dom in DOMAINS:
                        handles["test_" + dom].write(line)
                else:
                    handles["router_all"].write(line)
    finally:
        for h in handles.values():
            h.close()

    dups = sum(c - 1 for c in seen_uidg.values() if c > 1)
    if dups:
        raise SystemExit("落盘后发现 uid_g 重复 %d 条（唯一键失效）" % dups)
    for k, p in paths.items():
        if os.path.getsize(p) == 0:
            raise SystemExit("输出为空文件：%s" % p)
    return paths, {"counts": dict(cnt), "unique_uid_g": len(seen_uidg)}


def _router_query(rec):
    msgs = rec.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content") or ""
    return rec.get("ask") or rec.get("instruction") or ""


# ============================================================ 法条流登记

def register_statutes(root):
    st_dir = os.path.join(root, ST_REL)
    if not os.path.isdir(st_dir):
        return {"present": False}
    files = sorted(f for f in os.listdir(st_dir)
                   if f.endswith(".jsonl") and not f.startswith("_"))
    out = {"present": True, "dir": st_dir, "files": {}, "total": 0,
           "by_domain": collections.Counter(), "by_law_type": collections.Counter(),
           "by_status": collections.Counter(), "by_level": collections.Counter(),
           "sha1_set": set(), "uid_dup": 0, "missing_text_full": 0}
    uids = collections.Counter()
    for fn in files:
        n = 0
        for _, rec in iter_jsonl(os.path.join(st_dir, fn)):
            n += 1
            out["total"] += 1
            out["by_domain"][rec.get("domain") or "<empty>"] += 1
            out["by_law_type"][rec.get("law_type") or "<empty>"] += 1
            out["by_status"][rec.get("status") or "<empty>"] += 1
            out["by_level"][rec.get("level") or "<empty>"] += 1
            if rec.get("content_sha1"):
                out["sha1_set"].add(rec["content_sha1"])
            uids[rec.get("uid") or ""] += 1
            if not rec.get("text_full"):
                out["missing_text_full"] += 1
        out["files"][fn] = n
    out["uid_dup"] = sum(c - 1 for c in uids.values() if c > 1)
    return out


# ============================================================ 报告

def pct(a, b):
    return round(100.0 * a / b, 2) if b else 0.0


def render_md(args, stats, path):
    S = stats
    with open(path, "w", encoding="utf-8") as fh:
        W = lambda s="": fh.write(s + "\n")
        W("# 阶段 4：分层下采样 + 切分报告")
        W()
        W("> 生成脚本 `scripts/corpus/downsample_split.py`（确定性、幂等、输入只读）")
        W("> 生成时间 %s（服务器 %s）" % (S["generated_at"], S["host"]))
        W()
        W("## 0. 口径")
        W()
        W("- 输入：`%s/*.jsonl`（真实 QA 流）+ `%s/*.jsonl`（阶段 2c 派生流）；"
          "已排除 `%s`（合并副本，读它必然双计）"
          % (QA_REL, DERIVED_REL,
             "、".join(S["input"]["skipped_files"]) or "无"))
        W("- 池记录：**%d** = 真实 **%d** + 派生(合成) **%d**（`content_sha1` 全局唯一 → 样本键）"
          % (S["input"]["records"], S["input"]["real_records"],
             S["input"]["derived_records"]))
        W("- `uid` 撞号（旧口径 `uid_legacy`）：**%d** 条 —— 阶段 2 已把 `uid` 修为"
          "`<dataset>__<file_stem>:<source_index>`，与 `uid_g` 同口径，不再撞号；"
          "`uid_legacy` 仅作血缘回溯。"
          % S["input"]["uid_collision_records"])
        W("- **派生样本只进 train**：router/val/test 全部来自真实样本（评测公平性）。")
        W("- 切分方式：sha1 排序确定性抽取（无随机数）；四份 split 两两不相交")
        W("- 目标配比：" + "、".join("%s %d" % (k, v)
                                  for k, v in S["config"]["mix"].items()))
        W()
        W("## 1. 四份 split 规模（目标 vs 实得）")
        W()
        W("| split | 文件 | 目标 | 实得 |")
        W("|---|---|---|---|")
        W("| 训练集（A0 全域） | `data/train/train.jsonl` | %d | **%d** |"
          % (S["config"]["train_target"], S["splits"]["train"]))
        W("| 路由集 | `data/router/router_train.jsonl` | %d | **%d** |"
          % (S["config"]["router_size"], S["splits"]["router"]))
        W("| 验证集 | `data/dev/dev.jsonl` | %d | **%d** |"
          % (S["config"]["val_size"], S["splits"]["val"]))
        W("| 测试集 | `data/test/test.jsonl` | %d | **%d** |"
          % (S["config"]["test_size"], S["splits"]["test"]))
        W("| **合计占用** | | | **%d** |" % S["splits"]["total_used"])
        W()
        W("池剩余（未使用，可回溯）：**%d**" % S["splits"]["leftover"])
        W()
        W("## 2. 训练集分层（域）")
        W()
        W("| 域 | 目标 | 实得 | 占比 | 达成率 | 池可用 | 池使用率 |")
        W("|---|---|---|---|---|---|---|")
        for d in DOMAINS:
            x = S["train_domain"][d]
            W("| %s | %d | **%d** | %.1f%% | %.1f%% | %d | %.1f%% |"
              % (d, x["target"], x["achieved"], pct(x["achieved"],
                                                    S["splits"]["train"]),
                 100.0 * x["achieved"] / max(1, x["target"]),
                 x["pool_available"], x["pool_usage"]))
        W()
        W("## 3. 训练集分层（域 × 任务）—— 三层约束的留痕")
        W()
        W("- 单任务软上限 `--task-max-share %.2f`：单任务不得超过域配额的该比例。"
          % S["config"]["task_max_share"])
        W("- **派生组上限** `--derived-max-share %s`：合成样本合计不得超过域配额的该比例"
          "（防止「程序法专家只会背法条」）。"
          % ("禁用" if S["config"]["derived_max_share"] is None
             else "%.2f" % S["config"]["derived_max_share"]))
        W("- 放宽顺序：① 单任务上限 → ② 派生组上限；每级放宽都记进本表。")
        W()
        W("| 域 | 任务数 | 最大单任务份额 | 最大份额任务 | 单任务上限放宽 |")
        W("|---|---|---|---|---|")
        for d in DOMAINS:
            x = S["train_domain"][d]
            W("| %s | %d | **%.1f%%** | %s | %s |"
              % (d, x["n_tasks"], x["max_task_share_pct"], x["max_task_share_name"],
                 "**是**" if x["cap_relaxed"] else "否"))
        W()
        W("| 域 | 训练集 | 其中派生(合成) | 派生份额 | 派生池可用 | 派生上限 | 派生上限放宽 |")
        W("|---|---|---|---|---|---|---|")
        for d in DOMAINS:
            x = S["train_domain"][d]
            W("| %s | %d | %d | %.1f%% | %d | %s | %s |"
              % (d, x["achieved"], x["derived_achieved"], x["derived_share_pct"],
                 x["pool_derived_available"],
                 "-" if x["derived_cap"] is None else x["derived_cap"],
                 "**是**" if x["derived_cap_relaxed"] else "否"))
        W()
        for d in DOMAINS:
            x = S["train_domain"][d]
            if not (x["cap_relaxed"] or x["derived_cap_relaxed"]):
                continue
            W("**为什么 %s 放宽了**：训练池 %d 条 vs 目标 %d 条（%.2f 倍）；"
              "域内头号任务 `%s` 池内可用 %d 条（占池 %.1f%%）。"
              "判据是 `Σ min(cap, avail) ≥ target` —— 不成立时只能放宽，"
              "否则配额数学上不可达。实测最大单任务份额 %.1f%%。"
              % (d, x["pool_available"], x["target"],
                 x["pool_available"] / max(1, x["target"]),
                 x["max_task_share_name"], x["max_task_share_avail"],
                 100.0 * x["max_task_share_avail"] / max(1, x["pool_available"]),
                 x["max_task_share_pct"]))
            W()
        P = S["train_domain"]["procedural"]
        W("**程序法口径**：目标 %d 条，训练池 %d 条（其中派生 %d 条，占配额 %.1f%%）。"
          % (P["target"], P["pool_available"], P["pool_derived_available"],
             P["derived_share_pct"]))
        if not P["cap_relaxed"]:
            W("程序法**本轮无需放宽单任务上限**：阶段 2c 已用程序法条文派生"
              "「程序法条文任务」把池子加宽（README 3.1 的「诉讼法条文任务」来源），"
              "头号任务份额从 49.4%% 降到 %.1f%%（上限 %.0f%%）。"
              % (P["max_task_share_pct"], 100 * S["config"]["task_max_share"]))
        else:
            W("程序法仍放宽了单任务上限 —— 说明**派生补量仍不足以撑起题型多样性**，"
              "需要继续扩源（扩采真实程序法数据，而非加大派生比例，"
              "因为派生组本身有 %.0f%% 上限）。"
              % (100 * (S["config"]["derived_max_share"] or 0)))
        W()
        W("| 域\\|任务 | 配额 | 池可用（训练池口径） |")
        W("|---|---|---|")
        for k, v in sorted(S["task_alloc"].items(),
                           key=lambda x: (-int(x[0].split("|")[0] in ("procedural",)),
                                          -x[1])):
            W("| %s | %d | %d |" % (k, v, S["task_avail"].get(k, 0)))
        W()
        W("## 4. 训练集分层（来源）")
        W()
        W("| 来源 | 条数 | 占比 |")
        W("|---|---|---|")
        for k, v in S["train_source"].items():
            W("| %s | %d | %.2f%% |" % (k, v, pct(v, S["splits"]["train"])))
        W()
        W("## 5. 验证集 / 测试集分布（按目标配比分层）")
        W()
        W("| 域 | val | test |")
        W("|---|---|---|")
        for d in DOMAINS:
            W("| %s | %d | %d |" % (d, S["dev_detail"][d]["val"],
                                    S["dev_detail"][d]["test"]))
        W()
        W("## 6. 路由集")
        W()
        W("- 规模 **%d**（= 训练目标的 %.0f%%）" %
          (S["splits"]["router"], 100.0 * S["config"]["router_rate"]))
        W("- 每条带 `router_label`（多标签 `domains`）与 `router_query`（user 侧问题文本）")
        W("- 域分布：" + "、".join("%s %d" % (k, v)
                                for k, v in S["router_domain"].items()))
        W("- 多标签（跨域）占比：%.2f%%" % S["router_multilabel_pct"])
        W("- ⚠️ 阶段 7 所需的「贴近 CLaw 254 案分布的 **200–500 条人工标注**路由评测集」"
          "**必须新建**，不在本阶段产物内。")
        W()
        W("## 7. 法条条目流（检索语料，不进 SFT）")
        W()
        st = S["statutes"]
        if st.get("present"):
            W("- 条目总数 **%d**（`data/corpus/statute_items/`，阶段 2b 产物）" % st["total"])
            W("- 逐域：" + "、".join("%s %d" % (k, v)
                                   for k, v in st["by_domain"].items()))
            W("- 逐 level：" + "、".join("%s %d" % (k, v)
                                       for k, v in st["by_level"].items()))
            W("- 逐 status：" + "、".join("%s %d" % (k, v)
                                        for k, v in st["by_status"].items()))
            W("- **不做下采样**：它是检索语料（向量库主料），不是 SFT 样本 ——"
              "检索侧语料越多越好，砍它没有收益只会掉召回。")
            W("- 跨流重叠检查：QA 四份 split 的 `content_sha1` ∩ 法条条目 `content_sha1` = **%d**"
              % st["cross_flow_overlap"])
            W("- 若将来要「法条任务」样本（README 3.1 的程序法来源之一），"
              "须**另派生**并保持与 val/test disjoint —— 本阶段未做。")
        else:
            W("- 未发现法条条目目录。")
        W()
        W("## 8. 输出文件（逐个 SHA-256）")
        W()
        W("| 文件 | 行数 | 字节 | SHA-256 |")
        W("|---|---|---|---|")
        for p, meta in S["outputs"].items():
            W("| `%s` | %d | %d | `%s` |" % (p, meta["lines"], meta["bytes"],
                                             meta["sha256"][:32] + "…"))
        W()
        W("完整 SHA-256 见 `SPLIT_STATS.json` 的 `outputs`。")
        W()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--train-target", type=int, default=100000)
    ap.add_argument("--mix", default=",".join("%s=%d" % (k, v)
                                              for k, v in DEFAULT_MIX.items()))
    ap.add_argument("--router-rate", type=float, default=0.20,
                    help="路由集占训练目标的比例（0.20 → 20,000）")
    ap.add_argument("--router-size", type=int, default=0,
                    help="显式指定路由集规模；0 = 用 router-rate 推算")
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--task-alpha", type=float, default=1.0,
                    help="域内任务配额权重指数（1=等比分层，<1 抹平）")
    ap.add_argument("--task-max-share", type=float, default=0.35,
                    help="单任务占域配额的软上限（不可行时自动放宽并留痕）")
    ap.add_argument("--derived-max-share", type=float, default=0.15,
                    help="派生(合成)样本占域配额的组上限；<0 表示禁用该上限。"
                         "0.15 的取值依据：下限=把单任务份额压到 task-max-share 所需的最小补量"
                         "（实测约 0.04），上限=防止合成样本喧宾夺主；0.15 是两者之间的保守值")
    ap.add_argument("--derived-dir", default="",
                    help="派生流目录（默认 <root>/data/corpus/derived）")
    ap.add_argument("--min-task-samples", type=int, default=100,
                    help="每个有数据的任务型至少给多少条（不足则给满）")
    ap.add_argument("--train-dir", default="")
    ap.add_argument("--dev-dir", default="")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--router-dir", default="")
    ap.add_argument("--max-records", type=int, default=0,
                    help="仅开发调试用：截断输入（会使配比达不到目标）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只算配额与不相交性，不落盘")
    args = ap.parse_args()

    args.train_dir = args.train_dir or os.path.join(args.root, "data/train")
    args.dev_dir = args.dev_dir or os.path.join(args.root, "data/dev")
    args.test_dir = args.test_dir or os.path.join(args.root, "data/test")
    args.router_dir = args.router_dir or os.path.join(args.root, "data/router")

    mix = parse_mix(args.mix)
    missing = [d for d in DOMAINS if d not in mix]
    if missing:
        sys.exit("mix 缺少域：%s" % missing)
    router_size = args.router_size or int(round(args.router_rate * args.train_target))

    t0 = time.time()
    derived_max_share = None if args.derived_max_share < 0 else args.derived_max_share
    files, skipped = input_files(args.root, args.derived_dir)
    derived_files = [os.path.basename(p) for p, d in files if d]
    if not derived_files:
        print("!! 警告：未发现派生流（%s 为空）—— 程序法题型多样性回到纯真实数据，"
              "单任务上限可能需要放宽"
              % (args.derived_dir or os.path.join(args.root, DERIVED_REL)))

    recs, per_file, bad = load_light(files, args.max_records)
    by_key, groups, dev_detail, detail, per_source_file = build_splits(
        recs, mix, router_size, args.val_size, args.test_size,
        args.task_alpha, args.task_max_share, args.min_task_samples,
        derived_max_share)

    # ---------------- 统计
    domain_cnt = collections.Counter(r["domain"] for r in recs)
    # ★ 口径：任务池可用量必须按**训练池**算（已剔除 router/val/test）。
    #   若用全池量，会和配额口径不一致，把「池子本来就不够」误判成「分配不足」
    #   —— 实测踩过：程序法尾部任务被误报 9 条「样本偏少」。
    non_train = groups["router"] | groups["val"] | groups["test"]
    task_avail = collections.Counter(
        "%s|%s" % (r["domain"], r["task"])
        for r in recs if r["key"] not in non_train)
    train_domain = {}
    for d in DOMAINS:
        sub = [r for r in recs if r["domain"] == d and r["key"] in groups["train"]]
        tc = collections.Counter(r["task"] for r in sub)
        top = tc.most_common(1)
        info = detail["task_alloc_info"][d]
        d_ach = sum(1 for r in sub if r["derived"])
        train_domain[d] = {
            "target": mix[d],
            "achieved": len(sub),
            "pool_available": detail["domain"][d]["pool_available"],
            "pool_derived_available": detail["domain"][d]["pool_derived_available"],
            "derived_achieved": d_ach,
            "derived_share_pct": pct(d_ach, len(sub)),
            "derived_cap": info.get("derived_cap"),
            "derived_cap_relaxed": bool(info.get("derived_cap_relaxed")),
            "pool_usage": detail["domain"][d]["pool_usage"],
            "n_tasks": len(tc),
            "max_task_share_name": top[0][0] if top else "",
            "max_task_share_pct": pct(top[0][1], len(sub)) if top else 0.0,
            "max_task_share_avail": tc.most_common(1)[0][1] if tc else 0,
            "cap_binding": bool(info.get("cap_binding")),
            "cap_relaxed": bool(info.get("cap_relaxed")),
            "task_counts": dict(tc.most_common()),
        }
    task_alloc = {}
    for d in DOMAINS:
        sub = [r for r in recs if r["domain"] == d and r["key"] in groups["train"]]
        for t, c in collections.Counter(r["task"] for r in sub).items():
            task_alloc["%s|%s" % (d, t)] = c

    router_recs = [r for r in recs if r["key"] in groups["router"]]
    router_domain = dict(collections.Counter(r["domain"] for r in router_recs
                                            ).most_common())
    ml = sum(1 for r in router_recs if len(r["domains"]) > 1)

    train_src = collections.Counter(r["source"] for r in recs
                                    if r["key"] in groups["train"])

    # ---------------- 落盘
    paths, write_info = write_splits(args, by_key, groups, files)

    outputs = {}
    for p in sorted(set(paths.values())):
        n = 0
        with open(p, "r", encoding="utf-8") as fh:
            for _ in fh:
                n += 1
        outputs[p] = {"lines": n, "bytes": os.path.getsize(p),
                      "sha256": sha256_file(p)}

    # ---------------- 法条流登记 + 跨流重叠
    statutes = register_statutes(args.root)
    if statutes.get("present"):
        qa_shas = set()
        for r in recs:
            qa_shas.add(r["key"])
        statutes["cross_flow_overlap"] = len(qa_shas & statutes["sha1_set"])
        st_sha = statutes.pop("sha1_set")
        statutes["cross_flow_overlap_note"] = (
            "QA split 的 content_sha1 与法条条目 content_sha1 的交集")
        statutes["unique_content_sha1"] = len(st_sha)

    total_used = len(groups["train"]) + len(groups["router"]) + \
        len(groups["val"]) + len(groups["test"])
    stats = {
        "generated_by": "scripts/corpus/downsample_split.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "elapsed_sec": round(time.time() - t0, 1),
        "config": {
            "train_target": args.train_target,
            "mix": mix,
            "router_rate": args.router_rate,
            "router_size": router_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "task_alpha": args.task_alpha,
            "task_max_share": args.task_max_share,
            "derived_max_share": derived_max_share,
            "derived_dir": args.derived_dir or os.path.join(args.root, DERIVED_REL),
            "min_task_samples": args.min_task_samples,
            "dry_run": bool(args.dry_run),
        },
        "input": {
            "records": len(recs),
            "real_records": len(recs) - sum(1 for r in recs if r["derived"]),
            "derived_records": sum(1 for r in recs if r["derived"]),
            "files": per_file,
            "skipped_files": skipped,
            "unique_content_sha1": len(by_key),
            "uid_collision_records": len(recs) - len({r["legacy_uid"] for r in recs}),
            "bad": dict(bad),
            "per_source_file": per_source_file,
            "by_domain": dict(domain_cnt.most_common()),
        },
        "splits": {
            "train": len(groups["train"]),
            "val": len(groups["val"]),
            "test": len(groups["test"]),
            "router": len(groups["router"]),
            "total_used": total_used,
            "leftover": len(recs) - total_used,
            "write": write_info,
        },
        "train_domain": train_domain,
        "task_alloc": task_alloc,
        "task_avail": dict(task_avail),
        "task_avail_scope": "train_pool（已剔除 router/val/test）",
        "train_source": dict(train_src.most_common()),
        "dev_detail": dev_detail,
        "router_domain": router_domain,
        "router_multilabel_pct": pct(ml, len(router_recs)),
        "statutes": statutes,
        "outputs": outputs,
    }

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    if args.out_md:
        render_md(args, stats, args.out_md)

    print("POOL=%d (real=%d derived=%d)"
          % (len(recs), stats["input"]["real_records"],
             stats["input"]["derived_records"]))
    print("SPLITS train=%d val=%d test=%d router=%d leftover=%d"
          % (len(groups["train"]), len(groups["val"]), len(groups["test"]),
             len(groups["router"]), len(recs) - total_used))
    for d in DOMAINS:
        x = train_domain[d]
        print("  %-11s want=%d got=%d pool=%d use=%.1f%% maxtask=%s %.1f%% "
              "relaxed=%s derived=%d(%.1f%%) dcap=%s drelax=%s"
              % (d, x["target"], x["achieved"], x["pool_available"],
                 x["pool_usage"], x["max_task_share_name"],
                 x["max_task_share_pct"], x["cap_relaxed"],
                 x["derived_achieved"], x["derived_share_pct"],
                 x["derived_cap"], x["derived_cap_relaxed"]))
    print("DRY_RUN=%s elapsed=%.1fs" % (args.dry_run, stats["elapsed_sec"]))
    print("OK -> %s" % args.out_json)


if __name__ == "__main__":
    main()
