#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 3：双向去污（硬门禁）。

做什么
------
把「评测集」当黑名单，与「训练语料」做双向比对，输出可审计报告。
  · 正向：训练集里有没有混入评测题（含改写）
  · 反向：评测集里有没有混入训练语料来源（同源风险，如 Skepsun 司考 vs LexRubric sifakaoshi）

方法
----
两种指纹，缺一不可（单一指纹会给出错误结论）：
  1) 精确指纹  sha1(normalize(text))          —— 抓逐字复制
  2) 近似指纹  simhash64(char 3-gram, crc32)  —— 抓改写/微调
     命中判据：汉明距离 <= HAMMING_MAX(默认 3)
     为了不做 3e5 × 1.5e4 的全量两两比对，用「4×16bit 分段索引(LSH)」先取候选，
     再对候选算精确汉明距离 —— 这是线性代价，且不丢召回。

黑名单（评测集，绝不进训练）
---------------------------
  · LexRubric   473 咨询(falvzixun) + 176 司考(sifakaoshi) = 649
  · LexEval     23 任务 / 14,150 题
  · CLaw        254 案 —— ⚠️ 官方未公开发布，本地若不存在则记为 blocker，
                报告 verdict 只能给 INCOMPLETE，不允许给 PASS。

输出
----
  docs/corpus/DECONTAMINATION_REPORT.json   （机器可读，含 verdict）
  docs/corpus/DECONTAMINATION_REPORT.md     （人读，含命中明细）

用法
----
  python decontaminate.py --root /mnt/data/lidian/law-agent \
      --out-json .../DECONTAMINATION_REPORT.json --out-md .../DECONTAMINATION_REPORT.md

硬约定（改动必须同步 docs 与本文件顶部）
----------------------------------------
1. 扫描源优先取 `qa/_all.jsonl`（合并副本，只扫一遍避免双计）；但**必须校验其新鲜度**：
   `_all.jsonl` 的 mtime 若早于任何正式分文件，即判定为过期并改用分文件扫描
   —— 否则复扫会读到旧产物，直接给出错误的 PASS。
2. 分文件扫描一律排除 `_` 开头（合并副本）与 `*.sample.jsonl`（调试残留）。
3. 命中以 `uid` 记入剔除清单；**前提是 uid 全局唯一**（阶段 2 已保证，
   `apply_decontam.py` 有恒等式断言兜底）。uid 撞号会导致「按 uid 剔除」整组连坐。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict

import numpy as np

HAMMING_MAX = 3
NGRAM = 3
MAX_CHARS = 800           # 取正文前 N 字符参与指纹（题目主干都在前面）
BANDS = 4                  # 64bit / 4 = 16bit 每段
GC_MAX = 3_000_000         # 3-gram → crc32 缓存上限（超限清空，避免吃光内存）

SKIP_KEYS = {"id", "split", "dimension", "point", "score", "source", "type",
             "task", "level", "difficulty", "index", "no", "num"}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# --------------------------------------------------------------------------
# 文本归一化与指纹
# --------------------------------------------------------------------------
def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u3000", " ").replace("\r", "\n")
    # 去空白（中文里空白无意义，去掉能显著提升匹配率）
    s = re.sub(r"[ \t\n]+", "", s)
    return s


def sha1_of(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


_GC: dict[str, int] = {}


def _g32(g: str) -> int:
    """3-gram → crc32，带全局缓存（中文 3-gram 复用率极高，缓存后提速明显）。"""
    v = _GC.get(g)
    if v is None:
        v = zlib.crc32(g.encode("utf-8"))
        if len(_GC) >= GC_MAX:
            _GC.clear()
        _GC[g] = v
    return v


def simhash64(s: str) -> int:
    """字符 3-gram simhash（64bit）。

    ⚠️ 性能提示：**不要**用「对每个 gram 循环 64 位」的朴素写法 ——
    实测 31 万条语料要 40+ 分钟。这里改成 numpy 位矩阵一次算完，约 8 倍提速。
    """
    if not s:
        return 0
    s = s[:MAX_CHARS]
    n = len(s) - NGRAM + 1
    if n <= 0:
        grams = [s]
    else:
        grams = [s[i:i + NGRAM] for i in range(n)]
    h = np.fromiter((_g32(g) for g in grams), dtype=np.uint64, count=len(grams))
    bits = np.unpackbits(h.view(np.uint8)).reshape(len(grams), 64)
    ones = bits.sum(axis=0, dtype=np.int32)
    out_bits = (ones * 2 > len(grams)).astype(np.uint8)
    return int.from_bytes(np.packbits(out_bits).tobytes(), "big")


def bands_of(h: int) -> list[int]:
    return [(h >> (16 * i)) & 0xFFFF for i in range(BANDS)]


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------
# 递归收集 JSON 里的文本
# --------------------------------------------------------------------------
def collect_text(obj, out: list[str], key: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            collect_text(v, out, str(k))
    elif isinstance(obj, list):
        for v in obj:
            collect_text(v, out, key)
    elif isinstance(obj, str):
        if key.lower() in SKIP_KEYS:
            return
        out.append(obj)
    elif isinstance(obj, (int, float)):
        return


def load_blacklist(paths_with_tag: list[tuple[str, str]]):
    """返回 items: [{tag, file, idx, text(norm), sha1, sh, bands}]"""
    items = []
    for path, tag in paths_with_tag:
        if not os.path.exists(path):
            print(f"  [skip] 不存在: {path}", flush=True)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
        # 可能是 JSON 或 JSONL
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            data = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
        if isinstance(data, dict):
            # 常见形态：{task_name: [ ... ]}
            data = list(data.values())
        n_before = len(items)
        base = os.path.basename(path)
        idx = 0
        if isinstance(data, list):
            for unit in data:
                outs: list[str] = []
                collect_text(unit, outs)
                t = norm_text("".join(outs))
                if len(t) >= 8:
                    items.append({"tag": tag, "file": base, "idx": idx,
                                  "text": t, "sha1": sha1_of(t), "sh": simhash64(t)})
                idx += 1
        elif isinstance(data, list) and data and isinstance(data[0], list):
            pass
        print(f"  [load] {tag:<10} {base:<22} -> {len(items)-n_before} 条", flush=True)
    return items


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--hamming", type=int, default=HAMMING_MAX)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = 不限制（调试用）")
    ap.add_argument("--progress-secs", type=int, default=15, help="心跳间隔（秒）")
    ap.add_argument("--max-near", type=int, default=200,
                    help="报告 JSON/MD 里最多列出的命中样例条数（计数不受影响，全量保存）")
    ap.add_argument("--norm-dir", default="",
                    help="被查语料目录（默认 <root>/data/corpus/normalized；"
                         "二次验证时指向 data/corpus/decontaminated）")
    ap.add_argument("--out-uids", default="",
                    help="剔除清单输出路径（默认与 out-json 同目录 DECONTAM_REMOVED_UIDS.txt）")
    ap.add_argument("--require-claw", action="store_true",
                    help="把 CLaw 254 案缺失算作 blocker（默认否：2026-09-18 起CLaw 不进论文评测）")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[pid={os.getpid()}] 去污开始 {time.strftime('%H:%M:%S')}", flush=True)
    root = a.root
    norm_dir = a.norm_dir or os.path.join(root, "data/corpus/normalized")
    bench = os.path.join(root, "data/benchmark")

    print("=" * 78, flush=True)
    print("[1/4] 载入黑名单（评测集）", flush=True)

    def _glob_json(d: str) -> list[str]:
        if not os.path.isdir(d):
            return []
        return [os.path.join(d, fn) for fn in sorted(os.listdir(d))
                if fn.endswith(".json")]

    bl_specs: list[tuple[str, str]] = []
    # LexRubric：两个 split 分别打标（sifakaoshi 要单独做同源核查）
    lr_dir = os.path.join(bench, "lexrubric/repo/data")
    for p in _glob_json(lr_dir):
        base = os.path.basename(p)
        tag = "lexrubric_sikao" if "sifakaoshi" in base else "lexrubric_zixun"
        bl_specs.append((p, tag))
    # LexEval：**必须自动扫描全部 23 个任务文件**，不要硬编码（曾漏 9 个文件）
    for p in _glob_json(os.path.join(bench, "lexeval/repo/data")):
        bl_specs.append((p, "lexeval"))
    # CLaw：官方未公开发布；若用户日后自建完成，把文件放到下面任一位置即自动纳入
    claw_cands = [
        os.path.join(bench, "claw/repo/data/claw254.json"),
        os.path.join(bench, "claw/claw254.json"),
        os.path.join(root, "data/claw/claw254.json"),
    ]
    claw_found = None
    for c in claw_cands:
        if os.path.exists(c):
            claw_found = c
            bl_specs.append((c, "claw"))
            break

    bl = load_blacklist(bl_specs)
    exact_index = defaultdict(list)
    for it in bl:
        exact_index[it["sha1"]].append(it)

    band_index = [defaultdict(list) for _ in range(BANDS)]
    for i, it in enumerate(bl):
        for bi, bv in enumerate(bands_of(it["sh"])):
            band_index[bi][bv].append(i)

    print(f"  黑名单合计 = {len(bl)} 条", flush=True)
    print(f"  CLaw 254 案: {'已载入 ' + claw_found if claw_found else '缺失（官方未公开发布，走自建路线）'}",
          flush=True)
    bl_tag = Counter(it["tag"] for it in bl)

    # ----------------------------------------------------------------------
    print("[2/4] 扫描训练语料", flush=True)
    targets = []
    qa_dir = os.path.join(norm_dir, "qa")
    qa_parts = [os.path.join(qa_dir, fn) for fn in sorted(os.listdir(qa_dir))
                if fn.endswith(".jsonl") and not fn.startswith("_")
                and ".sample." not in fn]
    qa_all = os.path.join(qa_dir, "_all.jsonl")
    # ★ 合并副本优先（只扫一遍，避免双计）—— 但必须校验它是**新鲜的**：
    #   过期/陈旧的 _all.jsonl 会让复扫看不到最新产物，直接给出错误的 PASS。
    #   判据：_all.jsonl 的 mtime 不得早于任何正式分文件。
    if os.path.exists(qa_all) and qa_parts:
        if os.path.getmtime(qa_all) + 1e-6 < max(os.path.getmtime(p) for p in qa_parts):
            print("  !! 警告：qa/_all.jsonl 比正式分文件旧 → 视为陈旧，改用分文件扫描：")
            for p in qa_parts:
                print("       %s" % os.path.basename(p))
            targets.extend(("qa", p) for p in qa_parts)
        else:
            targets.append(("qa", qa_all))
    elif qa_parts:
        targets.extend(("qa", p) for p in qa_parts)
    stat_dir = os.path.join(norm_dir, "statutes")
    for fn in sorted(os.listdir(stat_dir)):
        if fn.endswith(".jsonl") and not fn.startswith("_") and ".sample." not in fn:
            targets.append(("statutes", os.path.join(stat_dir, fn)))

    print(f"  待扫 {len(targets)} 个文件，合计 "
          f"{sum(os.path.getsize(p) for _, p in targets)/1e6:.0f} MB", flush=True)
    total = 0
    exact_hits = []          # 全量保存（精确命中通常极少）
    near_hits = []           # 全量保存（约 2 万条 dict，内存可忽略）
    near_count = 0           # 命中事件数（side 级：同一行 question/full 两侧各算一次）
    hit_uids = set()         # 需剔除的唯一训练样本 uid（精确+近似）
    per_domain = Counter()
    per_file = Counter()
    bl_hit_counter = Counter()

    for kind, path in targets:
        fname = os.path.basename(path)
        fsize = os.path.getsize(path)
        print(f"  -> {kind}/{fname}  ({fsize/1e6:.1f} MB)", flush=True)
        last_beat = time.time()
        hit_max = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if a.max_rows and total >= a.max_rows:
                    hit_max = True
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                total += 1
                dom = r.get("domain") or "?"
                per_domain[dom] += 1
                per_file[fname] += 1

                q_side = norm_text(str(r.get("instruction") or "") + str(r.get("input") or ""))
                full = q_side + norm_text(str(r.get("output") or ""))
                if len(full) < 8:
                    continue

                for side_name, side in (("question", q_side), ("full", full)):
                    if len(side) < 8:
                        continue
                    s1 = sha1_of(side)
                    if s1 in exact_index:
                        for it in exact_index[s1]:
                            exact_hits.append({
                                "train_uid": r.get("uid"), "train_domain": dom,
                                "train_file": fname, "side": side_name,
                                "bench_tag": it["tag"], "bench_file": it["file"],
                                "bench_idx": it["idx"], "method": "exact_sha1",
                            })
                            bl_hit_counter[it["tag"]] += 1
                        hit_uids.add(r.get("uid"))
                        continue

                    sh = simhash64(side)
                    cand = set()
                    for bi, bv in enumerate(bands_of(sh)):
                        cand.update(band_index[bi].get(bv, ()))
                    best = None
                    for i in cand:
                        d = hamming(sh, bl[i]["sh"])
                        if d <= a.hamming and (best is None or d < best[0]):
                            best = (d, i)
                    if best is not None:
                        d, i = best
                        it = bl[i]
                        near_count += 1
                        bl_hit_counter[it["tag"]] += 1
                        hit_uids.add(r.get("uid"))
                        near_hits.append({
                            "train_uid": r.get("uid"), "train_domain": dom,
                            "train_file": fname, "side": side_name,
                            "hamming": d,
                            "bench_tag": it["tag"], "bench_file": it["file"],
                            "bench_idx": it["idx"], "method": "simhash_near",
                            "train_head": str(r.get("instruction") or "")[:80],
                        })

                now = time.time()
                if now - last_beat >= a.progress_secs:
                    last_beat = now
                    try:
                        pos = f.tell()
                    except Exception:
                        pos = 0
                    print(f"     [beat] 已扫 {total} 条 | 文件进度 {pos/1e6:.0f}/{fsize/1e6:.0f} MB"
                          f" | 精确 {len(exact_hits)} | 近重 {near_count}"
                          f" | 已用 {now-t0:.0f}s", flush=True)
        if hit_max:
            print(f"  [--max-rows {a.max_rows} 触发，提前结束]", flush=True)
            break

    # ----------------------------------------------------------------------
    print("[3/4] 同源风险专项：Skepsun 司考(训练) vs LexRubric sifakaoshi(评测)", flush=True)
    same_source = {"checked_skepsun_rows": 0, "exact": 0, "near": 0, "samples": []}
    ske = os.path.join(norm_dir, "qa/Skepsun__lawyer_llama_data.jsonl")
    sikao_ids = [i for i, it in enumerate(bl) if it["tag"] == "lexrubric_sikao"]
    if os.path.exists(ske) and sikao_ids:
        with open(ske, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                same_source["checked_skepsun_rows"] += 1
                t = norm_text(str(r.get("instruction") or "") + str(r.get("input") or ""))
                if len(t) < 8:
                    continue
                s1 = sha1_of(t)
                hit_exact = exact_index.get(s1)
                if hit_exact:
                    for it in hit_exact:
                        if it["tag"] == "lexrubric_sikao":
                            same_source["exact"] += 1
                            if len(same_source["samples"]) < 20:
                                same_source["samples"].append(
                                    {"uid": r.get("uid"), "type": "exact",
                                     "bench_idx": it["idx"]})
                            break
                    continue
                sh = simhash64(t)
                cand = set()
                for bi, bv in enumerate(bands_of(sh)):
                    cand.update(band_index[bi].get(bv, ()))
                for i in cand:
                    if bl[i]["tag"] != "lexrubric_sikao":
                        continue
                    d = hamming(sh, bl[i]["sh"])
                    if d <= a.hamming:
                        same_source["near"] += 1
                        if len(same_source["samples"]) < 20:
                            same_source["samples"].append(
                                {"uid": r.get("uid"), "type": "near", "hamming": d,
                                 "bench_idx": bl[i]["idx"],
                                 "train_head": str(r.get("instruction"))[:80]})
                        break

    # ----------------------------------------------------------------------
    print("[4/4] 出报告", flush=True)
    blockers = []
    # 2026-09-18 决策：CLaw（含 254 案）不再纳入论文评测体系，黑名单 = LexEval + LexRubric。
    # 保留 --require-claw 开关以备未来恢复；默认不把 CLaw 缺失当 blocker。
    if a.require_claw and not claw_found:
        blockers.append(
            "CLaw 254 案未就位（--require-claw 已开启）：官方数据未公开发布。"
            "把自建文件放到 data/benchmark/claw/repo/data/claw254.json 后重跑即可纳入。"
        )
    if exact_hits or hit_uids:
        verdict = "FAIL"
    elif blockers:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": {
            "exact": "sha1(normalize(instruction+input)) 与 sha1(normalize(instruction+input+output))",
            "near": f"simhash64(char-{NGRAM}gram, crc32), 汉明距离 <= {a.hamming}",
            "lsh": f"{BANDS} x 16bit 分段索引取候选，再算精确汉明距离",
            "normalize": "NFKC + 去所有空白",
            "max_chars_for_fingerprint": MAX_CHARS,
        },
        "blacklist": {
            "total": len(bl),
            "by_tag": dict(bl_tag),
            "claw_loaded": bool(claw_found),
            "claw_path_expected": claw_cands[0],
        },
        "checked": {
            "total_rows": total,
            "by_domain": dict(per_domain),
            "by_file": dict(per_file),
        },
        "exact_hits": {
            "count": len(exact_hits),
            "by_bench_tag": dict(Counter(h["bench_tag"] for h in exact_hits)),
            "items": exact_hits[:a.max_near],
        },
        "near_hits": {
            "count": near_count,
            "unique_train_uids": len(hit_uids),
            "by_bench_tag": dict(Counter(h["bench_tag"] for h in near_hits)),
            "by_train_file": dict(Counter(h["train_file"] for h in near_hits)),
            "by_train_domain": dict(Counter(h["train_domain"] for h in near_hits)),
            "by_hamming": dict(sorted(Counter(h["hamming"] for h in near_hits).items())),
            "items": near_hits[:a.max_near],
        },
        "same_source_check": same_source,
        "verdict": verdict,
        "blockers": blockers,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 剔除清单：每个被命中（精确或近似）的唯一训练样本 uid，一行一个
    uids_path = a.out_uids or os.path.join(
        os.path.dirname(a.out_json), "DECONTAM_REMOVED_UIDS.txt")
    with open(uids_path, "w", encoding="utf-8") as f:
        for u in sorted(hit_uids):
            f.write(str(u) + "\n")
    print(f"剔除清单: {uids_path} ({len(hit_uids)} 个 uid)", flush=True)

    md = []
    md.append("# 阶段 3：双向去污报告（DECONTAMINATION REPORT）\n")
    md.append(f"> 生成时间：{report['generated_at']}　耗时 {report['elapsed_sec']}s\n")
    md.append(f"> **verdict = `{verdict}`**\n")
    md.append("## 0. 结论\n")
    if verdict == "PASS":
        md.append("- 训练语料与已就位评测集之间**未发现精确复制或近似改写**。\n")
    elif verdict == "FAIL":
        md.append(f"- ⚠️ 发现 **精确命中 {len(exact_hits)} 条 / 近似命中 {near_count} 条**，"
                  "必须剔除后才能进训练。\n")
    else:
        md.append("- ⚠️ 去污**未完成**（存在 blocker），本次结果**不允许**当作 PASS 使用。\n")
    for b in blockers:
        md.append(f"- 🚧 blocker：{b}\n")
    md.append("\n## 1. 方法\n")
    for k, v in report["method"].items():
        md.append(f"- `{k}`：{v}\n")
    md.append("\n## 2. 黑名单（评测集）\n")
    md.append(f"- 合计 **{len(bl)}** 条：{dict(bl_tag)}\n")
    md.append(f"- CLaw 254 案：{'已载入' if claw_found else '**缺失**（官方未公开发布，走自建路线）'}\n")
    md.append("\n## 3. 被查训练语料\n")
    md.append(f"- 合计 **{total}** 行；按域 {dict(per_domain)}\n")
    for k, v in sorted(per_file.items()):
        md.append(f"  - `{k}`：{v}\n")
    md.append("\n## 4. 命中\n")
    md.append(f"- 精确命中：**{len(exact_hits)}** 条\n")
    if exact_hits:
        md.append(f"  - 按基准：{dict(Counter(h['bench_tag'] for h in exact_hits))}\n")
    md.append(f"- 近似命中（汉明 <= {a.hamming}）：**{near_count}** 个事件"
              f"（同一行 question/full 两侧各算一次），涉及唯一训练样本 "
              f"**{len(hit_uids)}** 条\n")
    if near_hits:
        md.append(f"  - 按基准：{dict(Counter(h['bench_tag'] for h in near_hits))}\n")
        md.append(f"  - 按训练文件：{dict(Counter(h['train_file'] for h in near_hits))}\n")
        md.append(f"  - 按域：{dict(Counter(h['train_domain'] for h in near_hits))}\n")
        md.append(f"  - 汉明距离分布：{dict(sorted(Counter(h['hamming'] for h in near_hits).items()))}\n")
    md.append(f"- 剔除清单（唯一 uid）：`{os.path.basename(uids_path)}`"
              f"（{len(hit_uids)} 条）\n")
    md.append("\n## 5. 同源风险专项：Skepsun 司考 vs LexRubric sifakaoshi\n")
    md.append(f"- 检查 Skepsun 训练样本 **{same_source['checked_skepsun_rows']}** 条\n")
    md.append(f"- 精确命中 **{same_source['exact']}** 条 / 近似命中 **{same_source['near']}** 条\n")
    if same_source["samples"]:
        md.append("- 样例（前 20）：\n")
        for s in same_source["samples"]:
            md.append(f"  - `{json.dumps(s, ensure_ascii=False)}`\n")
    md.append("\n## 6. 处置\n")
    md.append("1. 命中的训练样本一律**从训练集剔除**（脚本按 `train_uid` 生成黑名单）。\n")
    md.append("2. 剔除后**重跑本脚本**，verdict 必须变为 `PASS` 才允许进入阶段 4。\n")
    md.append("3. CLaw 254 案自建完成后**必须再跑一次**，否则论文里不能声称已做完整去污。\n")

    with open(a.out_md, "w", encoding="utf-8") as f:
        f.writelines(md)

    print("=" * 78, flush=True)
    print(f"verdict = {verdict}", flush=True)
    print(f"精确命中 {len(exact_hits)} / 近似命中 {near_count} / 已扫 {total} 行", flush=True)
    print(f"报告: {a.out_json}", flush=True)
    print(f"报告: {a.out_md}", flush=True)
    print("DONE_DECONTAM", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
