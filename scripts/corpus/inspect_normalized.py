#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2 抽样复核（只读）：把域打标结果摊开给人看。
用法：python scripts/corpus/inspect_normalized.py [jsonl 路径]
"""
import collections
import json
import random
import sys

P = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/data/lidian/law-agent/data/corpus/normalized/qa/_all.sample.jsonl"

recs = []
with open(P, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print("总条数: %d" % len(recs))
print()

print("=" * 100)
print("域 × 来源文件 交叉表")
print("=" * 100)
dom_of_file = collections.defaultdict(collections.Counter)
for r in recs:
    dom_of_file[r["source_file"]][r["domain"]] += 1
for f in sorted(dom_of_file):
    c = dom_of_file[f]
    tot = sum(c.values())
    parts = ", ".join("%s=%d(%.0f%%)" % (k, v, 100.0 * v / tot) for k, v in c.most_common())
    print("  %-52s n=%-6d %s" % (f, tot, parts))

print()
print("=" * 100)
print("域 × 判定依据 交叉表")
print("=" * 100)
dom_src = collections.defaultdict(collections.Counter)
for r in recs:
    dom_src[r["domain"]][r["domain_source"]] += 1
for d in sorted(dom_src):
    print("  %-12s %s" % (d, dict(dom_src[d].most_common())))

print()
print("=" * 100)
print("每域抽样 3 条（人工肉眼核对）")
print("=" * 100)
by_dom = collections.defaultdict(list)
for r in recs:
    by_dom[r["domain"]].append(r)
random.seed(42)
for d in sorted(by_dom):
    print("\n### 域 = %s  (n=%d)\n" % (d, len(by_dom[d])))
    for r in random.sample(by_dom[d], min(3, len(by_dom[d]))):
        print("  [%s / %s] task=%s kind=%s src=%s" % (
            r["source_dataset"], r["source_file"], r["task"], r["task_kind"], r["domain_source"]))
        print("    domains=%s  scores=%s" % (r["domains"], r["domain_evidence"].get("scores")))
        print("    evidence=%s" % json.dumps(r["domain_evidence"].get("evidence", {}),
                                             ensure_ascii=False)[:300])
        print("    ask   : %s" % r["ask"][:150].replace("\n", " "))
        print("    input : %s" % r["input"][:110].replace("\n", " "))
        print("    output: %s" % r["output"][:110].replace("\n", " "))
        print()

print("=" * 100)
print("可疑样本：走 fallback 的（最多 15 条）")
print("=" * 100)
fb = [r for r in recs if r["domain_source"] == "fallback"]
print("fallback 条数 = %d" % len(fb))
for r in fb[:15]:
    print("  %-40s task=%-22s ask=%s" % (r["source_file"], r["task"],
                                         r["ask"][:90].replace("\n", " ")))
print()
print("=" * 100)
print("多标签样本（最多 10 条）")
print("=" * 100)
ml = [r for r in recs if len(r["domains"]) > 1]
print("多标签条数 = %d" % len(ml))
for r in ml[:10]:
    print("  domains=%-26s scores=%s  ask=%s" % (
        r["domains"], r["domain_evidence"].get("scores"),
        r["ask"][:80].replace("\n", " ")))
