#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断去污后 QA 池的 uid 重复性质与 messages 结构（只读）。

回答三个问题：
  1. 20,947 条 uid 重复是「同内容重复」还是「uid 撞号（内容不同）」？
  2. content_sha1 全局重复有多少（真正的内容级重复）？
  3. `messages` 长度不等于 2 的 26,960 条是什么结构（是否含 system 消息）？

用法：
  python scripts/corpus/probe_split_inputs2.py --root /mnt/data/lidian/law-agent \
      --out-json /tmp/probe_split2.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

QA_REL = "data/corpus/decontaminated/qa"


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    qa = os.path.join(args.root, QA_REL)
    files = sorted(f for f in os.listdir(qa)
                   if f.endswith(".jsonl") and not f.startswith("_"))

    uid_rows = collections.defaultdict(list)
    sha_rows = collections.defaultdict(list)
    fkey_rows = collections.defaultdict(list)   # (source_dataset, source_file, source_index)

    msg_len_by_source = collections.defaultdict(collections.Counter)
    sys_msg_by_source = collections.defaultdict(collections.Counter)
    role_sig = collections.Counter()
    per_file_source = collections.Counter()
    total = 0

    for fn in files:
        for rec in iter_jsonl(os.path.join(qa, fn)):
            total += 1
            uid = rec.get("uid") or ""
            sha = rec.get("content_sha1") or ""
            src = rec.get("source_dataset") or "?"
            sfile = rec.get("source_file") or "?"
            sidx = rec.get("source_index")
            uid_rows[uid].append((src, sfile, sidx, sha, rec.get("task"),
                                  rec.get("domain")))
            sha_rows[sha].append((uid, src))
            fkey_rows[(src, sfile, sidx)].append(uid)
            per_file_source["%s|%s" % (src, sfile)] += 1
            msgs = rec.get("messages") or []
            msg_len_by_source[src][len(msgs)] += 1
            has_sys = int(bool(msgs) and msgs[0].get("role") == "system")
            sys_msg_by_source[src][has_sys] += 1
            role_sig["/".join(m.get("role", "?") for m in msgs[:5])] += 1

    # ---- 1. uid 重复性质
    dup_uids = {u: v for u, v in uid_rows.items() if len(v) > 1}
    dup_same_content = 0
    dup_diff_content = 0
    dup_cross_file = 0
    dup_file_pairs = collections.Counter()
    dup_examples = []
    for u, v in dup_uids.items():
        shas = {x[3] for x in v}
        if len(shas) == 1:
            dup_same_content += 1
        else:
            dup_diff_content += 1
            if len(dup_examples) < 6:
                dup_examples.append({
                    "uid": u,
                    "n": len(v),
                    "rows": [{"dataset": x[0], "file": x[1], "index": x[2],
                              "sha1": (x[3] or "")[:12], "task": x[4],
                              "domain": x[5]} for x in v[:6]],
                })
        fileset = {x[1] for x in v}
        if len(fileset) > 1:
            dup_cross_file += 1
        dup_file_pairs[tuple(sorted(fileset))] += 1

    # ---- 2. content_sha1 重复
    dup_sha = {s: v for s, v in sha_rows.items() if len(v) > 1 and s}
    dup_sha_rows = sum(len(v) - 1 for v in dup_sha.values())
    dup_sha_examples = []
    for s, v in list(dup_sha.items())[:6]:
        dup_sha_examples.append({"sha1": s[:12], "n": len(v),
                                 "uids": [x[0] for x in v[:6]],
                                 "datasets": sorted({x[1] for x in v})})

    # ---- 3. 消息结构
    role_sig_top = dict(role_sig.most_common(12))

    out = {
        "total_records": total,
        "unique_uid": len(uid_rows),
        "dup_uid_count": len(dup_uids),
        "dup_uid_extra_records": sum(len(v) - 1 for v in dup_uids.values()),
        "dup_uid_same_content": dup_same_content,
        "dup_uid_diff_content": dup_diff_content,
        "dup_uid_cross_source_file": dup_cross_file,
        "dup_uid_file_pairs": {" + ".join(k): v for k, v in
                               dup_file_pairs.most_common(20)},
        "dup_uid_diff_content_examples": dup_examples,
        "unique_content_sha1": len(sha_rows),
        "dup_sha1_groups": len(dup_sha),
        "dup_sha1_extra_records": dup_sha_rows,
        "dup_sha1_examples": dup_sha_examples,
        "per_source_file_records": dict(per_file_source.most_common()),
        "messages_len_by_source": {k: dict(v) for k, v in
                                   msg_len_by_source.items()},
        "system_msg_by_source": {k: dict(v) for k, v in
                                 sys_msg_by_source.items()},
        "role_signature_top": role_sig_top,
        "unique_file_key": len(fkey_rows),
        "dup_file_key": sum(1 for v in fkey_rows.values() if len(v) > 1),
    }
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("TOTAL=%d unique_uid=%d uid_dup=%d (extra_records=%d)"
          % (total, len(uid_rows), len(dup_uids),
             out["dup_uid_extra_records"]))
    print("uid_dup same_content=%d diff_content=%d cross_file=%d"
          % (dup_same_content, dup_diff_content, dup_cross_file))
    print("unique_content_sha1=%d dup_sha1_extra=%d"
          % (len(sha_rows), dup_sha_rows))
    print("dup_file_key=%d unique_file_key=%d"
          % (out["dup_file_key"], out["unique_file_key"]))
    print("role_sig_top=%s" % json.dumps(role_sig_top, ensure_ascii=False)[:400])
    print("msg_len_by_source=%s"
          % json.dumps({k: dict(v) for k, v in msg_len_by_source.items()},
                       ensure_ascii=False))
    print("OK -> %s" % args.out_json)


if __name__ == "__main__":
    main()
