#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 4: size up statute-corpus and large-judgment candidates found in round 3."""
import json
import urllib.error
import urllib.parse
import urllib.request

API = "https://hf-mirror.com/api/datasets"
TIMEOUT = 30


def get(url, accept=None, timeout=TIMEOUT):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "law-agent-probe/1.0")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


CANDIDATES = [
    "china-ai-law-challenge/cail2018",
    "twang2218/chinese-law-and-regulations",
    "Dusker/chinese-laws-pretrain",
    "wormtooth/MNBVC-judgment",
    "Kuugo/chinese_law_ft_dataset",
    "Kuugo/Chinese_Law",
    "Aiiluo/Chinese-Law-SFT-Dataset",
    "Dusker/lawyer-llama",
    "pandalla/chinese_law_examples",
    "Brench/chinese_law_data_rag_ft",
]

print("=" * 104)
print("[A] CANDIDATE SIZING")
print("=" * 104)
for ds in CANDIDATES:
    try:
        d = get(API + "/" + ds + "?blobs=true")
    except urllib.error.HTTPError as e:
        print("  %-46s HTTP %s (not proxied / missing)" % (ds, e.code))
        continue
    except Exception as e:
        print("  %-46s %s" % (ds, type(e).__name__))
        continue
    sib = d.get("siblings") or []
    tot = sum(s.get("size") or 0 for s in sib if isinstance(s.get("size"), int))
    lic = (d.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = ",".join(map(str, lic))
    print("  %-46s %8.1fMB files=%-4d lic=%-14s gated=%s" % (
        ds, tot / 1e6, len(sib), str(lic), str(d.get("gated"))))
    for s in sorted(sib, key=lambda x: -(x.get("size") or 0))[:3]:
        print("        %-58s %8.2f MB" % (s.get("rfilename"), (s.get("size") or 0) / 1e6))

print()
print("=" * 104)
print("[B] SEARCH: cail2019 / mnbvc / 法条")
print("=" * 104)
for t in ["cail2019", "cail2022", "mnbvc", "statute", "legal provision", "法条"]:
    u = API + "?" + urllib.parse.urlencode(
        {"search": t, "limit": 10, "sort": "downloads", "direction": -1})
    try:
        d = get(u)
    except Exception as e:
        print("  ERR [%s] %s" % (t, e))
        continue
    print("--- '%s' -> %d" % (t, len(d)))
    for x in d[:10]:
        print("    %-56s dl=%-8s likes=%s" % (x.get("id"), x.get("downloads"), x.get("likes")))

print()
print("PROBE4_DONE")
