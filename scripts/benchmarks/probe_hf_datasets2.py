#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 2: discover REAL dataset IDs via the search endpoint instead of guessing IDs.

Background: hf-mirror returns HTTP 401 for non-existent datasets (not 404),
so a guess-based probe is unreliable. The search endpoint tells the truth.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 25
BASE = "https://hf-mirror.com/api/datasets"


def get(url, accept=None, timeout=TIMEOUT):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "law-agent-probe/1.0")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(term, limit=25):
    u = BASE + "?" + urllib.parse.urlencode(
        {"search": term, "limit": limit, "sort": "downloads", "direction": -1})
    try:
        d = get(u)
    except Exception as e:
        print("  SEARCH-ERR [%s] %s: %s" % (term, type(e).__name__, e))
        return
    print("--- search '%s' -> %d hits" % (term, len(d)))
    if not d:
        return
    for x in d:
        print("    %-58s dl=%-9s likes=%-6s" % (
            x.get("id"), x.get("downloads"), x.get("likes")))


def probe_id(ds_id):
    url = BASE + "/" + ds_id + "?blobs=true"
    try:
        d = get(url)
    except urllib.error.HTTPError as e:
        return "%-52s HTTP %s" % (ds_id, e.code)
    except Exception as e:
        return "%-52s %s" % (ds_id, type(e).__name__)
    sib = d.get("siblings") or []
    total = sum(s.get("size") or 0 for s in sib if isinstance(s.get("size"), int))
    lic = (d.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = ",".join(map(str, lic))
    return "%-52s OK files=%-4d %9.1fMB lic=%-16s gated=%s" % (
        ds_id, len(sib), total / 1e6, str(lic), str(d.get("gated")))


print("=" * 104)
print("[A] SEARCH by keyword (find real IDs)")
print("=" * 104)
for t in ["law", "legal", "法律", "刑法", "判决", "司法考试", "cail"]:
    search(t)

print()
print("=" * 104)
print("[B] TARGETED ID CHECK (corrected / newly discovered)")
print("=" * 104)
for i in [
    "ShengbinYue/DISC-Law-SFT",
    "ShengbinYue/DISC-Law-SFT-Pair",
    "ShengbinYue/DISC-Law-SFT-Triplet",
    "Skepsun/lawyer_llama_data",
    "CSHaitao/LexEval",
    "CSHaitao/LegalAgentBench",
]:
    print("  " + probe_id(i))

print()
print("=" * 104)
print("[C] GITHUB search: legal datasets by stars")
print("=" * 104)
for q in ["法律+dataset", "chinese+legal+llm", "legal+judgment+chinese"]:
    u = "https://api.github.com/search/repositories?q=%s&sort=stars&per_page=12" % q
    try:
        d = get(u, accept="application/vnd.github+json")
    except Exception as e:
        print("  GH-ERR [%s] %s: %s" % (q, type(e).__name__, e))
        continue
    print("--- gh '%s' -> %d" % (q, d.get("total_count", 0)))
    for x in (d.get("items") or [])[:12]:
        print("    %-46s %6.1fMB  stars=%-6s %s" % (
            x.get("full_name"), (x.get("size") or 0) / 1024.0,
            x.get("stargazers_count"), (x.get("description") or "")[:60]))

print()
print("PROBE2_DONE")
