#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 3: nail down file structure + real field names of the top candidates,
plus probe newly discovered leads. Uses HTTP Range to peek without full download.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

API = "https://hf-mirror.com/api/datasets"
RES = "https://hf-mirror.com/datasets"
TIMEOUT = 30


def get(url, accept=None, timeout=TIMEOUT):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "law-agent-probe/1.0")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def peek(url, nbytes=2500, timeout=TIMEOUT):
    """Fetch only the first nbytes of a file via HTTP Range."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "law-agent-probe/1.0")
    req.add_header("Range", "bytes=0-%d" % nbytes)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(nbytes)
    except Exception as e:
        return "<peek failed: %s: %s>" % (type(e).__name__, e)
    return raw.decode("utf-8", errors="replace")


def list_files(ds_id):
    try:
        d = get(API + "/" + ds_id + "?blobs=true")
    except Exception as e:
        return None, "ERR %s: %s" % (type(e).__name__, e)
    sib = d.get("siblings") or []
    rows = []
    for s in sib:
        rows.append((s.get("rfilename"), s.get("size")))
    rows.sort(key=lambda x: -(x[1] or 0))
    lic = (d.get("cardData") or {}).get("license")
    return rows, "lic=%s" % lic


def section(t):
    print()
    print("=" * 104)
    print(t)
    print("=" * 104)


TARGETS = [
    "ShengbinYue/DISC-Law-SFT",
    "china-ai-law-challenge/cail2018",
    "china-ai-law-challenge/cail2019",
    "InternLM/InternLM-Law",
    "yuwenhan07/MSLR-Bench",
    "EternWang/LegalScope",
    "maheran123/LegalNER-19",
    "Skepsun/lawyer_llama_data",
]

section("[A] FILE STRUCTURE + SAMPLE PEEK (via HTTP Range, no full download)")
for ds in TARGETS:
    rows, note = list_files(ds)
    if rows is None:
        print("  %-38s %s" % (ds, note))
        continue
    print("  %-38s %s  files=%d  total=%.1fMB" % (
        ds, note, len(rows), sum((s or 0) for _, s in rows) / 1e6))
    for name, size in rows[:4]:
        print("      %-60s %8.2f MB" % (name, (size or 0) / 1e6))
    # peek into the largest json/jsonl-ish file only
    for name, size in rows[:4]:
        if name and (name.endswith(".json") or name.endswith(".jsonl")):
            txt = peek("%s/%s/resolve/main/%s" % (RES, ds, name))
            print("      --- head of %s ---" % name)
            print("      " + txt[:900].replace("\n", "\n      "))
            break

section("[B] EXTRA SEARCH: more keywords")
for t in ["judgment", "court", "lawyer", "LegalOne", "LawBench", "legal instruction",
          "chinese law", "criminal law"]:
    u = API + "?" + urllib.parse.urlencode(
        {"search": t, "limit": 12, "sort": "downloads", "direction": -1})
    try:
        d = get(u)
    except Exception as e:
        print("  ERR [%s] %s" % (t, e))
        continue
    print("--- '%s' -> %d" % (t, len(d)))
    for x in d[:12]:
        print("    %-56s dl=%-8s likes=%s" % (x.get("id"), x.get("downloads"), x.get("likes")))

section("[C] GitHub org: china-ai-law-challenge (all repos)")
try:
    d = get("https://api.github.com/orgs/china-ai-law-challenge/repos?per_page=50",
            accept="application/vnd.github+json")
    for x in d:
        print("    %-46s %8.1fMB stars=%-5s %s" % (
            x.get("full_name"), (x.get("size") or 0) / 1024.0,
            x.get("stargazers_count"), (x.get("description") or "")[:50]))
except Exception as e:
    print("    ERR %s: %s" % (type(e).__name__, e))

print()
print("PROBE3_DONE")
