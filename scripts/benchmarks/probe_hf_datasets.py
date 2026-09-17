#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe candidate legal datasets: HF mirror API (existence / size / license) + GitHub repos.

Read-only. Nothing is downloaded here - this is a feasibility probe only.
"""
import json
import urllib.request
import urllib.error

TIMEOUT = 25

HF_IDS = [
    "FudanDISC/DISC-Law-SFT",
    "FudanDISC/DISC-Law-SFT-Pair",
    "FudanDISC/DISC-Law-SFT-Triplet",
    "Skepsun/lawyer_llama_data",
    "Vicent0205/Chinese-Law-QA",
    "PKU-YuanGroup/ChatLaw",
    "THUIR/LEEC",
    "CSHaitao/LexEval",
    "CSHaitao/LegalAgentBench",
    "foggpoy/LexRubric",
    "OpenBMB/LawBench",
    "law-ai/LawGPT",
    "shibing624/lawyer_llama_data",
    "zjunlp/LawyerLLaMA",
    "SJTU-CL/Chinese-Law-QA",
]

GH_REPOS = [
    "FudanDISC/DISC-Law-SFT",
    "china-ai-law-challenge/cail2018",
    "china-ai-law-challenge/cail2019",
    "china-ai-law-challenge/cail2020",
    "pengxiao-song/LaWGPT",
    "PKU-YuanGroup/ChatLaw",
    "OpenBMB/LawBench",
    "CSHaitao/LegalAgentBench",
    "THUIR/LEEC",
]


def get(url, accept=None):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "law-agent-probe/1.0")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def probe_hf(ds_id):
    url = "https://hf-mirror.com/api/datasets/%s?blobs=true" % ds_id
    try:
        d = get(url)
    except urllib.error.HTTPError as e:
        return "ERR  %-42s HTTP %s" % (ds_id, e.code)
    except Exception as e:
        return "ERR  %-42s %s: %s" % (ds_id, type(e).__name__, e)

    sib = d.get("siblings") or []
    total = 0
    for s in sib:
        sz = s.get("size")
        if isinstance(sz, int):
            total += sz
    lic = (d.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    gated = d.get("gated")
    dl = d.get("downloads")
    return "OK   %-42s files=%-4d size=%9.1fMB license=%-18s gated=%-6s dl=%s" % (
        ds_id, len(sib), total / 1e6, str(lic), str(gated), str(dl))


def probe_gh(repo):
    url = "https://api.github.com/repos/%s" % repo
    try:
        d = get(url, accept="application/vnd.github+json")
    except urllib.error.HTTPError as e:
        return "ERR  %-40s HTTP %s" % (repo, e.code)
    except Exception as e:
        return "ERR  %-40s %s: %s" % (repo, type(e).__name__, e)
    lic = (d.get("license") or {}).get("spdx_id")
    return "OK   %-40s size=%8.1fMB license=%-14s stars=%-5s arch=%s" % (
        repo, (d.get("size") or 0) / 1024.0, str(lic), str(d.get("stargazers_count")),
        str(d.get("archived")))


def main():
    print("=" * 100)
    print("[A] HuggingFace mirror dataset probe  (endpoint: hf-mirror.com)")
    print("=" * 100)
    for i in HF_IDS:
        print(probe_hf(i))

    print()
    print("=" * 100)
    print("[B] GitHub repo probe  (api.github.com)")
    print("=" * 100)
    for r in GH_REPOS:
        print(probe_gh(r))
    print()
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
