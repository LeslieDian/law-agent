#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `docs/eval/README_BLOCK.md` 的内容原子地替换进 `README.md` 的 AUTO 标记区。

为什么要有这个脚本
--------------------------------------------------------------------------
过夜链跑完后要把真实评测数字写进 README。**让模型手改 markdown 是不可靠的**
（长文档里定位易错、容易连带改坏别处）。这里把"替换"变成确定性操作：
标记区 `<!-- AUTO_RESULTS_BEGIN ... -->` … `<!-- AUTO_RESULTS_END -->`（含标记本身）
整体替换为 block 文件内容。**标记之外一个字节都不动。**

用法
--------------------------------------------------------------------------
    python scripts/eval/patch_readme.py \
        --readme README.md --block docs/eval/README_BLOCK.md

    # 只看会改成什么，不落盘
    python scripts/eval/patch_readme.py --readme README.md --block X.md --dry-run
"""
from __future__ import annotations

import argparse
import sys

BEGIN = "<!-- AUTO_RESULTS_BEGIN"
END = "<!-- AUTO_RESULTS_END"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--block", default="docs/eval/README_BLOCK.md")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(a.readme, encoding="utf-8") as f:
        txt = f.read()
    with open(a.block, encoding="utf-8") as f:
        blk = f.read().strip()

    i = txt.find(BEGIN)
    j = txt.find(END)
    if i < 0 or j < 0:
        print("[FATAL] 在 %s 里找不到 AUTO 标记区。先手动插入一次：" % a.readme)
        print("        %s 由 scripts/eval/collect_results.py 自动生成，勿手改 -->" % BEGIN)
        print("        %s -->" % END)
        return 2
    if j < i:
        print("[FATAL] AUTO 标记顺序颠倒（END 在 BEGIN 之前）")
        return 2
    j += len(END)
    # 把 END 标记所在行剩下的 "-->" 一并吃掉
    rest_of_line = txt.find("\n", j)
    if rest_of_line < 0:
        rest_of_line = len(txt)
    if txt[j:rest_of_line].strip() in ("-->", "->"):
        j = rest_of_line

    old = txt[i:j]
    new = txt[:i] + blk + txt[j:]
    print("标记区：%d 字符 -> %d 字符" % (len(old), len(blk)))
    print("块内行数：%d" % blk.count("\n"))
    print("--- 新块前 3 行 ---")
    for line in blk.splitlines()[:3]:
        print("   " + line)
    print("--- 新块末 2 行 ---")
    for line in blk.splitlines()[-2:]:
        print("   " + line)

    if a.dry_run:
        print("[dry-run] 未落盘")
        return 0
    tmp = a.readme + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    import os
    os.replace(tmp, a.readme)
    print("已写回 %s（标记外内容未改动）" % a.readme)
    print("MARKER_README_PATCHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
