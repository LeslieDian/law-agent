#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2b：法条切条（整部法规 → 逐条法条），产出向量库主料。

输入：<root>/data/corpus/decontaminated/statutes/*.jsonl
      - twang2218__chinese-law-and-regulations.jsonl  **文档级**（22,771 部整部法规，任务 statute_doc）
      - pandalla__chinese_law_examples.jsonl          **条级**（984 条样例，任务 statute_item）
输出：<root>/data/corpus/statute_items/*.items.jsonl（统一「条级」schema）
报告：--out-json / --out-md

设计要点（均为实测驱动，改动请在此补充理由）
--------------------------------------------------------------------------
1. **只留国家级规范**：type ∈ {法律, 法律解释, 司法解释, 行政法规}。
   地方性法规 19,773 部（占文档级源的 87%）与部门规章、团体规定等不进向量库
   —— 它们是「某地/某部门的规则」，会污染全国性法律问答的检索结果。
   ⚠️ 宪法（含宪法修正案）默认**不保留**，理由：宪法条文极少被作为裁判依据引用、
   且修正案文本（"第三十二条 宪法序言第七自然段中…修改为…"）是**修改指令**而非规范条文，
   直接进向量库会严重误导检索。如需入库存量，用 --types 显式加上，请先看报告里的条数。

2. **锚点必须盯行首**：正文行首统一带一个半角空格（实测 indent_space 40,079 / 
   表意空格 0）。“依据本法第三十二条规定”这类**交叉引用**在正文中随处可见
   （实测全文本正则命中 27,394 次 vs 行首锚点唯一值 24,954 → 虚高约 2,440）。
   所以切条一律用 `^\\s*第X条` 行首锚点，绝不在文中间找「第X条」。

3. **剥离目录块**：303/400 部带「目 录」，其条目是章/节标题。目录里的章标题与正文
   顺序一致，若不剥离会把目录误当成章节归属来源。实现：见到「目 录」进入 TOC 模式，
   吞掉后续的标题类行与空行，直到第一行非标题正文出现。

4. **status 脏值清洗**：实测存在 `status="7"`（400 条里 7 条，全部落在
   「有关法律问题和重大问题的决定」类型上，属源数据字段错位）。本脚本把
   status ∉ {有效, 已修改, 已废止, 尚未生效} 的一律规范为「未知」，原值留存
   `status_raw` 并计数，**不静默丢弃**。注意 status="7" 的文档类型本身已被第 1 条过滤掉。

5. **历史版本一律保留**（硬约定第 5 条）：同一部法的多个版本条文各自保留，
   靠 `status` / `effective_from` / `effective_period` 区分；同一条号不同版本
   内容相同的，计入 `version_duplicates`，**不删**（删了会破坏版本链）。

6. **domain 打标走本阶段局部规则** `domain_source="statute_title_keyword"`：
   按**法规名**关键词判定（与阶段 2 QA 侧的六路加权打分**无关**，不要混用）。
   法规名是最强信号（「中华人民共和国刑法」→ criminal），比条文正文引用更可靠。
   命中不了 → general + `fallback`，占比在报告里登记。

7. **条目 uid 必须含源文件**：格式 `<dataset>__<file_stem>:<source_index>#<条号>`（`level="doc"`
   为 `…#__doc__`）。旧实现用 `<source_id>#<条号>`，而 `source_id` 只是文档在文件内的序号
   —— 与阶段 2 的 uid 撞号缺陷同源（漏了「源文件」这一维）。当前 twang2218 只有 1 个
   parquet 所以没暴露，但分片 parquet 或跨数据集同号时会**静默撞号**，下游按 uid 操作
   会整组连坐。改由 `doc_uid()` 统一生成，唯一性由构造成立，并由质检 C1 断言。

只读约定：不修改任何输入文件。
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_TYPES = ("法律", "法律解释", "司法解释", "行政法规")
KNOWN_STATUS = {"有效", "已修改", "已废止", "尚未生效"}
MIN_BODY_CHARS = 4          # 条文正文（不含条号）短于此视为空条，丢弃并计数

RE_ARTICLE = re.compile(
    r"^[ \t\u3000]*(第[〇零一二三四五六七八九十百千两0-9]+条(?:之[〇零一二三四五六七八九十]+)?)[ \t\u3000]*(.*)$")
RE_PART = re.compile(r"^[ \t\u3000]*第[〇零一二三四五六七八九十百千两0-9]+编[ \t\u3000]*(.*)$")
RE_CHAPTER = re.compile(r"^[ \t\u3000]*第[〇零一二三四五六七八九十百千两0-9]+章[ \t\u3000]*(.*)$")
RE_SECTION = re.compile(r"^[ \t\u3000]*第[〇零一二三四五六七八九十百千两0-9]+节[ \t\u3000]*(.*)$")
RE_TOC = re.compile(r"^[ \t\u3000]*目[ \t\u3000]*录[ \t\u3000]*$")
RE_TITLE_LIKE = re.compile(r"^[ \t\u3000]*(第[〇零一二三四五六七八九十百千两0-9]+[编章节]|序[ \t\u3000]*言|附[ \t\u3000]*则|目[ \t\u3000]*录)")
RE_ESC_NL = re.compile(r"\\\s*\n")   # 源数据里的转义换行 `\<换行>`，直接黏合

CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}

# 法规名 → 域（第一命中即用；顺序有意义：先程序法、再刑法、再民法）
DOMAIN_RULES = [
    ("procedural", ("诉讼法", "刑事诉讼法", "民事诉讼法", "行政诉讼法", "仲裁", "调解",
                    "程序", "执行", "证据", "监狱", "公证", "司法鉴定", "引渡", "管辖")),
    ("criminal", ("刑法", "犯罪", "刑事", "禁毒", "反有组织犯罪", "反恐怖", "反间谍",
                  "国家安全", "惩戒", "刑罚", "治安管理处罚")),
    ("civil", ("民法典", "民法", "物权", "合同", "婚姻", "继承", "收养", "侵权",
               "担保", "公司", "企业破产", "证券", "证券投资基金", "保险", "票据",
               "海商", "信托", "期货", "专利", "商标", "著作权", "知识产权", "反不正当竞争",
               "反垄断", "消费者权益", "产品质量", "劳动", "劳动合同", "就业", "土地承包",
               "农村土地", "城市房地产", "建筑", "招标投标", "涉外商事", "个人独资", "合伙")),
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def cn2int(s: str):
    """中文数字 → int（支持 十/百/千，如 一二三/二十/三百零五）。失败返回 None。"""
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def norm_space(s: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", s).strip()


def clean_text(s: str) -> str:
    s = RE_ESC_NL.sub("", s)
    s = re.sub(r"[ \t\u3000]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def label_int(label: str):
    """『第五十六条』→ 56；『第一百零九条之一』→ (109, 1)"""
    m = re.match(r"^第(.+?)条(?:之(.+))?$", label)
    if not m:
        return None, None
    num = cn2int(m.group(1))
    sub = None
    if m.group(2):
        sub = cn2int(m.group(2))
    return num, sub


def title_domain(title: str):
    for dom, keys in DOMAIN_RULES:
        for k in keys:
            if k in title:
                return dom, k
    return "general", None


# ---------------------------------------------------------------------------
# 切条核心
# ---------------------------------------------------------------------------
def classify_no_item(title: str, text: str) -> str:
    """切不出条文的文档归类（都要留痕，不静默丢）。

    实测三类为主：
      - 修正案：刑法修正案(九)(十) 等，正文用「一、二、」编号，是**修改指令**而非规范条文；
        并入向量库会造出"假法条"（内容是"将第X条修改为…"），且合并文本本就在正式法里
      - 附件/批准决议/决定：香港基本法附件、人大常委会批准国务院规定等，无条文结构
      - 其余：格式未识别 → 需人工过一眼（报告里带正文头 200 字）
    """
    if "修正案" in title:
        return "AMENDMENT_INSTRUCTION"
    if "附件" in title:
        return "ANNEX_DOC"
    if any(k in title for k in ("批准", "决议", "决定")):
        return "DECISION_OR_RATIFICATION"
    if len(text) < 200:
        return "TOO_SHORT_TO_CONTAIN_ARTICLE"
    return "FORMAT_UNRECOGNIZED"


def strip_header_for_fallback(title: str, text: str):
    """整篇兜底入库前，剥掉正文开头的「标题重复 + 发布信息块」。

    实测这类文书（最高法批复/解释、人大常委会决定）开头是：
        最高人民法院
        关于XX问题的批复
        (2021年5月11日最高人民法院审判委员会第1840次会议通过,自…施行)
        > 法释〔2021〕11号
        上海市高级人民法院：
        你院《…》收悉。经研究,批复如下:
    前 3~6 行是元信息（title 里已有），剥掉能提纯 embedding 内容；
    只剥**开头连续**的元信息行，一旦遇到别的行立即停止 —— 不做任何正文内改写。
    """
    lines = text.split("\n")
    title_compact = re.sub(r"[\s\u3000]+", "", title or "")
    skipped = 0
    i = 0
    while i < len(lines) and i <= 6:
        s = lines[i].strip()
        if not s:
            skipped += len(lines[i])
            i += 1
            continue
        if s.startswith(">") or re.match(r"^[（(].*[)）]$", s) or ("法释" in s and "号" in s):
            skipped += len(lines[i])
            i += 1
            continue
        comp = re.sub(r"[\s\u3000]+", "", s)
        if title_compact and comp and (comp in title_compact or title_compact in comp):
            skipped += len(lines[i])
            i += 1
            continue
        break
    return clean_text("\n".join(lines[i:])), skipped


def doc_uid(rec):
    """来源文档的全局唯一键：`<dataset>__<file_stem>:<source_index>`。

    ★ 为什么要它（2026-09-18 加固）：旧的条目 uid 是 `<source_id>#<条号>`，而
      `source_id` 是**文档在文件内的序号**，同样漏了「源文件」这一维。当前 twang2218
      只有 1 个 parquet → 暂时不撞号，但一旦换成分片 parquet（`train-00000-of-00002`）
      或另一个数据集用了同样的 id 命名，就会静默撞号，下游按 uid 操作会整组连坐。
      本格式与阶段 2 的 `uid`、阶段 4 的 `uid_g` 完全同口径，唯一性由构造成立。
    """
    return "%s__%s:%s" % (
        (rec.get("source_dataset") or "?").replace("/", "__"),
        os.path.splitext(os.path.basename(rec.get("source_file") or "?"))[0],
        rec.get("source_index"))


def whole_doc_item(rec, body, split_mode):
    """无条号结构的文书 → **整篇**作为一个检索单元入库。

    为什么不硬拆：这类文书（批复/答复/批准决定）用「一、二、」或纯叙述组织，
    没有条号锚点，硬拆只能靠猜；整篇通常仅数百字，本身就是完整、自洽的检索单元，
    对 RAG 反而比碎片更有用。为可审计，这类条目 `level="doc"`、
    `article_label=""`、`split_mode` 记录来源，绝不伪装成法条。
    """
    st = rec.get("statute") or {}
    title = norm_space(st.get("title", "") or "")
    dom, kw = title_domain(title)
    text_full = norm_space("%s %s" % (title, body))
    return {
        "uid": "%s#__doc__" % doc_uid(rec),
        "source_dataset": rec.get("source_dataset", ""),
        "source_file": rec.get("source_file", ""),
        "source_index": rec.get("source_index"),
        "source_id": rec.get("source_id", ""),
        "source_license": rec.get("source_license", ""),
        "source_grade": rec.get("source_grade", ""),
        "task": "statute_doc",
        "level": "doc",
        "split_mode": split_mode,
        "law_title": title,
        "law_type": st.get("type", ""),
        "law_type_source": "twang2218.type",
        "status": st.get("status_norm", ""),
        "status_raw": st.get("status_raw", ""),
        "publish_date": st.get("publish_date", ""),
        "effective_from": st.get("effective_from", ""),
        "effective_period": st.get("effective_period", ""),
        "office": st.get("office", ""),
        "office_level": st.get("office_level", ""),
        "office_category": st.get("office_category", ""),
        "part": "", "chapter": "", "section": "",
        "article_label": "", "article_no": None, "article_sub": None, "article_seq": 1,
        "text": body,
        "text_full": text_full,
        "char_len": len(body),
        "content_sha1": sha1(text_full),
        "domain": dom,
        "domains": [dom],
        "domain_source": "statute_title_keyword" if kw else "fallback",
        "domain_evidence": {"rule": "whole_doc_%s" % dom, "keyword": kw} if kw else {"fallback": True},
    }


def split_document(rec, args):
    """一部整部法规 → 逐条 item（生成器）。

    返回：(items, diag) 其中 diag 是本文件的诊断计数
    """
    st = rec.get("statute") or {}
    text = rec.get("output", "") or ""
    title = norm_space(st.get("title", "") or "")
    src_id = rec.get("source_id", "")

    diag = collections.Counter()
    items = []
    ctx = {"part": "", "chapter": "", "section": ""}
    cur = None            # 当前条文：{"label":…, "lines":[…]}
    toc_mode = False
    toc_seen = set()
    seq = 0

    def flush():
        nonlocal cur, seq
        if cur is None:
            return
        body = clean_text("\n".join(cur["lines"]))
        label = cur["label"]
        if len(body) < MIN_BODY_CHARS:
            diag["drop_body_too_short"] += 1
            cur = None
            return
        seq += 1
        num, sub = label_int(label)
        dom, kw = title_domain(title)
        text_full = norm_space("%s %s %s" % (title, label, body))
        items.append({
            "uid": "%s#%s" % (doc_uid(rec), label),
            "source_dataset": rec.get("source_dataset", ""),
            "source_file": rec.get("source_file", ""),
            "source_index": rec.get("source_index"),
            "source_id": src_id,
            "source_license": rec.get("source_license", ""),
            "source_grade": rec.get("source_grade", ""),
            "task": "statute_item",
            "level": "item",
            "law_title": title,
            "law_type": st.get("type", ""),
            "law_type_source": "twang2218.type",
            "status": st.get("status_norm", ""),
            "status_raw": st.get("status_raw", ""),
            "publish_date": st.get("publish_date", ""),
            "effective_from": st.get("effective_from", ""),
            "effective_period": st.get("effective_period", ""),
            "office": st.get("office", ""),
            "office_level": st.get("office_level", ""),
            "office_category": st.get("office_category", ""),
            "part": ctx["part"],
            "chapter": ctx["chapter"],
            "section": ctx["section"],
            "article_label": label,
            "article_no": num,
            "article_sub": sub,
            "article_seq": seq,
            "text": body,
            "text_full": text_full,
            "char_len": len(body),
            "content_sha1": sha1(text_full),
            "domain": dom,
            "domains": [dom],
            "domain_source": "statute_title_keyword" if kw else "fallback",
            "domain_evidence": {"rule": "article_item_%s" % dom,
                                "keyword": kw} if kw else {"fallback": True},
        })
        cur = None

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        if RE_TOC.match(line):
            toc_mode = True
            toc_seen = set()
            diag["toc_blocks"] += 1
            continue
        if toc_mode:
            s = line.strip()
            key = re.sub(r"[\s\u3000]+", "", s)
            if not s:
                diag["toc_lines_skipped"] += 1
                continue
            if RE_TITLE_LIKE.match(line):
                # 目录结束后正文会**重新从第一章开始**（实测：行政处罚法 TOC 列到第八章，
                # 正文再出现「第一章 总则」）。识别"标题在目录里已出现过"即判定正文重启，
                # 否则这一行会被目录模式吞掉，导致全法章节归属丢失（曾踩坑）。
                if len(toc_seen) >= 2 and key in toc_seen:
                    toc_mode = False
                    diag["toc_body_restart"] += 1
                    # 不 continue：本行按正文标题继续处理
                else:
                    toc_seen.add(key)
                    diag["toc_lines_skipped"] += 1
                    continue
            else:
                toc_mode = False
                # 不 continue：本行按正文内容继续处理

        m_art = RE_ARTICLE.match(line)
        if m_art:
            flush()
            cur = {"label": m_art.group(1), "lines": [m_art.group(2)]}
            diag["article_anchors"] += 1
            continue

        m_part = RE_PART.match(line)
        if m_part:
            flush()
            ctx["part"] = norm_space(m_part.group(0))
            ctx["chapter"], ctx["section"] = "", ""
            diag["part_lines"] += 1
            continue

        m_chap = RE_CHAPTER.match(line)
        if m_chap:
            flush()
            ctx["chapter"] = norm_space(m_chap.group(0))
            ctx["section"] = ""
            diag["chapter_lines"] += 1
            continue

        m_sect = RE_SECTION.match(line)
        if m_sect:
            flush()
            ctx["section"] = norm_space(m_sect.group(0))
            diag["section_lines"] += 1
            continue

        # 正文里的「附 则 / 序 言 / 第X编题名」等标题行：是块边界，绝不能粘进上一条条文
        # （实测 9,609 部带附则，若误粘会把「附 则」及其后段落算进最后一条）
        if RE_TITLE_LIKE.match(line) and line.strip():
            flush()
            diag["body_title_lines"] += 1
            continue

        if cur is not None:
            if line.strip():
                cur["lines"].append(line)
            else:
                diag["blank_in_article"] += 1
        else:
            diag["lines_before_first_article"] += 1

    flush()
    if not items:
        diag["doc_no_item"] += 1
    return items, diag


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/lidian/law-agent")
    ap.add_argument("--src-dir", default="",
                    help="默认 <root>/data/corpus/decontaminated/statutes")
    ap.add_argument("--out-dir", default="",
                    help="默认 <root>/data/corpus/statute_items")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help="保留的法规 type，逗号分隔；传 ALL 表示不过滤")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = 不限制（调试用）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--no-toc-skip", action="store_true")
    ap.add_argument("--no-doc-fallback", action="store_true",
                    help="关闭「无条号文书整篇入库」兜底（默认开启）")
    ap.add_argument("--min-doc-chars", type=int, default=60,
                    help="整篇兜底的最小长度（短于此视为无内容，不入库）")
    a = ap.parse_args()

    t0 = time.time()
    root = a.root
    src_dir = a.src_dir or os.path.join(root, "data/corpus/decontaminated/statutes")
    out_dir = a.out_dir or os.path.join(root, "data/corpus/statute_items")
    keep_types = set() if a.types.strip().upper() == "ALL" else {
        t.strip() for t in a.types.split(",") if t.strip()}

    print("=" * 78, flush=True)
    print("[1/4] 扫描源文件（文档级 → 逐条切分）", flush=True)
    print("      保留 type:", sorted(keep_types) or "ALL", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    doc_src = os.path.join(src_dir, "twang2218__chinese-law-and-regulations.jsonl")
    item_src = os.path.join(src_dir, "pandalla__chinese_law_examples.jsonl")

    stats = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": root,
        "src_dir": src_dir,
        "out_dir": out_dir,
        "keep_types": sorted(keep_types) if keep_types else "ALL",
        "min_body_chars": MIN_BODY_CHARS,
        "docs": {},
    }
    type_all = collections.Counter()      # 文档级源：所有 type
    type_kept = collections.Counter()
    type_dropped = collections.Counter()
    status_raw = collections.Counter()
    status_norm = collections.Counter()
    status_dirty = collections.Counter()
    status_dirty_samples = []
    status_dirty_src = collections.Counter()          # 过滤前：全量来源里的脏值
    status_dirty_src_by_type = collections.Counter()  # 脏值 × 类型
    dirty_kept_types = 0                              # 脏值里落在白名单类型上的条数（应为 0）
    no_item_reason = collections.Counter()            # 切不出条文的文档归类
    no_item_samples = []
    doc_fallback = 0                                  # 整篇兜底入库的文档数
    doc_fallback_by_type = collections.Counter()
    doc_fallback_skipped_chars = 0
    doc_fallback_samples = []
    domain_items = collections.Counter()
    domain_forced = collections.Counter()
    law_items = collections.Counter()
    law_skipped = []
    diag_all = collections.Counter()
    char_lens = []
    dup_uids = collections.Counter()
    dup_uid_samples = []
    seen_uids = set()
    version_dups = collections.Counter()
    samples = []
    total_items = 0
    out_file = os.path.join(out_dir, "twang2218__chinese-law-and-regulations.items.jsonl")

    def normalize_status(st_raw, title, dtype):
        """status 规范化：未知/数字脏值 → 未知，原值留痕"""
        s = norm_space(st_raw or "")
        if s in KNOWN_STATUS:
            return s
        status_dirty[s or "<empty>"] += 1
        if len(status_dirty_samples) < 10:
            status_dirty_samples.append(
                {"status_raw": s, "law_title": title, "law_type": dtype})
        return "未知"

    with open(out_file, "w", encoding="utf-8") as fo:
        n_docs = 0
        for rec in iter_jsonl(doc_src):
            n_docs += 1
            if a.max_docs and n_docs > a.max_docs:
                break
            st = dict(rec.get("statute") or {})
            dtype = norm_space(st.get("type", "") or "")
            title = norm_space(st.get("title", "") or "")
            type_all[dtype or "<empty>"] += 1
            status_raw[norm_space(st.get("status", "") or "") or "<empty>"] += 1

            # status 脏值：**在过滤前**统计全量来源，才能说清"脏值是否被白名单顺带清掉"
            s_raw_now = norm_space(st.get("status", "") or "")
            if s_raw_now not in KNOWN_STATUS:
                status_dirty_src[s_raw_now or "<empty>"] += 1
                status_dirty_src_by_type["%s|%s" % (s_raw_now or "<empty>",
                                                    dtype or "<empty>")] += 1
                if keep_types and dtype in keep_types:
                    dirty_kept_types += 1

            if keep_types and dtype not in keep_types:
                type_dropped[dtype or "<empty>"] += 1
                if len(law_skipped) < 12:
                    law_skipped.append({"title": title, "type": dtype,
                                        "reason": "TYPE_NOT_IN_WHITELIST"})
                continue
            type_kept[dtype] += 1

            st["status_raw"] = norm_space(st.get("status", "") or "")
            st["status_norm"] = normalize_status(st.get("status", ""), title, dtype)
            status_norm[st["status_norm"]] += 1
            rec = dict(rec)
            rec["statute"] = st

            items, diag = split_document(rec, a)
            diag_all.update(diag)
            if not items:
                raw_text = rec.get("output", "") or ""
                reason = classify_no_item(title, raw_text)
                # 整篇兜底：无条号结构的文书（批复/答复/批准决定/解释）整篇入库。
                # 但「修正案」例外 —— 内容是「将第X条修改为…」的修改指令，进库会造出假法条。
                if not a.no_doc_fallback and reason != "AMENDMENT_INSTRUCTION":
                    body, skipped = strip_header_for_fallback(title, raw_text)
                    if len(body) >= a.min_doc_chars:
                        it = whole_doc_item(rec, body, "whole_doc_fallback:" + reason)
                        if it["uid"] in seen_uids:
                            dup_uids[it["uid"]] += 1
                        else:
                            seen_uids.add(it["uid"])
                            fo.write(json.dumps(it, ensure_ascii=False) + "\n")
                            total_items += 1
                            doc_fallback += 1
                            doc_fallback_by_type["%s|%s" % (dtype, reason)] += 1
                            doc_fallback_skipped_chars += skipped
                            char_lens.append(it["char_len"])
                            domain_items[it["domain"]] += 1
                            if it["domain_source"] == "fallback":
                                domain_forced[it["domain"]] += 1
                            law_items[it["law_title"]] += 1
                            if len(doc_fallback_samples) < 10:
                                doc_fallback_samples.append(
                                    {"title": title, "type": dtype, "reason": reason,
                                     "chars": it["char_len"], "skipped_header": skipped,
                                     "head": body[:120].replace("\n", " ")})
                        continue
                diag_all["doc_no_item"] += 1
                no_item_reason["%s|%s" % (dtype, reason)] += 1
                if reason == "FORMAT_UNRECOGNIZED" and len(no_item_samples) < 25:
                    no_item_samples.append({
                        "title": title, "type": dtype, "status": st.get("status_norm", ""),
                        "chars": len(raw_text),
                        "head": clean_text(raw_text)[:200],
                    })
                if len(law_skipped) < 12:
                    law_skipped.append({"title": title, "type": dtype, "reason": reason})
                continue

            for it in items:
                if it["uid"] in seen_uids:
                    dup_uids[it["uid"]] += 1
                    if len(dup_uid_samples) < 5:
                        dup_uid_samples.append({"uid": it["uid"],
                                                "law": it["law_title"],
                                                "label": it["article_label"],
                                                "text_head": it["text"][:80]})
                    continue
                seen_uids.add(it["uid"])
                fo.write(json.dumps(it, ensure_ascii=False) + "\n")
                total_items += 1
                char_lens.append(it["char_len"])
                domain_items[it["domain"]] += 1
                if it["domain_source"] == "fallback":
                    domain_forced[it["domain"]] += 1
                law_items[it["law_title"]] += 1
                if len(samples) < 12 and it["article_seq"] <= 2 and n_docs % 150 == 0:
                    samples.append(it)

            if n_docs % 2000 == 0:
                print("      ... 已处理 %d 部，累计切出 %d 条 (%.1fs)"
                      % (n_docs, total_items, time.time() - t0), flush=True)

    stats["docs"]["twang2218__chinese-law-and-regulations"] = {
        "records_scanned": n_docs,
        "type_all": dict(type_all.most_common()),
        "type_kept": dict(type_kept.most_common()),
        "type_dropped": dict(type_dropped.most_common()),
        "items_out": total_items,
        "doc_level_fallback": {
            "docs": doc_fallback,
            "by_type_reason": dict(doc_fallback_by_type.most_common(20)),
            "header_chars_stripped_total": doc_fallback_skipped_chars,
            "samples": doc_fallback_samples,
        },
        "diagnostics": dict(diag_all.most_common()),
    }

    # --- 条级源：直接规范化进同一 schema ---
    print("[2/4] 条级源规范化（pandalla，已是逐条）", flush=True)
    item_out = os.path.join(out_dir, "pandalla__chinese_law_examples.items.jsonl")
    n_items_src = 0
    n_items_pass = 0
    n_drop_no_title = 0
    n_drop_no_title_fatal = 0
    with open(item_out, "w", encoding="utf-8") as fo:
        if os.path.exists(item_src):
            for rec in iter_jsonl(item_src):
                n_items_src += 1
                st = rec.get("statute") or {}
                title = norm_space(st.get("title", "") or "")
                if not title:
                    # 实测有 2 条 title 为空：退一步从 source_id（形如《商标法(2019修正)》-12）反推
                    n_drop_no_title += 1
                    title = norm_space(re.sub(r"-\d+$", "", rec.get("source_id", "") or ""))
                    if not title:
                        n_drop_no_title_fatal += 1
                        continue
                label = norm_space(st.get("article_no", "") or "")
                body = clean_text(rec.get("output", "") or "")
                # 条级源的正文**自带条号**（"第六条 法律、行政法规规定…"），
                # 与文档级源的约定（text 不含条号）对齐 → 剥掉行首条号
                body = RE_ARTICLE.sub(lambda m: m.group(2), body, count=1) \
                    if RE_ARTICLE.match(body) else body
                body = clean_text(body)
                if len(body) < MIN_BODY_CHARS:
                    continue
                dom, kw = title_domain(title)
                text_full = norm_space("%s %s %s" % (title, label, body))
                uid = "%s#%s" % (doc_uid(rec), label)
                if uid in seen_uids:
                    dup_uids[uid] += 1
                    continue
                seen_uids.add(uid)
                num, sub = label_int(label) if label else (None, None)
                it = {
                    "uid": uid,
                    "source_dataset": rec.get("source_dataset", ""),
                    "source_file": rec.get("source_file", ""),
                    "source_index": rec.get("source_index"),
                    "source_id": rec.get("source_id", ""),
                    "source_license": rec.get("source_license", ""),
                    "source_grade": rec.get("source_grade", ""),
                    "task": "statute_item",
                    "level": "item",
                    "law_title": title,
                    "law_type": norm_space(st.get("classification", "") or "") or "未知",
                    "law_type_source": "pandalla.classification（源无 type 字段）",
                    "status": "未知",
                    "status_raw": "",
                    "publish_date": "",
                    "effective_from": "",
                    "effective_period": "",
                    "office": "",
                    "office_level": "",
                    "office_category": "",
                    "part": "", "chapter": "", "section": "",
                    "article_label": label,
                    "article_no": num,
                    "article_sub": sub,
                    "article_seq": n_items_pass + 1,
                    "text": body,
                    "text_full": text_full,
                    "char_len": len(body),
                    "content_sha1": sha1(text_full),
                    "domain": dom,
                    "domains": [dom],
                    "domain_source": "statute_title_keyword" if kw else "fallback",
                    "domain_evidence": {"rule": "article_item_%s" % dom,
                                        "keyword": kw} if kw else {"fallback": True},
                }
                fo.write(json.dumps(it, ensure_ascii=False) + "\n")
                n_items_pass += 1
                total_items += 1
                char_lens.append(len(body))
                domain_items[dom] += 1
                if not kw:
                    domain_forced[dom] += 1
                law_items[title] += 1

    stats["docs"]["pandalla__chinese_law_examples"] = {
        "records_scanned": n_items_src, "items_out": n_items_pass,
        "title_missing_fallback_from_source_id": n_drop_no_title,
        "title_unrecoverable_dropped": n_drop_no_title_fatal,
        "note": "正文自带条号已剥离，与文档级源的 text（不含条号）对齐"}

    # --- 统计汇总 ---
    print("[3/4] 汇总裁剪/质量统计", flush=True)
    cl = sorted(char_lens)

    def q(x):
        return cl[min(len(cl) - 1, int(len(cl) * x))] if cl else None

    stats["items"] = {
        "total": total_items,
        "char_stats": {"min": cl[0] if cl else None, "p10": q(0.10), "p50": q(0.50),
                       "p90": q(0.90), "max": cl[-1] if cl else None},
        "domain_items": dict(domain_items.most_common()),
        "domain_fallback": dict(domain_forced.most_common()),
        "domain_fallback_share": round(
            sum(domain_forced.values()) / total_items, 4) if total_items else None,
        "status_raw_dist": dict(status_raw.most_common()),
        "status_after_norm": dict(status_norm.most_common()),
        "status_dirty_dist": dict(status_dirty.most_common()),
        "status_dirty_samples": status_dirty_samples,
        "status_dirty_in_source": {
            "total": sum(status_dirty_src.values()),
            "dist": dict(status_dirty_src.most_common()),
            "cross_type": dict(status_dirty_src_by_type.most_common(15)),
            "inside_whitelist": dirty_kept_types,
        },
        "no_item_classification": dict(no_item_reason.most_common()),
        "no_item_unrecognized_samples": no_item_samples,
        "top_laws_by_items": dict(law_items.most_common(25)),
        "unique_laws": len(law_items),
        "duplicate_uids": sum(dup_uids.values()),
        "duplicate_uid_samples": dup_uid_samples,
        "duplicate_uid_note": ("uid = <dataset>__<file_stem>:<source_index>#<条号>，重复说明该文档内"
                               "同一条号出现两次（修正案体例/附则重排），保留首次出现；"
                               "本计数是**被跳过的重复条号数**，产物本身 uid 唯一（质检 C1 断言）"),
        "version_duplicates_note": ("同条号多版本内容重复由 uid 区分，未删；只统计 uid 完全重复"),
        "skipped_law_samples": law_skipped,
    }
    stats["outputs"] = {
        "items_jsonl": [out_file, item_out],
        "note": "统一条级 schema；text=条文正文（不含条号），text_full=『法规名+条号+正文』（用于 embedding）",
    }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # --- Markdown 报告 ---
    print("[4/4] 写报告", flush=True)
    d = stats["docs"]["twang2218__chinese-law-and-regulations"]
    lines = []
    W = lines.append
    W("# 阶段 2b：法条切条报告（STATUTE ITEM SPLIT）\n")
    W("> 生成时间：%s　耗时 %.1fs" % (stats["generated_at"], time.time() - t0))
    W("> 脚本：`scripts/corpus/split_statutes.py`　核查方式：可重跑（幂等，只读输入）\n")
    W("## 1. 过滤（只留国家级规范）\n")
    W("| type | 源文档数 | 处置 |")
    W("|---|---|---|")
    for t, c in type_all.most_common():
        if t in keep_types:
            W("| %s | %d | ✅ 保留并切条 |" % (t, c))
        else:
            W("| %s | %d | ❌ 丢弃（非国家级规范/修改指令） |" % (t, c))
    W("")
    W("保留 %d 部 → 切出 **%d 条** 法条（另有条级源 %d 条）。"
      % (sum(type_kept.values()), d["items_out"], n_items_pass))
    W("")
    W("## 2. status 脏值清洗\n")
    W("**过滤前的全量来源**（21,787 部都算，不只白名单）：")
    W("")
    W("| 原始值 | 条数 | 处置 |")
    W("|---|---|---|")
    for s, c in status_raw.most_common():
        W("| `%s` | %d | %s |" % (s, c, s if s in KNOWN_STATUS else "脏值 → 规范为「未知」"))
    W("")
    sd = stats["items"]["status_dirty_in_source"]
    W("脏值合计 **%d** 部，按 值×类型 交叉（前 15）：" % sd["total"])
    W("")
    W("| status | 类型 | 条数 |")
    W("|---|---|---|")
    for k, c in sd["cross_type"].items():
        v, t = k.split("|", 1)
        W("| `%s` | %s | %d |" % (v, t, c))
    W("")
    W("→ **脏值落在白名单类型上的条数 = %d**。%s"
      % (sd["inside_whitelist"],
         "白名单内 status 100% 合法，向量库无需额外清洗。" if sd["inside_whitelist"] == 0
         else "⚠️ 白名单内仍有脏值，已规范为「未知」并留 status_raw 原值。"))
    W("")
    if stats["items"]["status_dirty_samples"]:
        W("脏值样例（前 10）：")
        for s in stats["items"]["status_dirty_samples"]:
            W("- `status=%s`　%s（type=%s）"
              % (s["status_raw"], s["law_title"][:40], s["law_type"]))
        W("")
    W("## 2b. 切不出条文的文档归类（不静默丢）\n")
    fb = d.get("doc_level_fallback") or {}
    W("其中 **%d 部**走「整篇兜底入库」（`level=\"doc\"`，`split_mode=whole_doc_fallback:*`）——"
      % fb.get("docs", 0))
    W("这类文书（最高法批复/答复、人大常委会决定、法规解释）正文用「一、二、」或纯叙述组织，"
      "没有条号锚点，硬拆只能靠猜；整篇仅数百字，本身就是自洽的检索单元。")
    W("入库时剥掉开头标题重复与发布信息块，累计剥掉 %d 字（不删任何正文）。"
      % fb.get("header_chars_stripped_total", 0))
    W("")
    W("| 类型\\归类 | 部数 | 处置 |")
    W("|---|---|---|")
    for k, c in (fb.get("by_type_reason") or {}).items():
        t, r = k.split("|", 1)
        W("| %s | %d | ✅ 整篇入库 |" % ("%s / %s" % (t, r), c))
    for k, c in stats["items"]["no_item_classification"].items():
        t, r = k.split("|", 1)
        W("| %s | %d | ❌ 丢弃（%s） |" % ("%s / %s" % (t, r), c, r))
    W("")
    W("**只有 `AMENDMENT_INSTRUCTION`（修正案）一律不进库**：正文是「将第X条修改为…」的"
      "**修改指令**而非规范条文，进向量库会造出「假法条」；合并后的正式法本已在库里。")
    W("")
    if fb.get("samples"):
        W("整篇入库样例：")
        for s in fb["samples"][:6]:
            W("- 《%s》（%s，%s，%d 字，剥头 %d 字）：%s"
              % (s["title"][:44], s["type"], s["reason"], s["chars"],
                 s["skipped_header"], s["head"][:100]))
        W("")
    if stats["items"]["no_item_unrecognized_samples"]:
        W("仍被丢弃且格式未识别的（需人工过一眼）：")
        for s in stats["items"]["no_item_unrecognized_samples"][:8]:
            W("- 《%s》（%s，%d 字）：%s" % (s["title"][:50], s["type"], s["chars"],
                                           s["head"][:110].replace("\n", " ")))
        W("")
    W("## 3. 域分布（法规名关键词规则，与阶段 2 QA 侧打分无关）\n")
    W("| 域 | 条数 | 占比 |")
    W("|---|---|---|")
    for dm, c in domain_items.most_common():
        W("| %s | %d | %.1f%% |" % (dm, c, 100.0 * c / max(1, total_items)))
    W("")
    W("`general` = 法规名未命中关键词（%d 条，%.1f%%）。**这不是「打标失败」**："
      % (domain_forced.get("general", 0),
         100.0 * domain_forced.get("general", 0) / max(1, total_items)))
    W("国家级法律里行政法/经济法/社会法（立法法、海关法、食品安全法…）本就占大头，"
      "按本项目「民法/刑法/程序法 + 通用」的三法域约定归 general 是正确的。")
    W("civil/criminal/procedural 合计 %d 条（%.1f%%）—— 这三类才是需要保证判准的。"
      % (sum(c for dm, c in domain_items.items() if dm != "general"),
         100.0 * sum(c for dm, c in domain_items.items() if dm != "general") / max(1, total_items)))
    W("")
    W("## 4. 切条诊断\n")
    W("| 指标 | 值 |")
    W("|---|---|")
    for k, v in d["diagnostics"].items():
        W("| %s | %d |" % (k, v))
    W("")
    W("## 5. 条文长度分布\n")
    W("| min | p10 | p50 | p90 | max |")
    W("|---|---|---|---|---|")
    cs = stats["items"]["char_stats"]
    W("| %s | %s | %s | %s | %s |" % (cs["min"], cs["p10"], cs["p50"], cs["p90"], cs["max"]))
    W("")
    W("## 6. 切条样例（每部法前 6 条里抽样 12 条）\n")
    for s in samples:
        W("- **%s %s**（%s / %s）" % (s["law_title"], s["article_label"],
                                     s["chapter"] or "—", s["status"]))
        W("  - %s" % s["text"][:220].replace("\n", " "))
    W("")
    W("## 7. 产出\n")
    W("- `data/corpus/statute_items/twang2218__chinese-law-and-regulations.items.jsonl`")
    W("- `data/corpus/statute_items/pandalla__chinese_law_examples.items.jsonl`")
    W("")
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 78, flush=True)
    print("DONE_SPLIT_STATUTES  items=%d  laws=%d  %.1fs"
          % (total_items, len(law_items), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
