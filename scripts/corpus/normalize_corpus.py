#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：语料归一化 + 域打标。

输入：data/corpus/raw/**            （阶段 1 落盘，**只读**，本脚本不修改任何 raw 文件）
配置：configs/corpus_adapters.yaml  （声明哪些文件进管线、用什么适配器、阈值与权重）
输出：data/corpus/normalized/
        qa/<slug>.jsonl         训练语料（统一 schema，含域标签）
        qa/_all.jsonl           合并流（可选 --no-merge 关闭）
        statutes/<slug>.jsonl   法条库（**保留历史版本与生效/失效区间**）
        STATS.json              逐来源 / 逐域 / 逐任务统计
        NORMALIZE_REPORT.json   验收报告（含与 target_mix 的差距）
        DEDUP_REPORT.json       重复与丢弃明细

用法：
    python scripts/corpus/normalize_corpus.py --list
    python scripts/corpus/normalize_corpus.py --limit-per-file 300 --suffix .sample
    python scripts/corpus/normalize_corpus.py

======================= 硬约定（改动必须同步 docs 与本文件顶部） =======================
1. 输入 raw 只读；同一份 raw 重跑，产出逐字节一致（无随机数）。
2. **`uid` 是全局唯一键，格式 `<dataset>__<file_stem>:<source_index>`**，与阶段 4 的
   `uid_g` 同口径。**绝不允许漏掉源文件**（旧实现漏了 → 20,947 条撞号，阶段 3 按 uid
   剔除时连带多删 2,271 条真实样本；详见 `uid_of()` 的说明）。生成后当场断言唯一，
   不唯一则 exit 3。
3. **`content_sha1` 是样本唯一键**（`sha1(instruction, input, output)`，全局唯一）。
4. 合并流 `_all.jsonl` **只合并本轮真正产出的文件**（以 `stats["sources"]` 登记为准）；
   目录里出现的其它 `*.jsonl`（如调试残留 `*.sample.jsonl`）一律**排除并告警**。
   历史缺陷：旧实现按「目录下所有非 `_` 开头 .jsonl」合并，把 3,598 条抽样残留吃进
   合并流（288,855 = 285,257 + 3,598）。
5. 调试抽样请用 `--suffix .sample`，**跑完把 `*.sample.jsonl` 移出 `data/corpus/`**，
   否则会污染下一次合并（现已由第 4 条兜底，但仍应清理）。
"""
import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import unicodedata

try:
    import yaml
except ImportError:
    sys.stderr.write("需要 pyyaml：envs/main/bin/python -m pip install pyyaml\n")
    raise

ROOT = "/mnt/data/lidian/law-agent"
RAW = os.path.join(ROOT, "data/corpus/raw")
OUT = os.path.join(ROOT, "data/corpus/normalized")

# =============================================================================
# 域定义
# =============================================================================
D_CRIMINAL = "criminal"
D_CIVIL = "civil"
D_PROCEDURAL = "procedural"
D_GENERAL = "general"          # 通用/其他（宪法、行政、无明确域、法条背诵）
DOMAINS = (D_CRIMINAL, D_CIVIL, D_PROCEDURAL, D_GENERAL)

# 程序法（**必须先于刑事/民事匹配**，因为「刑事诉讼法」含「刑事」、「民事诉讼法」不含「民法」）
PROCEDURAL_LAWS = (
    "民事诉讼法", "刑事诉讼法", "行政诉讼法", "海事诉讼特别程序法",
    "仲裁法", "人民调解法", "劳动争议调解仲裁法",
    "民事诉讼证据", "刑事诉讼规则", "办理刑事案件程序规定",
    "适用《中华人民共和国民事诉讼法》", "适用《中华人民共和国刑事诉讼法》",
    "适用《中华人民共和国行政诉讼法》", "民事诉讼法解释", "刑事诉讼法解释",
    "执行程序", "诉讼程序", "审判程序", "强制执行",
)

CRIMINAL_LAWS = (
    "刑法", "刑事", "反间谍法", "反恐怖主义法", "禁毒法", "监狱法",
    "国家安全法", "保守国家秘密法", "枪支管理法", "预防未成年人犯罪法",
    "关于办理", "刑事案件",
)

CIVIL_LAWS = (
    "民法", "民法典", "民法总则", "民法通则",
    "合同法", "物权法", "担保法", "侵权责任法", "婚姻法", "继承法", "收养法",
    "公司法", "合伙企业法", "个人独资企业法", "企业破产法", "破产法",
    "证券法", "证券投资基金法", "信托法", "保险法", "票据法", "海商法",
    "商标法", "专利法", "著作权法", "反不正当竞争法", "反垄断法",
    "消费者权益保护法", "产品质量法", "食品安全法",
    "劳动法", "劳动合同法", "社会保险法",
    "土地管理法", "城市房地产管理法", "农村土地承包法",
    "招标投标法", "建筑法", "旅游法", "电子商务法", "电子签名法",
    "建设工程", "买卖合同", "租赁合同", "借款合同", "民间借贷",
    "侵权责任", "婚姻家庭", "继承纠纷", "物业服务",
)

# 明确归入「通用/其他」的（行政法、宪法、程序无关的公共法）
GENERAL_LAWS = (
    "宪法", "立法法", "行政处罚法", "行政许可法", "行政强制法",
    "行政复议法", "治安管理处罚法", "道路交通安全法", "公务员法",
    "监察法", "政府信息公开", "地方性法规", "行政法规",
)

# 程序议题关键词（只用于问句，**不看判决书正文**）
PROC_TOPICS = (
    "管辖", "送达", "举证", "举证责任", "证明标准", "时效", "上诉", "抗诉",
    "再审", "申诉", "执行", "起诉", "受理", "立案", "回避", "证据",
    "简易程序", "普通程序", "独任", "合议", "审判委员会", "诉讼费用",
    "财产保全", "先予执行", "强制措施", "侦查", "审查起诉", "不起诉",
    "发回重审", "开庭", "审理期限", "延期审理", "中止审理", "终结诉讼",
    "缺席判决", "撤诉", "诉讼时效", "级别管辖", "地域管辖", "移送管辖",
    "指定管辖", "管辖异议", "非法证据排除", "取保候审", "逮捕", "拘留",
    "附带民事诉讼", "司法鉴定", "勘验", "证人出庭",
)

# 文书类型 → 域
DOC_TYPE_RULES = (
    ("刑事判决书", D_CRIMINAL), ("刑事裁定书", D_CRIMINAL),
    ("刑事附带民事", D_CRIMINAL), ("刑事判决", D_CRIMINAL),
    ("民事判决书", D_CIVIL), ("民事裁定书", D_CIVIL),
    ("民事调解书", D_CIVIL), ("民事判决", D_CIVIL),
    ("商事判决", D_CIVIL),
    ("行政判决书", D_PROCEDURAL), ("行政裁定书", D_PROCEDURAL),
    ("行政判决", D_PROCEDURAL),
)

# 任务型 → 域 强指示（默认值，可被 configs/corpus_adapters.yaml 覆盖）
TASK_DOMAIN_HINT = {
    "sent_pred": D_CRIMINAL,
    "crime_concept": D_CRIMINAL,
}

# 任务型 → 大类（决定程序法引用是否降权）
TASK_KIND = {
    # 判决/案例处理类：问的是实体结论，程序法引用属附带
    "jud_doc_sum": "case_analysis",
    "jud_read_compre": "case_analysis",
    "leg_ele_extra": "case_analysis",
    "leg_eve_detec": "case_analysis",
    "leg_case_cls": "case_analysis",
    "sim_case_match": "case_analysis",
    "op_sum": "case_analysis",
    "judgement_predit": "case_analysis",
    "sent_pred": "case_analysis",
    # 问答/考试类：程序议题本身就是考点
    "legal_question_answering": "qa",
    "exam": "exam",
    "legal_advice": "qa",
    "legal_counsel": "qa",
    "legal_counsel_multi_turn": "qa",
    "judical_examination": "exam",
    "judical_examination_v2": "exam",
    "crime_concept": "concept",
}

# 议题关键词（**只在问句里匹配**，用于救回「不引用法条」的考题/问答）
CRIMINAL_TOPICS = (
    "犯罪", "罪", "被告人", "定罪", "量刑", "有期徒刑", "拘役", "罚金",
    "主犯", "从犯", "正当防卫", "紧急避险", "犯罪构成", "自首", "立功",
    "累犯", "数罪并罚", "缓刑", "假释", "追诉", "刑事责任", "共犯",
    "未遂", "既遂", "犯罪中止", "刑事处罚", "罪行", "嫌疑",
)

CIVIL_TOPICS = (
    "合同", "侵权", "赔偿", "夫妻", "离婚", "继承", "遗嘱", "租赁",
    "借贷", "借款", "买卖", "债务", "债权", "物权", "所有权", "抵押",
    "质押", "股东", "股权", "破产", "保险", "票据", "商标", "专利",
    "著作权", "工伤", "房地产", "物业", "婚姻", "抚养", "财产分割",
    "违约责任", "法人", "代理权", "个人财产", "共同财产", "损害赔偿",
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LAWNAME_RE = re.compile(r"《([^》\n]{2,40})》")
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇0-9]+条(?:之[一二三四五六七八九十])?")


# =============================================================================
# 基础工具
# =============================================================================
def norm_ws(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[ \t\r\f\v]+", " ", s.replace("\u3000", " ")).strip()


def sha1(*parts, n=None):
    h = hashlib.sha1()
    for p in parts:
        h.update(norm_ws(p).encode("utf-8"))
        h.update(b"\x1e")
    d = h.hexdigest()
    return d[:n] if n else d


def uid_of(dataset, source_file, source_index):
    """全局唯一键：`<dataset>__<file_stem>:<source_index>`（与阶段 4 的 `uid_g` 同口径）。

    ★ 为什么必须带 source_file（2026-09-18 修的真实缺陷）：
      旧实现是 `<dataset>:<source_id>`，**漏了源文件**。DISC-Law-SFT 的
      `DISC-Law-SFT-Pair-QA-released.jsonl` 与 `-Triplet-QA-released.jsonl` 共用同一套
      `source_id`(行号) → 20,947 条记录撞号到同一个 uid（内容却完全不同）。
      撞号的杀伤力不在统计虚高，而在**按 uid 操作会整组连坐**：
      阶段 3 去污按 uid 剔除时，本该删 14,218 组、实际删了 16,489 条 —— 连带多删
      2,271 条真实样本（方向是多删，保守，无污染风险，但数字被改动）。
      `source_index` 在单份文件内严格唯一 → 本格式的全局唯一性由构造成立，且稳定可复现。
    """
    return "%s__%s:%s" % ((dataset or "?").replace("/", "__"),
                          os.path.splitext(os.path.basename(source_file or "?"))[0],
                          source_index)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_records(path):
    """json-array / jsonl / parquet 自适应。"""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    head = read_text(path)[:4096]
    if head.lstrip().startswith("["):
        return json.loads(read_text(path))
    out = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def law_domain(name):
    """法条名 → 域。顺序敏感：程序法优先。"""
    n = norm_ws(name)
    for kw in PROCEDURAL_LAWS:
        if kw in n:
            return D_PROCEDURAL
    for kw in CRIMINAL_LAWS:
        if kw in n:
            return D_CRIMINAL
    for kw in CIVIL_LAWS:
        if kw in n:
            return D_CIVIL
    for kw in GENERAL_LAWS:
        if kw in n:
            return D_GENERAL
    return None


def extract_laws(*texts):
    """抽取《...》法条名（去重，保序）。"""
    seen, out = set(), []
    for t in texts:
        if not t:
            continue
        for m in LAWNAME_RE.finditer(str(t)):
            name = norm_ws(m.group(1))
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def doc_type_of(text):
    if not text:
        return None
    for pat, dom in DOC_TYPE_RULES:
        if pat in text:
            return (pat, dom)
    return None


# =============================================================================
# 域打标
# =============================================================================
def label_domain(rec, cfg):
    """加权打分定域。返回 (primary, multilabel_list, source, evidence)

    ★ `domain_source` 必须是「**对胜出域**贡献最大的那类信号」，不是「所有域加总最大的信号」。
      之前写成全局加总，导致 topic_keyword 因为同时对三个域加分、总和虚高而虚占 47.5% 的
      domain_source，把 statute_ref / doc_type 这些更可靠的信号盖掉了，审计链就断了。
    """
    w = cfg["domain_labeling"]["weights"]
    ml_ratio = cfg["domain_labeling"]["multilabel_ratio"]
    proc_disc = cfg["domain_labeling"]["case_analysis_proc_discount"]

    scores = collections.defaultdict(float)
    ev = collections.defaultdict(list)
    contrib = collections.defaultdict(float)              # 全局（仅用于统计展示）
    contrib_dom = collections.defaultdict(lambda: collections.defaultdict(float))  # [域][信号]

    kind = TASK_KIND.get(rec["task"], "qa")

    def add(dom, amount, signal, note=None):
        scores[dom] += amount
        contrib[signal] += amount
        contrib_dom[dom][signal] += amount
        if note:
            ev[signal].append(note)

    # --- 1. 引用法条 ---
    for law in rec["refs_question"]:
        d = law_domain(law)
        if d:
            add(d, w["statute_ref_in_question"], "statute_ref_in_question", "%s→%s" % (law, d))
    for law in rec["refs_answer"]:
        d = law_domain(law)
        if d:
            add(d, w["statute_ref_in_answer"], "statute_ref_in_answer", "%s→%s" % (law, d))

    # --- 2. 文书类型 ---
    dt = doc_type_of(rec["input"])
    if dt:
        add(dt[1], w["doc_type"], "doc_type", "%s→%s" % dt)

    # --- 3. 任务型强指示 ---
    hint = rec.get("task_domain_hint")
    if hint:
        add(hint, w["task_hint"], "task_hint", hint)

    # --- 4. 罪名释义 ---
    if rec["task"] == "crime_concept":
        add(D_CRIMINAL, w["crime_concept"], "crime_concept", "罪名释义")

    # --- 5. 议题关键词（只在问句里找，各域权重见 configs/corpus_adapters.yaml）---
    # 大量司考/问答样本不引用具体法条（「下列哪个选项不属于法官应当遵守的司法礼仪」），
    # 只靠引用法条会把它们全丢到 general。关键词权重刻意低于引用法条，只做兜底与平局裁决。
    kw_cfg = (
        (D_PROCEDURAL, PROC_TOPICS, w.get("topic_procedural_each", 0.8),
         w.get("topic_procedural_cap", 3.0)),
        (D_CRIMINAL, CRIMINAL_TOPICS, w.get("topic_criminal_each", 0.5),
         w.get("topic_criminal_cap", 2.0)),
        (D_CIVIL, CIVIL_TOPICS, w.get("topic_civil_each", 0.5),
         w.get("topic_civil_cap", 2.0)),
    )
    for dom, kws, each, cap in kw_cfg:
        hits = [k for k in kws if k in rec["ask"]]
        if hits:
            add(dom, min(cap, each * len(hits)), "topic_keyword",
                "%s:%s" % (dom, "+".join(hits[:8])))

    # --- 6. 判决类任务：程序法引用降权（判决问的是实体结论）---
    # 降权同时要按同比例削弱它在 contrib_dom 里的权重，否则证据链与实际得分不一致
    if kind == "case_analysis" and scores.get(D_PROCEDURAL):
        before = scores[D_PROCEDURAL]
        scores[D_PROCEDURAL] *= proc_disc
        k = proc_disc
        for sig in list(contrib_dom[D_PROCEDURAL]):
            contrib_dom[D_PROCEDURAL][sig] *= k
        ev["proc_discount"].append("%.2f→%.2f" % (before, scores[D_PROCEDURAL]))

    if not scores or max(scores.values()) <= 0:
        return D_GENERAL, [D_GENERAL], "fallback", {"fallback": True}

    mx = max(scores.values())
    primary = max(scores.items(), key=lambda kv: (kv[1], -DOMAINS.index(kv[0])))[0]
    labels = [d for d in DOMAINS if scores.get(d, 0) >= mx * ml_ratio and scores.get(d, 0) > 0]
    if primary not in labels:
        labels = [primary] + labels

    # ★ domain_source = 对**胜出域**贡献最大的信号
    per_dom = contrib_dom.get(primary) or {}
    src = max(per_dom.items(), key=lambda kv: kv[1])[0] if per_dom else "fallback"
    return primary, labels, src, {
        "scores": {k: round(v, 3) for k, v in scores.items() if v},
        "evidence": {k: v[:6] for k, v in ev.items()},
        "contrib": {k: round(v, 3) for k, v in contrib.items()},
        "contrib_primary": {k: round(v, 3) for k, v in sorted(
            per_dom.items(), key=lambda kv: -kv[1])},
    }


# =============================================================================
# 适配器：把各来源的原始记录 → 统一中间结构
# =============================================================================
def _base(ds_id, fname, idx, sid, task, instruction, inp, output, system="", refs=None):
    return {
        "source_dataset": ds_id,
        "source_file": fname,
        "source_index": idx,
        "source_id": str(sid) if sid is not None else "",
        "task": task,
        "system": norm_ws(system),
        "instruction": norm_ws(instruction),
        "input": norm_ws(inp),
        "output": norm_ws(output),
        "references": refs or [],
    }


def a_disc_law_sft(ds_id, fname, recs):
    """ShengbinYue/DISC-Law-SFT：jsonl，字段 id/input/output（Triplet 另有 reference）"""
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        sid = r.get("id", "")
        task = str(sid).rsplit("-", 1)[0] if "-" in str(sid) else "legal_question_answering"
        refs = []
        ref = r.get("reference")
        if isinstance(ref, str) and ref.strip().startswith("["):
            try:
                refs = [norm_ws(x) for x in json.loads(ref)]
            except Exception:
                refs = []
        elif isinstance(ref, list):
            refs = [norm_ws(x) for x in ref]
        yield _base(ds_id, fname, i, sid, task, "", r.get("input", ""), r.get("output", ""),
                    refs=refs)


def a_disc_law_sft_dusker(ds_id, fname, recs):
    """Dusker/lawyer-llama：json-array，字段 system/instruction/input/output"""
    task_map = {
        "DISC-Law-SFT-Triplet.json": "judgement_predit",
        "fakao_gpt4.json": "exam",
        "zixun_gpt4.json": "legal_advice",
        "kg_crime_llama.json": "crime_concept",
    }
    task = task_map.get(os.path.basename(fname), "qa")
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        yield _base(ds_id, fname, i, "%s-%d" % (task, i), task,
                    r.get("instruction", ""), r.get("input", ""), r.get("output", ""),
                    system=r.get("system", ""))


def a_skepsun(ds_id, fname, recs):
    """Skepsun/lawyer_llama_data：json-array，字段 input/instruction/output/source/prefix"""
    smap = {
        "judical_examination.json": "judical_examination",
        "judical_examination_v2.json": "judical_examination_v2",
        "legal_advice.json": "legal_advice",
        "legal_counsel_with_article_v2.json": "legal_counsel",
        "legal_counsel_multi_turn_with_article_v2.json": "legal_counsel_multi_turn",
    }
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        src = norm_ws(r.get("source", ""))
        task = smap.get(src, "qa")
        yield _base(ds_id, fname, i, "%s-%d" % (task, i), task,
                    r.get("instruction", ""), r.get("input", ""), r.get("output", ""),
                    system=r.get("prefix", ""))


def a_twang2218(ds_id, fname, recs):
    """twang2218：法规**文档级**，保留历史版本与生效区间（硬约定第 5 条）"""
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        yield _base(ds_id, fname, i, "%s-%d" % (r.get("type", "law"), i), "statute_doc",
                    r.get("title", ""), "", r.get("content", ""),
                    refs=[],
                    system="", ) | {
            "statute": {
                "title": norm_ws(r.get("title", "")),
                "type": norm_ws(r.get("type", "")),
                "status": norm_ws(r.get("status", "")),
                "office": norm_ws(r.get("office", "")),
                "office_level": norm_ws(r.get("office_level", "")),
                "office_category": norm_ws(r.get("office_category", "")),
                "publish_date": str(r.get("publish_date") or ""),
                "effective_from": str(r.get("effective_date") or ""),
                "effective_period": norm_ws(r.get("effective_period", "")),
            }
        }


def a_pandalla(ds_id, fname, recs):
    """pandalla：条级法条，字段 title/classification/num/contents"""
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        yield _base(ds_id, fname, i, "%s-%d" % (norm_ws(r.get("title", ""))[:20], i),
                    "statute_item", "", "", norm_ws(r.get("contents", "")), ) | {
            "statute": {
                "title": norm_ws(r.get("title", "")),
                "classification": norm_ws(r.get("classification", "")),
                "article_no": norm_ws(r.get("num", "")),
                "level": "item",
            }
        }


ADAPTERS = {
    "disc_law_sft": a_disc_law_sft,
    "disc_law_sft_dusker": a_disc_law_sft_dusker,
    "skepsun": a_skepsun,
    "twang2218": a_twang2218,
    "pandalla": a_pandalla,
}


# =============================================================================
# 归一化主流程
# =============================================================================
def build_record(raw, cfg, ds_meta):
    """中间结构 → 最终统一 schema"""
    task = raw["task"]
    instruction = raw["instruction"]
    inp = raw["input"]
    output = raw["output"]

    # ask = 问句主干（用于程序议题判定）；DISC 系没有 instruction，指令嵌在 input 头部
    if instruction:
        ask = instruction
    else:
        ask = inp[:300]

    refs_q = extract_laws(instruction, ask)
    refs_a = extract_laws(raw.get("system", ""), output, *raw.get("references", []))
    # 去重：问句里已出现的，答案侧不重复计入
    qs = set(refs_q)
    refs_a = [x for x in refs_a if x not in qs]

    rec = {
        "source_dataset": raw["source_dataset"],
        "source_file": raw["source_file"],
        "source_index": raw["source_index"],
        "source_id": raw["source_id"],
        "source_license": ds_meta.get("license", ""),
        "source_grade": ds_meta.get("license_grade", ""),
        "task": task,
        "task_kind": TASK_KIND.get(task, "qa"),
        "system": raw.get("system", ""),
        "instruction": instruction,
        "input": inp,
        "output": output,
        "references": raw.get("references", []),
        "refs_question": refs_q,
        "refs_answer": refs_a,
        "ask": ask,
        "task_domain_hint": cfg["task_domain_hint"].get(task),
        "lang": "zh",
        "synthetic": False,
        "replay": False,
        "char_len": {"instruction": len(instruction), "input": len(inp), "output": len(output)},
        "normalized_at": cfg["_now"],
    }

    d, labels, src, ev = label_domain(rec, cfg)
    rec["domain"] = d
    rec["domains"] = labels
    rec["domain_source"] = src
    rec["domain_evidence"] = ev

    # ★ uid = 全局唯一键，**必须含源文件**（见 uid_of 的说明：旧实现漏 source_file
    #   导致 20,947 条撞号，阶段 3 按 uid 剔除时连带多删 2,271 条真实样本）。
    #   `uid_legacy` 保留旧口径，仅作审计/血缘回溯用，下游一律不再使用。
    rec["uid"] = uid_of(raw["source_dataset"], raw["source_file"],
                        raw["source_index"])
    rec["uid_legacy"] = "%s:%s" % (raw["source_dataset"].replace("/", "__"),
                                   raw["source_id"])
    rec["content_sha1"] = sha1(instruction, inp, output)
    rec["case_sha1"] = sha1(inp[-400:]) if inp else ""
    if raw.get("statute"):
        rec["statute"] = raw["statute"]

    # 渲染成规范 chat 格式，消除下游拼接歧义
    msgs = []
    if rec["system"]:
        msgs.append({"role": "system", "content": rec["system"]})
    user = (instruction + "\n" + inp).strip() if instruction else inp
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": output})
    rec["messages"] = msgs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=os.path.join(ROOT, "configs/corpus_adapters.yaml"))
    ap.add_argument("--sources", default=os.path.join(ROOT, "configs/corpus_sources.yaml"))
    ap.add_argument("--only", nargs="*", default=[], help="只处理这些 dataset id")
    ap.add_argument("--limit-per-file", type=int, default=0, help="每个文件最多取 N 条（抽样验证用）")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，如 .sample")
    ap.add_argument("--no-merge", action="store_true", help="不生成合并流 _all.jsonl")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.adapters, encoding="utf-8"))
    src_cfg = yaml.safe_load(open(a.sources, encoding="utf-8"))
    meta_by_id = {s["id"]: s for s in (src_cfg.get("sources") or [])}
    cfg["_now"] = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(os.path.join(OUT, "qa"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "statutes"), exist_ok=True)

    dropped = {(d["dataset"], d["file"]): d for d in (cfg.get("drop_files") or [])}
    filters = cfg.get("filters") or {}

    stats = {
        "generated_at": cfg["_now"],
        "drop_files": cfg.get("drop_files") or [],
        "keep_files_note": cfg.get("keep_files") or [],
        "sources": {},
        "totals": collections.Counter(),
        "domain_by_source": {},
        "domain_source_dist": collections.Counter(),
        "task_dist": collections.Counter(),
        "multilabel_count": 0,
    }
    dedup = {"global_content_dup": 0, "internal": {}, "examples": []}
    seen_content = {}
    seen_uid = {}
    uid_dup = {"count": 0, "samples": []}
    drop_by_task = collections.defaultdict(collections.Counter)

    for s in cfg.get("sources") or []:
        ds_id = s["id"]
        if a.only and ds_id not in a.only:
            continue
        adapter = ADAPTERS.get(s["adapter"])
        if adapter is None:
            print("  !! 未知 adapter: %s" % s["adapter"])
            continue
        slug = ds_id.replace("/", "__")
        role = s.get("role", "qa")
        meta = meta_by_id.get(ds_id, {})

        print("\n=== %s [role=%s, adapter=%s] ===" % (ds_id, role, s["adapter"]))
        if a.list:
            for f in s["include"]:
                mark = "  [DROP]" if (ds_id, f) in dropped else ""
                print("    %s%s" % (f, mark))
            continue

        out_lines = []
        sub = collections.Counter()
        ds_domain = collections.Counter()

        for fname in s["include"]:
            fpath = os.path.join(RAW, slug, fname)
            if not os.path.exists(fpath):
                print("    !! 缺失: %s" % fpath)
                sub["missing_file"] += 1
                continue
            if (ds_id, fname) in dropped:
                d = dropped[(ds_id, fname)]
                print("    [DROP] %s ← %s" % (fname, d.get("verdict", "dropped")))
                sub["dropped_by_dedup"] += 1
                dedup["internal"][fname] = {"dropped": True, "reason": d.get("reason", "")}
                continue

            recs = load_records(fpath)
            if a.limit_per_file:
                recs = recs[: a.limit_per_file]
            n_in = 0
            n_out = 0
            for raw in adapter(ds_id, fname, recs):
                n_in += 1
                # --- 质量过滤（每条丢弃都按「任务型 × 原因」计数，便于判断阈值是否过严）---
                if filters.get("drop_empty_output", True) and not raw["output"]:
                    sub["drop_empty_output"] += 1
                    drop_by_task[raw["task"]]["empty_output"] += 1
                    continue
                if len(raw["output"]) < filters.get("min_output_chars", 0):
                    sub["drop_output_too_short"] += 1
                    drop_by_task[raw["task"]]["output_too_short"] += 1
                    continue
                if role == "qa" and len(raw["input"]) + len(raw["instruction"]) < filters.get(
                        "min_input_chars", 0):
                    sub["drop_input_too_short"] += 1
                    drop_by_task[raw["task"]]["input_too_short"] += 1
                    continue
                if filters.get("require_cjk", True):
                    probe = raw["input"] + raw["instruction"] + raw["output"]
                    if not CJK_RE.search(probe):
                        sub["drop_no_cjk"] += 1
                        continue

                rec = build_record(raw, cfg, meta)

                # --- 全局精确去重 ---
                key = rec["content_sha1"]
                if key in seen_content:
                    sub["drop_content_dup"] += 1
                    dedup["global_content_dup"] += 1
                    if len(dedup["examples"]) < 20:
                        dedup["examples"].append({
                            "source": ds_id, "file": fname, "id": rec["source_id"],
                            "same_as": seen_content[key],
                        })
                    continue
                seen_content[key] = "%s:%s" % (ds_id, rec["source_id"])

                # --- uid 全局唯一性：当场暴露，绝不留给下游 ---
                # 撞号的危害不是统计虚高，而是「按 uid 剔除/合并」会整组连坐（见 uid_of 说明）。
                if rec["uid"] in seen_uid:
                    uid_dup["count"] += 1
                    if len(uid_dup["samples"]) < 20:
                        uid_dup["samples"].append({
                            "uid": rec["uid"], "file": fname,
                            "source_id": rec["source_id"],
                            "first_seen_in": seen_uid[rec["uid"]],
                        })
                else:
                    seen_uid[rec["uid"]] = fname

                ds_domain[rec["domain"]] += 1
                stats["domain_source_dist"][rec["domain_source"]] += 1
                stats["task_dist"][rec["task"]] += 1
                if len(rec["domains"]) > 1:
                    stats["multilabel_count"] += 1
                out_lines.append(rec)
                n_out += 1

            sub["files_read"] += 1
            print("    %-52s in=%-7d out=%-7d" % (fname, n_in, n_out))
            stats["totals"]["in"] += n_in
            stats["totals"]["out"] += n_out
            stats["totals"]["files"] += 1

        target_dir = os.path.join(OUT, "statutes" if role == "statutes" else "qa")
        outp = os.path.join(target_dir, slug + a.suffix + ".jsonl")
        with open(outp, "w", encoding="utf-8") as fh:
            for r in out_lines:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("    -> %s  (%d 条)" % (outp, len(out_lines)))

        stats["sources"][ds_id] = {
            "slug": slug,
            "role": role,
            "adapter": s["adapter"],
            "received": dict(sub),
            "written": len(out_lines),
            "output": outp,
            "domain_dist": dict(ds_domain),
        }
        stats["domain_by_source"][ds_id] = dict(ds_domain)

    if a.list:
        return 0

    # --- uid 唯一性硬断言（放在合并/汇总之前，fail loud） ---
    stats["uid_unique"] = {"total": sum(s["written"] for s in stats["sources"].values()),
                           "duplicates": uid_dup["count"],
                           "samples": uid_dup["samples"]}
    if uid_dup["count"]:
        print("\n" + "!" * 78)
        print("!!! uid 不唯一：%d 条撞号 —— 下游按 uid 剔除会整组连坐，拒绝继续" % uid_dup["count"])
        for s in uid_dup["samples"][:10]:
            print("    uid=%s  file=%s  first_seen_in=%s"
                  % (s["uid"], s["file"], s["first_seen_in"]))
        print("!" * 78)
        return 3

    # 合并 qa 流
    if not a.no_merge:
        allp = os.path.join(OUT, "qa", "_all" + a.suffix + ".jsonl")
        qa_dir = os.path.join(OUT, "qa")
        # ★ 只合并**本轮真正产出**的文件（以 stats["sources"] 登记的 output 为准）。
        #   历史缺陷：旧实现按「目录下所有非 `_` 开头的 *.jsonl」合并，会把调试遗留的
        #   `*.sample.jsonl` 一起吃进合并流 —— 实测 `_all.jsonl` 288,855 行
        #   = 285,257（三个正式文件）+ 3,598（三个 .sample.jsonl 残留）。
        expected = sorted(os.path.basename(s["output"])
                          for s in stats["sources"].values()
                          if s.get("role") == "qa")
        stray = sorted(f for f in os.listdir(qa_dir)
                       if f.endswith(".jsonl") and not f.startswith("_")
                       and f not in expected)
        if stray:
            print("\n!! 警告：qa 输出目录有本轮未产出的 .jsonl（**已排除，未合并**）：")
            for f in stray:
                print("     %s" % f)
            print("     若为调试残留（如 *.sample.jsonl），建议移出 data/corpus/ 之外。")
        n = 0
        with open(allp, "w", encoding="utf-8") as out:
            for fn in expected:
                p = os.path.join(qa_dir, fn)
                if not os.path.exists(p):
                    print("!! 期望产出缺失，合并流将不完整：%s" % p)
                    continue
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        out.write(line)
                        n += 1
        print("\n合并流 -> %s (%d 条，来自 %d 个文件)" % (allp, n, len(expected)))
        stats["merged_qa"] = {"path": allp, "rows": n, "files": expected,
                              "stray_not_merged": stray}

    # --- 汇总：域分布 ---
    dom_total = collections.Counter()
    for ds, dd in stats["domain_by_source"].items():
        m = stats["sources"].get(ds, {}).get("role", "qa")
        if m != "qa":
            continue
        for k, v in dd.items():
            dom_total[k] += v
    stats["qa_domain_total"] = dict(dom_total)

    # --- 与 target_mix 对比 ---
    # target_mix 形如 {criminal: {target: 30000, share: 0.30}, ..., total: 100000}
    # 兼容两种写法：{target: N, share: x} 或直接给数字
    target = (src_cfg.get("target_mix") or {})

    def _t_of(v):
        if v is None:
            return None
        if isinstance(v, dict):
            return v.get("target")
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    cmp_rows = []
    _tot = max(1, sum(dom_total.values()))
    for dom in DOMAINS:
        t = _t_of(target.get(dom))
        have = dom_total.get(dom, 0)
        cmp_rows.append({"domain": dom, "target": t, "have": have,
                         "ratio_have": round(have / _tot, 4),
                         "delta": (None if t is None else have - t),
                         "gap": (None if t is None else max(0, t - have))})
    stats["target_comparison"] = cmp_rows
    stats["target_mix_total"] = _t_of(target.get("total"))
    stats["target_mix_raw"] = {k: (v if not isinstance(v, dict) else dict(v))
                               for k, v in target.items()}
    stats["total_gap_to_target"] = sum(
        r["gap"] for r in cmp_rows if r["gap"] is not None)

    stats["totals"] = dict(stats["totals"])
    stats["domain_source_dist"] = dict(stats["domain_source_dist"])
    stats["task_dist"] = dict(stats["task_dist"].most_common(40))
    stats["drop_by_task"] = {k: dict(v) for k, v in
                             sorted(drop_by_task.items(),
                                    key=lambda kv: -sum(kv[1].values()))}

    with open(os.path.join(OUT, "STATS" + a.suffix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "DEDUP_REPORT" + a.suffix + ".json"), "w", encoding="utf-8") as fh:
        json.dump(dedup, fh, ensure_ascii=False, indent=2)

    # --- 控制台汇总 ---
    print("\n" + "=" * 78)
    print("域分布（qa 流）")
    print("=" * 78)
    tot = max(1, sum(dom_total.values()))
    for row in stats["target_comparison"]:
        dom, v, t = row["domain"], row["have"], row["target"]
        print("  %-12s %8d  (%5.1f%%)   目标=%-8s 差=%-8s 缺口=%s" % (
            dom, v, 100.0 * v / tot, t if t is not None else "-",
            row["delta"] if row["delta"] is not None else "-",
            row["gap"] if row["gap"] is not None else "-"))
    print("  %-12s %8d   （target_mix.total=%s，总缺口=%s）" % (
        "合计", sum(dom_total.values()), stats.get("target_mix_total"),
        stats.get("total_gap_to_target")))
    print("域占比: %s" % json.dumps(
        {r["domain"]: r["ratio_have"] for r in stats["target_comparison"]},
        ensure_ascii=False))
    print("domain_source 分布: %s" % dict(stats["domain_source_dist"]))
    print("多标签条数: %d" % stats["multilabel_count"])
    print("全局内容重复丢弃: %d" % dedup["global_content_dup"])
    print("\n按任务型统计的丢弃（前 12）:")
    for t, c in list(stats["drop_by_task"].items())[:12]:
        print("  %-26s %s" % (t, c))
    print("\n逐来源域分布:")
    for ds, dd in stats["domain_by_source"].items():
        print("  %-42s %s" % (ds, dd))
    print("\n产出: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
