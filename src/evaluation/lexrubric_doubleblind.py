#!/usr/bin/env python3
"""LexRubric 双盲判分器（接项目 Judge 配置）。

与上游 `code/eval.py` 的关系
------------------------------------------------------------------
**严格保留**的部分（否则分数不可比）：
  * 提示词模板：原样读取上游 `code/prompt.txt`，不做任何改写
  * 占位符替换：`{conversation}` / `{rubric_item}` / `{direction}`
  * 方向判定：`direction = "不是" if point >= 0 else "是"`
  * 计分规则：`total_score = Σ(point for item if criteria_met)`
    （注意负分项：criteria_met=True 时才把负分加进去，即扣分）
  * 对话拼装：`f"user: {question}\\nassistant: {answer}"`
  * 思考链剥离：答案中 `</think>` 之后的内容才算正式回复

**替换/增强**的部分：
  * 判分者：从「单一 DashScope 模型」换成项目 `configs/judge.yaml` 里的
    Gemini-2.5-Pro + DeepSeek-R1 双 Judge
  * 双盲：候选答案做自指标识脱敏，Judge 不知道答案来自哪个系统
  * 落盘：每条 rubric 的原始回复（raw_response）全部保留，可追溯到
  * 汇总：分 split / 分维度 + 双 Judge 一致性（Pearson/Spearman）
    + percentile bootstrap 95% 区间
  * 续跑：按 (case, judge) 断点续跑，JSONL 追加写入

用法
------------------------------------------------------------------
  # 1) 先干跑，不发请求，只检查提示词与调用量
  python -m src.evaluation.lexrubric_doubleblind \
      --lexrubric-root data/benchmark/lexrubric \
      --judge-config configs/judge.yaml \
      --out outputs/judge/lexrubric/smoke \
      --use-reference --limit 3 --dry-run

  # 2) 正式评测（需 GEMINI_API_KEY / DEEPSEEK_API_KEY）
  python -m src.evaluation.lexrubric_doubleblind \
      --lexrubric-root data/benchmark/lexrubric \
      --answers outputs/rag/E5/answers.jsonl \
      --judge-config configs/judge.yaml \
      --out outputs/judge/lexrubric/E5 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---- 允许直接以脚本方式运行 ----
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.judge import (  # noqa: E402
    SAFE_HIDE_LABELS,
    BlindContext,
    JudgeCall,
    build_judges,
)
from src.evaluation.aggregate import (  # noqa: E402
    bootstrap_ci,
    inter_judge_agreement,
    mean,
    summarize_scores,
)

DATASET_NAME = "LexRubric"

# 两个 split 独立登记，绝不混算；与论文表格口径一致
SPLITS = [
    {"key": "legal_consultation", "label": "法律咨询",
     "file": "data/falvzixun.json", "paper_n": 473, "paper_avg_items": 22.41},
    {"key": "judicial_exam", "label": "司法考试",
     "file": "data/sifakaoshi.json", "paper_n": 176, "paper_avg_items": 9.86},
]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class RubricItem:
    idx: int
    criterion: str
    point: float
    dimension: str

    @property
    def direction(self) -> str:
        """★ 原样复刻上游逻辑：正分 → '不是'，负分 → '是'。"""
        return "不是" if self.point >= 0 else "是"


@dataclass
class Case:
    case_id: str          # 全局唯一，带数据集前缀，避免与 LexEval 撞号
    split: str
    raw_id: Any
    question: str
    reference_answer: str
    rubrics: list[RubricItem]

    @property
    def max_score(self) -> float:
        return sum(r.point for r in self.rubrics if r.point > 0)

    @property
    def min_score(self) -> float:
        return sum(r.point for r in self.rubrics if r.point < 0)


@dataclass
class Answer:
    case_id: str
    system: str = "unknown"
    answer: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #

def _read_json_or_jsonl(p: Path) -> Any:
    """LexRubric 是标准 JSON；这里仍保留 JSONL 回退，便于扩展。"""
    txt = p.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        recs = []
        for line in txt.splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
        return recs


def load_cases(root: Path) -> list[Case]:
    cases: list[Case] = []
    data_dir = root / "repo"
    for sp in SPLITS:
        p = data_dir / sp["file"]
        if not p.exists():
            raise FileNotFoundError(f"缺少数据文件: {p}")
        raw = _read_json_or_jsonl(p)
        if not isinstance(raw, list):
            raise ValueError(f"{p} 顶层不是列表")
        for rec in raw:
            rid = rec.get("ID", rec.get("id"))
            if rid is None:
                continue
            items = []
            for i, it in enumerate(rec.get("rubrics") or []):
                items.append(RubricItem(
                    idx=i,
                    criterion=str(it.get("criterion", "")),
                    point=float(it.get("point", 0) or 0),
                    dimension=str(it.get("dimension", "")),
                ))
            cases.append(Case(
                case_id=f"{DATASET_NAME.lower()}::{sp['key']}::{rid}",
                split=sp["key"],
                raw_id=rid,
                question=str(rec.get("question", "")),
                reference_answer=str(rec.get("answer", "")),
                rubrics=items,
            ))
    return cases


def load_answers(path: Path) -> dict[str, Answer]:
    """读取候选答案。接受 JSONL 或 JSON 数组。

    每条至少要有 `id`（原始 ID）或 `case_id`，以及 `answer`。
    可选 `system` 用来标识来源（**不会传给 Judge**，仅用于落盘归档）。
    """
    txt = path.read_text(encoding="utf-8")
    recs: list[dict] = []
    try:
        obj = json.loads(txt)
        recs = obj if isinstance(obj, list) else list(obj.values())
    except Exception:  # noqa: BLE001
        for line in txt.splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    out: dict[str, Answer] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        cid = r.get("case_id")
        if not cid:
            rid = r.get("id", r.get("ID"))
            split = r.get("split", "legal_consultation")
            if rid is None:
                continue
            cid = f"{DATASET_NAME.lower()}::{split}::{rid}"
        out[str(cid)] = Answer(
            case_id=str(cid),
            system=str(r.get("system", "unknown")),
            answer=str(r.get("answer", r.get("model_response", ""))),
            extra={k: v for k, v in r.items()
                   if k not in ("case_id", "id", "ID", "split", "system",
                                "answer", "model_response")},
        )
    return out


def load_judge_cfg(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(txt) or {}
    except Exception:  # noqa: BLE001
        # PyYAML 缺失时的极简回退：只够读 judges 列表的 name/provider
        print("[warn] PyYAML 不可用，改用极简解析（仅能识别 name/provider/model）",
              file=sys.stderr)
        cfg: dict = {"judges": []}
        cur: dict | None = None
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("- name:"):
                cur = {"name": s.split(":", 1)[1].strip().strip('"')}
                cfg["judges"].append(cur)
            elif cur is not None and ":" in s and not s.startswith("#"):
                k, _, v = s.partition(":")
                v = v.split("#")[0].strip().strip('"')
                if v:
                    cur[k.strip()] = v
        return cfg


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #

def load_template(root: Path) -> str:
    p = root / "repo" / "code" / "prompt.txt"
    if not p.exists():
        raise FileNotFoundError(f"缺少上游提示词模板: {p}")
    return p.read_text(encoding="utf-8")


def strip_think(text: str) -> str:
    """原样复刻上游：`</think>` 之后才算正式回复。"""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def build_prompt(template: str, conversation: str, item: RubricItem) -> str:
    """原样复刻上游的替换顺序与占位符。"""
    return (template
            .replace("{conversation}", conversation)
            .replace("{rubric_item}", item.criterion)
            .replace("{direction}", item.direction))


# --------------------------------------------------------------------------- #
# 判分
# --------------------------------------------------------------------------- #

def score_case(
    case: Case,
    answer: Answer,
    template: str,
    judges: dict[str, Any],
    blind: BlindContext,
    max_workers: int,
) -> dict[str, Any]:
    """对单个案例，用全部 Judge 逐条 rubric 打分。"""
    filtered = strip_think(answer.answer)
    scrubbed = blind.scrub(filtered)
    conversation = f"user: {case.question}\nassistant: {scrubbed}"

    jobs = [(jn, j, item) for item in case.rubrics for jn, j in judges.items()]

    # 每个 Judge 的逐条结果
    per_judge: dict[str, list[dict]] = {jn: [] for jn in judges}
    calls: list[JudgeCall] = []

    def _one(jn: str, j: Any, item: RubricItem):
        prompt = build_prompt(template, conversation, item)
        call = j.ask_json(prompt, case_id=case.case_id)
        return jn, item, call

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, jn, j, item) for jn, j, item in jobs]
        for f in as_completed(futs):
            jn, item, call = f.result()
            calls.append(call)
            met = False
            explanation = ""
            parse_error = not call.ok
            if call.ok and isinstance(call.parsed, dict):
                v = call.parsed.get("criteria_met")
                met = v if isinstance(v, bool) else str(v).strip().lower() == "true"
                explanation = str(call.parsed.get("explanation", ""))
            per_judge[jn].append({
                "criterion_idx": item.idx,
                "criterion": item.criterion,
                "point": item.point,
                "dimension": item.dimension,
                "direction": item.direction,
                "criteria_met": met,
                "explanation": explanation,
                "parse_error": parse_error,
                "judge_error": call.error,
                "attempts": call.attempts,
                "latency_s": call.latency_s,
                "raw_response": call.raw_text,
            })

    # ★ 计分：与原版完全一致的规则
    totals: dict[str, float] = {}
    dim_totals: dict[str, dict[str, float]] = {}
    for jn, items in per_judge.items():
        items.sort(key=lambda x: x["criterion_idx"])
        totals[jn] = sum(i["point"] for i in items if i["criteria_met"])
        d: dict[str, float] = {}
        for i in items:
            if i["criteria_met"]:
                d[i["dimension"]] = d.get(i["dimension"], 0.0) + i["point"]
        dim_totals[jn] = d

    return {
        "case_id": case.case_id,
        "split": case.split,
        "raw_id": case.raw_id,
        "system": answer.system,
        "question": case.question,
        "candidate_answer": filtered,
        "candidate_answer_scrubbed": scrubbed,
        "reference_answer": case.reference_answer,
        "max_score": case.max_score,
        "min_score": case.min_score,
        "per_judge_total": totals,
        "per_judge_dimension_total": dim_totals,
        "mean_total": mean(list(totals.values())) if totals else None,
        "criterion_evaluations": per_judge,
        "n_criteria": len(case.rubrics),
        "parse_errors": sum(1 for cs in calls if not cs.ok),
        "judge_calls": len(calls),
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def aggregate(results: list[dict], judges: list[str]) -> dict:
    """分 split 汇总 + Judge 两两一致性 + bootstrap 区间。"""
    out: dict[str, Any] = {"by_split": {}, "overall": {}}

    def _block(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        blk: dict[str, Any] = {"n": len(rs)}
        for jn in judges:
            vals = [r["per_judge_total"].get(jn, 0.0) for r in rs]
            blk[jn] = summarize_scores(vals)
        # 双 Judge 取均值后的主指标
        meanvals = [r["mean_total"] for r in rs if r.get("mean_total") is not None]
        blk["judge_mean"] = summarize_scores(meanvals)
        # 归一化（得分 / 满分），便于跨 split 比较
        norm = [r["mean_total"] / r["max_score"] * 100
                for r in rs if r.get("mean_total") is not None and r["max_score"] > 0]
        if norm:
            blk["normalized_pct"] = summarize_scores(norm)
        blk["parse_error_cases"] = sum(1 for r in rs if r.get("parse_errors"))
        return blk

    splits = sorted({r["split"] for r in results})
    for sp in splits:
        out["by_split"][sp] = _block([r for r in results if r["split"] == sp])
    out["overall"] = _block(results)

    if len(judges) >= 2:
        a, b = judges[0], judges[1]
        out["inter_judge"] = inter_judge_agreement({
            a: [r["per_judge_total"].get(a, 0.0) for r in results],
            b: [r["per_judge_total"].get(b, 0.0) for r in results],
        })
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="LexRubric 双盲判分")
    ap.add_argument("--lexrubric-root", required=True,
                    help="LexRubric 数据集根目录（含 repo/ 与 MANIFEST.json）")
    ap.add_argument("--answers", default="",
                    help="候选答案 JSONL/JSON；与 --use-reference 二选一")
    ap.add_argument("--use-reference", action="store_true",
                    help="用数据集自带参考答案当候选答案（用于自检判分链路）")
    ap.add_argument("--judge-config", default="configs/judge.yaml")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--judges", default="", help="逗号分隔，只跑指定 Judge")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个案例")
    ap.add_argument("--split", default="", choices=["", "legal_consultation", "judicial_exam"])
    ap.add_argument("--dry-run", action="store_true",
                    help="不发任何请求，只生成提示词样例与调用量预估")
    a = ap.parse_args()

    root = Path(a.lexrubric_root).resolve()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"{DATASET_NAME} 双盲判分")
    print("=" * 74)

    # ---- 上游资产指纹（可复现性） ----
    template = load_template(root)
    tpl_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    upstream_eval = root / "repo" / "code" / "eval.py"
    eval_sha = hashlib.sha256(upstream_eval.read_bytes()).hexdigest() if upstream_eval.exists() else ""
    print(f"上游 prompt.txt sha256 = {tpl_sha[:16]}...")
    print(f"上游 eval.py   sha256 = {eval_sha[:16]}...")

    # ---- 载入案例 ----
    cases = load_cases(root)
    if a.split:
        cases = [c for c in cases if c.split == a.split]
    n_items = sum(len(c.rubrics) for c in cases)
    print(f"载入案例 {len(cases)} 个, rubric 细则 {n_items} 条")
    for sp in SPLITS:
        sub = [c for c in cases if c.split == sp["key"]]
        if sub:
            print(f"  {sp['label']}: {len(sub)} 例（论文自述 {sp['paper_n']}）")

    # ---- 载入答案 ----
    if a.use_reference:
        answers = {c.case_id: Answer(c.case_id, "reference", c.reference_answer) for c in cases}
        print("答案来源: 数据集自带参考答案（自检模式）")
    elif a.answers:
        answers = load_answers(Path(a.answers))
        print(f"答案来源: {a.answers}（{len(answers)} 条）")
    else:
        print("[FATAL] 必须指定 --answers 或 --use-reference", file=sys.stderr)
        return 2

    missing = [c.case_id for c in cases if c.case_id not in answers]
    if missing:
        print(f"[warn] {len(missing)} 个案例没有对应答案，将被跳过（示例 {missing[:3]}）")
    cases = [c for c in cases if c.case_id in answers]
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        print("[FATAL] 没有可评测的案例", file=sys.stderr)
        return 2

    # ---- 盲评上下文 ----
    blind = BlindContext(
        seed=42,
        hide_labels=SAFE_HIDE_LABELS,      # 法律文本：只脱敏自指表述，不碰正文
        anonymize_system_name=True,
        shuffle_answer_order=True,
        scrub_model_family=False,          # 关闭泛化替换，避免误伤法律用语
    )

    # ---- Judge ----
    cfg = load_judge_cfg(Path(a.judge_config))
    judge_cfgs = cfg.get("judges", []) or []
    if a.judges:
        want = {s.strip() for s in a.judges.split(",") if s.strip()}
        judge_cfgs = [j for j in judge_cfgs if j.get("name") in want]
    if not judge_cfgs:
        print("[FATAL] judge 配置为空", file=sys.stderr)
        return 2

    # ---- 干跑 ----
    if a.dry_run:
        probe_case = cases[0]
        probe_ans = answers[probe_case.case_id]
        conv = f"user: {probe_case.question}\nassistant: {strip_think(probe_ans.answer)}"
        sample = build_prompt(template, conv, probe_case.rubrics[0])
        (out_dir / "prompts_preview.txt").write_text(
            "# 第 1 例 / 第 1 条 rubric 的完整提示词\n"
            + "=" * 70 + "\n" + sample + "\n" + "=" * 70
            + f"\n\n# 负分项示例（direction 应为 '是'）\n",
            encoding="utf-8")
        neg = next((r for c in cases for r in c.rubrics if r.point < 0), None)
        if neg:
            with (out_dir / "prompts_preview.txt").open("a", encoding="utf-8") as f:
                f.write(build_prompt(template, conv, neg))

        est = sum(len(c.rubrics) for c in cases) * len(judge_cfgs)
        meta = {
            "dry_run": True,
            "dataset": DATASET_NAME,
            "cases": len(cases),
            "rubric_items": sum(len(c.rubrics) for c in cases),
            "judges": [j.get("name") for j in judge_cfgs],
            "estimated_api_calls": est,
            "prompt_template_sha256": tpl_sha,
            "upstream_eval_sha256": eval_sha,
            "blind_audit": blind.audit,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[dry-run] 未发送任何请求")
        print(f"  预计 API 调用次数: {est}")
        print(f"  提示词样例 -> {out_dir/'prompts_preview.txt'}")
        print(f"  元信息     -> {out_dir/'meta.json'}")
        print("DRY_RUN_OK")
        return 0

    judges = build_judges(
        judge_cfgs,
        temperature=0.0,
        max_retries=int((cfg.get("protocol") or {}).get("max_retries", 3)),
    )
    print("Judge: " + ", ".join(f"{n}({j.model})" for n, j in judges.items()))

    # ---- 逐例判分 ----
    jl = out_dir / "judgments.jsonl"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["case_id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [c for c in cases if c.case_id not in done]
    print(f"待处理 {len(todo)} 例（已跳过 {len(done)} 例）")

    results: list[dict] = []
    for i, case in enumerate(todo, 1):
        res = score_case(case, answers[case.case_id], template, judges, blind, a.workers)
        results.append(res)
        with jl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        (raw_dir / f"{case.split}__{case.raw_id}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 5 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {case.case_id}  "
                  f"mean={res['mean_total']:.1f}/{res['max_score']:.1f}  "
                  f"parse_err={res['parse_errors']}")

    # ---- 汇总（含历史结果） ----
    all_res = []
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                all_res.append(json.loads(line))

    summary = aggregate(all_res, list(judges.keys()))
    summary["meta"] = {
        "dataset": DATASET_NAME,
        "cases": len(all_res),
        "judges": {n: j.model for n, j in judges.items()},
        "prompt_template_sha256": tpl_sha,
        "upstream_eval_sha256": eval_sha,
        "blind_audit": blind.audit,
        "scoring_rule": "total_score = sum(point if criteria_met)",
        "direction_rule": "point>=0 -> '不是'; point<0 -> '是'",
        "aggregation": "mean over cases; normalized_pct = mean_total/max_score*100",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caveat": "Judge 为现行模型版本，分数与 LexRubric 论文表格不可直接等同，"
                  "仅作同口径参考比较。",
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    for sp, blk in summary.get("by_split", {}).items():
        jm = blk.get("judge_mean", {})
        np_ = blk.get("normalized_pct", {})
        print(f"{sp}: n={blk.get('n')}  judge_mean={jm.get('mean')}  "
              f"归一化={np_.get('mean')}%  95%CI={np_.get('ci95')}")
    ov = summary["overall"].get("judge_mean", {})
    print(f"overall: n={summary['overall'].get('n')}  judge_mean={ov.get('mean')}")
    if "inter_judge" in summary:
        print(f"双 Judge 一致性: {summary['inter_judge']['pairwise']}")
    print(f"汇总 -> {out_dir/'SUMMARY.json'}")
    print("=" * 74)
    print("LEXRUBRIC_EVAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
