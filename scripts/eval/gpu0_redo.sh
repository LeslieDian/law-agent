#!/bin/bash
# ===========================================================================
# GPU0 重跑通道 v2（2026-09-21 10:10 修订）
#
# v1 回顾：原 overnight gpu0_lane 等待 12h 超时抢跑 → OOM 事故，产物已归档。
#   v1 修复 = 双条件等待（A0 MARKER 且 GPU0 显存 < 10GB），无超时抢跑路径。
#
# v2 新增两处（用户要求压时间 + 砍掉不挡路的实验）：
#   ★ [Q5 助推] base 生成题跑完后，若 GPU1 的 MoE 生成题（Q5）还没跑完，
#     GPU0 用 run_inference 的**断点续跑**机制分担剩余题：
#       - 种子拷贝主 answers 到 splitB 目录（续跑自动跳过已完成 case_id）
#       - 两进程**各写各的文件**（禁区的"同文件双写"不发生）
#       - splitB 完成后：union 按 case_id 去重，unique ≥ 14148 才原子替换主文件
#         并停掉 GPU1 的 Q5（其剩余题已全覆盖）；不满足则回退自然完成，零损失
#       - case_id 是位置制（lexeval::split::idx），splitB 用**同一份 GEN 输入**
#         保证 id 与主文件同构
#   ★ [三专家后置] criminal/civil/procedure 内部集只喂"域专业化"分析图、
#     不进主表 → 挪到 MARKER_OVERNIGHT_DONE **之后**跑，不挡论文主结果。
#     分数由每小时 stage_sync 的 collect_results.py 自动补进 README。
# ===========================================================================
cd /mnt/data/lidian/law-agent || exit 1
P=envs/main/bin/python
D=data/benchmark/lexeval/repo/data
OBJ="$D/1_1.json,$D/1_2.json,$D/1_3.json,$D/2_1.json,$D/2_2.json,$D/2_3.json,$D/2_4.json,$D/2_5.json,$D/3_1.json,$D/3_2.json,$D/3_3.json,$D/3_4.json,$D/3_5.json,$D/3_6.json,$D/4_1.json,$D/4_2.json,$D/6_1.json,$D/6_2.json,$D/6_3.json"
GEN="$D/5_1.json,$D/5_2.json,$D/5_3.json,$D/5_4.json"
O=logs/overnight_redo
mkdir -p "$O"

run_stage(){
  local name="$1" logf="$2"; shift 2
  echo "=== [$name] START $(date '+%F %T') ==="
  "$@" > "$logf" 2>&1
  local rc=$?
  local okline=$(tr '\r' '\n' < "$logf" | grep -oE "ok=[0-9]+ err=[0-9]+" | tail -1)
  echo "[$name] rc=$rc  $okline"
  local okn=$(echo "$okline" | grep -oE "ok=[0-9]+" | cut -d= -f2)
  local errn=$(echo "$okline" | grep -oE "err=[0-9]+" | cut -d= -f2)
  if [ -n "$errn" ] && [ -n "$okn" ] && [ "$errn" -gt "$okn" ]; then
    echo "[$name] ★WARN 错误数超过成功数，产物可能不可用：$okline"
  fi
  return $rc
}

# ---------- 稳健等待：A0 链结束 且 GPU0 真空闲 ----------
echo "=== 等待 A0 链结束 且 GPU0 显存 < 10GB（最长 24h） $(date '+%F %T') ==="
for i in $(seq 1 1440); do
  M=$(grep -c MARKER_FULL_INFER_A0_DONE logs/eval_A0_chain.log 2>/dev/null); M=${M:-0}
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  if [ "${M:-0}" -ge 1 ] && [ "${MEM:-99999}" -lt 10000 ]; then break; fi
  sleep 60
done
echo "等待结束：MARKER=$M GPU0_used=${MEM}MiB $(date '+%F %T')"
sleep 30   # 显存彻底释放

# ---------- base 三阶段 ----------
run_stage base_lexrubric "$O/O1_base_lexrubric.log" \
  $P scripts/train/run_inference.py --task lexrubric \
     --adapter none --gpu 0 --batch-size 32 --max-new-tokens 1536 \
     --system-name base --out outputs/infer/lexrubric_base \
     --report-json docs/eval/FULL_INFER_lexrubric_base.json \
     --report-md docs/eval/FULL_INFER_lexrubric_base.md

run_stage base_obj "$O/O2_base_lexeval_obj.log" \
  $P scripts/train/run_inference.py --task lexeval \
     --adapter none --gpu 0 --batch-size 32 --max-new-tokens 256 --input "$OBJ" \
     --system-name base --out outputs/infer/lexeval_base \
     --report-json docs/eval/FULL_INFER_lexeval_obj_base.json \
     --report-md docs/eval/FULL_INFER_lexeval_obj_base.md

run_stage base_gen "$O/O3_base_lexeval_gen.log" \
  $P scripts/train/run_inference.py --task lexeval \
     --adapter none --gpu 0 --batch-size 32 --max-new-tokens 1536 --input "$GEN" \
     --system-name base --out outputs/infer/lexeval_base \
     --report-json docs/eval/FULL_INFER_lexeval_gen_base.json \
     --report-md docs/eval/FULL_INFER_lexeval_gen_base.md

# ---------- [v2] MoE Q5 助推（GPU0 空出后分担 MoE 生成题，省约 2h） ----------
MGEN=outputs/infer/lexeval_moe_L2/answers.jsonl
MROWS=$(wc -l < "$MGEN" 2>/dev/null); MROWS=${MROWS:-0}
if [ "$MROWS" -lt 13900 ]; then
  echo "=== [Q5助推] MoE lexeval 行数=$MROWS < 13900，GPU0 分担剩余生成题 $(date '+%F %T') ==="
  mkdir -p outputs/infer/lexeval_moe_L2_splitB
  cp -f "$MGEN" outputs/infer/lexeval_moe_L2_splitB/answers.jsonl
  echo "种子拷贝完成：splitB 起点行数=$(wc -l < outputs/infer/lexeval_moe_L2_splitB/answers.jsonl)"
  run_stage q5_assist "$O/O3b_q5_assist.log" \
    $P scripts/train/run_inference.py --task lexeval \
       --moe-gate models/moe/L2_gate/gate_weights.pt --gpu 0 --batch-size 32 \
       --max-new-tokens 1536 --input "$GEN" --system-name moe_L2 \
       --out outputs/infer/lexeval_moe_L2_splitB \
       --report-json docs/eval/FULL_INFER_lexeval_gen_moe_L2_splitB.json \
       --report-md docs/eval/FULL_INFER_lexeval_gen_moe_L2_splitB.md
  # ---- 合并判定：union(主, splitB) 按 case_id 去重（只要无 error 的行） ----
  $P - <<'PYX'
import json, os
A = "outputs/infer/lexeval_moe_L2/answers.jsonl"
B = "outputs/infer/lexeval_moe_L2_splitB/answers.jsonl"
def load(p):
    out = {}
    if not os.path.exists(p): return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("error"): continue          # 错误行不参与合并
            cid = r.get("case_id")
            if cid and cid not in out: out[cid] = line
    return out
u = load(A); na = len(u)
u.update(load(B))
print("union unique = %d (主文件自身 %d)" % (len(u), na))
# 期望：客观 11400 + 生成 2750 = 14150；容 2 条误差
if len(u) >= 14148:
    with open(A + ".merged", "w", encoding="utf-8") as f:
        f.write("\n".join(u.values()) + "\n")
    os.replace(A + ".merged", A)                  # 原子替换
    open("outputs/infer/lexeval_moe_L2/.ASSIST_MERGED", "w").write("ok")
    print("MERGED_OK rows=%d" % len(u))
else:
    print("MERGE_SKIP union<14148 —— 不动主文件，GPU1 自然跑完（回退零损失）")
PYX
  if [ -f outputs/infer/lexeval_moe_L2/.ASSIST_MERGED ]; then
    APID=$(ps -eo pid,args | grep "run_inference" | grep -- "--out outputs/infer/lexeval_moe_L2 " | grep -v grep | awk '{print $1}' | head -1)
    if [ -n "$APID" ]; then
      kill "$APID" && echo "已停 GPU1 的 Q5（pid=$APID，剩余题已由 splitB 全覆盖）"
    else
      echo "GPU1 的 Q5 已自行结束（无需停）"
    fi
    sleep 5
  fi
else
  echo "=== [Q5助推] MoE 已完成（行数=$MROWS），跳过 ==="
fi

# ---------- 等 MoE 行完成（GPU1 队列）再收尾 ----------
echo "=== 等待 MoE 队列完成（MARKER 或 13900 行，最长 20h） ==="
for i in $(seq 1 1200); do
  GM=$(grep -c MARKER_GPU1_QUEUE_DONE logs/gpu1_queue_moe_chain.log 2>/dev/null); GM=${GM:-0}
  MR=$(wc -l < outputs/infer/lexeval_moe_L2/answers.jsonl 2>/dev/null || echo 0)
  if [ "${GM:-0}" -ge 1 ] && [ "${MR:-0}" -ge 13900 ]; then echo "MoE 完成 (rows=$MR)"; break; fi
  sleep 60
done
echo "MoE 等待结束：MARKER=$GM rows=$MR $(date '+%F %T')"

# ---------- 收尾：客观打分 → 汇总 → 出图 → 主标志 ----------
for pair in "A0:outputs/infer/lexeval_A0_unified_qwen3_8b" \
            "moe_L2:outputs/infer/lexeval_moe_L2" \
            "base:outputs/infer/lexeval_base"; do
  ans="${pair#*:}/answers.jsonl"; out="${pair%%:*}"
  if [ -s "$ans" ]; then
    $P scripts/eval/score_answers.py --task lexeval --answers "$ans" \
       --metrics all --out "outputs/score/lexeval_$out.json" > "$O/score_lexeval_$out.log" 2>&1
    echo "  lexeval[$out] rc=$?"
  fi
done
for pair in "A0:outputs/infer/A0_internal_test" \
            "moe_L2:outputs/infer/internal_moe_L2"; do
  ans="${pair#*:}/answers.jsonl"; out="${pair%%:*}"
  if [ -s "$ans" ]; then
    $P scripts/eval/score_answers.py --task internal_test --answers "$ans" --metrics all \
       --group-by-domain data/test/test.jsonl \
       --out "outputs/score/internal_$out.json" > "$O/score_internal_$out.log" 2>&1
    echo "  internal[$out] rc=$?"
  fi
done

$P scripts/eval/collect_results.py > "$O/collect.log" 2>&1
echo "collect rc=$? $(grep MARKER_COLLECT_DONE "$O/collect.log")"
$P scripts/eval/make_figures.py > "$O/figures.log" 2>&1
echo "figures rc=$? $(grep MARKER_FIGURES_DONE "$O/figures.log")"

echo "MARKER_OVERNIGHT_DONE $(date '+%F %T')"

# ---------- [v2] 三专家内部集：主标志之后补跑（只喂分析图，不挡主表） ----------
for dom in criminal civil procedure; do
  [ -f "models/adapters/$dom/adapter_model.safetensors" ] || { echo "跳过 $dom（适配器缺）"; continue; }
  run_stage "internal_$dom" "$O/O4_internal_$dom.log" \
    $P scripts/train/run_inference.py --task internal_test \
       --adapter "models/adapters/$dom" --gpu 0 --batch-size 32 --max-new-tokens 1024 \
       --system-name "$dom" --out "outputs/infer/internal_$dom" \
       --report-json "docs/eval/FULL_INFER_internal_$dom.json" \
       --report-md "docs/eval/FULL_INFER_internal_$dom.md"
  ans="outputs/infer/internal_$dom/answers.jsonl"
  if [ -s "$ans" ]; then
    $P scripts/eval/score_answers.py --task internal_test --answers "$ans" --metrics all \
       --group-by-domain data/test/test.jsonl \
       --out "outputs/score/internal_$dom.json" > "$O/score_internal_$dom.log" 2>&1
    echo "  internal[$dom] score rc=$?"
  fi
done
$P scripts/eval/collect_results.py > "$O/collect2.log" 2>&1
echo "collect2 rc=$? （专家行由每小时 stage_sync 自动刷进 README）"
echo "=== GPU0 通道全部完成（含专家后置） $(date '+%F %T') ==="
