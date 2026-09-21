#!/bin/bash
# =============================================================================
# 过夜无人值守链（2026-09-20 建立）
#
# 前提：GPU1 上 logs/_run/gate_run.sh 正在跑（门控训练 → 自动 gpu1_queue.sh 出 MoE 结果），
#       GPU0 上 A0 三阶段评测链正在跑（末尾打 MARKER_FULL_INFER_A0_DONE）。
#       两条链都**不归本脚本管**，本脚本只接手它们之后剩下的活。
#
# 本脚本做四件事（两条并行通道 + 收尾）：
#   通道 A（判分，仅 API、不占 GPU）：
#     A0 → MoE-L2 → base，逐个：等该系统的 LexRubric 答案齐 → 冒烟 8 条 → 全量 MiniMax-M3 判分
#   通道 B（GPU0）：等 A0 链跑完 → base 的 lexrubric/obj/gen 三阶段
#                   → （选做）三专家内部验证集 1000 题
#   收尾：客观打分（lexeval + internal，CPU 秒级）→ collect_results → make_figures
#   标志：MARKER_OVERNIGHT_DONE / MARKER_OVERNIGHT_NO_KEY
#
# 设计取舍
#   1) 判分通道与生成通道**并行**：判分走 HTTP，不碰 GPU，硬等生成结束纯属浪费。
#   2) 判分**串行**（一次一个系统）：避免 3 路 × 10 workers 的风控风险，
#      且单路失败不影响后续系统。
#   3) 每个系统判分前先跑 8 条冒烟：API key 错 / case_id 对不上 / 覆盖率 0
#      这三个致命问题要在"烧掉 12335 次调用"之前暴露。
#   4) 全脚本 set +e：任何一步失败都记录并继续，能出的结果一定要出全。
# =============================================================================
cd /mnt/data/lidian/law-agent || exit 1
set +e
export PYTHONUTF8=1

P=envs/main/bin/python
D=data/benchmark/lexeval/repo/data
OBJ="$D/1_1.json,$D/1_2.json,$D/1_3.json,$D/2_1.json,$D/2_2.json,$D/2_3.json,$D/2_4.json,$D/2_5.json,$D/3_1.json,$D/3_2.json,$D/3_3.json,$D/3_4.json,$D/3_5.json,$D/3_6.json,$D/4_1.json,$D/4_2.json,$D/6_1.json,$D/6_2.json,$D/6_3.json"
GEN="$D/5_1.json,$D/5_2.json,$D/5_3.json,$D/5_4.json"
O=logs/overnight
mkdir -p "$O" logs/eval docs/eval docs/figures outputs/score outputs/judge/lexrubric

N_RUBRIC=649
N_OBJ=11400
N_GEN=2750
WAIT_H_RUBRIC=16          # 等某个系统 LexRubric 答案齐的最长小时数

log(){ echo "[$(date '+%F %T')] $*"; }
nl(){ wc -l < "$1" 2>/dev/null || echo 0; }

# 证据留存 —— 无头服务器没有"屏幕截图"，用 GPU/进程/产出快照代替，进论文附录
snap(){
  { echo ""
    echo "================ EVIDENCE [$1]  $(date '+%F %T') ================"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
    echo "--- running ---"
    ps -eo pid,etimes,args | grep -E "[r]un_inference|[t]rain_moe_gate|[l]exrubric_doubleblind" | cut -c1-140
    echo "--- answers.jsonl ---"
    for f in outputs/infer/*/answers.jsonl; do
      [ -f "$f" ] && printf "%8s  %s\n" "$(nl "$f")" "$f"
    done
  } >> docs/eval/EVIDENCE.txt 2>&1
  log "evidence -> docs/eval/EVIDENCE.txt [$1]"
}

wait_file(){   # wait_file <path> <min_lines> <max_hours> <desc>
  local f=$1 n=$2 h=$3 d=$4 i max
  max=$(( h * 720 ))        # 每 tick 5s
  for i in $(seq 1 "$max"); do
    [ "$(nl "$f")" -ge "$n" ] && { log "READY $d ($(nl "$f")/$n)"; return 0; }
    sleep 5
  done
  log "[WARN] TIMEOUT 等 $d：$(nl "$f")/$n"
  return 1
}

run_stage(){   # run_stage <tag> <logfile> <cmd...>
  local tag=$1 lg=$2; shift 2
  log ">>> STAGE $tag 开始"
  ( "$@" ) > "$lg" 2>&1
  local rc=$?
  log "<<< STAGE $tag rc=$rc  ($(sed 's/\r/\n/g' "$lg" 2>/dev/null | grep -E 'rc=|条/s|MARKER' | tail -1))"
  snap "after_$tag"
  return $rc
}

# ---------------------------------------------------------------------------
# 通道 A：判分（API）
# ---------------------------------------------------------------------------
judge_one(){   # judge_one <sys> <answers.jsonl>
  local sys=$1 ans=$2
  wait_file "$ans" "$N_RUBRIC" "$WAIT_H_RUBRIC" "$sys LexRubric 答案" || return 1

  log "--- $sys 冒烟判分 8 条（先验 key/覆盖率，避免白烧 12k 次调用）"
  rm -rf "outputs/judge/lexrubric/_smoke_$sys"
  $P -m src.evaluation.lexrubric_doubleblind \
     --lexrubric-root data/benchmark/lexrubric --answers "$ans" \
     --judge-config configs/judge.yaml --judges minimax-m3 --workers 4 --limit 8 \
     --out "outputs/judge/lexrubric/_smoke_$sys" > "$O/judge_${sys}_smoke.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q LEXRUBRIC_EVAL_DONE "$O/judge_${sys}_smoke.log"; then
    log "[FAIL] $sys 冒烟判分失败 rc=$rc —— 跳过该系统全量判分"
    sed 's/\r/\n/g' "$O/judge_${sys}_smoke.log" | tail -25 >> docs/eval/JUDGE_FAILURES.txt 2>&1
    echo "JUDGE_SMOKE_FAIL_$sys" >> "$O/judge_results.txt"
    return 1
  fi
  log "[OK] $sys 冒烟通过：$(sed 's/\r/\n/g' "$O/judge_${sys}_smoke.log" | grep -E 'overall:' | tail -1)"

  log "--- $sys 全量判分（$N_RUBRIC 条 case，workers=10）"
  $P -m src.evaluation.lexrubric_doubleblind \
     --lexrubric-root data/benchmark/lexrubric --answers "$ans" \
     --judge-config configs/judge.yaml --judges minimax-m3 --workers 10 \
     --out "outputs/judge/lexrubric/$sys" > "$O/judge_${sys}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "outputs/judge/lexrubric/$sys/SUMMARY.json" ]; then
    log "[OK] $sys 全量判分完成 → outputs/judge/lexrubric/$sys/SUMMARY.json"
    echo "JUDGE_OK_$sys" >> "$O/judge_results.txt"
  else
    log "[FAIL] $sys 全量判分 rc=$rc"
    echo "JUDGE_FAIL_$sys" >> "$O/judge_results.txt"
    sed 's/\r/\n/g' "$O/judge_${sys}.log" | tail -25 >> docs/eval/JUDGE_FAILURES.txt 2>&1
  fi
  snap "after_judge_$sys"
}

judge_lane(){
  log "=== 通道 A（判分）启动 ==="
  export MINIMAX_API_KEY=$(grep -oE 'MINIMAX_API_KEY=[^ ]*' /home/lidian/.bashrc | head -1 | cut -d= -f2- | tr -d '"')
  if [ -z "$MINIMAX_API_KEY" ]; then
    log "[FATAL] 取不到 MINIMAX_API_KEY —— 判分通道放弃"
    echo "MARKER_OVERNIGHT_NO_KEY"
    return 1
  fi
  log "MINIMAX_API_KEY 已取到（len=${#MINIMAX_API_KEY}）"
  : > "$O/judge_results.txt"
  judge_one A0     outputs/infer/lexrubric_A0_unified_qwen3_8b/answers.jsonl
  judge_one moe_L2 outputs/infer/lexrubric_moe_L2/answers.jsonl
  judge_one base   outputs/infer/lexrubric_base/answers.jsonl
  log "=== 通道 A 结束 ==="
  cat "$O/judge_results.txt" 2>/dev/null
}

# ---------------------------------------------------------------------------
# 通道 B：GPU0（base 三阶段 + 三专家内部集）
# ---------------------------------------------------------------------------
gpu0_lane(){
  log "=== 通道 B（GPU0）启动：先等 A0 三阶段链跑完 ==="
  local i gone=0
  for i in $(seq 1 8640); do                                   # 最长 12h
    grep -q MARKER_FULL_INFER_A0_DONE logs/eval_A0_chain.log 2>/dev/null && break
    # ★ 必须"连续 2 分钟无 run_inference 进程"才认定链已结束：
    #   链在**阶段之间**有短暂空档（打印分隔行、写报告），单次 pgrep miss 就抢跑
    #   会和 A0 的下一阶段同时在 GPU0 上、直接 OOM。踩过，故留 24 tick 消抖。
    if pgrep -f "[r]un_inference.py --task le" >/dev/null; then
      gone=0
    else
      gone=$((gone + 1))
    fi
    [ "$gone" -ge 24 ] && break
    sleep 5
  done
  if grep -q MARKER_FULL_INFER_A0_DONE logs/eval_A0_chain.log 2>/dev/null; then
    log "A0 链正常结束（见 MARKER_FULL_INFER_A0_DONE）"
  else
    log "[WARN] 未见 A0 链 MARKER 且无 run_inference 进程 —— 按 GPU0 已空闲继续"
  fi
  # ★ OOM 事故修复（2026-09-21）：上面的超时抢跑曾在 A0 生成进程仍占 63GB 时开跑，
  #   base 三阶段 + 三专家内部集大面积 CUDA OOM、数据报废。现在要求 GPU0 显存
  #   真的空下来（<10GB）才继续，最多再等 6h。
  for i in $(seq 1 360); do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    [ "${MEM:-99999}" -lt 10000 ] && break
    sleep 60
  done
  log "GPU0 显存 ${MEM:-?}MiB，开始通道 B 任务"
  snap "gpu0_lane_start"
  sleep 20                                                     # 等显存彻底释放

  # ---- base 三阶段 ----
  run_stage base_lexrubric logs/eval/O1_base_lexrubric.log \
    $P scripts/train/run_inference.py --task lexrubric \
       --adapter none --gpu 0 --batch-size 32 --max-new-tokens 1536 \
       --system-name base --out outputs/infer/lexrubric_base \
       --report-json docs/eval/FULL_INFER_lexrubric_base.json \
       --report-md docs/eval/FULL_INFER_lexrubric_base.md

  run_stage base_obj logs/eval/O2_base_lexeval_obj.log \
    $P scripts/train/run_inference.py --task lexeval \
       --adapter none --gpu 0 --batch-size 32 --max-new-tokens 256 --input "$OBJ" \
       --system-name base --out outputs/infer/lexeval_base \
       --report-json docs/eval/FULL_INFER_lexeval_obj_base.json \
       --report-md docs/eval/FULL_INFER_lexeval_obj_base.md

  run_stage base_gen logs/eval/O3_base_lexeval_gen.log \
    $P scripts/train/run_inference.py --task lexeval \
       --adapter none --gpu 0 --batch-size 32 --max-new-tokens 1536 --input "$GEN" \
       --system-name base --out outputs/infer/lexeval_base \
       --report-json docs/eval/FULL_INFER_lexeval_gen_base.json \
       --report-md docs/eval/FULL_INFER_lexeval_gen_base.md

  # ---- 选做：三专家内部验证集（1000 题，cap 1024），用于"域专业化"分析 ----
  for dom in criminal civil procedure; do
    [ -f "models/adapters/$dom/adapter_model.safetensors" ] || { log "跳过 $dom（适配器缺）"; continue; }
    run_stage "internal_$dom" "logs/eval/O4_internal_$dom.log" \
      $P scripts/train/run_inference.py --task internal_test \
         --adapter "models/adapters/$dom" --gpu 0 --batch-size 32 --max-new-tokens 1024 \
         --system-name "$dom" --out "outputs/infer/internal_$dom" \
         --report-json "docs/eval/FULL_INFER_internal_$dom.json" \
         --report-md "docs/eval/FULL_INFER_internal_$dom.md"
  done
  log "=== 通道 B 结束 ==="
}

# ---------------------------------------------------------------------------
# 收尾：客观打分 → 汇总 → 出图
# ---------------------------------------------------------------------------
finalize(){
  log "=== 收尾：客观打分（CPU，秒级）==="
  local pair ans out
  # lexeval：客观 + 生成同在一份 answers.jsonl，score_answers 按 gold 形态自动分流
  for pair in "A0:outputs/infer/lexeval_A0_unified_qwen3_8b" \
              "moe_L2:outputs/infer/lexeval_moe_L2" \
              "base:outputs/infer/lexeval_base"; do
    ans="${pair#*:}/answers.jsonl"
    out="${pair%%:*}"
    if [ -s "$ans" ]; then
      $P scripts/eval/score_answers.py --task lexeval --answers "$ans" \
         --metrics all --out "outputs/score/lexeval_$out.json" > "$O/score_lexeval_$out.log" 2>&1
      log "  lexeval[$out] rc=$?  $(grep -E '选择题 Accuracy|ROUGE-L' "$O/score_lexeval_$out.log" | tr '\n' ' ')"
    else
      log "  lexeval[$out] 跳过（answers 空）"
    fi
  done
  # internal：A0 目录名是历史遗留的 A0_internal_test
  for pair in "A0:outputs/infer/A0_internal_test" \
              "moe_L2:outputs/infer/internal_moe_L2" \
              "criminal:outputs/infer/internal_criminal" \
              "civil:outputs/infer/internal_civil" \
              "procedure:outputs/infer/internal_procedure"; do
    ans="${pair#*:}/answers.jsonl"
    out="${pair%%:*}"
    if [ -s "$ans" ]; then
      $P scripts/eval/score_answers.py --task internal_test --answers "$ans" --metrics all \
         --group-by-domain data/test/test.jsonl \
         --out "outputs/score/internal_$out.json" > "$O/score_internal_$out.log" 2>&1
      log "  internal[$out] rc=$?  $(grep -E 'ROUGE-L|法条引用命中率' "$O/score_internal_$out.log" | tr '\n' ' ')"
    else
      log "  internal[$out] 跳过（answers 空）"
    fi
  done

  log "=== 收尾：汇总矩阵 ==="
  $P scripts/eval/collect_results.py > "$O/collect.log" 2>&1
  log "collect rc=$?  $(grep MARKER_COLLECT_DONE "$O/collect.log")"
  sed 's/\r/\n/g' "$O/collect.log" | grep -E '^\[ok\]|^\[skip\]' 

  log "=== 收尾：出图 ==="
  $P scripts/eval/make_figures.py > "$O/figures.log" 2>&1
  log "figures rc=$?  $(grep MARKER_FIGURES_DONE "$O/figures.log")"
  sed 's/\r/\n/g' "$O/figures.log" | grep -E '^\[OK\]|^\[FAIL\]|^\[skip\]'

  snap "final"
  log "产物清单："
  ls -la docs/figures/ 2>&1 | tail -8
  echo "MARKER_OVERNIGHT_DONE $(date '+%F %T')"
}

# ---------------------------------------------------------------------------
log "############ 过夜链启动 $(date '+%F %T') ############"
snap "start"
echo "=== judge_results ===" > "$O/judge_results.txt"

judge_lane > "$O/judge_lane.log" 2>&1 &
J=$!
log "judge_lane pid=$J"

gpu0_lane > "$O/gpu0_lane.log" 2>&1 &
G=$!
log "gpu0_lane pid=$G"

wait $G
log "gpu0_lane 退出 rc=$?"
wait $J
log "judge_lane 退出 rc=$?"

finalize
log "############ 过夜链结束 $(date '+%F %T') ############"
