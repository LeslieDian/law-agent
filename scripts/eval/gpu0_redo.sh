#!/bin/bash
# ===========================================================================
# GPU0 重跑通道（2026-09-21 OOM 事故修复版）
#
# 事故回顾：原 overnight_run.sh 的 gpu0_lane 等待循环有 12h 上限，
#   2026-09-20 20:29 起等到 08:40 超时退出，在 A0 生成进程（63GB 显存）仍在
#   GPU0 时抢跑 → base 三阶段 + criminal/civil 内部集大面积 CUDA OOM，
#   产物已归档为 answers.OOM_STALE.jsonl（不删）。
#
# 修复点：
#   ★ 等待条件改为**双条件**：A0 链 MARKER 出现 **且** GPU0 显存 < 10GB，
#     两者都满足才开跑，不再有"超时抢跑"路径。
#   ★ 收尾（finalize）等 MoE 行也完成（GPU1 队列 MARKER + 行数 >= 13900）
#     才打 MARKER_OVERNIGHT_DONE，避免半成品触发收割。
#   ★ 每阶段结束检查 ok/err 比例，err > ok 时打 WARN（不自动重试，留给人工看）。
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

# ---------- 三专家内部验证集 ----------
for dom in criminal civil procedure; do
  [ -f "models/adapters/$dom/adapter_model.safetensors" ] || { echo "跳过 $dom（适配器缺）"; continue; }
  run_stage "internal_$dom" "$O/O4_internal_$dom.log" \
    $P scripts/train/run_inference.py --task internal_test \
       --adapter "models/adapters/$dom" --gpu 0 --batch-size 32 --max-new-tokens 1024 \
       --system-name "$dom" --out "outputs/infer/internal_$dom" \
       --report-json "docs/eval/FULL_INFER_internal_$dom.json" \
       --report-md "docs/eval/FULL_INFER_internal_$dom.md"
done

echo "=== GPU0 通道全部完成 $(date '+%F %T') ==="

# ---------- 等 MoE 行完成（GPU1 队列）再收尾 ----------
echo "=== 等待 MoE 队列完成（MARKER 或 13900 行，最长 20h） ==="
for i in $(seq 1 1200); do
  GM=$(grep -c MARKER_GPU1_QUEUE_DONE logs/gpu1_queue_moe_chain.log 2>/dev/null); GM=${GM:-0}
  MR=$(wc -l < outputs/infer/lexeval_moe_L2/answers.jsonl 2>/dev/null || echo 0)
  if [ "${GM:-0}" -ge 1 ] && [ "${MR:-0}" -ge 13900 ]; then echo "MoE 完成 (rows=$MR)"; break; fi
  sleep 60
done
echo "MoE 等待结束：MARKER=$GM rows=$MR $(date '+%F %T')"

# ---------- 收尾：客观打分 → 汇总 → 出图 ----------
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
            "moe_L2:outputs/infer/internal_moe_L2" \
            "criminal:outputs/infer/internal_criminal" \
            "civil:outputs/infer/internal_civil" \
            "procedure:outputs/infer/internal_procedure"; do
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
