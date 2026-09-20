#!/bin/bash
# ===========================================================================
# GPU1 队列：门控训练完成后**自动接续**（无需人在场）
#
# 设计取舍（2026-09-20 21:00 修订，与 scripts/eval/overnight_run.sh 配套）：
#   ★ 本队列**只跑 MoE 这一行**；E0 base 那两阶段已挪到 GPU0（overnight.sh 通道 B）——
#     GPU0 上的 A0 链约 04:5x 结束，之后空出来，两卡分头跑比串在一条队列上快近一倍。
#   ⛔ **千万不要把 base 再放回这里**：会和 GPU0 上的 base 进程写同一个 answers.jsonl。
#      单行 JSON 含 reference（截断 4000 字）可比 4KB 大，超过 PIPE_BUF 后 O_APPEND
#      不再是原子写 → 行交错、JSONL 损坏。本项目明令禁区。
#   ★ 新增 Q5 = MoE 的 LexEval 生成题（2,750, cap1536）：MoE 是本文主方法，
#     必须有完整三项（LexRubric / 客观 / 生成）才与 A0 可比。base 的生成题由 GPU0 跑。
#
# 顺序：MoE 冒烟(门禁) -> MoE 内部1k -> MoE LexRubric -> MoE 客观11400 -> MoE 生成2750
# 结束标志：MARKER_GPU1_QUEUE_DONE / MARKER_GPU1_QUEUE_MOE_SMOKE_FAIL
# ===========================================================================
cd /mnt/data/lidian/law-agent || exit 1
mkdir -p logs/eval docs/eval docs/moe outputs/infer

# ★ 独占标记：并发跑两份队列会让两个进程写同一个 answers.jsonl。
#   单行 JSON 含 reference（截断 4000 字）可比 4KB 大，超过 PIPE_BUF 后 O_APPEND 不再原子
#   → 行交错、JSONL 损坏（本项目明令禁区）。
#   守卫脚本 scripts/eval/gpu1_after_gate.sh 用**这个 PID 文件**做去重判断
#   —— 只查指定 PID 的 /proc/<pid>/cmdline，**不用 pgrep -f**（后者会被自身命令行误匹配）。
echo $$ > logs/_run/gpu1_queue.pid
trap 'rm -f logs/_run/gpu1_queue.pid' EXIT

P=envs/main/bin/python
D=data/benchmark/lexeval/repo/data
OBJ="$D/1_1.json,$D/1_2.json,$D/1_3.json,$D/2_1.json,$D/2_2.json,$D/2_3.json,$D/2_4.json,$D/2_5.json,$D/3_1.json,$D/3_2.json,$D/3_3.json,$D/3_4.json,$D/3_5.json,$D/3_6.json,$D/4_1.json,$D/4_2.json,$D/6_1.json,$D/6_2.json,$D/6_3.json"
GEN="$D/5_1.json,$D/5_2.json,$D/5_3.json,$D/5_4.json"
GATE=models/moe/L2_gate/gate_weights.pt
GPU=1

echo "=== GPU1 队列启动 $(date '+%F %T') ==="
if [ ! -f "$GATE" ]; then
  echo "ABORT: 门控权重不存在 $GATE"
  echo "MARKER_GPU1_QUEUE_FAIL"
  exit 2
fi
ls -la "$GATE"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ---------------------------------------------------------------- Q1 冒烟
echo ""
echo "=== [Q1] MoE 冒烟 2 题（门禁：验证 --moe-gate 加载路径） $(date '+%T') ==="
rm -rf outputs/infer/_moe_smoke
$P scripts/train/run_inference.py --task internal_test \
  --moe-gate "$GATE" --gpu $GPU --limit 2 --batch-size 2 --max-new-tokens 256 \
  --out outputs/infer/_moe_smoke --no-resume \
  --report-json docs/eval/SMOKE_MOE_internal.json > logs/eval/Q1_moe_smoke.log 2>&1
Q1=$?
echo "Q1 rc=$Q1"
sed 's/\r/\n/g' logs/eval/Q1_moe_smoke.log | grep -vE '^\s*$|Loading weights' | tail -28
if [ $Q1 -ne 0 ]; then
  echo "MoE 加载/生成路径不通 —— 后续 MoE 阶段跳过"
  echo "MARKER_GPU1_QUEUE_MOE_SMOKE_FAIL"
  MOE_OK=0
else
  MOE_OK=1
  echo "MoE 冒烟通过"
fi

# ------------------------------------------------------- Q2 MoE 内部验证集
if [ $MOE_OK -eq 1 ]; then
  echo ""
  echo "=== [Q2] MoE 内部验证集 1k（与 A0 同口径，出分域成绩） $(date '+%T') ==="
  $P scripts/train/run_inference.py --task internal_test \
    --moe-gate "$GATE" --gpu $GPU --batch-size 32 --max-new-tokens 1024 \
    --system-name moe_L2 --out outputs/infer/internal_moe_L2 \
    --report-json docs/eval/FULL_INFER_internal_moe_L2.json \
    --report-md docs/eval/FULL_INFER_internal_moe_L2.md > logs/eval/Q2_moe_internal.log 2>&1
  echo "Q2 rc=$? / 已生成 $(wc -l < outputs/infer/internal_moe_L2/answers.jsonl 2>/dev/null || echo 0) 条"
  tail -3 logs/eval/Q2_moe_internal.log
fi

# --------------------------------------------------- Q3 MoE LexRubric 649
if [ $MOE_OK -eq 1 ]; then
  echo ""
  echo "=== [Q3] MoE LexRubric 649 (cap 1536) $(date '+%T') ==="
  $P scripts/train/run_inference.py --task lexrubric \
    --moe-gate "$GATE" --gpu $GPU --batch-size 32 --max-new-tokens 1536 \
    --system-name moe_L2 --out outputs/infer/lexrubric_moe_L2 \
    --report-json docs/eval/FULL_INFER_lexrubric_moe_L2.json \
    --report-md docs/eval/FULL_INFER_lexrubric_moe_L2.md > logs/eval/Q3_moe_lexrubric.log 2>&1
  echo "Q3 rc=$? / 已生成 $(wc -l < outputs/infer/lexrubric_moe_L2/answers.jsonl 2>/dev/null || echo 0) / 649"
  tail -3 logs/eval/Q3_moe_lexrubric.log
fi

# ------------------------------------------- Q4 MoE LexEval 客观题 11400
if [ $MOE_OK -eq 1 ]; then
  echo ""
  echo "=== [Q4] MoE LexEval 客观题 11400 (cap 256) $(date '+%T') ==="
  $P scripts/train/run_inference.py --task lexeval \
    --moe-gate "$GATE" --gpu $GPU --batch-size 32 --max-new-tokens 256 \
    --input "$OBJ" --system-name moe_L2 --out outputs/infer/lexeval_moe_L2 \
    --report-json docs/eval/FULL_INFER_lexeval_obj_moe_L2.json \
    --report-md docs/eval/FULL_INFER_lexeval_obj_moe_L2.md > logs/eval/Q4_moe_lexeval_obj.log 2>&1
  echo "Q4 rc=$? / 已生成 $(wc -l < outputs/infer/lexeval_moe_L2/answers.jsonl 2>/dev/null || echo 0) / 11400"
  tail -3 logs/eval/Q4_moe_lexeval_obj.log
fi

# ------------------------------------------- Q5 MoE LexEval 生成题 2750
if [ $MOE_OK -eq 1 ]; then
  echo ""
  echo "=== [Q5] MoE LexEval 生成题 2750 (cap 1536) $(date '+%T') ==="
  $P scripts/train/run_inference.py --task lexeval \
    --moe-gate "$GATE" --gpu $GPU --batch-size 32 --max-new-tokens 1536 \
    --input "$GEN" --system-name moe_L2 --out outputs/infer/lexeval_moe_L2 \
    --report-json docs/eval/FULL_INFER_lexeval_gen_moe_L2.json \
    --report-md docs/eval/FULL_INFER_lexeval_gen_moe_L2.md > logs/eval/Q5_moe_lexeval_gen.log 2>&1
  echo "Q5 rc=$? / 已生成 $(wc -l < outputs/infer/lexeval_moe_L2/answers.jsonl 2>/dev/null || echo 0) / 13900(累计)"
  tail -3 logs/eval/Q5_moe_lexeval_gen.log
fi

# ------------------------------------------- Q6 A0 内部验证集统一 cap 重跑
# ★ 2026-09-20 21:00 发现的口径不一致（必须修，否则内部集那张 5 系统对比表不可比）：
#   A0 的 internal_test 是早先按 `--max-new-tokens 512` 跑的
#   （见 docs/eval/FULL_INFER_internal_A0.json 的 config），
#   而本队列 Q2 的 moe_L2 与 overnight_run.sh 里三专家的 internal_test 都用 **1024**。
#   实测 A0 那份 1000 条里 output_tokens 的 p99 = max = 512（确有样本顶到上限），
#   且 gold 字符数 p90=503 / p99=1099 → **512 会系统性截断**，1024 才够。
#   ⇒ 统一到 1024，A0 重跑一遍；旧答案按项目约定归档为 answers.cap512_STALE.jsonl（不删）。
#   放在队列末尾（Q5 之后），不拖慢本文主方法 MoE 那一行。
echo ""
echo "=== [Q6] A0 内部验证集重跑（统一 cap=1024，修 512/1024 口径不一致） $(date '+%T') ==="
rm -rf outputs/infer/A0_internal_test_c1024
$P scripts/train/run_inference.py --task internal_test \
  --adapter models/adapters/A0_unified_qwen3_8b --gpu $GPU \
  --batch-size 32 --max-new-tokens 1024 --no-resume \
  --system-name A0_unified_qwen3_8b --out outputs/infer/A0_internal_test_c1024 \
  --report-json docs/eval/FULL_INFER_internal_A0_c1024.json \
  --report-md docs/eval/FULL_INFER_internal_A0_c1024.md > logs/eval/Q6_a0_internal_c1024.log 2>&1
Q6=$?
NEW=$(wc -l < outputs/infer/A0_internal_test_c1024/answers.jsonl 2>/dev/null || echo 0)
echo "Q6 rc=$Q6 / 已生成 $NEW 条"
tail -3 logs/eval/Q6_a0_internal_c1024.log
if [ "$Q6" -eq 0 ] && [ "$NEW" -ge 900 ]; then
  # 原子换位：旧答案归档（不删），新答案落到规范路径，下游收集/打分只认那一个路径
  if [ -f outputs/infer/A0_internal_test/answers.jsonl ]; then
    mv -f outputs/infer/A0_internal_test/answers.jsonl \
          outputs/infer/A0_internal_test/answers.cap512_STALE.jsonl
  fi
  cp -f outputs/infer/A0_internal_test_c1024/answers.jsonl \
        outputs/infer/A0_internal_test/answers.jsonl
  echo "Q6 口径统一完成：A0_internal_test 现为 cap=1024（旧文件已归档 cap512_STALE）"
  echo "MARKER_A0_INTERNAL_C1024_OK"
else
  echo "MARKER_A0_INTERNAL_C1024_FAIL（保留 512 旧答案，未做换位）"
fi

echo ""
echo "=== 产出清单 ==="
for d in internal_moe_L2 lexrubric_moe_L2 lexeval_moe_L2 A0_internal_test A0_internal_test_c1024; do
  printf "  %-26s %s\n" "$d" "$(wc -l < outputs/infer/$d/answers.jsonl 2>/dev/null || echo 0)"
done
echo "MARKER_GPU1_QUEUE_DONE $(date '+%F %T')"
