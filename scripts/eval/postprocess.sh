#!/bin/bash
# =============================================================================
# 收尾后处理（幂等，可反复跑）：去重 → 客观打分 → 汇总矩阵 → 出图 → 打包
#
# 为什么单独一个脚本
# ---------------------------------------------------------------------------
# 1) **去重保险**：answers.jsonl 是追加写，万一有两个进程同时写（或人工重跑），
#    会出现同一 case_id 多行 → 打分被重复计数。这里按 case_id 保序去重后才打分。
#    顺便把"坏行/重复行"数记进证据文件，异常可追溯。
# 2) **与过夜链解耦**：链里的 finalize 已经跑过一次；本脚本可以再跑一次覆盖结果，
#    不依赖链活着（链的 finalize 若被打断，这里能补齐）。
# 3) **一步出包**：把该带回本地的东西打成 `logs/artifacts_<ts>.tar.gz`，
#    本地端一条命令下载即可，省掉一长串 --download。
#
# 用法：bash scripts/eval/postprocess.sh
# 标志：MARKER_POSTPROCESS_DONE
# =============================================================================
cd /mnt/data/lidian/law-agent || exit 1
set +e
export PYTHONUTF8=1

P=envs/main/bin/python
O=logs/overnight
mkdir -p "$O" docs/eval docs/figures outputs/score

log(){ echo "[$(date '+%F %T')] $*"; }
nl(){ wc -l < "$1" 2>/dev/null || echo 0; }

# ---------------------------------------------------------------- 1) 去重
dedup(){
  local f=$1
  [ -s "$f" ] || { log "  跳过（文件空）：$f"; return 0; }
  "$P" - "$f" <<'PY'
import json, sys
p = sys.argv[1]
seen, out, bad, dup = set(), [], 0, 0
for line in open(p, encoding="utf-8"):
    s = line.strip()
    if not s:
        continue
    try:
        o = json.loads(s)
    except Exception:
        bad += 1
        continue
    cid = o.get("case_id")
    if cid in seen:
        dup += 1
        continue
    seen.add(cid)
    out.append(s)
with open(p, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print("  dedup %-58s kept=%-6d duplicates=%-5d bad_lines=%d" % (p, len(out), dup, bad))
PY
}

log "=== 1) answers.jsonl 去重（保序，按 case_id）==="
{
  echo ""
  echo "================ DEDUP REPORT $(date '+%F %T') ================"
  for f in outputs/infer/*/answers.jsonl; do
    [ -f "$f" ] && dedup "$f"
  done
} 2>&1 | tee -a docs/eval/EVIDENCE.txt

# ------------------------------------------------- 2) 客观打分（覆盖重跑）
log "=== 2) 客观打分 ==="
for pair in "A0:outputs/infer/lexeval_A0_unified_qwen3_8b" \
            "moe_L2:outputs/infer/lexeval_moe_L2" \
            "base:outputs/infer/lexeval_base"; do
  ans="${pair#*:}/answers.jsonl"; out="${pair%%:*}"
  if [ -s "$ans" ]; then
    "$P" scripts/eval/score_answers.py --task lexeval --answers "$ans" \
       --metrics all --out "outputs/score/lexeval_$out.json" > "$O/score_lexeval_$out.log" 2>&1
    log "  lexeval[$out] rc=$?  $(grep -E '选择题 Accuracy|ROUGE-L' "$O/score_lexeval_$out.log" | tr '\n' ' ')"
  else
    log "  lexeval[$out] 跳过（answers 空）"
  fi
done

for pair in "A0:outputs/infer/A0_internal_test" \
            "moe_L2:outputs/infer/internal_moe_L2" \
            "criminal:outputs/infer/internal_criminal" \
            "civil:outputs/infer/internal_civil" \
            "procedure:outputs/infer/internal_procedure"; do
  ans="${pair#*:}/answers.jsonl"; out="${pair%%:*}"
  if [ -s "$ans" ]; then
    "$P" scripts/eval/score_answers.py --task internal_test --answers "$ans" --metrics all \
       --group-by-domain data/test/test.jsonl \
       --out "outputs/score/internal_$out.json" > "$O/score_internal_$out.log" 2>&1
    log "  internal[$out] rc=$?  $(grep -E 'ROUGE-L|法条引用命中率' "$O/score_internal_$out.log" | tr '\n' ' ')"
  else
    log "  internal[$out] 跳过（answers 空）"
  fi
done

# --------------------------------------------------------- 3) 汇总 + 出图
log "=== 3) 汇总矩阵 ==="
"$P" scripts/eval/collect_results.py > "$O/collect.log" 2>&1
log "collect rc=$?  $(grep MARKER_COLLECT_DONE "$O/collect.log")"
sed 's/\r/\n/g' "$O/collect.log" | grep -E '^\[ok\]|^\[skip\]'

log "=== 4) 出图 ==="
"$P" scripts/eval/make_figures.py > "$O/figures.log" 2>&1
log "figures rc=$?  $(grep MARKER_FIGURES_DONE "$O/figures.log")"
sed 's/\r/\n/g' "$O/figures.log" | grep -E '^\[OK\]|^\[FAIL\]|^\[skip\]'

# ------------------------------------------------------------- 5) 打包
log "=== 5) 打包产物 ==="
TS=$(date '+%Y%m%d_%H%M')
TAR="logs/artifacts_${TS}.tar.gz"
LIST=logs/_artifact_list.txt
: > "$LIST"
add(){ for p in "$@"; do [ -e "$p" ] && echo "$p" >> "$LIST"; done; }
add docs/eval/RESULTS_MATRIX.json docs/eval/RESULTS_MATRIX.md docs/eval/README_BLOCK.md
add docs/eval/EVIDENCE.txt docs/eval/JUDGE_FAILURES.txt
add docs/moe/L2_GATE.json docs/moe/L2_GATE.md docs/moe/ROUTER_L1.json
add docs/moe/VERIFY_MIXTURE_GPU_FP32_36L.json docs/moe/VERIFY_MIXTURE_GPU_4BIT.json
add logs/overnight logs/overnight_chain.log logs/moe_gate_chain.log logs/gpu1_after_gate_chain.log
add docs/figures docs/train outputs/score
for g in docs/eval/FULL_INFER_*.json docs/eval/FULL_INFER_*.md \
         outputs/judge/lexrubric/*/SUMMARY.json outputs/judge/lexrubric/*/meta.json; do
  add $g
done
tar czf "$TAR" -T "$LIST"
log "包大小：$(ls -la "$TAR" | awk '{print $5}') 字节，含 $(tar tzf "$TAR" | wc -l) 个文件"
echo "ARTIFACT_TAR=$TAR"
echo "MARKER_POSTPROCESS_DONE $(date '+%F %T')"
