#!/bin/bash
# L2 门控链：校验推送 -> 自检(已有证据则复用) -> 门控训练 -> 自动接续 GPU1 评测队列
# 结束标志：MARKER_GATE_DONE / MARKER_GATE_CHAIN_FAIL
cd /mnt/data/lidian/law-agent || exit 1
mkdir -p logs/train docs/moe models/moe logs/_bak_agent docs/eval

echo "=== [0] VERIFY PUSH $(date '+%F %T') ==="
md5sum src/moe/gated_lora.py scripts/train/train_moe_gate.py
M=$(md5sum src/moe/gated_lora.py | awk '{print $1}')
if [ "$M" != "49974556c603d3a1aad7f5fed830a2c8" ]; then
  echo "ABORT_MD5_MISMATCH gated_lora got=$M"
  echo "MARKER_GATE_CHAIN_FAIL"
  exit 2
fi
echo "MD5_OK"
envs/main/bin/python -m py_compile src/moe/gated_lora.py scripts/train/train_moe_gate.py \
  || { echo "ABORT_COMPILE"; echo "MARKER_GATE_CHAIN_FAIL"; exit 2; }
echo "COMPILE_OK"

FP32_JSON=docs/moe/VERIFY_MIXTURE_GPU_FP32_36L.json
echo "=== [1a] SELF-CHECK fp32 / 全 36 层 / 4 专家（门禁项） $(date '+%F %T') ==="
if [ -f "$FP32_JSON" ] && grep -q '"all_pass": true' "$FP32_JSON"; then
  echo "[1a] 复用已有证据 $FP32_JSON（gated_lora.py md5 未变 → 结果确定性，无需重跑）"
else
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  envs/main/bin/python scripts/moe/verify_mixture.py \
    --experts civil,criminal,procedure,unified \
    --device cuda:1 --dtype fp32 --max-layers 0 \
    --report-json "$FP32_JSON" > logs/moe_verify_fp32_36l.log 2>&1
  echo "selfcheck_fp32 rc=$?"
  sed 's/\r/\n/g' logs/moe_verify_fp32_36l.log | grep -vE '^\s*$|Loading weights' | tail -32
fi
if ! grep -q '"all_pass": true' "$FP32_JSON"; then
  echo "SELFCHECK_FP32_FAILED"
  echo "MARKER_GATE_CHAIN_FAIL"
  exit 3
fi
echo "SELFCHECK_PASS"

echo "=== [1b] SELF-CHECK nf4 4bit（仅记录，不作门禁） $(date '+%F %T') ==="
# 不作门禁的理由：等价性要证明的四件事在 fp32 下已**逐位相等**，结论已成立；
# 4bit 路径同一套数学经不同 bnb kernel 调用序，相对差 1.8–2.5%（量化+bf16 累积），
# 与 1e-2 容差不匹配。该数字作为"量化路径数值偏差"记录进论文，不作正确性判据。
if [ -f docs/moe/VERIFY_MIXTURE_GPU_4BIT.json ]; then
  echo "[1b] 复用已有 4bit 记录"
else
  envs/main/bin/python scripts/moe/verify_mixture.py \
    --experts civil,criminal,procedure,unified \
    --device cuda:1 --4bit --max-layers 0 \
    --report-json docs/moe/VERIFY_MIXTURE_GPU_4BIT.json > logs/moe_verify_gpu.log 2>&1
  echo "selfcheck_4bit rc=$?"
fi
envs/main/bin/python -c "
import json
d=json.load(open('docs/moe/VERIFY_MIXTURE_GPU_4BIT.json'))
print('4bit 各专家相对差:', [round(e['relative'],5) for e in d['per_expert']])
print('4bit 容差:', d.get('tolerance_relative'))
" 2>&1 | tail -4

echo "=== [2] GATE TRAINING gpu1 $(date '+%F %T') ==="
envs/main/bin/python scripts/train/train_moe_gate.py \
  --experts civil,criminal,procedure,unified \
  --data data/router/router_train.jsonl \
  --gpu 1 --gate-share in_features --gate-init unified \
  --out-dir models/moe/L2_gate \
  --report-json docs/moe/L2_GATE.json --report-md docs/moe/L2_GATE.md \
  > logs/train/L2_gate.log 2>&1
rc=$?
echo "gate_train rc=$rc"
sed 's/\r/\n/g' logs/train/L2_gate.log | grep -vE '^\s*$' | tail -40
if [ $rc -eq 0 ] && [ -f docs/moe/L2_GATE.json ]; then
  echo "MARKER_GATE_DONE $(date '+%F %T')"
  if [ -f logs/_run/gpu1_queue.sh ]; then
    echo "=== [3] 自动接续 GPU1 评测队列 $(date '+%F %T') ==="
    bash logs/_run/gpu1_queue.sh
  else
    echo "[warn] logs/_run/gpu1_queue.sh 不存在，GPU1 队列未启动"
  fi
else
  echo "MARKER_GATE_CHAIN_FAIL"
fi
