#!/bin/bash
# =============================================================================
# 守卫：等门控权重落盘 → 自动接手跑 GPU1 评测队列
#
# 为什么要它（2026-09-20 事故复盘）
# ---------------------------------------------------------------------------
# `logs/_run/gate_run.sh` 把"[3] 接续评测队列"写在**链的最后一步**。
# 那次门控训练被外部打断 → train_moe_gate rc≠0 → 链直接打 `MARKER_GATE_CHAIN_FAIL`
# 退出，**[3] 永远不会执行**，于是 MoE 那一整行评测被**静默落下**（日志里只有一句 FAIL，
# 而 GPU 上又确实有训练在跑，很容易误判成"一切正常"）。
#
# 教训：**"训练完自动接评测"不该依赖链条自己活着**。这里把这份职责独立成守卫：
#   只要 `gate_weights.pt` 出现就接手，与链的生死无关。
#
# 去重：只认 `logs/_run/gpu1_queue.pid` 这个 PID 文件（不用 pgrep -f，
#       否则会被本脚本自身命令行里的 "gpu1_queue.sh" 字面量误匹配）。
#
# 标志：MARKER_GATE_QUEUE_DONE / MARKER_GATE_QUEUE_ALREADY_RUNNING
#       / MARKER_GATE_WEIGHTS_MISSING / MARKER_GATE_WEIGHTS_TIMEOUT
# =============================================================================
cd /mnt/data/lidian/law-agent || exit 1
set +e

GATE=models/moe/L2_gate/gate_weights.pt
PIDF=logs/_run/gpu1_queue.pid
log(){ echo "[$(date '+%F %T')] $*"; }

log "守卫启动：等 $GATE（最长 12h）"
i=0
while [ $i -lt 8640 ]; do
  [ -f "$GATE" ] && break
  if ! pgrep -f "[t]rain_moe_gate" >/dev/null; then
    log "训练进程已消失，宽限 120s 等写盘 ..."
    sleep 120
    [ -f "$GATE" ] && break
    log "[WARN] 训练进程消失且权重未出现 —— 放弃守卫"
    echo "MARKER_GATE_WEIGHTS_MISSING"
    exit 3
  fi
  i=$((i + 1))
  sleep 5
done

if [ ! -f "$GATE" ]; then
  log "[WARN] 等权重超时"
  echo "MARKER_GATE_WEIGHTS_TIMEOUT"
  exit 3
fi
log "权重已出现：$(ls -la "$GATE")"

# ---- 去重：已有队列在跑就不重复起（保护 answers.jsonl 不被双写） ----
if [ -f "$PIDF" ]; then
  OTHER=$(cat "$PIDF" 2>/dev/null)
  if [ -n "$OTHER" ] && [ -d "/proc/$OTHER" ] && \
     tr '\0' ' ' < "/proc/$OTHER/cmdline" 2>/dev/null | grep -q "gpu1_queue"; then
    log "[SKIP] 已有 gpu1_queue 在跑（pid=$OTHER）—— 守卫退出，交给它"
    echo "MARKER_GATE_QUEUE_ALREADY_RUNNING"
    exit 0
  fi
  log "pidfile 里的 pid=$OTHER 已不存活，清理后继续"
  rm -f "$PIDF"
fi

log "启动 GPU1 评测队列"
bash logs/_run/gpu1_queue.sh
rc=$?
log "gpu1_queue rc=$rc"
echo "MARKER_GATE_QUEUE_DONE rc=$rc"
