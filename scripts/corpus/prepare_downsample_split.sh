#!/usr/bin/env bash
# 阶段 4：分层下采样 + train/val/test/路由集切分（幂等包装）。
# 语法自检 -> 切分 -> 质检门禁（硬卡口：四份 split 两两不相交）-> 打印结论。
#
# 用法：
#   bash scripts/corpus/prepare_downsample_split.sh             # 全量（10 万配比）
#   bash scripts/corpus/prepare_downsample_split.sh --dry-run   # 只算配额，不落盘
#   bash scripts/corpus/prepare_downsample_split.sh --task-alpha 0.8
#
# 产物（都是 configs 里写死的路径，本脚本不引入新路径）：
#   data/train/train.jsonl            # 10 万，A0 全域（qlora_unified.yaml）
#   data/train/{civil,criminal,procedure,general}.jsonl   # 分域（adapters_router.yaml）
#   data/dev/dev.jsonl                # 1,000 验证集（qlora_unified.yaml）
#   data/test/test.jsonl              # 1,000 测试集
#   data/router/router_train.jsonl    # 20,000 路由训练集
#   docs/corpus/SPLIT_{STATS,REPORT,VERIFY}.*
set -euo pipefail

ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$ROOT/docs/corpus"

mkdir -p "$DOCS"

echo "=== [1/4] 语法自检 ==="
"$PY" -m py_compile "$HERE/downsample_split.py"
"$PY" -m py_compile "$HERE/verify_split.py"

echo "=== [2/4] 分层下采样 + 切分 ==="
"$PY" "$HERE/downsample_split.py" \
  --root "$ROOT" \
  --out-json "$DOCS/SPLIT_STATS.json" \
  --out-md   "$DOCS/SPLIT_REPORT.md" \
  "$@"

if [[ " $* " == *" --dry-run "* ]]; then
  echo "=== [3/4] 干跑模式，跳过质检与落盘校验 ==="
  echo "DONE_PREPARE_DOWNSAMPLE_SPLIT (dry-run)"
  exit 0
fi

echo "=== [3/4] 质检门禁（硬卡口：四份 split 两两不相交）==="
"$PY" "$HERE/verify_split.py" \
  --root "$ROOT" \
  --expect "$DOCS/SPLIT_STATS.json" \
  --out-json "$DOCS/SPLIT_VERIFY.json" \
  --out-md   "$DOCS/SPLIT_VERIFY.md"

echo "=== [4/4] 结论 ==="
"$PY" - <<'PYEOF'
import json, os
ROOT = os.environ.get("ROOT", "/mnt/data/lidian/law-agent")
s = json.load(open(os.path.join(ROOT, "docs/corpus/SPLIT_STATS.json"), encoding="utf-8"))
v = json.load(open(os.path.join(ROOT, "docs/corpus/SPLIT_VERIFY.json"), encoding="utf-8"))
print("verdict    =", v["verdict"])
print("splits     =", {k: s["splits"][k] for k in ("train", "val", "test", "router")})
print("leftover   =", s["splits"]["leftover"])
for k in ("criminal", "civil", "procedural", "general"):
    x = s["train_domain"][k]
    print("  %-11s %d/%d pool=%d use=%.1f%% maxtask=%.1f%%(%s) cap_relaxed=%s"
          % (k, x["achieved"], x["target"], x["pool_available"], x["pool_usage"],
             x["max_task_share_pct"], x["max_task_share_name"], x["cap_relaxed"]))
print("disjoint   =", {k: (x["uid_g"], x["content_sha1"])
                       for k, x in v["checks"]["C3_disjoint"].items()})
print("statutes   =", s["statutes"].get("total"), "cross_flow_overlap =",
      s["statutes"].get("cross_flow_overlap"))
print("warnings   =", v["warnings"])
print("failures   =", v["failures"])
PYEOF

echo "DONE_PREPARE_DOWNSAMPLE_SPLIT"
