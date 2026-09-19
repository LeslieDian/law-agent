#!/usr/bin/env bash
# =============================================================================
# 阶段 3 第二遍 b：对**清洗镜像**复扫，verdict 必须 = PASS（★ 门禁证据）
# =============================================================================
# 为什么必须单独跑这一遍：
#   第一遍（prepare_decontaminate.sh）在 normalized 全量上找命中 → 产剔除清单；
#   应用剔除（apply_decontam.py）后再复扫镜像，**复扫 PASS 才是论文可引用的门禁证据**。
#   「第一遍命中不是失败，是流程的一部分」——见 README 10.6。
#
# ★ 必须与第一遍带同一个 `--statutes-near`（默认 audit）：
#   法条流的近似命中按硬约定 4 只记录不剔除，所以镜像里会**仍然存在**这些近似命中。
#   若复扫用 `remove`，会把它们重新算成命中 → 复扫必然 FAIL（假故障）。
#
# 三断言的**数值**必须逐条核对（不能只看 verdict）：
#   1) 复扫 verdict = PASS 且 near_hits.count = 0
#   2) DECONTAM_REMOVED_UIDS_PASS.txt 是**空文件**
#   3) 恒等式：首扫 checked − 复扫 checked == 首扫 unique 剔除 uid 数
#
# 输出：
#   docs/corpus/DECONTAMINATION_REPORT_PASS.json / .md
#   docs/corpus/DECONTAM_REMOVED_UIDS_PASS.txt   （**必须是空文件**）
# =============================================================================
set -euo pipefail

export ROOT="${ROOT:-/mnt/data/lidian/law-agent}"
export PY="${PY:-$ROOT/envs/main/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCS="$ROOT/docs/corpus"
export STATUTES_NEAR="${STATUTES_NEAR:-audit}"

mkdir -p "$DOCS"

echo "=== [1/3] 语法自检 ==="
"$PY" -m py_compile "$HERE/decontaminate.py"

echo "=== [2/3] 复扫清洗镜像（--statutes-near $STATUTES_NEAR） ==="
date '+%Y-%m-%d %H:%M:%S'
"$PY" "$HERE/decontaminate.py" \
  --root "$ROOT" \
  --norm-dir data/corpus/decontaminated \
  --out-json "$DOCS/DECONTAMINATION_REPORT_PASS.json" \
  --out-md   "$DOCS/DECONTAMINATION_REPORT_PASS.md" \
  --out-uids "$DOCS/DECONTAM_REMOVED_UIDS_PASS.txt" \
  --statutes-near "$STATUTES_NEAR"

echo "=== [3/3] 门禁结论 ==="
"$PY" - <<'PYEOF'
import json, os, sys

docs = os.environ["DOCS"]
r = json.load(open(os.path.join(docs, "DECONTAMINATION_REPORT_PASS.json"),
                   encoding="utf-8"))
first = json.load(open(os.path.join(docs, "DECONTAMINATION_REPORT.json"),
                       encoding="utf-8"))
uids = os.path.join(docs, "DECONTAM_REMOVED_UIDS_PASS.txt")
n_uids = sum(1 for x in open(uids, encoding="utf-8") if x.strip())

a = first["checked"]["total_rows"]
b = r["checked"]["total_rows"]
# ★ 注意命名陷阱：`near_hits.unique_train_uids` 实际是**精确+近似合计**的唯一 uid 数
#   （脚本里 hit_uids 由两条路径共同写入）。新报告额外给出 removal_summary，优先用它。
rs = first.get("removal_summary") or {}
removed = rs.get("unique_uids_to_remove", first["near_hits"]["unique_train_uids"])

print("首扫被查行数        =", a)
print("复扫被查行数        =", b)
print("复扫 verdict        =", r["verdict"])
print("复扫 精确命中       =", r["exact_hits"]["count"])
print("复扫 近似命中(剔除) =", r["near_hits"]["count"])
audit = r.get("statutes_near_audit", {})
print("复扫 法条近似(留痕) =", audit.get("count"),
      "(policy=%s，应为只记录不剔除)" % audit.get("policy"))
print("二次剔除清单行数    =", n_uids, "(必须为 0)")
print()
print("恒等式：%d - %d = %d  vs 首扫剔除 uid 数 %d -> %s"
      % (a, b, a - b, removed, "OK" if a - b == removed else "MISMATCH"))

ok = (r["verdict"] == "PASS" and n_uids == 0
      and r["near_hits"]["count"] == 0 and r["exact_hits"]["count"] == 0
      and a - b == removed)
if not ok:
    print("!! 门禁未通过")
    sys.exit(3)
print("门禁通过：复扫 PASS + 清单为空 + 恒等式闭环")
PYEOF

echo "DONE_PREPARE_DECONTAM_PASS"
