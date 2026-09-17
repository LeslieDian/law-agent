"""评测结果的统计汇总（纯标准库实现，无 numpy 依赖）。

对应 `configs/judge.yaml` 的 `statistics` 段：
    inter_judge: ["pearson", "spearman"]
    ci_method: "bootstrap"
    ci_level: 0.95
    bootstrap_iters: 10000
    paired_test.pairs: [["E0","E5"]]

方法学说明（写论文时需注明）：
  * 所有区间均为 percentile bootstrap，重采样对象是**案例**而非分数，
    这样能保留案例之间的相关性。
  * 成对比较使用 paired bootstrap，即每次重采样同一组案例索引。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
# 基础统计量
# --------------------------------------------------------------------------- #

def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """线性插值分位数，q ∈ [0,1]；输入必须已升序。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _ranks(xs: Sequence[float]) -> list[float]:
    """平均秩（正确处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")
    mx, my = mean(x[:n]), mean(y[:n])
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")
    return pearson(_ranks(x[:n]), _ranks(y[:n]))


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    values: Sequence[float],
    iters: int = 10000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """单样本均值的 percentile bootstrap 置信区间。返回 (lo, hi)。"""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    alpha = (1.0 - level) / 2.0
    return percentile(stats, alpha), percentile(stats, 1.0 - alpha)


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_gain: float
    ci_lo: float
    ci_hi: float
    p_value: float
    win: int
    tie: int
    loss: int

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "mean_gain": round(self.mean_gain, 4),
            "ci95": [round(self.ci_lo, 4), round(self.ci_hi, 4)],
            "p_value": round(self.p_value, 6),
            "win_tie_loss": [self.win, self.tie, self.loss],
        }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    iters: int = 10000,
    level: float = 0.95,
    seed: int = 42,
    tol: float = 1e-9,
) -> PairedResult:
    """成对比较：b - a 的均值增益及其 bootstrap 区间与双侧 p 值。

    p 值口径：零假设「增益均值为 0」下，重采样增益分布中
    与 0 距离不小于观测值的比例（标准 paired bootstrap 置换近似）。
    """
    n = min(len(a), len(b))
    if n == 0:
        return PairedResult(0, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), 0, 0, 0)

    diffs = [b[i] - a[i] for i in range(n)]
    obs = mean(diffs)

    # 中心化后重采样，得到零假设下的增益分布
    centered = [d - obs for d in diffs]
    rng = random.Random(seed)
    boot_centered = []
    boot_raw = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        s_raw = 0.0
        s_cen = 0.0
        for i in idx:
            s_raw += diffs[i]
            s_cen += centered[i]
        boot_raw.append(s_raw / n)
        boot_centered.append(s_cen / n)

    boot_raw.sort()
    alpha = (1.0 - level) / 2.0
    ci_lo = percentile(boot_raw, alpha)
    ci_hi = percentile(boot_raw, 1.0 - alpha)

    ge = sum(1 for v in boot_centered if abs(v) >= abs(obs))
    p = (ge + 1) / (iters + 1)

    win = sum(1 for d in diffs if d > tol)
    loss = sum(1 for d in diffs if d < -tol)
    tie = n - win - loss

    return PairedResult(n, mean(a[:n]), mean(b[:n]), obs, ci_lo, ci_hi, p, win, tie, loss)


# --------------------------------------------------------------------------- #
# 便捷汇总
# --------------------------------------------------------------------------- #

def summarize_scores(scores: Sequence[float], iters: int = 10000, seed: int = 42) -> dict:
    lo, hi = bootstrap_ci(scores, iters=iters, seed=seed)
    return {
        "n": len(scores),
        "mean": round(mean(scores), 4),
        "stdev": round(stdev(scores), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "min": round(min(scores), 4) if scores else None,
        "max": round(max(scores), 4) if scores else None,
    }


def inter_judge_agreement(
    by_judge: dict[str, Sequence[float]],
    case_ids: Iterable[str] | None = None,
) -> dict:
    """计算多 Judge 之间的一致性（Pearson / Spearman 两两配对）。"""
    names = sorted(by_judge.keys())
    out: dict = {"judges": names, "pairwise": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out["pairwise"][f"{a} vs {b}"] = {
                "pearson": round(pearson(by_judge[a], by_judge[b]), 4),
                "spearman": round(spearman(by_judge[a], by_judge[b]), 4),
            }
    return out
