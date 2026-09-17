"""双盲（blind）处理工具。

设计目标：让 Judge 无法从输入中推断出候选答案来自哪个系统，
从而避免「系统名偏见」污染评分。

与 `configs/judge.yaml` 的 `blindness` 段一一对应：
    anonymize_system_name: true
    hide_labels: ["本系统", "基础模型", "E0", "E5"]
    shuffle_answer_order: true
    seed: 42
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

# 安全脱敏：只抹掉「自指本系统」的表述。
# 用于法律文本类任务 —— 法律答案里出现「民法典」「刑法」等词必须原样保留，
# 因此**绝不**在这里做模型家族名的泛化替换。
SAFE_HIDE_LABELS: tuple[str, ...] = (
    "本系统", "本文方法", "我们的方法", "基础模型", "本模型",
    "our system", "our method", "our approach", "this system",
)

# 默认需要屏蔽的自指/系统标识词。可由 configs/judge.yaml 覆盖。
DEFAULT_HIDE_LABELS: tuple[str, ...] = (
    "本系统", "基础模型", "本文方法", "我们的方法", "our system", "our method",
    "E0", "E1", "E2", "E3", "E4", "E5",
    "A1", "A2", "A3", "A4", "A5", "A6",
    "Qwen", "GPT", "DeepSeek", "Gemini", "Claude", "GLM", "Kimi", "Baichuan",
    "ChatGLM", "LawGPT", "LexiLaw", "HanFei", "ChatLaw", "通义", "文心",
)

# 模型名/系统名常见的后缀写法。⚠️ 仅在 scrub_model_family=True 时启用，
# 因为它会误伤「 pro 」「 base 」这类可能出现在普通英文里的片段。
_MODEL_SUFFIX_RE = re.compile(
    r"(?i)\b(?:[-_ ]?(?:instruct|chat|base|thinking|preview|turbo|max|flash|pro|8b|7b|13b|14b|32b|70b|0\.6b|4b))\b"
)


@dataclass
class BlindContext:
    """一次盲评会话的上下文，负责稳定的匿名编号与顺序扰动。"""

    seed: int = 42
    hide_labels: tuple[str, ...] = DEFAULT_HIDE_LABELS
    anonymize_system_name: bool = True
    shuffle_answer_order: bool = True
    # 是否额外做「模型家族名 + 规格后缀」的泛化替换。
    # 法律/医学等专业文本任务建议保持 False，避免误伤正文。
    scrub_model_family: bool = False
    _anon_cache: dict[str, str] = field(default_factory=dict, init=False)
    _hidden_found: dict[str, int] = field(default_factory=dict, init=False)

    # ---------- 匿名编号 ----------

    def anon_id(self, system_name: str) -> str:
        """把系统名映射为一个稳定、无信息的编号（如 SYS-7K3Q）。

        用 seed 加盐的哈希保证：
          * 同一系统在整轮评测中拿到同一个编号（可追溯）
          * 编号本身不含任何顺序/规模线索（不可反推）
        """
        if not self.anonymize_system_name:
            return system_name
        if system_name not in self._anon_cache:
            h = hashlib.sha256(f"{self.seed}:{system_name}".encode("utf-8")).hexdigest()
            # 取 4 位大写十六进制，避免形似英文单词
            self._anon_cache[system_name] = "SYS-" + h[:4].upper()
        return self._anon_cache[system_name]

    # ---------- 文本去标识 ----------

    def scrub(self, text: str) -> str:
        """从答案正文里抹掉自指系统名与（可选的）模型家族名。

        只做替换、不改动任何实质内容，且记录命中次数，便于审计。
        """
        if not text:
            return text
        out = text
        for label in self.hide_labels:
            if not label:
                continue
            pattern = re.compile(re.escape(label), re.IGNORECASE)
            n = len(pattern.findall(out))
            if n:
                self._hidden_found[label] = self._hidden_found.get(label, 0) + n
                out = pattern.sub("[已脱敏]", out)
        if self.anonymize_system_name and self.scrub_model_family:
            out = _MODEL_SUFFIX_RE.sub("", out)
        return out

    # ---------- 顺序扰动 ----------

    def order(self, items: list[str], key: str = "") -> list[str]:
        """用确定性随机源打乱候选顺序（同 key 结果可复现）。"""
        if not self.shuffle_answer_order or len(items) <= 1:
            return list(items)
        rng = random.Random(f"{self.seed}:{key}")
        idx = list(range(len(items)))
        rng.shuffle(idx)
        return [items[i] for i in idx]

    @property
    def audit(self) -> dict:
        """审计信息：本次盲评究竟脱敏了什么。"""
        return {
            "seed": self.seed,
            "anonymize_system_name": self.anonymize_system_name,
            "shuffle_answer_order": self.shuffle_answer_order,
            "systems_anonymized": dict(self._anon_cache),
            "labels_hidden_counts": dict(self._hidden_found),
        }


def stable_case_id(dataset: str, index: int, raw_id: str | None = None) -> str:
    """生成跨数据集唯一、可复现的 case id。

    前缀带数据集名，保证 LexRubric 与 LexEval 的 id 永不冲突（两者不可混排）。
    """
    prefix = dataset.strip().lower()
    if raw_id:
        return f"{prefix}::case-{raw_id}"
    return f"{prefix}::case-{index:05d}"
