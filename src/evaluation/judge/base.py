"""Judge 基类：统一的调用、重试、JSON 解析与结果封装。

与 `configs/judge.yaml` 的 `protocol` 段对应：
    output_format: json
    retry_on_parse_error: true
    max_retries: 3

铁律：重试**只重发请求**，绝不修改被测答案的内容。
"""
from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .jsonparse import JsonParseError, extract_json


@dataclass
class JudgeCall:
    """一次 Judge 调用的完整记录（含原始输出，便于事后审计）。"""

    judge_name: str
    case_id: str
    ok: bool
    attempts: int
    latency_s: float
    raw_text: str = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None
    prompt_sha256: str = ""
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseJudge(abc.ABC):
    """所有 Judge 的公共外壳。

    子类只需实现 `_invoke(prompt) -> (text, usage)`。
    """

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_sleep_s: float = 3.0,
        timeout_s: float = 180.0,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_sleep_s = retry_sleep_s
        self.timeout_s = timeout_s

    # ---------- 子类实现 ----------

    @abc.abstractmethod
    def _invoke(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """发一次请求，返回 (回复文本, usage 信息)。"""

    # ---------- 公共逻辑 ----------

    def ask_json(self, prompt: str, case_id: str, salt: str = "") -> JudgeCall:
        """发请求并解析为 JSON；解析失败按配置重试。"""
        import hashlib

        p_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        last_err: str | None = None
        last_raw = ""
        t0 = time.time()
        usage: dict[str, Any] = {}

        for attempt in range(1, self.max_retries + 1):
            # 重试时对提示词做轻微扰动提示，避免命中缓存得到同样的坏输出。
            # 注意：绝不改动被测答案本身。
            p = prompt if attempt == 1 else (
                prompt + "\n\n[重试提醒] 上一次输出无法被解析为 JSON。"
                "请只输出一个合法 JSON 对象，不要包含 markdown 代码围栏或任何解释文字。"
            )
            try:
                text, usage = self._invoke(p)
                last_raw = text or ""
                parsed = extract_json(text)
                if not isinstance(parsed, dict):
                    parsed = {"_value": parsed}
                return JudgeCall(
                    judge_name=self.name, case_id=case_id, ok=True, attempts=attempt,
                    latency_s=round(time.time() - t0, 3), raw_text=last_raw,
                    parsed=parsed, prompt_sha256=p_sha, model=self.model, usage=usage,
                )
            except JsonParseError as e:
                last_err = f"JsonParseError: {e}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep_s * attempt)

        return JudgeCall(
            judge_name=self.name, case_id=case_id, ok=False, attempts=self.max_retries,
            latency_s=round(time.time() - t0, 3), raw_text=last_raw,
            error=last_err, prompt_sha256=p_sha, model=self.model, usage=usage,
        )

    # ---------- 落盘 ----------

    @staticmethod
    def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
        """把一条判定结果追加写入 JSONL（不覆盖，便于断点续跑）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load_done_case_ids(path: str | Path, judge_name: str | None = None) -> set[str]:
        """读取已有 JSONL，返回已成功评过的 case_id 集合（用于续跑）。"""
        done: set[str] = set()
        p = Path(path)
        if not p.exists():
            return done
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not rec.get("ok"):
                    continue
                if judge_name and rec.get("judge_name") != judge_name:
                    continue
                cid = rec.get("case_id")
                if cid:
                    done.add(cid)
        return done
