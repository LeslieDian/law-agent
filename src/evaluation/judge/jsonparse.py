"""从 LLM 回复里稳健地抽出 JSON。

Judge 输出经常带 markdown 代码围栏、前后说明文字、或尾随逗号，
直接 json.loads 失败率很高。这里做分层容错解析。
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class JsonParseError(ValueError):
    """所有分层策略都无法解析出 JSON。"""


def _try_load(s: str):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def extract_json(text: str):
    """尽力从文本中解析出 JSON 对象/数组。

    策略依次为：
      1. 整体直接解析
      2. 取 markdown 代码围栏内的内容
      3. 取第一个 '{' 到最后一个 '}'（或 '[' 到 ']'）
      4. 在 3 的基础上容忍尾随逗号

    全部失败则抛 JsonParseError。
    """
    if text is None:
        raise JsonParseError("输入为 None")

    s = text.strip()
    if not s:
        raise JsonParseError("输入为空")

    got = _try_load(s)
    if got is not None:
        return got

    for m in _FENCE_RE.finditer(s):
        got = _try_load(m.group(1).strip())
        if got is not None:
            return got

    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = s.find(opener), s.rfind(closer)
        if i != -1 and j > i:
            chunk = s[i:j + 1]
            got = _try_load(chunk)
            if got is not None:
                return got
            got = _try_load(_TRAILING_COMMA_RE.sub(r"\1", chunk))
            if got is not None:
                return got

    raise JsonParseError(f"无法解析 JSON，前 200 字符: {s[:200]!r}")
