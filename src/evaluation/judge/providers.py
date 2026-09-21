"""具体 Judge 提供方实现。

覆盖 `configs/judge.yaml` 里的两类 provider：
  * provider: google              -> GeminiJudge（google-genai SDK）
  * provider: openai_compatible   -> OpenAICompatJudge（openai SDK，DeepSeek 走这条）

SDK 全部延迟导入：没装 SDK 也能 import 本模块跑单测。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from .base import BaseJudge


class GeminiJudge(BaseJudge):
    """Google Gemini 系列（如 gemini-2.5-pro）。"""

    def __init__(self, name: str, model: str, api_key_env: str = "GEMINI_API_KEY", **kw: Any) -> None:
        super().__init__(name=name, model=model, **kw)
        self.api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise RuntimeError(f"环境变量 {self.api_key_env} 未设置")
            from google import genai  # 延迟导入
            self._client = genai.Client(api_key=key)
        return self._client

    def _invoke(self, prompt: str) -> tuple[str, dict[str, Any]]:
        client = self._ensure()
        from google.genai import types  # 延迟导入

        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        text = getattr(resp, "text", None) or ""
        usage: dict[str, Any] = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "completion_tokens": getattr(um, "candidates_token_count", None),
                "total_tokens": getattr(um, "total_token_count", None),
            }
        return text, usage


class OpenAICompatJudge(BaseJudge):
    """任何 OpenAI 兼容端点（DeepSeek、vLLM、MiniMax、本地 8000 端口等）。

    ★ 多 key 轮换（2026-09-21 事故后新增）
      事故：base 全量判分用了一把「Token Plan 已耗尽」的 key 一路跑下去，
      judgments 里 31.8% 的 rubric 判定是 429 失败后按「未满足」计入，
      导致 A0(0.2%) / MoE(4.1%) / base(31.8%) 失败率差 20 倍、系统间不可比。

      做法：`api_key_env` 是**主 key 的变量名**；若同名加后缀 `_2`/`_3`/…（到 `_9`）
      也存在，则一并纳入轮换。任一 key 撞限流 / 配额 / 传输错误，就换下一把 key
      **重发同一条请求** —— 提示词、模型、temperature、max_tokens 全部不变，
      只有「用哪把 key 发出去」不同，因此**判分口径零变化**。
      失败 key 进入递进冷却（15min 起、每次翻倍、6h 封顶），冷却期内不再作首选；
      只有所有 key 都在冷却时才回头试最早到期的那个。
      单 key 场景与改造前行为一致（仅仅多了一层异常包裹）。
    """

    KEY_COOLDOWN_BASE_S = 900.0
    KEY_COOLDOWN_MAX_S = 6 * 3600.0
    MAX_KEY_SLOTS = 9

    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url_env: str = "DEEPSEEK_BASE_URL",
        base_url_default: str = "https://api.deepseek.com",
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kw)
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.base_url_default = base_url_default
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self._clients: dict[int, Any] = {}
        self._keys: list[str] = []
        self._key_idx = 0
        self._mute_until: dict[int, float] = {}
        self._fail_streak: dict[int, int] = {}
        self._lock = threading.RLock()   # _load_keys/_client_for 会互相调用，必须可重入

    # ---------- key 装载与轮换 ----------

    def _load_keys(self) -> list[str]:
        """主 key + `_2`/`_3`/… 依次读取，去重后返回（结果缓存）。"""
        if self._keys:
            return self._keys
        names = [self.api_key_env] + [
            "%s_%d" % (self.api_key_env, i) for i in range(2, self.MAX_KEY_SLOTS + 1)
        ]
        keys: list[str] = []
        for n in names:
            v = os.environ.get(n, "").strip()
            if v and v not in keys:
                keys.append(v)
        if not keys:
            raise RuntimeError(f"环境变量 {self.api_key_env} 未设置")
        with self._lock:
            if not self._keys:
                self._keys = keys
        return self._keys

    def _client_for(self, idx: int):
        with self._lock:
            c = self._clients.get(idx)
            if c is None:
                keys = self._load_keys()
                base = os.environ.get(self.base_url_env, "").strip() or self.base_url_default
                from openai import OpenAI  # 延迟导入
                c = OpenAI(api_key=keys[idx % len(keys)], base_url=base, timeout=self.timeout_s)
                self._clients[idx] = c
            return c

    def _order_keys(self) -> list[int]:
        """本轮尝试顺序：未冷却的 key 优先，其次按冷却到期时间先后。"""
        n = len(self._load_keys())
        with self._lock:
            start = self._key_idx % n
            now = time.time()
            mute = dict(self._mute_until)

        def rot(lst: list[int]) -> list[int]:
            return [i for i in lst if i >= start] + [i for i in lst if i < start]

        hot = [i for i in range(n) if mute.get(i, 0.0) <= now]
        if hot:
            return rot(hot)
        return sorted(range(n), key=lambda i: mute.get(i, 0.0))

    def _penalize(self, idx: int) -> None:
        with self._lock:
            streak = self._fail_streak.get(idx, 0) + 1
            self._fail_streak[idx] = streak
            cool = min(self.KEY_COOLDOWN_BASE_S * (2 ** (streak - 1)), self.KEY_COOLDOWN_MAX_S)
            self._mute_until[idx] = time.time() + cool
            self._key_idx = idx          # 下次从下一把开始（_order_keys 会 rotate）

    def _reward(self, idx: int) -> None:
        with self._lock:
            self._key_idx = idx
            self._fail_streak[idx] = 0
            self._mute_until.pop(idx, None)

    def key_status(self) -> list[dict[str, Any]]:
        """各 key 槽位的冷却状态（0 槽=主 key），供日志审计"实际用了哪把 key"。"""
        n = len(self._load_keys())
        now = time.time()
        with self._lock:
            return [
                {"slot": i, "cooldown_s_left": int(max(0.0, self._mute_until.get(i, 0.0) - now))}
                for i in range(n)
            ]

    def _invoke(self, prompt: str) -> tuple[str, dict[str, Any]]:
        self._load_keys()                # 未配置就早失败，别等到发请求
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        last_exc: Exception | None = None
        for idx in self._order_keys():
            try:
                resp = self._client_for(idx).chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001  —— 换 key 重发，最后一次的异常向上抛
                last_exc = e
                self._penalize(idx)
                continue
            self._reward(idx)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage: dict[str, Any] = {}
            if getattr(resp, "usage", None) is not None:
                u = resp.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
            return text, usage
        assert last_exc is not None
        raise last_exc


def build_judges(judge_cfgs: list[dict[str, Any]], **common: Any) -> dict[str, BaseJudge]:
    """按 configs/judge.yaml 的 judges 列表构造 Judge 实例。

    common 可传 temperature / max_retries 等公共参数。
    """
    out: dict[str, BaseJudge] = {}
    for jc in judge_cfgs:
        name = jc["name"]
        provider = (jc.get("provider") or "").lower()
        model = jc.get("model") or name
        if provider in ("google", "gemini"):
            out[name] = GeminiJudge(
                name=name, model=model,
                api_key_env=jc.get("api_key_env", "GEMINI_API_KEY"),
                **common,
            )
        elif provider in ("openai_compatible", "openai", "deepseek"):
            out[name] = OpenAICompatJudge(
                name=name, model=model,
                api_key_env=jc.get("api_key_env", "DEEPSEEK_API_KEY"),
                base_url_env=jc.get("base_url_env", "DEEPSEEK_BASE_URL"),
                base_url_default=jc.get("base_url", "https://api.deepseek.com"),
                max_tokens=jc.get("max_tokens"),
                extra_body=jc.get("extra_body"),
                **common,
            )
        else:
            raise ValueError(f"未知 provider: {provider!r}（judge={name}）")
    return out
