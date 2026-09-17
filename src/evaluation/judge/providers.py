"""具体 Judge 提供方实现。

覆盖 `configs/judge.yaml` 里的两类 provider：
  * provider: google              -> GeminiJudge（google-genai SDK）
  * provider: openai_compatible   -> OpenAICompatJudge（openai SDK，DeepSeek 走这条）

SDK 全部延迟导入：没装 SDK 也能 import 本模块跑单测。
"""
from __future__ import annotations

import os
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
    """任何 OpenAI 兼容端点（DeepSeek、vLLM、本地 8000 端口等）。"""

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
        self._client = None

    def _ensure(self):
        if self._client is None:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise RuntimeError(f"环境变量 {self.api_key_env} 未设置")
            base = os.environ.get(self.base_url_env, "").strip() or self.base_url_default
            from openai import OpenAI  # 延迟导入
            self._client = OpenAI(api_key=key, base_url=base, timeout=self.timeout_s)
        return self._client

    def _invoke(self, prompt: str) -> tuple[str, dict[str, Any]]:
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        resp = client.chat.completions.create(**kwargs)
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
