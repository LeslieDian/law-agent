"""双盲 Judge 基础设施。

子模块：
  anonymize   —— 盲评（匿名编号、去标识、顺序扰动）
  jsonparse   —— 稳健 JSON 抽取
  base        —— Judge 基类、调用记录、JSONL 落盘与续跑
  providers   —— Gemini / OpenAI 兼容（DeepSeek）实现

注意：本包只提供「怎么问、怎么解析、怎么落盘」，
具体某个基准的提示词与打分口径由各自的 runner 负责，
不同基准的结果**不得混排**。
"""
from .anonymize import DEFAULT_HIDE_LABELS, SAFE_HIDE_LABELS, BlindContext, stable_case_id
from .base import BaseJudge, JudgeCall
from .jsonparse import JsonParseError, extract_json
from .providers import GeminiJudge, OpenAICompatJudge, build_judges

__all__ = [
    "BlindContext",
    "DEFAULT_HIDE_LABELS",
    "SAFE_HIDE_LABELS",
    "stable_case_id",
    "BaseJudge",
    "JudgeCall",
    "JsonParseError",
    "extract_json",
    "GeminiJudge",
    "OpenAICompatJudge",
    "build_judges",
]
