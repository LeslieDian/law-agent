"""配置加载：YAML + ${ENV} / ${ENV:-default} 展开。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .io import project_path

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: str | Path | None = None, override: bool = False) -> None:
    """极简 .env 加载，避免额外依赖。已存在的环境变量默认不被覆盖。"""
    env_path = Path(path) if path else project_path(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def _expand(node: Any) -> Any:
    """递归展开 ${ENV} 占位符。未定义且无默认值时抛错，避免静默用错密钥。"""
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if not isinstance(node, str):
        return node

    def _sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise KeyError(f"环境变量 {name} 未设置（配置中引用了 ${{{name}}}）")
            return default
        return value

    return _ENV_PATTERN.sub(_sub, node)


def load_config(path: str | Path, use_dotenv: bool = True) -> dict:
    """加载 YAML 配置并展开环境变量。path 可为项目根目录相对路径。"""
    if use_dotenv:
        load_dotenv()
    p = Path(path)
    if not p.is_absolute():
        p = project_path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand(raw)
