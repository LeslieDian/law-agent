"""通用 IO 与哈希工具。所有脚本统一从这里取路径和读写函数。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

# 项目根目录（src/common/io.py -> 上溯三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """相对项目根目录拼接绝对路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """计算文件 SHA-256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """计算文本 SHA-256，用于 prompt / 输出指纹。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PUNCT = re.compile(r"[\s\u3000·、，。；：！？（）【】《》“”‘’\-—_/\\|,.!?()\[\]{}<>\"'`~@#$%^&*+=]+")


def normalize_text(text: str) -> str:
    """规范化文本：去标点空白、全角转半角、转小写。用于 L2 级查重。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:            # 全角空格
            continue
        if 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            ch = chr(code - 0xFEE0)
        out.append(ch)
    return _PUNCT.sub("", "".join(out)).lower()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """逐行读取 JSONL，跳过空行。"""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """写入 JSONL，返回写入条数。自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict) -> None:
    """追加单条记录（推理输出用，支持断点续跑）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
