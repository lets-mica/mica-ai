"""路径解析工具。

约定：
- 模型根目录默认位于 ``<cap>/model/``，可被环境变量 ``MICA_MODELS_DIR`` 覆盖
- 仓库根 ``mica-ai/`` 通过 ``mica_root()`` 取得，供 ``download.py`` 拼绝对路径
"""

from __future__ import annotations

import os
from pathlib import Path

# mica-ai 的当前版本号（与根 pom.xml 的 <revision> 保持一致）
MICA_VERSION = "2026.06.01"


def mica_root() -> Path:
    """返回仓库根目录 ``mica-ai/`` 的绝对路径。

    ``model-tools/common/paths.py`` 上溯两级即仓库根。
    """
    return Path(__file__).resolve().parents[2]


def cap_models_dir(cap: str, *, env_override: bool = True) -> Path:
    """获取某个能力（ppocr / tts / voice / speaker / intent）的模型根目录。

    优先级：
      1. 环境变量 ``MICA_MODELS_DIR``（若设置，目录为 ``$MICA_MODELS_DIR/<cap>``）
      2. ``<repo>/model-tools/<cap>/model``（默认）
    """
    if env_override and (env_root := os.environ.get("MICA_MODELS_DIR")):
        return Path(env_root).expanduser().resolve() / cap
    return mica_root() / "model-tools" / cap / "model"


def ensure_dir(path: os.PathLike | str) -> Path:
    """创建目录（已存在则跳过），并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
