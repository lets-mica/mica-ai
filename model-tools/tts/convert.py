"""整理 Kokoro-82M 模型目录。

ModelScope 仓库的 voices 子目录里同时有 ``.bin`` 和 ``.npy``，但 mica-ai-tts
只需要 ``.bin``。这里把 ``model_dynamic.onnx`` / ``config.json`` / ``voices/*.bin``
扁平化到 ``model/`` 根下，方便 Java 端直接配置。

输入：
    model-tools/tts/model/kokoro-82m-v1.1-onnx/model_dynamic.onnx
    model-tools/tts/model/kokoro-82m-v1.1-onnx/config.json
    model-tools/tts/model/kokoro-82m-v1.1-onnx/voices/*.bin
    model-tools/tts/model/kokoro-82m-v1.1-onnx/voices/*.npy

输出：
    model-tools/tts/model/model_dynamic.onnx
    model-tools/tts/model/config.json
    model-tools/tts/model/voices/*.bin
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, fail, ok, step, warn
from common.progress import info


def flatten(root: Path) -> None:
    src = root / "kokoro-82m-v1.1-onnx"
    if not src.exists():
        fail(f"未找到 {src}，请先执行 download.py")
        raise SystemExit(1)

    out = ensure_dir(root / "out")

    for name in ("model_dynamic.onnx", "config.json"):
        s = src / name
        if not s.exists():
            fail(f"源文件缺失: {s}")
            raise SystemExit(1)
        step(f"拷贝 {name}")
        shutil.copy2(s, out / name)

    # voices/*.bin
    src_voices = src / "voices"
    if not src_voices.exists():
        fail(f"未找到 voices 目录: {src_voices}")
        raise SystemExit(1)

    out_voices = ensure_dir(out / "voices")
    bin_files = sorted(src_voices.glob("*.bin"))
    if not bin_files:
        fail(f"voices 目录下没有 .bin 文件: {src_voices}")
        raise SystemExit(1)

    step(f"拷贝 voices/*.bin （共 {len(bin_files)} 个）")
    for f in bin_files:
        shutil.copy2(f, out_voices / f.name)

    npy_count = sum(1 for _ in src_voices.glob("*.npy"))
    if npy_count:
        warn(f"忽略 {npy_count} 个 .npy 备份文件（mica-ai-tts 不需要）")

    ok(f"整理完成: {out} （voices={len(bin_files)} 个音色）")


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 Kokoro 目录")
    cap_models_dir("tts")  # 触达环境变量校验
    root = cap_models_dir("tts")
    step(f"Kokoro 目录整理，根: {root}")
    flatten(root)


if __name__ == "__main__":
    main()
