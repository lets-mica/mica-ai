"""下载 mica-ai-tts 的 Kokoro-82M 模型。

Kokoro-82M 是 mica-ai-tts 当前唯一支持的 TTS 引擎：
  - 82M 参数，纯 ONNX Runtime 推理
  - 中英双语（默认音色 zf_001/zm_001 中文效果最佳）
  - 完全离线

下载源：
  - ModelScope（默认，国内 CDN，速度 5-10MB/s，免登录）
  - HuggingFace（备选）

用法：
    python download.py                     # 等价 --source modelscope
    python download.py --source modelscope
    python download.py --source huggingface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DownloadSource,
    DownloadSpec,
    download_model,
    step,
)
from common.progress import info

KOKORO_MODEL_SCOPE_ID = "KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX"
KOKORO_HUGGINGFACE_ID = "KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX"

KOKORO_REQUIRED = (
    "model_dynamic.onnx",
    "config.json",
    "voices/af.bin",
)


def _download_kokoro(source: DownloadSource):
    spec = DownloadSpec(
        target_subdir="kokoro-82m-v1.1-onnx",
        modelscope_id=KOKORO_MODEL_SCOPE_ID,
        huggingface_id=KOKORO_HUGGINGFACE_ID,
        required_files=KOKORO_REQUIRED,
    )
    return download_model("tts", spec, source=source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 mica-ai-tts 的 Kokoro-82M 模型")
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="modelscope",
        help="下载源（默认 modelscope，国内 CDN）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step("准备下载 Kokoro-82M TTS 模型")

    source = DownloadSource(args.source)
    info(f"下载源: {source.value}")
    _download_kokoro(source)
    info("下载完成。下一步：python convert.py")


if __name__ == "__main__":
    main()
