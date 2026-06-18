"""下载 Kokoro-82M（v1.1 dynamic/static ONNX）。

ModelScope 镜像，国内速度极快。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DownloadSource,
    DownloadSpec,
    cap_models_dir,
    download_model,
    ok,
    step,
)
from common.progress import info

MODEL_SCOPE_ID = "KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX"
HUGGINGFACE_ID = "KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX"

REQUIRED_FILES = (
    "model_dynamic.onnx",
    "config.json",
    "voices/af.bin",  # 抽样校验 voices 目录
)


def build_spec() -> DownloadSpec:
    return DownloadSpec(
        target_subdir="kokoro-82m-v1.1-onnx",
        modelscope_id=MODEL_SCOPE_ID,
        huggingface_id=HUGGINGFACE_ID,
        required_files=REQUIRED_FILES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Kokoro-82M（v1.1）")
    parser.add_argument(
        "--source",
        default=DownloadSource.MODELSCOPE.value,
        choices=[s.value for s in DownloadSource],
    )
    args = parser.parse_args()

    step(f"开始下载 Kokoro-82M / source={args.source}")
    info(f"目标目录: {cap_models_dir('tts') / 'kokoro-82m-v1.1-onnx'}")
    download_model("tts", build_spec(), source=DownloadSource(args.source))
    ok("Kokoro 下载完成。下一步：python convert.py 整理 voices 目录。")


if __name__ == "__main__":
    main()
