"""下载 SenseVoiceSmall（多语种 ASR 模型）。

ModelScope ID: iic/SenseVoiceSmall
官方 ONNX 化流程见 https://github.com 的 SenseVoice-ONNX 项目。
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

MODEL_SCOPE_ID = "iic/SenseVoiceSmall"
REQUIRED_FILES = (
    "configuration.json",
    "model.pt",  # 上游 PyTorch 权重
)


def build_spec() -> DownloadSpec:
    return DownloadSpec(
        target_subdir="SenseVoiceSmall",
        modelscope_id=MODEL_SCOPE_ID,
        required_files=REQUIRED_FILES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 SenseVoiceSmall")
    parser.add_argument(
        "--source",
        default=DownloadSource.MODELSCOPE.value,
        choices=[s.value for s in DownloadSource],
    )
    args = parser.parse_args()

    step(f"开始下载 SenseVoiceSmall / source={args.source}")
    info(f"目标目录: {cap_models_dir('voice') / 'SenseVoiceSmall'}")
    download_model("voice", build_spec(), source=DownloadSource(args.source))
    ok("SenseVoiceSmall 下载完成。下一步：python convert.py 导出 ONNX。")


if __name__ == "__main__":
    main()
