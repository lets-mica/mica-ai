"""下载 OpenCV Zoo YuNet + SFace 人脸识别模型（Apache-2.0，可商用）。

OpenCV Zoo 把模型以裸 .onnx 文件直接托管在 GitHub，无需转换。
- YuNet:  ~340 KB，人脸检测
- SFace:  ~89 MB，512d Embedding

默认 source = direct（GitHub raw 直链）
可选 source = modelscope / huggingface（如果未来有镜像）
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
    info,
    ok,
    step,
)


# OpenCV Zoo GitHub raw 直链（裸 .onnx，无需解压）
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def build_specs(source: str) -> list[DownloadSpec]:
    if source == DownloadSource.MODELSCOPE.value:
        # OpenCV Zoo 暂无官方 ModelScope 镜像；如未来上架可填 ID
        raise NotImplementedError(
            "OpenCV Zoo 当前在 ModelScope 上暂无镜像，请使用 --source direct 或 huggingface"
        )
    if source == DownloadSource.HUGGINGFACE.value:
        # OpenCV Zoo 暂无官方 HuggingFace 仓库；如未来上架可填 ID
        raise NotImplementedError(
            "OpenCV Zoo 当前在 HuggingFace 上暂无镜像，请使用 --source direct"
        )
    # default: DIRECT，从 GitHub raw 下载
    return [
        DownloadSpec(
            target_subdir="yunet_raw",
            url=YUNET_URL,
            url_archive="none",  # 裸文件，无需解压
            required_files=("face_detection_yunet_2023mar.onnx",),
        ),
        DownloadSpec(
            target_subdir="sface_raw",
            url=SFACE_URL,
            url_archive="none",
            required_files=("face_recognition_sface_2021dec.onnx",),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 OpenCV Zoo 人脸识别模型 (Apache-2.0)")
    parser.add_argument(
        "--source",
        default=DownloadSource.DIRECT.value,
        choices=[s.value for s in DownloadSource],
        help="下载来源（默认 direct，从 OpenCV Zoo GitHub 下载）",
    )
    args = parser.parse_args()

    step(f"开始下载 OpenCV Zoo YuNet + SFace / source={args.source}")
    info(f"目标根目录: {cap_models_dir('face')}")

    specs = build_specs(args.source)
    targets = download_model(
        "face",
        specs,
        source=DownloadSource(args.source),
        skip_if_exists=True,
    )
    for t in targets:
        info(f"  ✓ {t}")

    ok("YuNet + SFace 下载完成。下一步：python convert.py 拷贝到 model/out/。")


if __name__ == "__main__":
    main()