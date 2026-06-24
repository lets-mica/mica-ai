"""下载 InsightFace buffalo_l 人脸识别模型包。

InsightFace 官方已经预打包好了 buffalo_l.zip（包含 det_10g.onnx +
w600k_r50.onnx + 2d106det.onnx + genderage.onnx），直接从 GitHub release
下载解压即可，不需要 PyTorch 转换。

默认 source = direct（GitHub release）
可选 source = modelscope（如果 ModelScope 上有该镜像）
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DownloadSource,
    DownloadSpec,
    cap_models_dir,
    download_model,
    fail,
    info,
    ok,
    step,
)
from common.progress import warn


# buffalo_l.zip 官方直链（含完整 4 个 ONNX）
BUFFALO_L_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def build_spec(source: str) -> DownloadSpec:
    if source == DownloadSource.MODELSCOPE.value:
        # ModelScope 上的 insightface buffalo_l 镜像 ID（如果存在）。
        # 注意：ModelScope 上的 ID 可能会变，如果不存在请改用 --source direct。
        return DownloadSpec(
            target_subdir="buffalo_l_raw",
            modelscope_id="damo/cv_resnet_face-recognition_arcface",  # 仅识别部分
            required_files=(),  # 识别模型的 ONNX 文件名未知，由后续 convert.py 自行查找
        )
    # 默认 direct
    return DownloadSpec(
        target_subdir="buffalo_l_raw",
        url=BUFFALO_L_ZIP_URL,
        url_archive="zip",
        required_files=("det_10g.onnx", "w600k_r50.onnx"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 InsightFace buffalo_l 模型包")
    parser.add_argument(
        "--source",
        default=DownloadSource.DIRECT.value,
        choices=[s.value for s in DownloadSource],
        help="下载来源（默认 direct，从 GitHub release 下载 buffalo_l.zip）",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="保留下载的 zip 包（默认解压后删除）",
    )
    args = parser.parse_args()

    step(f"开始下载 InsightFace buffalo_l / source={args.source}")
    info(f"目标目录: {cap_models_dir('face') / 'buffalo_l_raw'}")

    if args.source == DownloadSource.DIRECT.value:
        # DIRECT 模式：先下载 zip，extractall 后 downloader 会自动展平单层目录
        results = download_model(
            "face",
            build_spec(args.source),
            source=DownloadSource.DIRECT,
            skip_if_exists=True,
        )
        target_dir = results[0]
        # downloader 已经解压并删除了 zip，但如果 --keep-zip 由用户手动控制则不在脚本管
        if not args.keep_zip:
            zip_in_target = target_dir / "buffalo_l.zip"
            if zip_in_target.exists():
                zip_in_target.unlink()
                info(f"已清理 zip 包: {zip_in_target}")
    else:
        # ModelScope 模式
        warn("ModelScope 模式只下载识别部分（det + rec 不在同一仓库）")
        warn("建议改用 --source direct 一键下载完整 buffalo_l 包")
        results = download_model(
            "face",
            build_spec(args.source),
            source=DownloadSource(args.source),
            skip_if_exists=True,
        )
        target_dir = results[0]

    ok("buffalo_l 下载完成。下一步：python convert.py 拷贝到 model/out/。")


if __name__ == "__main__":
    main()
