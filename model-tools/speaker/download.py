"""下载 ERes2Net / ERes2NetV2 声纹识别模型。

ModelScope 提供了多个 ERes2Net 系列模型，默认下载 V2（更准）。
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

# ERes2NetV2（推荐） / ERes2Net base 备选
MODEL_SCOPE_VARIANTS: dict[str, str] = {
    "eres2netv2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    "eres2netv2_en": "iic/speech_eres2netv2_sv_en_voxceleb1023_16k",
    "eres2net": "iic/speech_eres2net_sv_zh-cn_16k-common",
    "eres2net_base_200k": "iic/speech_eres2net_base_200k_sv_zh-cn_16k-common",
}


def build_spec(variant: str) -> DownloadSpec:
    if variant not in MODEL_SCOPE_VARIANTS:
        raise SystemExit(f"未知 variant: {variant}，可选 {list(MODEL_SCOPE_VARIANTS.keys())}")
    return DownloadSpec(
        target_subdir=f"{variant}_raw",
        modelscope_id=MODEL_SCOPE_VARIANTS[variant],
        required_files=("configuration.json",),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 ERes2Net 声纹模型")
    parser.add_argument(
        "--variant",
        default="eres2netv2",
        choices=list(MODEL_SCOPE_VARIANTS.keys()),
        help="模型变体（默认 eres2netv2）",
    )
    parser.add_argument(
        "--source",
        default=DownloadSource.MODELSCOPE.value,
        choices=[s.value for s in DownloadSource],
    )
    args = parser.parse_args()

    step(f"开始下载 ERes2Net / variant={args.variant} / source={args.source}")
    info(f"目标目录: {cap_models_dir('speaker') / (args.variant + '_raw')}")
    download_model("speaker", build_spec(args.variant), source=DownloadSource(args.source))
    ok("ERes2Net 下载完成。下一步：python convert.py 导出 ONNX。")


if __name__ == "__main__":
    main()
