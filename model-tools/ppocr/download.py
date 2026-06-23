"""下载 PP-OCRv6 的检测 + 识别 ONNX 模型。

PP-OCRv6 的官方分发在 Baidu BCE（PaddleX），国内下载速度良好。
对其它 PaddleOCR 系列模型（v3 / v4）也兼容，按需修改 SPEC 即可。

模型尺寸：tiny（默认）、small、medium

用法：
    python download.py            # 下载默认 PP-OCRv6 server
    python download.py --spec tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 ``python download.py`` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DownloadSource, DownloadSpec, cap_models_dir, download_model, ok, step
from common.progress import info

# PaddlePaddle 官方 ONNX 推理包（不同规格）
PADDLE_OCR_URLS: dict[str, dict[str, str]] = {
    "medium": {
        "det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_medium_det_onnx.tar",
        "rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_medium_rec_0515_onnx.tar",
    },
    "small": {
        "det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_small_det_onnx.tar",
        "rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_small_rec_0515_onnx.tar",
    },
    "tiny": {
        "det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_det_onnx.tar",
        "rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_rec_0515_onnx.tar",
    },
}


def build_specs(spec: str) -> list[DownloadSpec]:
    """构造检测 + 识别的两个 DownloadSpec。"""
    if spec not in PADDLE_OCR_URLS:
        raise SystemExit(f"未知 spec: {spec}，可选 {list(PADDLE_OCR_URLS.keys())}")
    urls = PADDLE_OCR_URLS[spec]
    return [
        DownloadSpec(
            target_subdir=f"PP-OCRv6_{spec}_det/raw",
            url=urls["det"],
            url_archive="tar",
            required_files=("inference.onnx",),
        ),
        DownloadSpec(
            target_subdir=f"PP-OCRv6_{spec}_rec/raw",
            url=urls["rec"],
            url_archive="tar",
            required_files=("inference.onnx",),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 PP-OCRv6 模型（det + rec）")
    parser.add_argument(
        "--spec",
        default="tiny",
        choices=list(PADDLE_OCR_URLS.keys()),
        help="模型规格（默认 tiny，与参考项目 ppocrv6_onnx 一致）。"
             "medium / server 需配套不同字典，参见 README。",
    )
    parser.add_argument(
        "--source",
        default=DownloadSource.DIRECT.value,
        choices=[s.value for s in DownloadSource],
        help="下载源（默认 direct，因官方分发不在 ModelScope）",
    )
    args = parser.parse_args()

    step(f"开始下载 PP-OCRv6 / spec={args.spec} / source={args.source}")
    specs = build_specs(args.spec)
    info(f"目标根目录: {cap_models_dir('ppocr')}")
    info(f"将下载 {len(specs)} 个文件: {[s.target_subdir for s in specs]}")

    download_model("ppocr", specs, source=DownloadSource(args.source))
    ok("PP-OCRv6 下载完成。下一步：python convert.py 整理目录。")


if __name__ == "__main__":
    main()
