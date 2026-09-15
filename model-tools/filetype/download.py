"""下载 Google Magika standard_v3_3 模型（Apache-2.0，可商用）。

三个文件均托管在 google/magika GitHub 仓库：

- model.onnx                          （≈3.1 MB，ONNX 推理图，输入 int64 [1,2048]）
- config.min.json                     （超参 + 阈值 + overwrite_map）
- content_types_kb.min.json           （在 python/src/magika/config/，约 45 KB）

下载后落到 ``model-tools/filetype/model/magika_v3/`` 目录。
默认 source = direct（GitHub raw 直链）。

实现说明：三个文件共用同一 ``magika_v3`` 目录，但 ``DownloadSpec`` 的
``skip_if_exists`` 会基于目录是否非空判定，导致第二次 spec 进入时被跳过。
因此本脚本不走 ``download_model``，而是逐文件 ``_direct_download``，并
由 ``required_files`` 控制是否跳过单个文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DownloadSource,
    cap_models_dir,
    ensure_dir,
    info,
    ok,
    step,
    warn,
)
from common.downloader import _direct_download


MAGIKA_BASE = "https://github.com/google/magika/raw/main"

MAGIKA_FILES: list[tuple[str, str, str | None]] = [
    (
        f"{MAGIKA_BASE}/assets/models/standard_v3_3/model.onnx",
        "model.onnx",
        None,
    ),
    (
        f"{MAGIKA_BASE}/assets/models/standard_v3_3/config.min.json",
        "config.min.json",
        None,
    ),
    (
        f"{MAGIKA_BASE}/python/src/magika/config/content_types_kb.min.json",
        "content_types_kb.min.json",
        "2788e78d638b1bff0a0743d6d2ee582bbbbca6f0c6dc2be48d62ada9af23f760",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Google Magika 文件类型检测模型 (Apache-2.0)")
    parser.add_argument(
        "--source",
        default=DownloadSource.DIRECT.value,
        choices=[s.value for s in DownloadSource],
        help="下载来源（默认 direct，从 Google Magika GitHub 下载）",
    )
    args = parser.parse_args()

    if args.source != DownloadSource.DIRECT.value:
        warn(f"source={args.source} 暂未适配，本脚本只支持 direct")

    target_dir = ensure_dir(cap_models_dir("filetype") / "magika_v3")
    step(f"开始下载 Google Magika standard_v3_3 -> {target_dir}")

    for url, fname, sha in MAGIKA_FILES:
        dst = target_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            info(f"  已存在，跳过: {dst}")
            continue
        _direct_download(url, target_dir, sha256=sha, archive="none")

    ok("Magika 下载完成。下一步：python convert.py 校验并拷贝到 model/out/。")


if __name__ == "__main__":
    main()