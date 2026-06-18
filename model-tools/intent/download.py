"""下载 chinese-bert-wwm-ext 预训练模型（用于 intent 微调）。"""

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

MODEL_SCOPE_ID = "AI-ModelScope/chinese-bert-wwm-ext"
HUGGINGFACE_ID = "hfl/chinese-bert-wwm-ext"
REQUIRED_FILES = (
    "config.json",
    "vocab.txt",
)


def build_spec() -> DownloadSpec:
    return DownloadSpec(
        target_subdir="chinese-bert-wwm-ext",
        modelscope_id=MODEL_SCOPE_ID,
        huggingface_id=HUGGINGFACE_ID,
        required_files=REQUIRED_FILES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 chinese-bert-wwm-ext")
    parser.add_argument(
        "--source",
        default=DownloadSource.MODELSCOPE.value,
        choices=[s.value for s in DownloadSource],
    )
    args = parser.parse_args()

    step(f"开始下载 chinese-bert-wwm-ext / source={args.source}")
    info(f"目标目录: {cap_models_dir('intent') / 'chinese-bert-wwm-ext'}")
    download_model("intent", build_spec(), source=DownloadSource(args.source))
    ok("预训练模型下载完成。下一步：python train.py 开始微调。")


if __name__ == "__main__":
    main()
