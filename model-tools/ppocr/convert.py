"""把下载后的 PP-OCRv6 原始目录整理为 mica-ai-ppocr 期望的扁平结构。

输入：
    model-tools/ppocr/model/PP-OCRv6_<spec>_det/raw/inference.onnx
    model-tools/ppocr/model/PP-OCRv6_<spec>_rec/raw/inference.onnx
    model-tools/ppocr/model/PP-OCRv6_<spec>_rec/raw/rec_char_dict.txt  (v6 默认无)

输出：
    model-tools/ppocr/model/det/inference.onnx
    model-tools/ppocr/model/rec/inference.onnx
    model-tools/ppocr/model/rec_char_dict.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, ok, step, fail, warn
from common.progress import info

DICT_FILE_NAME = "rec_char_dict.txt"


def find_raw_dirs(root: Path, spec: str) -> tuple[Path | None, Path | None]:
    """定位 ``PP-OCRv6_<spec>_<det|rec>/raw`` 目录。"""
    det_dir = root / f"PP-OCRv6_{spec}_det" / "raw"
    rec_dir = root / f"PP-OCRv6_{spec}_rec" / "raw"

    # 兼容某些包会带中间层（PP-OCRv6_server_det/inference.onnx）
    if not det_dir.exists():
        candidate = root / f"PP-OCRv6_{spec}_det"
        if (candidate / "inference.onnx").exists():
            det_dir = candidate
    if not rec_dir.exists():
        candidate = root / f"PP-OCRv6_{spec}_rec"
        if (candidate / "inference.onnx").exists():
            rec_dir = candidate

    return det_dir if det_dir.exists() else None, rec_dir if rec_dir.exists() else None


def flatten(root: Path, spec: str) -> None:
    src_root = root
    det_raw, rec_raw = find_raw_dirs(src_root, spec)
    if not det_raw or not rec_raw:
        fail(f"未找到 det/rec 原始目录，请先执行 download.py。查找路径：{src_root}")
        raise SystemExit(1)

    out_root = ensure_dir(root / "out")
    det_out = ensure_dir(out_root / "det")
    rec_out = ensure_dir(out_root / "rec")

    step(f"拷贝 det: {det_raw}/inference.onnx -> {det_out}/inference.onnx")
    shutil.copy2(det_raw / "inference.onnx", det_out / "inference.onnx")

    step(f"拷贝 rec: {rec_raw}/inference.onnx -> {rec_out}/inference.onnx")
    shutil.copy2(rec_raw / "inference.onnx", rec_out / "inference.onnx")

    # 字符字典：v6 不一定自带；如缺失给出明确提示
    dict_src = rec_raw / DICT_FILE_NAME
    if not dict_src.exists():
        # 退而求其次，到 rec_raw 的父目录或 rec_raw 同级的 inference.yml 中找
        warn(
            f"未在 {rec_raw} 找到 {DICT_FILE_NAME}。"
            f"mica-ai-ppocr 必需，请从 PaddleOCR/ppocr/utils/ 下载同名文件并放到 {rec_out}"
        )
    else:
        shutil.copy2(dict_src, out_root / DICT_FILE_NAME)
        info(f"拷贝字符字典 -> {out_root / DICT_FILE_NAME}")

    ok(f"整理完成，输出在 {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 PP-OCRv6 目录为 mica-ai-ppocr 期望结构")
    parser.add_argument("--spec", default="server", help="与 download.py 保持一致")
    args = parser.parse_args()

    root = cap_models_dir("ppocr")
    step(f"PP-OCRv6 目录整理，根目录: {root}, spec={args.spec}")
    flatten(root, args.spec)


if __name__ == "__main__":
    main()
