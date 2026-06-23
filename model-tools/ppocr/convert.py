"""把下载后的 PP-OCRv6 原始目录整理为 mica-ai-ppocr 期望的扁平结构。

输入：
    model-tools/ppocr/model/PP-OCRv6_<spec>_det/raw/inference.onnx
    model-tools/ppocr/model/PP-OCRv6_<spec>_rec/raw/inference.onnx
    model-tools/ppocr/model/PP-OCRv6_<spec>_rec/raw/inference.yml

输出：
    model-tools/ppocr/model/out/det/inference.onnx
    model-tools/ppocr/model/out/rec/inference.onnx
    model-tools/ppocr/model/out/rec_char_dict.txt
    model-tools/ppocr/model/out/rec_char_dict_<spec>.txt    # 按 spec 留底

字典来源优先级：
1. 从 inference.yml 的 PostProcess.character_dict 抽取（v6 标准做法）
2. 从 raw/rec_char_dict.txt 拷贝（兼容老式分发包）
3. 仍找不到则在 model-tools/ppocr/ 父目录查找
"""

from __future__ import annotations

import argparse
import re
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


_CHAR_DICT_RE = re.compile(r"character_dict:\n((?:\s*-\s.*\n)+)")
_VOCAB_DIM_RE = re.compile(r"^\s*vocab_size:\s*(\d+)", re.M)


def _yaml_char_dict(text: str) -> list[str]:
    """从 inference.yml 文本中解析 PostProcess.character_dict 列表。"""
    m = _CHAR_DICT_RE.search(text)
    if not m:
        return []
    items: list[str] = []
    for line in m.group(1).splitlines():
        line = line.rstrip()
        if not line.strip().startswith("-"):
            continue
        val = line.split("-", 1)[1].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        items.append(val)
    return items


def _onnx_vocab(onnx_path: Path) -> int | None:
    """读取 onnx 输出最后一维，作为 vocab 维度。"""
    try:
        import onnx  # type: ignore
    except ImportError:
        return None
    try:
        m = onnx.load(str(onnx_path))
    except Exception as e:  # noqa: BLE001
        warn(f"onnx 解析失败 {onnx_path}: {e}")
        return None
    try:
        dim = m.graph.output[0].type.tensor_type.shape.dim[-1]
        return dim.dim_value if dim.dim_value > 0 else None
    except Exception:
        return None


def extract_dict_from_yml(rec_raw: Path) -> list[str] | None:
    """从 rec 模型的 inference.yml 抠出 character_dict。

    失败返回 None（由调用方决定走 fallback）。
    """
    yml = rec_raw / "inference.yml"
    if not yml.is_file():
        return None
    try:
        text = yml.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 部分 PaddleX 包用 GBK 保存过 yml；兜底一次
        try:
            text = yml.read_text(encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None
    items = _yaml_char_dict(text)
    if not items:
        return None
    # 去除可能的空行/None
    return [ch for ch in items if ch is not None]


def write_dict(items: list[str], dst: Path) -> None:
    """把字典项逐行写到目标文件（每行一个字符）。"""
    ensure_dir(dst.parent)
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for ch in items:
            f.write(ch + "\n")


def resolve_dict(
    rec_raw: Path, root: Path, spec: str
) -> tuple[Path | None, list[str] | None, int | None]:
    """按优先级查找字典。

    Returns: (dst_path_or_None, items_or_None, onnx_vocab)
    """
    items = extract_dict_from_yml(rec_raw)
    if items:
        return None, items, None  # 由 caller 决定落盘位置

    # fallback 1: raw 目录下的 rec_char_dict.txt
    fallback = rec_raw / DICT_FILE_NAME
    if fallback.is_file():
        lines = [ln.strip() for ln in fallback.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return fallback, lines, None

    # fallback 2: 模型目录根
    fallback2 = root / f"PP-OCRv6_{spec}_rec" / DICT_FILE_NAME
    if fallback2.is_file():
        lines = [ln.strip() for ln in fallback2.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return fallback2, lines, None

    return None, None, None


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

    # ---- 字典抽取 ----
    default_dict_path = out_root / DICT_FILE_NAME
    spec_dict_path = out_root / f"rec_char_dict_{spec}.txt"

    fallback_used, items, _ = resolve_dict(rec_raw, src_root, spec)

    if items:
        # 优先 yml 抽取
        if fallback_used is None:
            info(f"从 {rec_raw}/inference.yml 抽取 character_dict ({len(items)} 字符)")
        else:
            info(f"从 {fallback_used} 拷贝字典 ({len(items)} 字符)")
        # 写入默认与 spec 两个副本
        write_dict(items, default_dict_path)
        info(f"写出 -> {default_dict_path}")
        if default_dict_path.resolve() != spec_dict_path.resolve():
            write_dict(items, spec_dict_path)
            info(f"留底 -> {spec_dict_path}")
    else:
        # 没有 yml、也没有 rec_char_dict.txt：提示用户
        warn(
            f"未在 {rec_raw} 找到 inference.yml 内的 character_dict，"
            f"也没找到 raw/rec_char_dict.txt。"
        )
        warn(
            "mica-ai-ppocr 必需字典。请从 PaddleOCR 仓库 ppocr/utils/dict/ "
            "下载与本 spec 对应的字符字典，或重新下载完整 PaddleX 模型包（带 inference.yml）。"
        )
        return  # 模型已就位，缺字典留给上层处理

    # ---- vocab 校验 ----
    vocab = _onnx_vocab(rec_out / "inference.onnx")
    if vocab is not None:
        # PaddleOCR 约定：onnx vocab = len(dict) + 2（blank + 末尾特殊 token）
        # Java 端 CtcLabelDecoder 已用 idx < chars.length 保护，
        # 因此"差 1"是预期且无害的。这里只打印警告，不阻塞。
        delta = vocab - (len(items) + 1)
        if delta == 0:
            info(f"vocab 校验: onnx={vocab}, dict+blank={len(items) + 1} ✓")
        elif delta == 1:
            info(
                f"vocab 校验: onnx={vocab}, dict+blank={len(items) + 1} "
                f"(差 1，PaddleOCR 末尾预留 1 个特殊 token，Java 端越界保护已自动跳过，无影响)"
            )
        elif delta < 0:
            warn(f"vocab 校验异常: onnx={vocab}, dict+blank={len(items) + 1}（字典超出模型维度！）")
        else:
            warn(f"vocab 校验: onnx={vocab}, dict+blank={len(items) + 1}（差 {delta}，需补占位行）")

    ok(f"整理完成，输出在 {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 PP-OCRv6 目录为 mica-ai-ppocr 期望结构")
    parser.add_argument("--spec", default="tiny", help="与 download.py 保持一致，默认 tiny")
    args = parser.parse_args()

    root = cap_models_dir("ppocr")
    step(f"PP-OCRv6 目录整理，根目录: {root}, spec={args.spec}")
    flatten(root, args.spec)


if __name__ == "__main__":
    main()
