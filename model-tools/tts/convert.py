"""整理 mica-ai-tts 下载后的 Kokoro-82M 模型目录。

Kokoro 仓库结构 (ModelScope/HuggingFace 下载后)：
    kokoro-82m-v1.1-onnx/
        model_dynamic.onnx
        config.json
        voices/
            af.bin
            ... (其它音色)

本脚本把 ``voices/*.bin`` 抽到 ``out/voices/``，便于 mica-ai-tts 部署。

用法：
    python convert.py
    python convert.py --quantize    # 动态量化 model_dynamic.onnx (optional)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, fail, ok, step
from common.progress import info


def _convert_kokoro(root: Path, *, quantize: bool) -> None:
    src = root / "kokoro-82m-v1.1-onnx"
    if not src.exists():
        fail(f"未找到 {src}，请先执行 download.py")
        sys.exit(1)

    out = ensure_dir(root / "out")
    step(f"整理 Kokoro-82M: {src} -> {out}")

    for name in ("model_dynamic.onnx", "config.json"):
        s = src / name
        if not s.exists():
            fail(f"  缺失: {s}")
            sys.exit(1)
        shutil.copy2(s, out / name)
        info(f"  ✓ {name}  ({s.stat().st_size // 1024 // 1024} MB)")

    src_voices = src / "voices"
    if not src_voices.exists():
        fail(f"  缺失: {src_voices}")
        sys.exit(1)
    out_voices = ensure_dir(out / "voices")
    bin_files = sorted(src_voices.glob("*.bin"))
    if not bin_files:
        fail("  voices/ 下没有 .bin 音色")
        sys.exit(1)

    for f in bin_files:
        shutil.copy2(f, out_voices / f.name)
    info(f"  ✓ voices/*.bin (共 {len(bin_files)} 个音色)")

    if quantize:
        try:
            from common.onnx_utils import quantize_dynamic
            target = out / "model_dynamic.int8.onnx"
            quantize_dynamic(out / "model_dynamic.onnx", target)
        except ImportError:
            info("  未安装 onnxruntime，跳过量化（pip install onnxruntime）")
        except Exception as e:
            info(f"  量化失败: {e}（跳过）")

    ok(f"Kokoro 整理完成: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 mica-ai-tts Kokoro-82M 模型")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="动态量化 model_dynamic.onnx（INT8，可减小约 60% 体积）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = cap_models_dir("tts")
    _convert_kokoro(root, quantize=args.quantize)
    ok("完成。")


if __name__ == "__main__":
    main()
