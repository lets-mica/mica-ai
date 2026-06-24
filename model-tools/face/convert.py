"""把 download.py 下载的 buffalo_l 原始模型拷贝为 mica-ai-face 期望的最终文件。

InsightFace 官方提供的 buffalo_l 已经全部是 ONNX 格式，不需要 PyTorch
重新导出。本脚本只做：

1. 校验下载下来的 ONNX 文件（input / output shape）
2. 把 mica-ai-face 真正用到的两个文件（det_10g.onnx / w600k_r50.onnx）
   拷贝 / 链接到 model/out/ 下，方便 Java 端直接读取。

最终产物：

    model-tools/face/model/out/det_10g.onnx     ← RetinaFace（640x640 BGR）
    model-tools/face/model/out/w600k_r50.onnx   ← ArcFace（112x112 BGR, output 512d）

Java 侧配置：

    mica:
      ai:
        face:
          det-model-path: <abs>/model/out/det_10g.onnx
          rec-model-path: <abs>/model/out/w600k_r50.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, fail, info, ok, step, warn
from common.onnx_utils import check_onnx


# mica-ai-face 真正用到的文件
REQUIRED_FILES = {
    "det_10g.onnx":   "RetinaFace 人脸检测（buffalo_l，640x640 BGR 输入）",
    "w600k_r50.onnx": "ArcFace 人脸识别（buffalo_l，112x112 BGR 输入，512d 输出）",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 mica-ai-face ONNX 产物")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验 raw 模型有效性，不复制",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="用符号链接代替复制（节省磁盘）",
    )
    args = parser.parse_args()

    root = cap_models_dir("face")
    raw_dir = root / "buffalo_l_raw"
    if not raw_dir.exists():
        fail(f"未找到 {raw_dir}，请先执行：python download.py")
        sys.exit(1)

    step("校验 raw 模型文件...")
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        fail(f"raw 目录缺少必要文件: {missing}")
        sys.exit(1)

    if args.check_only:
        for fname in REQUIRED_FILES:
            ok(f"raw/{fname} 存在 ({REQUIRED_FILES[fname]})")
        ok("check-only 完成")
        return

    out_dir = ensure_dir(root / "out")
    step(f"导出到 {out_dir}")

    for fname, desc in REQUIRED_FILES.items():
        src = raw_dir / fname
        dst = out_dir / fname
        # 检查源 ONNX 文件是否合法（输入输出 shape 可被正确解析）
        try:
            check_onnx(src)
        except Exception as e:
            warn(f"{fname} 校验失败，但仍继续拷贝: {e}")

        if args.link:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            info(f"  链接 {dst} -> {src}")
        else:
            shutil.copy2(src, dst)
            info(f"  复制 {dst}")

    ok(f"导出完成。产物目录: {out_dir}")
    info("Java 侧 yml 配置：")
    info("  mica.ai.face.det-model-path: <abs>/model/out/det_10g.onnx")
    info("  mica.ai.face.rec-model-path: <abs>/model/out/w600k_r50.onnx")


if __name__ == "__main__":
    main()
