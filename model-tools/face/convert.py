"""把 download.py 下载的 OpenCV Zoo 原始模型拷贝为 mica-ai-face 期望的最终文件。

OpenCV Zoo 模型本身就是 ONNX，不需要 PyTorch 转换。本脚本只做：

1. 校验下载下来的 ONNX 文件（input / output shape）
2. 把 mica-ai-face 真正用到的两个文件（YuNet + SFace）
   拷贝 / 链接到 model/out/ 下，方便 Java 端直接读取。

最终产物：

    model-tools/face/model/out/face_detection_yunet_2023mar.onnx     ← YuNet（320x320 RGB）
    model-tools/face/model/out/face_recognition_sface_2021dec.onnx   ← SFace（112x112 RGB, 512d）

Java 侧配置：

    mica:
      ai:
        face:
          det-model-path: <abs>/model/out/face_detection_yunet_2023mar.onnx
          rec-model-path: <abs>/model/out/face_recognition_sface_2021dec.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, info, ok, step, warn
from common.onnx_utils import check_onnx


# mica-ai-face 真正用到的文件
REQUIRED_FILES = {
    "face_detection_yunet_2023mar.onnx":
        "YuNet 人脸检测（OpenCV Zoo，320x320 RGB 输入，输出 5 关键点）",
    "face_recognition_sface_2021dec.onnx":
        "SFace 人脸识别（OpenCV Zoo，112x112 RGB 输入，512d 输出）",
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

    # 两个原始模型目录
    raw_dirs = {
        "face_detection_yunet_2023mar.onnx": root / "yunet_raw",
        "face_recognition_sface_2021dec.onnx": root / "sface_raw",
    }
    missing_dirs = [d for d in raw_dirs.values() if not d.exists()]
    if missing_dirs:
        warn(f"未找到 {missing_dirs}，请先执行：python download.py")
        sys.exit(1)

    step("校验 raw 模型文件...")
    for fname, raw_dir in raw_dirs.items():
        src = raw_dir / fname
        if not src.exists():
            warn(f"raw 目录缺少 {src}")
            sys.exit(1)
        try:
            check_onnx(src)
        except Exception as e:
            warn(f"{fname} 校验失败，但仍继续: {e}")

    if args.check_only:
        for fname, desc in REQUIRED_FILES.items():
            ok(f"raw/{fname} 存在 ({desc})")
        ok("check-only 完成")
        return

    out_dir = ensure_dir(root / "out")
    step(f"导出到 {out_dir}")

    for fname, raw_dir in raw_dirs.items():
        src = raw_dir / fname
        dst = out_dir / fname
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
    info("  mica.ai.face.det-model-path: <abs>/model/out/face_detection_yunet_2023mar.onnx")
    info("  mica.ai.face.rec-model-path: <abs>/model/out/face_recognition_sface_2021dec.onnx")


if __name__ == "__main__":
    main()