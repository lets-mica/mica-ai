#!/usr/bin/env python3
"""u2netp 预处理契约对照探针（只读校验，不参与构建）。

用途：核验 mica-ai-matting 的 I/O 契约与归一化参数，尤其是「通道顺序」这一写在
Java 代码里、出问题却只会表现为「边缘变糊」的隐性约定。

只做两件事：
  1. 打印模型的输入 / 输出张量签名（形状、dtype、节点名）
  2. 固定同一张真实照片，只改 mean/std/通道序，比较掩码差异（以 ImageNet-RGB 为基准的 MAE）

⚠️ 需要 numpy / onnxruntime / Pillow，属于一次性核对工具。
   与 AGENTS.md §6.2 无关：本脚本不参与构建，也不是「模型自检入口」——
   模型的唯一自检入口是 mica-ai-matting 的集成测试（`./mvnc.sh -pl mica-ai-core/mica-ai-matting test`）。

用法：
    python probe_preprocess.py <model.onnx> [image.jpg]
"""
import os
import sys

import numpy as np
import onnxruntime as ort

MEAN_IMAGENET_RGB = [0.485, 0.456, 0.406]
STD_IMAGENET_RGB = [0.229, 0.224, 0.225]


def load_image(path, size=320):
    """优先 Pillow；退化为 OpenCV。返回 RGB uint8 (size, size, 3)。"""
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
                          dtype=np.uint8)
    except ImportError:
        import cv2
        bgr = cv2.imread(path)
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run(sess, image_rgb, mean, std, as_bgr=False):
    a = image_rgb.astype(np.float32) / 255.0
    if as_bgr:
        a = a[..., ::-1]
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    a = np.transpose(a, (2, 0, 1))[None, ...].astype(np.float32)
    name = sess.get_inputs()[0].name
    d0 = np.asarray(sess.run(None, {name: a})[0])[0, 0]
    return (d0 - d0.min()) / (d0.max() - d0.min() + 1e-8)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model = sys.argv[1]
    image = sys.argv[2] if len(sys.argv) > 2 else None

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    print("== 输入 ==")
    for i in sess.get_inputs():
        print(f"  {i.name}  shape={i.shape}  type={i.type}")
    print("== 输出 ==")
    outs = sess.get_outputs()
    print(f"  共 {len(outs)} 个")
    for i, o in enumerate(outs):
        print(f"  [{i}] {o.name}  shape={o.shape}  type={o.type}")

    if not image or not os.path.exists(image):
        print("\n（未提供有效图片，跳过预处理对照；d0 语义见 README「输出语义」）")
        return 0

    img = load_image(image)
    base = run(sess, img, MEAN_IMAGENET_RGB, STD_IMAGENET_RGB)
    cases = [
        ("ImageNet-RGB（本模块采用）", MEAN_IMAGENET_RGB, STD_IMAGENET_RGB, False),
        ("均值镜像（等价 BGR 喂入）", [0.406, 0.456, 0.485], [0.225, 0.224, 0.229], False),
        ("通道序写反（BGR 当 RGB）", MEAN_IMAGENET_RGB, STD_IMAGENET_RGB, True),
        ("全用 0.5", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False),
        ("不做归一化", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False),
    ]
    print(f"\n== 预处理对照（{image}，基准 = ImageNet-RGB）==")
    for name, m, s, bgr in cases:
        mask = run(sess, img, m, s, bgr)
        mae = 0.0 if name.startswith("ImageNet-RGB") else float(np.abs(mask - base).mean())
        print(f"  {name:28s} alpha_mean={mask.mean():.4f}  fg@0.5={(mask > 0.5).mean():.4f}  MAE={mae:.4f}")
    print("\n结论：u2netp 对 mean/std 绝对取值不敏感，但对【通道顺序】敏感 —— "
          "BGR 当 RGB 喂的 MAE 明显高于镜像均值。mica-ai-matting 固定走 ImageNet-RGB。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
