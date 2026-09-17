"""定位 PP-DocLayoutV3 ONNX 输出坐标所在的空间（letterbox vs 原图）。

背景：docs/layout-tracking.md 与实测数据互相矛盾 —— 同一张 1654x2339 的图，
一次测得 box 上限 ~565（= letterbox content 宽度 566），另一次测得 y2=4255（远超 800）。
两者比值恰为 scale^2（2.924^2 = 8.55），怀疑是 feed 的 scale_factor 语义不同导致。

本脚本对同一张图做受控实验：只改 scale_factor / im_shape 的喂法，观察输出坐标边界。

用法：
    python3 probe_coord_space.py            # 默认用 PaddleX 官方 demo 图
    python3 probe_coord_space.py <img_path>
"""
import os
import sys
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = os.environ.get(
    "LAYOUT_ONNX",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "models", "model.onnx"),
)
MEAN = np.array([0.8286, 0.8281, 0.8282], dtype=np.float32)
STD = np.array([0.1889, 0.1889, 0.1889], dtype=np.float32)
TARGET = 800
# 注意：WSL 的 /tmp 是 tmpfs，VM 一停就被清空，缓存必须放持久目录
DEMO_CACHE = os.path.expanduser("~/.cache/mica-ai-layout/layout_demo.jpg")


def load_image(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        urllib.request.urlretrieve(url, path)
    return Image.open(path).convert("RGB")


def letterbox(img):
    """PaddleX 风格 letterbox：等比缩放到 800 内，贴左上角，右下补 114 灰。"""
    w0, h0 = img.size
    r = min(TARGET / w0, TARGET / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    canvas = Image.new("RGB", (TARGET, TARGET), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (0, 0))
    arr = np.array(canvas).astype(np.float32) / 255.0
    chw = ((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return chw, r, nw, nh, w0, h0


def run(sess, chw, im_shape, scale_factor):
    outs = sess.run(None, {
        "image": chw,
        "im_shape": np.array([im_shape], dtype=np.float32),
        "scale_factor": np.array([scale_factor], dtype=np.float32),
    })
    boxes, count, order = outs
    valid = boxes[boxes[:, 1] > 0.5] if boxes.size else boxes
    return boxes, count, order, valid


def summarize(tag, boxes, count, order, w0, h0, r):
    if boxes.size == 0:
        print(f"  {tag}: EMPTY")
        return
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]
    top = boxes[np.argsort(-boxes[:, 1])][:3]
    print(f"  {tag}:")
    print(f"    boxes.shape={boxes.shape} count={count} order.shape={order.shape}")
    print(f"    raw x2 range=[{x2.min():.1f}, {x2.max():.1f}]  y2 range=[{y2.min():.1f}, {y2.max():.1f}]")
    for row in top:
        print(f"    cls={int(row[0]):2d} score={row[1]:.4f} "
              f"box=({row[2]:.1f},{row[3]:.1f},{row[4]:.1f},{row[5]:.1f}) slot={int(row[6])}")
    # 三种候选空间下的"合理性"检查：最高分框应在原图范围内
    print(f"    top1 若视作原图坐标 -> 越界? "
          f"x2>={w0}: {top[0][4] > w0}, y2>={h0}: {top[0][5] > h0}")
    print(f"    top1 若视作 letterbox 坐标(÷{r:.4f}) -> "
          f"x2={top[0][4] / r:.0f} (原图宽 {w0}), y2={top[0][5] / r:.0f} (原图高 {h0})")


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEMO_CACHE
    img = load_image(img_path)
    chw, r, nw, nh, w0, h0 = letterbox(img)
    print(f"image={img_path} orig={w0}x{h0} -> letterbox {nw}x{nh} (r={r:.4f}, 1/r={1 / r:.4f})")
    print(f"若输出为 letterbox 空间: x2 上限应 ≈ {nw}, y2 上限应 ≈ {nh}（content 区）"
          f"；若为原图空间: x2 上限应 ≈ {w0}, y2 上限应 ≈ {h0}\n")

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    im_shape_ok = [float(h0), float(w0)]
    scale_recip = [1.0 / r, 1.0 / r]
    scale_r = [r, r]
    scale_one = [1.0, 1.0]

    cases = [
        ("A scale=1/r (当前 Java 用法? doc 待定)", im_shape_ok, scale_recip),
        ("B scale=1.0", im_shape_ok, scale_one),
        ("C scale=r", im_shape_ok, scale_r),
        ("D scale=1/r, im_shape 恒为 800", [800.0, 800.0], scale_recip),
    ]
    results = {}
    for tag, ims, sf in cases:
        boxes, count, order, valid = run(sess, chw, ims, sf)
        summarize(f"{tag} [im_shape={ims} scale={[round(v, 4) for v in sf]}]",
                  boxes, count, order, w0, h0, r)
        results[tag] = boxes
        print()

    print("=== 交叉验证 ===")
    a_boxes = results[cases[0][0]]      # scale = 1/r
    b_boxes = results[cases[1][0]]      # scale = 1.0
    c_boxes = results[cases[2][0]]      # scale = r
    print(f"  A(1/r) vs B(1.0) 逐行一致? {np.allclose(a_boxes, b_boxes)}")
    # 只比坐标列 [2:6]：class_id/score/slot 不参与缩放，一起比会得到虚假的大偏差
    coords_b = b_boxes[:, 2:6]
    coords_a_over_r = a_boxes[:, 2:6] / r
    print(f"  B.coords == A.coords/r 逐行成立? {np.allclose(coords_b, coords_a_over_r, atol=1.0)}"
          f"  max|diff|={np.abs(coords_b - coords_a_over_r).max():.4f} px")
    print(f"  A/C 坐标比 中位数={np.median(a_boxes[:, 4] / np.where(c_boxes[:, 4] > 1, c_boxes[:, 4], np.nan)):.4f}"
          f"  (= r^2={r ** 2:.4f} ⟺ C 相对 A 被放大 1/r^2，符合 out = orig / scale_factor)")
    print(f"  slot_id / score 三组一致? slot={np.array_equal(a_boxes[:, 6], c_boxes[:, 6])} "
          f"score={np.allclose(a_boxes[:, 1], c_boxes[:, 1])}")

    print("\n=== 结论 ===")
    a_max = (a_boxes[:, 4].max(), a_boxes[:, 5].max())
    b_max = (b_boxes[:, 4].max(), b_boxes[:, 5].max())
    print(f"  输出坐标 = 原图坐标 / scale_factor（模型不做 letterbox 反算）")
    print(f"  · 喂 1/r -> letterbox 画布坐标: A.x2max={a_max[0]:.1f} (x2max 应 ≲ {nw}、y2max 应 ≲ {nh})")
    print(f"  · 喂 1.0 -> 原图坐标:          B.x2max={b_max[0]:.1f} (应 ≲ {w0})、y2max={b_max[1]:.1f} (应 ≲ {h0})")
    print(f"  · 喂 r   -> 被放大 1/r^2 倍（Java 当前写法，错误）")
    print(f"  Java 口径: im_shape=[1,2]={{origH,origW}}, scale_factor=[1,2]={{1/scale,1/scale}},"
          f" 还原 orig = (out - pad) / scale, 再 clip 到原图")


if __name__ == "__main__":
    main()
