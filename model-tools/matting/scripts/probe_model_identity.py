#!/usr/bin/env python3
"""rembg U2Net 族模型的「身份 / 行为」对照探针（只读校验，不参与构建）。

用途：核验 rembg release 里的 u2netp.onnx 到底是不是 u2net_human_seg 的权重。

背景：rembg 官方 release 的 `u2netp.onnx` 只有 4.36MB，而 U-2-Net 官方仓库的 u2netp
权重是 4.7MB（u2netp.pth）；同时 HuggingFace 上被明确命名为 U-2-Net-Human-Seg 的
onnx 与本机 u2netp.onnx 逐字节相同（sha256 一致）。因此需要行为旁证。

判据：
  1. 人像图 → human_seg 变体应给出干净的人体轮廓
  2. 非人像图（如静物 / 风景）→ human_seg 变体应「找不出人」，整体激活面积显著更小；
     通用显著性（u2net）则应稳定圈出主体
  3. 若两个模型对同一输入给出几乎相同的掩码 → 行为上也同源

⚠️ 需要 numpy / onnxruntime / Pillow，属于一次性核对工具。
   与 AGENTS.md §6.2 无关：本脚本不参与构建，也不是「模型自检入口」——
   模型的唯一自检入口是 mica-ai-matting 的集成测试
   （`./mvnc.sh -pl mica-ai-core/mica-ai-matting test`）。

用法：
    python probe_model_identity.py <modelA.onnx> [modelB.onnx] <image1> [image2] ...
"""
import sys

import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 320


def load_rgb(path):
    """优先 Pillow；退化为 OpenCV。返回 RGB uint8 (320, 320, 3)。"""
    try:
        from PIL import Image
        return np.asarray(
            Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR),
            dtype=np.uint8)
    except ImportError:
        import cv2
        bgr = cv2.imread(path)
        bgr = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb):
    a = image_rgb.astype(np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.transpose(a, (2, 0, 1))[None, ...].astype(np.float32)


def predict(sess, image_rgb):
    """返回 (d0 原始张量 320x320, min-max 归一化后的 320x320)。"""
    name = sess.get_inputs()[0].name
    out = np.asarray(sess.run(None, {name: preprocess(image_rgb)})[0])[0, 0]
    span = out.max() - out.min()
    norm = (out - out.min()) / (span if span > 1e-8 else 1e-8)
    return out, norm


def describe(tag, model, image_path, images):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    print(f"\n== {tag} ==")
    print(f"  file       : {model}")
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print(f"  inputs     : {[(i.name, i.shape) for i in ins]}")
    print(f"  outputs    : {len(outs)} 个, 首个={outs[0].name}")
    for path in images:
        rgb = load_rgb(path)
        raw, norm = predict(sess, rgb)
        # 前景占比：以 0.5 为阈值
        fg = float((norm > 0.5).mean())
        print(f"  [{path}]")
        print(f"      raw  min/max = {raw.min():.6f} / {raw.max():.6f}   {_verdict_sigmoid(raw)}")
        print(f"      norm 前景占比(>0.5) = {fg:6.2%}   均值 = {norm.mean():.4f}")
        describe.last_norm = getattr(describe, "last_norm", {})
        describe.last_norm[(tag, path)] = norm
    return sess


def _verdict_sigmoid(raw):
    if raw.min() >= -1e-6 and raw.max() <= 1.0 + 1e-6:
        return "→ 已在 [0,1]（疑似内置 Sigmoid）"
    return "→ 超出 [0,1]（疑似 logits，需手动 sigmoid）"


def compare(tag_a, tag_b, images):
    store = describe.last_norm
    print("\n== 两模型掩码差异（同一输入，min-max 归一化后） ==")
    for path in images:
        a = store.get((tag_a, path))
        b = store.get((tag_b, path))
        if a is None or b is None:
            continue
        mae = float(np.abs(a - b).mean())
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print(f"  [{path}]  MAE={mae:.6f}  相关系数={corr:.6f}"
              f"   {'→ 行为高度一致（同源嫌疑）' if mae < 0.02 else '→ 行为存在差异'}")


def main():
    args = sys.argv[1:]
    models = [a for a in args if a.endswith(".onnx")]
    images = [a for a in args if not a.endswith(".onnx")]
    if not models or not images:
        print(__doc__)
        return 1

    for idx, model in enumerate(models):
        describe(f"model{idx}:{model.split('/')[-1]}", model, model, images)
    if len(models) == 2:
        compare(f"model0:{models[0].split('/')[-1]}",
                f"model1:{models[1].split('/')[-1]}", images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
