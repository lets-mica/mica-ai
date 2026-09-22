#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mica-ai-textline 契约探针（只读，不参与构建）。

用途：核验 PP-LCNet 文本行方向分类 ONNX 的
  1. I/O 契约（输入名 / 形状 / 输出名 / 类数）
  2. 类别语义（索引 0 = 0_degree、索引 1 = 180_degree）
  3. 通道顺序敏感性（BGR vs RGB）
  4. 真实数据上的置信度水平与鲁棒性

⚠️ 这不是构建 / 自检入口。本仓库模型的唯一自检入口是各模块的集成测试：
    mvn -pl mica-ai-core/mica-ai-textline -am test
本脚本只在「换模型 / 换导出 / 排查判定异常」时由人工手动运行。

依赖：onnxruntime / numpy / opencv-python
用法：
    python probe_contract.py <model.onnx> [<more.onnx> ...]
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

# 官方 inference.yml 的 PreProcess 契约
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_W, IN_H = 160, 80
LABELS = ["0_degree", "180_degree"]

STD_WARN = "[WARN]"
STD_FAIL = "[FAIL]"


def signature(path):
    """打印模型 I/O 签名。"""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print(f"\n{'=' * 72}\n{os.path.basename(path)}  ({os.path.getsize(path):,} B)")
    print(f"  sha256 = {sha256(path)}")
    print("  --- INPUTS ---")
    for i in ins:
        print(f"    name={i.name!r} shape={i.shape} type={i.type}")
    print("  --- OUTPUTS ---")
    for o in outs:
        print(f"    name={o.name!r} shape={o.shape} type={o.type}")
    return sess


def sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def preprocess(bgr, order):
    """按 inference.yml 契约预处理：resize(160,80) -> /255 -> normalize -> CHW。"""
    img = cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32)
    if order == "RGB":
        x = x[:, :, ::-1]
    x = (x / 255.0 - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def infer(sess, bgr, order):
    """返回 (logits, probs)。"""
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: preprocess(bgr, order)})[0][0]
    e = np.exp(logits - logits.max())
    return logits, e / e.sum()


def render(text, h=80, w=600, color=(0, 0, 0), bg=(255, 255, 255), scale=1.6):
    """渲染干净的合成文本行（用于通道顺序等对照实验）。"""
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(img, text, (15, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3, cv2.LINE_AA)
    return img


def check_contract(sess, path):
    """校验 I/O 契约是否与预期一致。"""
    bad = 0
    shape = sess.get_inputs()[0].shape
    if len(shape) != 4 or shape[1] != 3 or shape[2] != IN_H or shape[3] != IN_W:
        print(f"  {STD_FAIL} 输入形状与预期 [N,3,{IN_H},{IN_W}] 不符: {shape}")
        bad += 1
    out_shape = sess.get_outputs()[0].shape
    if len(out_shape) != 2 or out_shape[1] != 2:
        print(f"  {STD_FAIL} 输出形状与预期 [N,2] 不符: {out_shape}")
        bad += 1
    if bad == 0:
        print(f"  [OK] 契约匹配：输入 [N,3,{IN_H},{IN_W}]、输出 [N,2]")
    return bad


def check_class_semantics(sess):
    """校验索引 0 = 0_degree、索引 1 = 180_degree。"""
    upright = render("Hello World 123")
    flipped = cv2.rotate(upright, cv2.ROTATE_180)
    bad = 0
    for label, img, expect_idx in (("upright", upright, 0), ("flipped", flipped, 1)):
        logits, probs = infer(sess, img, "BGR")
        got = int(probs.argmax())
        flag = "" if got == expect_idx else f"  {STD_FAIL}"
        if got != expect_idx:
            bad += 1
        print(f"  {label:8s} logits=[{logits[0]:+.4f},{logits[1]:+.4f}] "
              f"p({LABELS[1]})={probs[1]:.4f} -> {LABELS[got]}{flag}")
    if bad == 0:
        print("  [OK] 类别语义正确（0=0_degree, 1=180_degree）")
    return bad


def check_channel_order(sess):
    """对照 BGR / RGB，判断模型是否对通道顺序敏感。"""
    upright = render("Hello World 123")
    flipped = cv2.rotate(upright, cv2.ROTATE_180)
    results = {}
    for order in ("BGR", "RGB"):
        for label, img in (("upright", upright), ("flipped", flipped)):
            results[(order, label)] = infer(sess, img, order)[1]
    same = all(
        int(results[("BGR", lb)].argmax()) == int(results[("RGB", lb)].argmax())
        for lb in ("upright", "flipped")
    )
    print(f"  BGR p(180): upright={results[('BGR','upright')][1]:.4f} flipped={results[('BGR','flipped')][1]:.4f}")
    print(f"  RGB p(180): upright={results[('RGB','upright')][1]:.4f} flipped={results[('RGB','flipped')][1]:.4f}")
    if same:
        print("  [OK] BGR / RGB 结论一致 —— 本任务对通道顺序不敏感")
    else:
        print(f"  {STD_WARN} BGR / RGB 结论不同 —— 必须按官方约定用 BGR，并核查 normalization")
    return 0 if same else 1


def check_robustness(sess, sample):
    """在退化输入上检查鲁棒性（缩小 / 模糊 / JPEG 压缩）。"""
    if sample is None or not os.path.exists(sample):
        print("  (跳过：未提供真实样例图)")
        return 0
    base = cv2.imread(sample)
    if base is None:
        print("  (跳过：样例图无法解码)")
        return 0
    small = cv2.resize(base, (max(8, base.shape[1] // 4), max(8, base.shape[0] // 4)))
    cases = {
        "raw": base,
        "downscale": cv2.resize(small, (base.shape[1], base.shape[0])),
        "blur": cv2.GaussianBlur(base, (5, 5), 0),
        "jpeg-q30": cv2.imdecode(cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1),
    }
    bad = 0
    for name, img in cases.items():
        for lbl, im, expect_idx in (("raw", img, 1), ("rot180", cv2.rotate(img, cv2.ROTATE_180), 0)):
            logits, probs = infer(sess, im, "BGR")
            got = int(probs.argmax())
            flag = "" if got == expect_idx else f"  {STD_FAIL}"
            if got != expect_idx:
                bad += 1
            print(f"  {name:11s} {lbl:7s} logits=[{logits[0]:+.4f},{logits[1]:+.4f}] "
                  f"p({LABELS[1]})={probs[1]:.4f} -> {LABELS[got]}{flag}")
    if bad == 0:
        print("  [OK] 退化输入下判定全部正确")
    return bad


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("错误：请至少给出一个 .onnx 路径")
        return 2

    sample = os.environ.get("TEXTLINE_SAMPLE")
    if sample is None:
        for cand in ("textline_rot180_demo.jpg", "real_flipped.jpg"):
            if os.path.exists(cand):
                sample = cand
                break

    total_bad = 0
    for path in argv[1:]:
        if not os.path.exists(path):
            print(f"{STD_FAIL} 文件不存在: {path}")
            total_bad += 1
            continue
        sess = signature(path)
        print("  --- 契约校验 ---")
        total_bad += check_contract(sess, path)
        print("  --- 类别语义 ---")
        total_bad += check_class_semantics(sess)
        print("  --- 通道顺序敏感性 ---")
        total_bad += check_channel_order(sess)
        print("  --- 退化输入鲁棒性 ---")
        total_bad += check_robustness(sess, sample)

    print(f"\n{'=' * 72}")
    if total_bad == 0:
        print("[OK] 全部检查通过")
    else:
        print(f"{STD_FAIL} 共 {total_bad} 项检查未通过")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
