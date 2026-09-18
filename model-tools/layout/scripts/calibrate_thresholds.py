"""PP-DocLayoutV3 阈值标定探针（对齐 Java 端后处理链路）。

用途：在真实文档图上扫描 scoreThreshold，输出每个区域的 label / score / 是否被
NMS 或「整页 image 伪框」规则丢弃，用于选定 mica-ai-layout 的默认阈值。

Java 端对应实现：
  - 坐标反算：LayoutPostProcessor.toOrig
  - NMS：LayoutPostProcessor.nms（同类 0.6 / 异类 0.98）
  - 整页伪框：LayoutPostProcessor.isOverlargeImageBox（横 0.82 / 竖 0.93）
  - 阅读顺序：LayoutPostProcessor.decodeReadingOrder（跳过类不占号）

用法（WSL / Linux）：
    python3 calibrate_thresholds.py [图片路径]

默认图片为同级 demo_doc.jpg（官方 demo 1654x2339）。
模型路径默认 ../models/model.onnx，可用 LAYOUT_ONNX 覆盖。
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LAYOUT_ONNX", os.path.join(HERE, "..", "models", "model.onnx"))
IMAGE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "demo_doc.jpg")

INPUT_SIDE = 800
PAD_VALUE = 114.0
MEAN = np.array([0.8286, 0.8281, 0.8282], dtype=np.float32)
STD = np.array([0.1889, 0.1889, 0.1889], dtype=np.float32)

LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference", "reference_content",
    "seal", "table", "text", "vertical_text", "vision_footnote",
]

# 对齐 PaddleX LayoutAnalysisProcess.SKIP_ORDER_LABELS
SKIP_ORDER_LABELS = {
    "figure_title", "vision_footnote", "image", "chart", "table",
    "header", "header_image", "footer", "footer_image", "footnote", "aside_text",
}

NMS_SAME, NMS_DIFF = 0.6, 0.98
OVERLARGE_LANDSCAPE, OVERLARGE_PORTRAIT = 0.82, 0.93


def letterbox(bgr):
    """右下补边到 800x800，返回 CHW float 张量与反算参数。"""
    h, w = bgr.shape[:2]
    scale = min(INPUT_SIDE / h, INPUT_SIDE / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((INPUT_SIDE, INPUT_SIDE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = ((rgb - MEAN) / STD).transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(chw), scale, nh, nw


def to_orig(v, pad, scale, limit):
    return min(max(int(round((float(v) - pad) / scale)), 0), limit)


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    bgr = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("无法读取图片: " + IMAGE)
    orig_h, orig_w = bgr.shape[:2]
    print(f"图片 {IMAGE}  {orig_w}x{orig_h}")

    chw, scale, _, _ = letterbox(bgr)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    boxes = sess.run(None, {
        "image": chw,
        "im_shape": np.array([[orig_h, orig_w]], dtype=np.float32),
        "scale_factor": np.array([[1.0 / scale, 1.0 / scale]], dtype=np.float32),
    })[0]

    # 反 letterbox + clip，保留全部分数（不预过滤，供扫阈值用）
    cands = []
    for i, row in enumerate(boxes):
        cls = int(round(float(row[0])))
        if cls < 0 or cls >= len(LABELS):
            continue
        x1 = to_orig(row[2], 0, scale, orig_w)
        y1 = to_orig(row[3], 0, scale, orig_h)
        x2 = to_orig(row[4], 0, scale, orig_w)
        y2 = to_orig(row[5], 0, scale, orig_h)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cands.append({"cls": cls, "label": LABELS[cls], "score": float(row[1]),
                      "box": (x1, y1, x2, y2), "order": float(row[6]), "idx": i})

    print(f"原始有效 query（score>0 且框合法）: {len(cands)}")
    print("\n=== 分数分布（前 30 高）===")
    for c in sorted(cands, key=lambda c: -c["score"])[:30]:
        print(f"  {c['label']:<18} score={c['score']:.4f} box={c['box']}")

    # 扫阈值：观察每个阈值下「过滤后 → 整页伪框丢弃 → NMS」的存活情况
    area_limit = (OVERLARGE_LANDSCAPE if orig_w > orig_h else OVERLARGE_PORTRAIT) * orig_w * orig_h
    print("\n=== 阈值扫描（对齐 Java 后处理链路）===")
    print(f"  {'阈值':<8}{'过阈值':<8}{'整页丢弃':<10}{'NMS后':<8}标签分布")
    for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        kept = [c for c in cands if c["score"] >= th]
        kept.sort(key=lambda c: -c["score"])
        before_nms = len(kept)
        dropped_page = 0
        after = []
        for c in kept:
            if c["label"] == "image":
                x1, y1, x2, y2 = c["box"]
                if (x2 - x1) * (y2 - y1) > area_limit:
                    dropped_page += 1
                    continue
            after.append(c)
        # 贪心 NMS
        final, removed = [], set()
        for i, cur in enumerate(after):
            if i in removed:
                continue
            final.append(cur)
            for j in range(i + 1, len(after)):
                if j in removed:
                    continue
                lim = NMS_SAME if cur["label"] == after[j]["label"] else NMS_DIFF
                if iou(cur["box"], after[j]["box"]) > lim:
                    removed.add(j)
        dist = {}
        for c in final:
            dist[c["label"]] = dist.get(c["label"], 0) + 1
        label_str = ", ".join(f"{k}×{v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  {th:<8}{before_nms:<8}{dropped_page:<10}{len(final):<8}{label_str}")

    # 对照：官方 draw_threshold 0.4 下的完整明细
    print("\n=== 阈值 0.4 下的最终结果（含阅读顺序）===")
    kept = sorted([c for c in cands if c["score"] >= 0.4], key=lambda c: -c["score"])
    after = [c for c in kept
             if not (c["label"] == "image"
                     and (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) > area_limit)]
    final, removed = [], set()
    for i, cur in enumerate(after):
        if i in removed:
            continue
        final.append(cur)
        for j in range(i + 1, len(after)):
            if j in removed:
                continue
            lim = NMS_SAME if cur["label"] == after[j]["label"] else NMS_DIFF
            if iou(cur["box"], after[j]["box"]) > lim:
                removed.add(j)

    ordered = sorted(range(len(final)), key=lambda i: (final[i]["order"], -final[i]["score"], i))
    rank = {}
    oi = 1
    for pos in ordered:
        if final[pos]["label"] in SKIP_ORDER_LABELS:
            rank[pos] = -1
        else:
            rank[pos] = oi
            oi += 1
    for i, c in enumerate(final):
        print(f"  [read={rank[i]:>3}] {c['label']:<18} score={c['score']:.4f} box={c['box']}")
    print(f"\n参与编号区域: {sum(1 for v in rank.values() if v > 0)}，跳过类: {sum(1 for v in rank.values() if v < 0)}")


if __name__ == "__main__":
    main()
