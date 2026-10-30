# model-tools/detect

`mica-ai-detect` 的模型资产（通用目标检测，COCO 80 类）。

| 文件 | 大小 | 输入 | 来源 | License |
|------|------|------|------|---------|
| `object_detection_nanodet_2022nov.onnx` | 3.62MB | `[1,3,416,416]` | [opencv/opencv_zoo · object_detection_nanodet](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache-2.0 ✅ 可商用 |
| `object_detection_yolox_2022nov.onnx` | 34.2MB | `[1,3,640,640]` | [opencv/opencv_zoo · object_detection_yolox](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox) | Apache-2.0 ✅ 可商用 |

## 获取方式

```bash
git clone --depth=1 --filter=blob:none --sparse https://github.com/opencv/opencv_zoo.git opencv_zoo
cd opencv_zoo
git sparse-checkout set models/object_detection_nanodet models/object_detection_yolox
git lfs pull
# 拷到本目录
cp models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx  <repo>/model-tools/detect/models/
cp models/object_detection_yolox/object_detection_yolox_2022nov.onnx      <repo>/model-tools/detect/models/
```

> 两个文件都是 GitHub LFS 托管，`curl` 直下只会拿到 130 字节的指针文件，必须 `git lfs pull`。

## I/O 契约（实测，勿凭文件名推断）

```
object_detection_nanodet_2022nov.onnx
  输入  input.1   [1, 3, 416, 416]
  输出  792 [1, 2704, 80]   cls  stride 8
        814 [1,  676, 80]   cls  stride 16
        836 [1,  169, 80]   cls  stride 32
        795 [1, 2704, 32]   bbox DFL stride 8
        817 [1,  676, 32]   bbox DFL stride 16
        839 [1,  169, 32]   bbox DFL stride 32

object_detection_yolox_2022nov.onnx
  输入  images    [1, 3, 640, 640]
  输出  output    [1, 8400, 85]   (= 80×80 + 40×40 + 20×20 个锚点，stride 8/16/32)
```

- **NanoDet 该导出只有 3 个 stride（8/16/32）**，不是常见的 4 stride，输出是 **6 个**张量（不是 8 个）。
- YOLOX 输出 85 维为 `cx,cy,w,h,obj` + 80 类，**sigmoid 已固化在导出图内**（数值直接在 0~1）。
- 预处理差异见 [mica-ai-detect/README.md](../../mica-ai-core/mica-ai-detect/README.md#io-格式)。

## 替换模型

1. 先按 `AGENTS.md` §6.1 自检 License（必须允许商用）
2. 覆盖本目录文件 → 跑 `mvn -pl mica-ai-core/mica-ai-detect -am test` 确认可加载且输出契约一致
3. 若输出形态（张量个数 / stride / 归一化）变化，先更新 `mica-ai-detect/README.md` 的 I/O 契约与 `ObjectDetector` 适配，再入库
