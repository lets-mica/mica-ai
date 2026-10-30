# mica-ai-detect

通用目标检测（COCO 80 类），纯 **ONNX Runtime + openpnp/opencv** 实现，零 Python、零 PyTorch、零 Spring 依赖，JDK 8 兼容。

支持两种模型族（合约一致，切换只改配置）：

| 族 | 模型 | 输入 | 体积 | 特点 |
|----|------|------|------|------|
| `NANODET_PLUS`（默认） | `object_detection_nanodet_2022nov.onnx` | 416×416 | 3.62MB | 轻量、CPU 友好，适合高帧率流 |
| `YOLOX_S` | `object_detection_yolox_2022nov.onnx` | 640×640 | 34.2MB | 精度更高，CPU 代价更大 |

两个模型均来自 [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)（**Apache-2.0**，可商用），权重位于 `model-tools/detect/models/`。

## 快速开始

```java
// NanoDet-Plus（默认族）
try (DetectEngine engine = DetectEngine.create(DetectConfig.builder()
        .modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
        .build())) {
    DetectResult result = engine.detectPath("street.jpg");
    for (DetectBox box : result.getBoxes()) {
        System.out.printf("%s %.3f [%.0f,%.0f,%.0f,%.0f]%n",
            box.label(engine.getLabels()), box.getScore(),
            box.getX1(), box.getY1(), box.getX2(), box.getY2());
    }
}

// 切 YOLOX-s：family + inputSize 必须成对修改
DetectConfig yolox = DetectConfig.builder()
    .family(DetectModelFamily.YOLOX_S)
    .inputSize(640)
    .scoreThreshold(0.5f)
    .nmsThreshold(0.5f)
    .build();
```

`DetectEngine` 实现 `AutoCloseable`，**必须** try-with-resources（内部持有 ONNX session 与会话选项）。

## 核心组件

| 类 | 职责 |
|----|------|
| `DetectEngine` | 门面：`detect(Mat)` / `detectBytes(byte[])` / `detectPath(String)`，配置与类别名 |
| `ObjectDetector` | ONNX 加载、预处理、解码、NMS；构造期校验「配置边长 == 模型张量边长」并快速失败 |
| `DetectConfig` | 模型族、路径、阈值、类别名、ORT 会话参数的 builder + `validate()` |
| `DetectModelFamily` | 模型族枚举，显式固化「输入边长 / 归一化 / padding / 输出形态 / NMS 语义」契约 |
| `Nms` | 贪心 NMS（类别无关 / 按类别），纯算法无 OpenCV 依赖，便于单测 |
| `DetectImageUtils` | letterbox、BGR→RGB、CHW 张量化，零 Spring 依赖 |
| `DetectBox` / `DetectResult` | 检测框（原图绝对像素坐标）与单帧结果快照 |

## I/O 格式

### NanoDet-Plus

- **输入** 名称 `input.1`，形状 `[1,3,416,416]`，NCHW float：
  1. BGR → RGB
  2. letterbox：`ratio = min(416/w, 416/h)`，缩放后居中填充（**填充值 0**，`INTER_AREA`）
  3. 归一化 `(x - mean) / std`，`mean = [103.53,116.28,123.675]`、`std = [57.375,57.12,58.395]`，**未做 /255**（官方统计量本身是 0~255 量纲）
- **输出** 6 个 `[1,N,D]`：stride 8/16/32 各有 1 个 cls（`D=80`）与 1 个 bbox（`D=32`），按锚点数 `N`（2704 / 676 / 169）配对
- **解码**：bbox 为 DFL 分布（4 边 × 8 bin），softmax + 期望投影 × stride；锚点 `cx = x*stride + 0.5*(stride-1)`
- **NMS**：类别无关（对齐 `cv2.dnn.NMSBoxes`）

> ⚠️ 该导出**只有 3 个 stride**（8/16/32），没有 stride 64；输出是 6 个张量而非 8 个。

### YOLOX-s

- **输入** 名称 `images`，形状 `[1,3,640,640]`，NCHW float：
  1. BGR → RGB
  2. letterbox：居中填充（**填充值 114**，`INTER_LINEAR`）
  3. **不做归一化**，保留 0~255
- **输出** 1 个 `[1,8400,85]`：`(cx,cy,w,h,obj,80 类)`，已在导出图中做过 sigmoid
- **解码**：`cx=(gx+grid)*stride`、`w=exp(gw)*stride`，得分 `obj × cls`
- **NMS**：按类别（对齐 `cv2.dnn.NMSBoxesBatched`）

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `family` | `NANODET_PLUS` | 模型族，决定预处理 / 解码 / NMS 契约 |
| `modelPath` | 空 | 为空时按 `family` 解析 `classpath:mica-ai/models/detect/<family>.onnx` |
| `inputSize` | `416` | 必须等于模型张量边长（YOLOX 用 640），否则构造期抛异常 |
| `scoreThreshold` | `0.35` | 低于该值的候选框直接丢弃 |
| `nmsThreshold` | `0.6` | IoU 大于该值的重叠框被抑制 |
| `maxDetections` | `100` | 单帧最多输出框数 |
| `labels` | `null` | 为空时用内置 COCO 80 类（`CocoLabels.COCO_80`，另有 `COCO_80_ZH`） |
| `onnx` | defaults | ORT 会话参数（`intraOpNumThreads` / `device` 等） |

## Spring Boot Starter

```yaml
mica:
  ai:
    detect:
      enabled: true
      model-path: model-tools/detect/models/object_detection_nanodet_2022nov.onnx
      family: NANODET_PLUS
      input-size: 416
      score-threshold: 0.35
      nms-threshold: 0.6
      max-detections: 100
      onnx:
        intra-op-num-threads: 2
```

容器关闭时由 `destroyMethod = "close"` 自动释放。

## 测试

```bash
mvn -pl mica-ai-core/mica-ai-detect -am test                 # 单测 + 真模型集成测试
mvn -pl mica-ai-core/mica-ai-detect -am test -Dmica.ai.detect.externalModel=/abs/path/model.onnx
```

集成测试覆盖：输入边长 / 输出张量个数契约、真图推理后坐标与得分合法性、空图返回 `null`、`inputSize` 与模型不符时快速失败。

## 模型获取

模型随仓库分发于 `model-tools/detect/models/`（单文件均 <50MB）。替换模型时请先按 AGENTS.md §6.1 校验 License 是否允许商用。
