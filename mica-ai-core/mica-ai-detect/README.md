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
| `SlicedDetectEngine` | 切片推理门面：切片 → 逐片推理 → 坐标回映射 → 跨片合并，**不持有模型** |
| `ObjectDetector` | ONNX 加载、预处理、解码、NMS；构造期校验「配置边长 == 模型张量边长」并快速失败 |
| `DetectConfig` | 模型族、路径、阈值、类别名、ORT 会话参数的 builder + `validate()` |
| `SlicedDetectConfig` | 切片配置：委托 `DetectConfig` + 重叠率 / 切片数上限 / 合并策略 |
| `DetectModelFamily` | 模型族枚举，显式固化「输入边长 / 归一化 / padding / 输出形态 / NMS 语义」契约 |
| `Nms` | 贪心 NMS（类别无关 / 按类别），纯算法无 OpenCV 依赖，便于单测 |
| `DetectImageUtils` | letterbox、BGR→RGB、CHW 张量化，零 Spring 依赖 |
| `SliceGrid` / `Slice` | 切片网格计算（纯几何）与切片值对象 |
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

## 切片推理

> 解决「整图缩放后小目标看不见」的问题：2000×1500 的图 letterbox 到 416 后，
> 一个 30×30 的远处目标只剩约 8px，模型根本检不出；切片后它按原像素推理，清晰可见。

**零新增模型、零新增依赖** —— 切片推理复用同一份 `DetectConfig` 与同一个 `DetectEngine`。

### 快速开始

```java
try (DetectEngine engine = DetectEngine.create(DetectConfig.builder()
        .modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
        .build())) {

    SlicedDetectEngine sliced = SlicedDetectEngine.create(engine,
        SlicedDetectConfig.builder()
            .detectConfig(engine.getConfig())
            .overlapRatio(0.25f)
            .maxSlices(64)
            .mergeStrategy(SliceMergeStrategy.IOU_MERGE)
            .build());

    DetectResult result = sliced.detect(bgr);   // 原图坐标系
}
```

`SlicedDetectEngine` **不持有模型**（底层 `DetectEngine` 由调用方持有与释放），因此不实现 `AutoCloseable`；
无内部可变状态，可作为单例共享。

### 三条硬规则

| 规则 | 原因 |
|------|------|
| **切片边长恒等于模型输入边长**，不是配置项 | letterbox 缩放比 `min(size/w, size/h)` 只有在切片是正方形且边长 == 模型输入边长时才恒为 1、无填充，每片与整图推理的预处理契约才完全一致 |
| **边缘切片补齐到同样大小**（填充值随模型族） | 边缘切片不足一整块时 letterbox 会把它**放大**，小目标优势直接归零 |
| **重叠是必需的**（默认 0.25） | 没有重叠时，边界物体会被切成两半，两半的 IoU 常低于 NMS 阈值，跨片合并压不住 → 输出「半个物体」 |

### 切片网格

- 步进 `step = 边长 - round(边长 × overlapRatio)`（416 / 0.25 → 312）
- 起点按步进递增；**末列 / 末行回退**到 `边界 - 边长`，保证图像边缘被完整覆盖且不越界
- 例：1200 宽 → 起点 `0 / 312 / 624 / 784`；1500 高 → 起点 `0 / 312 / 624 / 936 / 1084`

### 跨片合并

| 策略 | 说明 |
|------|------|
| `IOU_MERGE`（默认） | 复用整图的 `Nms.suppress`，类别互斥语义随模型族（NanoDet 类别无关 / YOLOX 按类别） |
| `KEEP_ALL` | 保留各片原始框，由调用方自行做跨片去重 |

### 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `detectConfig` | **必填** | 委托的整图检测配置（模型路径 / 族 / 阈值 / 标签） |
| `overlapRatio` | `0.25` | 切片重叠比例，`(0, 1)` |
| `maxSlices` | `64` | 切片数上限，超限抛 `MicaAiException` 并提示调大上限或换更大输入边长的模型 |
| `mergeStrategy` | `IOU_MERGE` | 跨片合并策略 |

> ⚠️ **算力代价**：切片推理的开销约为「整图 × 切片数」。1200×1500 的图在 416 切片下是 20 片。
> `maxSlices` 是显式门槛，超限直接失败并给出可操作的提示，而不是悄悄跑满。

---

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `family` | `NANODET_PLUS` | 模型族，决定预处理 / 解码 / NMS 契约 |
| `modelPath` | **必填** | 模型路径，支持 `classpath:` 前缀；**模型不随 jar 分发**，未配置时启动即失败并给出提示 |
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

切片推理的测试分两层：

- `SliceGridTest` / `SlicedDetectConfigTest` / `SlicedDetectEngineTest`：**零 ONNX** 的纯逻辑测试 —— 网格计算与末列回退、坐标回映射、跨片合并、边界裁剪、切片数上限；其中 `SlicedDetectEngineTest` 用假检测源驱动完整链路
- `SlicedDetectIntegrationTest`：加载真 ONNX 跑完整切片推理，验证结果落在原图坐标系内

> 「切片比整图多检出多少小目标」取决于真实拍摄画面，用 example 端点
> `POST /detect/sliced-compare` 在实拍图上量化（返回只在切片出现的框与算力代价倍数）。

## 模型获取

模型随仓库分发于 `model-tools/detect/models/`（单文件均 <50MB）。替换模型时请先按 AGENTS.md §6.1 校验 License 是否允许商用。
