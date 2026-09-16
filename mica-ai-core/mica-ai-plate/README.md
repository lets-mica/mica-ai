# mica-ai-plate

> [HyperLPR3](https://github.com/szad670401/HyperLPR)（standard_v20230229）中国车牌识别的 Java 复刻：纯 ONNX Runtime + openpnp/opencv，零 Python，Apache-2.0 可商用。

## 1. 模型规格

| 文件 | 来源 | License | 大小 | 说明 |
|------|------|---------|------|------|
| `y5fu_320x_sim.onnx` | [HyperLPR3 v20230229](https://github.com/szad670401/HyperLPR) | Apache 2.0 | ≈6.8 MB | 车牌检测（YOLOv5 多任务，320 输入） |
| `y5fu_640x_sim.onnx` | 同上 | Apache 2.0 | ≈7.4 MB | 车牌检测（640 输入，大图可选） |
| `rpv3_mdict_160_r3.onnx` | 同上 | Apache 2.0 | ≈4.6 MB | CRNN + SVTR 车牌字符识别（77 token CTC） |
| `litemodel_cls_96x_r1.onnx` | 同上 | Apache 2.0 | ≈0.9 MB | 车牌颜色分类（3 类：黄 / 蓝 / 绿） |

模型已直接入库：[`model-tools/plate/models/`](../../model-tools/plate/README.md)（无需下载脚本）。

### 模型 I/O

**检测（y5fu）**

| 项 | 值 |
|----|----|
| 输入节点 | `input`，shape `[1, 3, 320, 320]`（640 模型为 `[1, 3, 640, 640]`），dtype `float32` |
| 输出节点 | `output`，shape `[1, 6300, 15]`（640 模型为 `[1, 25200, 15]`），dtype `float32` |
| 输出含义 | 每行 15 维：`[cx, cy, w, h, obj_conf, 4 关键点 (8 维), 2 层分类 score]`；layer 0 = 单层车牌，layer 1 = 双层车牌 |

**识别（rpv3）**

| 项 | 值 |
|----|----|
| 输入节点 | `data`，shape `[1, 3, 48, 160]`，dtype `float32`，归一化 `(x - 127.5) / 127.5`，**不做** BGR→RGB 转换 |
| 输出节点 | `output`，shape `[1, 20, 78]`，dtype `float32`（logits，未 softmax） |
| 输出含义 | 20 个时间步 × 78 类（77 字符 + 1 冗余位）；CTC 解码：argmax → 去重 → 去 blank(0) |

**颜色分类（litemodel）**

| 项 | 值 |
|----|----|
| 输入节点 | `data`，shape `[1, 3, 96, 96]`，dtype `float32` |
| 输出节点 | `output`，shape `[1, 3]`，dtype `float32`；index 0 = 黄 / 1 = 蓝 / 2 = 绿 |

## 2. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`PlatePipeline`](src/main/java/net/dreamlu/mica/ai/plate/pipeline/PlatePipeline.java) | 实现 `AutoCloseable`，对外提供 `recognizePath` / `recognizeBytes` / `recognize(Mat)` |
| 配置 | [`PlateConfig`](src/main/java/net/dreamlu/mica/ai/plate/PlateConfig.java) | Builder 模式，含三模型路径、输入尺寸、阈值、`maxPlates`、`onnx` |
| 检测 | [`PlateDetector`](src/main/java/net/dreamlu/mica/ai/plate/detection/PlateDetector.java) | letterBox 预处理 + 推理 + xywh2xyxy + conf*class + NMS + 坐标反算 |
| 对齐 | [`PlateAligner`](src/main/java/net/dreamlu/mica/ai/plate/alignment/PlateAligner.java) | 4 点透视校正（对齐 Python `get_rotate_crop_image`），高宽比 ≥ 1.5 时旋转 90°（双层牌） |
| 识别 | [`PlateRecognizer`](src/main/java/net/dreamlu/mica/ai/plate/recognition/PlateRecognizer.java) | 变宽 resize + 右侧 padding 到 160 + 推理 + CTC 解码 |
| 字典 | [`PlateDictionary`](src/main/java/net/dreamlu/mica/ai/plate/recognition/PlateDictionary.java) | 77 token（对齐 `hyperlpr3/common/tokenize.py`），blank index = 0 |
| 分类 | [`PlateClassifier`](src/main/java/net/dreamlu/mica/ai/plate/recognition/PlateClassifier.java) | 仅在车牌号无法判定类型时调用（黄 / 蓝 / 绿三选一） |
| 结果 | [`PlateResult`](src/main/java/net/dreamlu/mica/ai/plate/model/PlateResult.java) | 车牌号 / 类型 / 检测与识别置信度 / 框 / 4 关键点 |
| 类型 | [`PlateType`](src/main/java/net/dreamlu/mica/ai/plate/model/PlateType.java) | 10 类枚举，对齐 Python `typedef.py`（索引一致） |

### 处理流程

```
Path / Bytes / Mat
      │
      ▼
[PlatePipeline.recognize]
   ├─ [PlateDetector.detect]      letterBox(320/640) → ONNX → 15 维后处理 + NMS
   │                               → bbox + 4 关键点 + layerNum（单/双层）
   ├─ [PlateAligner.rotateCrop]   4 点透视校正；h/w ≥ 1.5 → 旋转 90°（双层）
   ├─ 双层牌（layerNum == 1）      按 h * 0.4 切上下两行，各识别一次后拼接
   │  单层牌                       整图识别一次
   ├─ [PlateRecognizer]           CTC 解码 → plateCode + confidence
   ├─ [PlatePipeline.codeFilter]  车牌号规则判型（WJ / 8 位 / 学 / 港澳 / 警 / 粤Z）
   │   └─ UNKNOWN 时 → [PlateClassifier] 颜色分类兜底（黄/蓝/绿，双层 → YELLOW_DOUBLE）
   ▼
List<PlateResult>
```

## 3. 车牌类型（PlateType）

对齐 HyperLPR3 `common/typedef.py`，枚举 `ordinal()` 与 Python 常量值一致：

| 枚举 | Python 常量 | 值 | 说明 |
|------|------------|----|------|
| `UNKNOWN` | `UNKNOWN` | -1* | 未知（交给颜色分类器兜底） |
| `BLUE` | `BLUE` | 0 | 蓝牌 |
| `YELLOW_SINGLE` | `YELLOW_SINGLE` | 1 | 黄牌单层 |
| `WHITE_POLICE` | `WHILE_SINGLE` | 2 | 白牌（警牌 / WJ） |
| `GREEN` | `GREEN` | 3 | 绿牌新能源 |
| `HONG_KONG_MACAO` | `BLACK_HK_MACAO` | 4 | 黑牌港澳 |
| `HK_SINGLE` | `HK_SINGLE` | 5 | 香港单层 |
| `HK_DOUBLE` | `HK_DOUBLE` | 6 | 香港双层 |
| `MACAO_SINGLE` | `MACAO_SINGLE` | 7 | 澳门单层 |
| `MACAO_DOUBLE` | `MACAO_DOUBLE` | 8 | 澳门双层 |
| `YELLOW_DOUBLE` | `YELLOW_DOUBLE` | 9 | 黄牌双层 |

> \* `UNKNOWN` 在 Python 中为 `-1`，Java 枚举 ordinal 从 0 起，语义对齐、数值不强制一致；判型规则见 `PlatePipeline.codeFilter`（对齐 `typedef.py::code_filter`）。`HK_*` / `MACAO_*` 为完整语义保留位，当前 `code_filter` 规则不会产出，归属 `HONG_KONG_MACAO`。

### 判型规则（codeFilter，对齐 Python `code_filter`）

按顺序命中即返回：

1. `WJ` 开头 → `WHITE_POLICE`
2. 长度 = 8 → `GREEN`
3. 含 `学` → `BLUE`
4. 含 `港` / `澳` → `HONG_KONG_MACAO`
5. 含 `警` → `WHITE_POLICE`
6. `粤Z` 开头 → `HONG_KONG_MACAO`
7. 其余 → `UNKNOWN` → 颜色分类器兜底（黄/蓝/绿，双层牌 → `YELLOW_DOUBLE`）

## 4. 快速开始

### 纯 Java（零 Spring）

```java
PlateConfig config = PlateConfig.builder()
    .detectionModelPath("models/y5fu_320x_sim.onnx")
    .recognitionModelPath("models/rpv3_mdict_160_r3.onnx")
    .classificationModelPath("models/litemodel_cls_96x_r1.onnx")
    .build();
try (PlatePipeline pipeline = PlatePipeline.create(config)) {
    List<PlateResult> results = pipeline.recognizePath("car.jpg");
    for (PlateResult r : results) {
        // r.getPlateCode()  == "津B6H920"
        // r.getPlateType()  == PlateType.BLUE
        // r.getRecognitionConfidence() == 0.998
    }
}
```

### Spring Boot Starter

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-plate-spring-boot-starter</artifactId>
</dependency>
```

```yaml
mica:
  ai:
    plate:
      detection-model-path: classpath:models/y5fu_320x_sim.onnx
      recognition-model-path: classpath:models/rpv3_mdict_160_r3.onnx
      classification-model-path: classpath:models/litemodel_cls_96x_r1.onnx
      detection-input-size: 320      # 320 / 640
      detection-confidence-threshold: 0.25
      detection-nms-threshold: 0.5
      max-plates: 5
      device: cpu                    # cpu / gpu
```

```java
@RestController
@RequiredArgsConstructor
public class DemoController {
    private final PlatePipeline pipeline;

    @PostMapping("/plate")
    public List<PlateResult> recognize(@RequestParam MultipartFile file) throws IOException {
        return pipeline.recognizeBytes(file.getBytes());
    }
}
```

### 配置项（`mica.ai.plate` 前缀，见 `PlateProperties`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `20230229` | 版本标识（日志用） |
| `detection-model-path` / `recognition-model-path` / `classification-model-path` | 无 | 三模型路径，必填，支持 `classpath:` |
| `detection-input-size` | `320` | 检测输入边长（仅 320 / 640） |
| `detection-confidence-threshold` | `0.25` | 检测置信度阈值 |
| `detection-nms-threshold` | `0.5` | NMS IoU 阈值 |
| `max-plates` | `5` | 单图最多返回车牌数 |
| `device` | `cpu` | `cpu` / `gpu`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | 0 = ORT 默认 |
| `onnx.inter-op-threads` | `0` | 0 = ORT 默认 |

### 输出字段（`PlateResult`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `plateCode` | `String` | 车牌号（双层牌为上下两行拼接） |
| `plateType` | `PlateType` | 10 类枚举，见上表；JSON 序列化用 `getCode()`（小写下划线） |
| `detectionConfidence` | `float` | 检测框得分（obj_conf × class score） |
| `recognitionConfidence` | `float` | CTC 解码字符概率均值（双层牌为两行平均） |
| `boundingBox` | `int[4]` | 原图坐标 `[x1, y1, x2, y2]` |
| `landmarks` | `int[4][2]` | 4 角点，左上 / 右上 / 右下 / 左下 |

## 5. License

- 代码：Apache License 2.0
- 模型：HyperLPR3 v20230229，[Apache License 2.0](https://github.com/szad670401/HyperLPR/blob/master/LICENSE)，**可商用** ✅
