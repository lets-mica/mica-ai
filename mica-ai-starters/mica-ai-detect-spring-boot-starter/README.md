# mica-ai-detect-spring-boot-starter

> [OpenCV Zoo](https://github.com/opencv/opencv_zoo) 通用目标检测（COCO 80 类）Spring Boot Starter，基于 [mica-ai-detect](../../mica-ai-core/mica-ai-detect/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `DetectEngine` Bean，对外提供 `detect` / `detectBytes` / `detectPath` 三组 API。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-detect-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 两个模型**都已随仓库分发**：NanoDet-Plus 3.62MB（默认，416×416）/ YOLOX-s 34.2MB（640×640，精度更高）。

---

## 2. 配置项（`mica.ai.detect` 前缀）

```yaml
mica:
  ai:
    detect:
      enabled: true
      family: NANODET_PLUS                                # NANODET_PLUS / YOLOX_S
      model-path: model-tools/detect/models/object_detection_nanodet_2022nov.onnx
      input-size: 416                                     # 必须与模型张量边长一致
      score-threshold: 0.35
      nms-threshold: 0.6
      max-detections: 100
      onnx:
        device: CPU                                       # CPU / GPU（枚举）
        intra-op-num-threads: 0                           # 0 = ORT 默认
        inter-op-num-threads: 0
      # 切片推理（与整图检测共用同一份模型，零新增模型资产）
      sliced:
        enabled: true                                     # false 时不装配 SlicedDetectEngine
        overlap-ratio: 0.25                               # 切片重叠比例，(0, 1)
        max-slices: 64                                    # 切片数上限，超限即失败
        merge-strategy: IOU_MERGE                         # IOU_MERGE / KEEP_ALL
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `DetectEngine` Bean |
| `family` | `NANODET_PLUS` | 模型族，决定**预处理 / 解码 / NMS 契约**；`NANODET_PLUS` / `YOLOX_S` |
| `model-path` | **必填** | 模型路径，支持 `classpath:`；**模型不随 jar 分发**，未配置时启动即失败并给出提示 |
| `input-size` | `416` | 模型输入边长，必须与模型张量形状一致（NanoDet 416 / YOLOX 640），否则启动即抛 `MicaAiException` |
| `score-threshold` | `0.35` | 置信度阈值，低于该值的候选框直接丢弃 |
| `nms-threshold` | `0.6` | NMS 的 IoU 阈值 |
| `max-detections` | `100` | 单帧最多输出的检测框数量 |
| `labels` | `null` | 类别名（按 classId 索引）；为空时使用内置 COCO 80 类 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举，GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |
| `sliced.enabled` | `true` | 是否装配 `SlicedDetectEngine` Bean |
| `sliced.overlap-ratio` | `0.25` | 切片重叠比例，`(0, 1)`；重叠是必需的，否则边界物体会被切成两半 |
| `sliced.max-slices` | `64` | 切片数上限，超限抛 `MicaAiException` 并提示调整 |
| `sliced.merge-strategy` | `IOU_MERGE` | 跨片合并策略：`IOU_MERGE`（跨片去重）/ `KEEP_ALL`（保留各片原始框） |

> 切片边长**不是配置项**：它恒等于 `input-size`（模型输入边长）。切片边长若更小，
> letterbox 会把切片放大，小目标优势归零；若更大则浪费算力且超出模型输入。

### ⚠️ 切换模型族必须成对改两项

两族的**预处理 / 解码 / NMS 契约不同**（letterbox 填充值、归一化、输出张量形态都不一样），只改其一会因输入边长与模型张量形状不符而启动失败：

```yaml
mica:
  ai:
    detect:
      family: YOLOX_S
      model-path: model-tools/detect/models/object_detection_yolox_2022nov.onnx
      input-size: 640          # ← 必须跟着改
      score-threshold: 0.5     # YOLOX 官方 conf 0.5
      nms-threshold: 0.5       # YOLOX 官方 NMS 0.5
```

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class DetectController {

    private final DetectEngine detectEngine;

    /** 上传图片 → 返回带类别与置信度的检测框 */
    @PostMapping("/detect/image")
    public DetectResult detect(@RequestParam("file") MultipartFile file) throws IOException {
        return detectEngine.detectBytes(file.getBytes());
    }
}
```

### 公开 API（`DetectEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `detect(Mat)` | `DetectResult` | 检测 BGR 图；空图返回 `null` |
| `detectBytes(byte[])` | `DetectResult` | 检测图片字节（png / jpg 等） |
| `detectPath(String)` | `DetectResult` | 检测图片文件；文件不存在抛 `MicaAiException` |
| `getLabels()` | `String[]` | 类别名数组（默认 COCO 80 类） |
| `modelInputSize()` / `outputCount()` | `int` | 模型自检（NanoDet 416 / 6，YOLOX 640 / 1） |

`DetectResult` 提供 `getWidth()` / `getHeight()` / `getBoxes()` / `size()`；
`DetectBox` 提供 `getX1() / getY1() / getX2() / getY2() / getScore() / getClassId()` 与 `width() / height() / area() / iou(other) / label(labels)`。

### 切片推理（`SlicedDetectEngine`）

`SlicedDetectEngine` Bean 由 Starter 自动装配，复用已有的 `DetectEngine` Bean（不重复加载模型）：

```java
@RestController
@RequiredArgsConstructor
public class SlicedDetectController {

    private final SlicedDetectEngine slicedDetectEngine;

    /** 上传大图 → 切片推理 → 返回原图坐标系的检测框 */
    @PostMapping("/detect/sliced")
    public DetectResult sliced(@RequestParam("file") MultipartFile file) throws IOException {
        return slicedDetectEngine.detectBytes(file.getBytes());
    }
}
```

| 方法 | 返回 | 说明 |
|------|------|------|
| `detect(Mat)` | `DetectResult` | 切片检测 BGR 图；空图返回 `null` |
| `detectBytes(byte[])` | `DetectResult` | 切片检测图片字节 |
| `sliceCount(width, height)` | `int` | 纯计算：给定尺寸会被切成几片（不推理） |
| `getConfig()` | `SlicedDetectConfig` | 当前切片配置 |
| `getSource()` | `DetectSource` | 底层检测能力 |

> `SlicedDetectEngine` 不持有模型、不实现 `AutoCloseable`；底层 session 的生命周期仍由 `DetectEngine` Bean 管理。

---

## 4. 与多目标跟踪组合

检测框直接喂给 ByteTracker 即得稳定 track id：

```java
List<BBox> boxes = detectEngine.detectBytes(bytes).getBoxes().stream()
    .map(b -> new BBox(b.getX1(), b.getY1(), b.getX2(), b.getY2(), b.getScore()))
    .collect(Collectors.toList());
List<Track> tracks = byteTracker.update(boxes);
```

见 [`mica-ai-track-spring-boot-starter`](../mica-ai-track-spring-boot-starter/README.md)。

---

## 5. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `DetectEngine` Bean：

```java
@Bean(destroyMethod = "close")
public DetectEngine myEngine(DetectProperties props) {
    return DetectEngine.create(DetectAutoConfiguration.toConfig(props));
}
```

⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 6. 测试覆盖

- `DetectPropertiesTest`：默认值、字段全量映射、`onnx` 实例共享、切片配置映射
- [`DetectIntegrationTest`](../../mica-ai-core/mica-ai-detect/src/test/java/net/dreamlu/mica/ai/detect/DetectIntegrationTest.java)：真模型推理、输出节点数与输入边长自检、空图兜底、模型缺失时跳过
- [`SlicedDetectEngineTest`](../../mica-ai-core/mica-ai-detect/src/test/java/net/dreamlu/mica/ai/detect/SlicedDetectEngineTest.java) / [`SlicedDetectIntegrationTest`](../../mica-ai-core/mica-ai-detect/src/test/java/net/dreamlu/mica/ai/detect/SlicedDetectIntegrationTest.java)：切片网格、坐标回映射、跨片合并、切片数上限（假检测源零 ONNX + 真模型各一层）
