# mica-ai-layout-spring-boot-starter

> [PaddleOCR PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) 文档版面分析 Spring Boot Starter，基于 [mica-ai-layout](../../mica-ai-core/mica-ai-layout/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `LayoutPipeline` Bean，对外提供 `detectPath` / `detectBytes` / `detect(Mat)` 三个 API，返回 25 类版面区域（含 V3 阅读顺序）。

---

## ⚠️ 模型分发说明

PP-DocLayoutV3 模型 `model.onnx` **125 MB**，超出 AGENTS.md §6.2 的「<50MB 入库」约定，**不随仓库分发**。

- **首次使用**：本地下载 / 拷贝 `model.onnx` 到任意目录（如 `model-tools/layout/models/`），将 `mica.ai.layout.enabled` 显式设为 `true` 并指定 `model-path`
- **未配置 / 未启用时**：starter 默认 `enabled=true` 但 `model-path` 为空会 **fail-fast** 抛 `MicaAiException`；`mica.ai.example` 的 `application.yml` 默认 `enabled=false` 作为参考
- **集成测试**：对应模块的 `LayoutIntegrationTest` 在模型缺失时用 `Assumptions` 整类跳过，不阻塞 CI

详见 [`model-tools/layout/README.md`](../../model-tools/layout/README.md) 的「分发状态」。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-layout-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 125MB，需注意仓库 / 制品体积策略。

---

## 2. 配置项（`mica.ai.layout` 前缀）

```yaml
mica:
  ai:
    layout:
      enabled: true                                          # 默认 true；新克隆的仓库里设 false 以避免启动失败
      model-version: v3                                      # 版本标识（日志用）
      model-path: model-tools/layout/models/model.onnx       # 必填，支持 classpath:
      max-side-length: 800                                   # letterbox 方形边长，必须等于模型输入边长
      score-threshold: 0.4
      score-ratio: 0.6                                       # 相对阈值：max(score-threshold, top1 × 该值)；0=关闭
      layout-nms: true
      nms-threshold: 0.6                                     # 同类 IoU
      nms-diff-class-threshold: 0.98                         # 异类 IoU（0.98 ⇒ 类间几乎不互斥）
      max-detections: 100
      skip-order-labels:                                     # 不参与阅读顺序编号的标签（可选，缺省=PaddleX 11 类）
        - figure_title
        - image
      class-score-thresholds:                                # per-class 阈值覆盖（可选）
        14: 0.5                                              # image 类更严
        21: 0.5                                              # table 类更严
      mean: [0.8286, 0.8281, 0.8282]
      std:  [0.1889, 0.1889, 0.1889]
      onnx:
        device: CPU                                          # CPU / GPU（枚举）
        intra-op-num-threads: 0                              # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；模型缺失时请显式设为 `false` |
| `model-version` | `v3` | 版本标识（日志用） |
| `model-path` | `classpath:mica-ai/models/layout/v3/model.onnx` | 模型路径，**必填**，支持 `classpath:` |
| `max-side-length` | `800` | letterbox 方形边长；**必须等于模型输入边长**，不一致启动即抛 `MicaAiException` |
| `score-threshold` | `0.4` | 全局置信度阈值 |
| `score-ratio` | `0`（关闭） | **相对阈值系数**：有效阈值 = `max(score-threshold, top1 × score-ratio)`。实测 `0.6` 可抑制同页长尾误检，同时兼容单列裁剪图（详见模块 README「阈值标定」） |
| `class-score-thresholds` | 空 | per-class 阈值覆盖；key=classId, value=threshold |
| `layout-nms` | `true` | 是否启用 NMS（导出的 ONNX 未内置 NMS） |
| `nms-threshold` | `0.6` | 同类框 IoU 阈值 |
| `nms-diff-class-threshold` | `0.98` | 异类框 IoU 阈值（0.98 ⇒ 类间几乎不互斥，允许版面区域嵌套） |
| `max-detections` | `100` | 单图最多返回版面区域数 |
| `skip-order-labels` | PaddleX 11 类名单 | 不参与阅读顺序编号的标签 code；配空列表表示所有类别都参与编号。未识别的 code 启动即抛异常 |
| `mean` / `std` | `[0.8286,0.8281,0.8282]` / `[0.1889,0.1889,0.1889]` | 归一化参数（**非专家不要改**，与训练时保持一致） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举） |

> `enabled=false` 时不装配 `LayoutPipeline` Bean；`model-path` 缺失启动会 **fail-fast**。

---

## 3. 使用示例

### 3.1 REST 端点：上传图片 → 返回版面区域

```java
@RestController
@RequiredArgsConstructor
public class LayoutController {

    private final LayoutPipeline pipeline;

    @PostMapping("/layout/detect")
    public List<LayoutResult> detect(@RequestParam("file") MultipartFile file) throws IOException {
        return pipeline.detectBytes(file.getBytes());
    }
}
```

### 3.2 业务用法

```java
@Service
@RequiredArgsConstructor
public class DocumentService {

    private final LayoutPipeline pipeline;

    public List<LayoutResult> analyzeLayout(byte[] imageBytes) {
        List<LayoutResult> regions = pipeline.detectBytes(imageBytes);
        // 按 V3 readingOrder 升序遍历 = 推荐阅读顺序
        return regions.stream()
            .sorted(Comparator.comparingInt(LayoutResult::getReadingOrder))
            .collect(Collectors.toList());
    }
}
```

### 3.3 输出字段（`LayoutResult`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `label` | `LayoutLabel` | 25 类枚举 |
| `labelCode` | `String` | 字符串标签，与官方 `label_list` 完全一致 |
| `boundingBox` | `int[4]` | 原图坐标 `[x1, y1, x2, y2]`（已反 letterbox + clip） |
| `score` | `float` | 置信度（模型直接输出，无需再 softmax） |
| `index` | `int` | 模型输出行号（300 个 query 中的下标） |
| `order` | `int` | 按 score 降序过滤后的返回序号，从 0 起 |
| `readingOrder` | `int` | **V3 专属**：对齐 PaddleX `order` 的阅读顺序编号，**从 1 开始**且连续；跳过类（11 类）与「模型未输出该列」时为 `-1` |

---

## 4. 自定义 Pipeline

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `LayoutPipeline` Bean：

```java
@Configuration
public class MyLayoutConfig {

    @Bean
    public LayoutPipeline myPipeline(LayoutProperties props) {
        LayoutConfig cfg = LayoutConfig.builder()
            .modelPath(props.getModelPath())
            .maxSideLength(props.getMaxSideLength())
            .scoreThreshold(props.getScoreThreshold())
            .build();
        return LayoutPipeline.create(cfg);
    }
}
```

只要返回 `LayoutPipeline` 即可被 Starter 识别为自定义实现，**业务代码零改动**。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype / plate / layout 四个 Starter，
并提供了 `/layout/detect` 与 `/layout/detect-bytes` 端点；对 layout 能力默认 `enabled=false`
（模型未随仓库分发），本地放好模型后置 `true` 即可启用。