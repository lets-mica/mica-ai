# mica-ai-classify-spring-boot-starter

> [PaddleClas PULC](https://github.com/PaddlePaddle/PaddleClas) 轻量图像分类 Spring Boot Starter，基于 [mica-ai-classify](../../mica-ai-core/mica-ai-classify/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 注入 `ClassifyEngine` Bean，对外提供 `classify(Mat)` / `classifyBytes` / `classifyPath`，输出 Top-K + 全量概率；单标签与多标签（`table_attribute`）共用同一套 API。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-classify-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；推荐 JDK 8+（Spring Boot 2.7.x）。
> 4 个 PULC 模型（4.14–6.79MB）已随仓库分发于 `model-tools/classify/models/`，无需额外下载。

---

## 2. 配置项（`mica.ai.classify` 前缀）

```yaml
mica:
  ai:
    classify:
      enabled: true
      model-path: model-tools/classify/models/text_image_orientation.onnx
      labels-path: model-tools/classify/models/text_image_orientation_labels.txt
      input-width: 224
      input-height: 224
      resize-short: 256          # table_attribute 用 0（直接 resize 224）
      top-k: 1
      multi-label: false
      onnx:
        device: CPU
        intra-op-num-threads: 0
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `text_image_orientation` | 模型标识；`model-path` 为空时按 `classpath:mica-ai/models/classify/<version>.onnx` 解析 |
| `model-path` | 空 | 模型路径，支持 `classpath:`；**实际部署请显式指向 `model-tools/classify/models/*.onnx`** |
| `labels-path` | **必填** | 与模型强耦合的 label_list（一行一个标签）；缺失启动即抛 `MicaAiException` |
| `input-width` / `input-height` | `224` / `224` | 必须等于模型输入宽高 |
| `resize-short` | `256` | 短边缩放目标；`0` 关闭短边缩放 + 中心裁剪（`table_attribute` 官方配置） |
| `channel-order` | `RGB` | `RGB` / `BGR`（PaddleClas 官方为 RGB） |
| `mean` / `std` | ImageNet | RGB 3 通道归一化参数，非专家不要改 |
| `interpolation` | `LINEAR` | `LINEAR` / `NEAREST` / `CUBIC` |
| `multi-label` | `false` | 每个输出独立 sigmoid（表格 6 属性） |
| `output-is-probability` | `true` | 官方 ONNX 已内置 softmax / sigmoid；裸 logits 导出时置 `false` |
| `top-k` | `1` | 返回前 K 个类别 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（GPU 需换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> `labels-path` **没有 classpath 兜底**：label_list 与模型必须成对配置，避免"模型换了、标签没换"的静默错位。

### 切换 PULC 模型

只改配置，不改 Java 代码：

```yaml
# 清晰度
mica.ai.classify.model-path: model-tools/classify/models/clarity_assessment.onnx
mica.ai.classify.labels-path: model-tools/classify/models/clarity_assessment_labels.txt

# 表格 6 属性（多标签）
mica.ai.classify.model-path: model-tools/classify/models/table_attribute.onnx
mica.ai.classify.labels-path: model-tools/classify/models/table_attribute_labels.txt
mica.ai.classify.resize-short: 0
mica.ai.classify.multi-label: true
mica.ai.classify.top-k: 6
```

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class ClassifyController {

    private final ClassifyEngine classifyEngine;

    /** 整图方向：返回 0 / 90 / 180 / 270 */
    @PostMapping("/classify/orientation")
    public ClassifyResult orientation(@RequestParam("file") MultipartFile file) throws IOException {
        return classifyEngine.classifyBytes(file.getBytes());
    }

    /** 表格属性：按属性名取概率 */
    @PostMapping("/classify/table")
    public Map<String, Float> table(@RequestParam("file") MultipartFile file) throws IOException {
        ClassifyResult r = classifyEngine.classifyBytes(file.getBytes());
        Map<String, Float> attrs = new LinkedHashMap<>();
        for (String label : classifyEngine.labels()) {
            attrs.put(label, r.scoreOf(label));
        }
        return attrs;
    }
}
```

### 公开 API（`ClassifyEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `classify(Mat)` | `ClassifyResult` | 原图分类；null / 空图返回 null |
| `classifyBytes(byte[])` | `ClassifyResult` | 字节入参，自动释放临时 Mat |
| `classifyPath(String)` | `ClassifyResult` | 文件路径入参 |
| `labels()` | `String[]` | label_list 副本 |
| `classCount()` | `int` | 类别数 |
| `isMultiLabel()` | `boolean` | 是否多标签 |
| `modelInputWidth()` / `modelInputHeight()` | `int` | 模型输入宽高 |

`ClassifyResult`：`top()` / `topLabel()` / `topScore()` / `scoreAt(int)` / `scoreOf(String)` / `labelsAbove(float)`，
不持有原生内存，无需 close。

---

## 4. 多模型共存

一个 `ClassifyEngine` 只绑定一个模型。需要同时用多个 PULC 模型时，自行声明多个 Bean，例如：

```java
@Configuration
public class MyClassifyConfig {

    @Bean(destroyMethod = "close")
    public ClassifyEngine tableClassifyEngine() {
        return ClassifyEngine.create(ClassifyConfig.builder()
            .modelPath("model-tools/classify/models/table_attribute.onnx")
            .labelsPath("model-tools/classify/models/table_attribute_labels.txt")
            .resizeShort(0)
            .multiLabel(true)
            .topK(6)
            .build());
    }
}
```

Starter 的默认 `classifyEngine` 走 `@ConditionalOnMissingBean`，自定义同名类型 Bean 会接管默认装配。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 测试覆盖

- `ClassifyPropertiesTest`：默认值、字段全量映射、`onnx` 实例共享、`labels-path` 必填校验
- [`ClassifyIntegrationTest`](../../mica-ai-core/mica-ai-classify/src/test/java/net/dreamlu/mica/ai/classify/ClassifyIntegrationTest.java)：4 个模型真实推理、多标签、label 数 / 输入尺寸不符快速失败
