# mica-ai-textline-spring-boot-starter

> [PP-LCNet](https://github.com/PaddlePaddle/PaddleX) 文本行方向分类（0° / 180°）Spring Boot Starter，基于 [mica-ai-textline](../../mica-ai-core/mica-ai-textline/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `TextLineEngine` Bean，对外提供 `classify*`（只判定方向）与 `uprightBytes`（判定 + 转正）两组 API，典型用途是 **OCR 流水线的前置转正**。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-textline-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `PP-LCNet_x1_0_textline_ori.onnx` 仅 **6.46 MB**，**已随仓库分发**，无需额外下载。

---

## 2. 配置项（`mica.ai.textline` 前缀）

```yaml
mica:
  ai:
    textline:
      enabled: true
      model-version: PP-LCNet_x1_0_textline_ori
      model-path: model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx  # 支持 classpath:
      input-width: 160                    # 必须等于模型输入宽（NCHW 最后一维）
      input-height: 80                    # 必须等于模型输入高
      channel-order: BGR                  # BGR（官方约定）/ RGB（本任务实测不敏感）
      interpolation: LINEAR               # LINEAR / NEAREST / CUBIC
      upside-down-threshold: 0.5          # 判定为倒置的最低概率
      output-is-probability: false        # 官方 ONNX 输出为裸 logits
      mean: [0.485, 0.456, 0.406]         # PaddleX NormalizeImage 默认值
      std:  [0.229, 0.224, 0.225]
      onnx:
        device: CPU                       # CPU / GPU（枚举）
        intra-op-num-threads: 0           # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `TextLineEngine` Bean |
| `model-version` | `PP-LCNet_x1_0_textline_ori` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/textline/PP-LCNet_x1_0_textline_ori.onnx` | 模型路径，支持 `classpath:`；**指向本地文件即可切换到外置模型** |
| `input-width` | `160` | 模型输入宽（NCHW 最后一维）；**必须等于模型输入宽**，不一致启动即抛 `MicaAiException` |
| `input-height` | `80` | 模型输入高（NCHW 倒数第二维）；**必须等于模型输入高**。⚠️ 官方是 **80×160**（H×W），不是 160×80 |
| `channel-order` | `BGR` | `BGR`（官方 PaddleX 约定）/ `RGB`。本任务判定结构朝向而非颜色，实测两者结论一致 |
| `mean` / `std` | `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | 归一化参数（PaddleX `NormalizeImage` 默认值） |
| `interpolation` | `LINEAR` | 缩放到模型尺寸的插值：`LINEAR` / `NEAREST` / `CUBIC` |
| `upside-down-threshold` | `0.5` | 判定为「倒置」的最低概率。**实测真实扫描件上倒置态仅得 `p≈0.58`**（合成图可达 0.73），误旋转代价高时建议调到 `0.6~0.7` |
| `output-is-probability` | `false` | 模型输出是否已是 softmax 后的概率；官方导出为**裸 logits**，需本模块 softmax |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举，GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> 与 layout 不同，本能力模型**已入库**，默认配置即可启动，无需 `enabled: false` 兜底。

### 切换到轻量版模型

`PP-LCNet_x0_25_textline_ori`（约 0.96MB）与内置的 x1_0 版 I/O 契约**完全一致**，
因此**只改 `model-path` 就行**，无需改代码：

```yaml
mica:
  ai:
    textline:
      # 指向你自行下载的模型文件（约 0.96MB，未入库）
      model-path: /data/models/PP-LCNet_x0_25_textline_ori.onnx
```

模型获取方式与契约说明见 [`model-tools/textline/README.md`](../../model-tools/textline/README.md)。

---

## 3. 使用示例

### 3.1 REST 端点：上传单行文本图

```java
@RestController
@RequiredArgsConstructor
public class TextLineController {

    private final TextLineEngine textlineEngine;

    /** 只判定方向，返回 JSON */
    @PostMapping(value = "/textline/classify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> classify(@RequestPart("file") MultipartFile file) throws IOException {
        TextLineOrientationResult r = textlineEngine.classifyBytes(file.getBytes());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("orientation", r.getOrientation().name());   // DEGREE_0 / DEGREE_180
        body.put("label", r.getOrientation().getLabel());     // 0_degree / 180_degree
        body.put("score", r.getScore());
        body.put("upsideDown", r.isUpsideDown());
        body.put("angle", r.angle());                         // 0 / 180
        return body;
    }

    /** 倒置才旋转，方向正常原样回传；返回 PNG */
    @PostMapping(value = "/textline/upright", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] upright(@RequestPart("file") MultipartFile file) throws IOException {
        return textlineEngine.uprightBytes(file.getBytes());
    }
}
```

### 3.2 业务用法：OCR 前置转正流水线

```java
@Service
@RequiredArgsConstructor
public class OcrPreprocessService {

    private final TextLineEngine textlineEngine;

    /**
     * 对每一条已定位的文本行做「无脑转正」：倒置则旋转 180°，正常则原样返回。
     * 方向正常时返回的就是入参同一引用，零拷贝、零重编码。
     */
    public byte[] uprightLine(byte[] lineBytes) {
        return textlineEngine.uprightBytes(lineBytes);
    }

    /**
     * 需要区分「确定倒置 / 拿不准」时，读 score 自行决策（例如低于阈值走人工复核）。
     * 注意：这里的阈值来自 TextLineConfig，业务侧也可拿到 score 后再定策略。
     */
    public boolean needsManualReview(byte[] lineBytes) {
        TextLineOrientationResult r = textlineEngine.classifyBytes(lineBytes);
        return r.getScore() < 0.7f;
    }

    /** 拿到 Mat 自行续接后续 OpenCV 处理 */
    public Mat uprightMat(Mat lineBgr) {
        // ⚠️ 方向正常时返回的就是入参本身，注意不要重复 release
        return textlineEngine.rotateIfUpsideDown(lineBgr);
    }
}
```

### 3.3 公开 API（`TextLineEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `classify(Mat)` / `classifyBytes(byte[])` / `classifyPath(String)` | `TextLineOrientationResult` | 只判定方向；输入为 `null` / 空图时返回 `null` |
| `rotateIfUpsideDown(Mat)` | `Mat` | 倒置则返回旋转 180° 的**新 Mat**；方向正常时返回**入参本身** |
| `uprightBytes(byte[])` | `byte[]` | 倒置则旋转 180° 输出 PNG；方向正常时**原样返回输入字节** |
| `modelInputWidth()` / `modelInputHeight()` / `classCount()` | `int` | 模型自检：官方模型应为 160 / 80 / 2 |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `TextLineEngine` Bean：

```java
@Configuration
public class MyTextLineConfig {

    @Bean(destroyMethod = "close")
    public TextLineEngine myEngine(TextLineProperties props) {
        TextLineConfig cfg = TextLineConfig.builder()
            .modelPath(props.getModelPath())
            .inputWidth(props.getInputWidth())
            .inputHeight(props.getInputHeight())
            .upsideDownThreshold(0.7f)          // 更保守：拿不准就不转
            .build();
        return TextLineEngine.create(cfg);
    }
}
```

只要返回 `TextLineEngine` 即可被 Starter 识别为自定义实现，**业务代码零改动**。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype / plate / layout / matting / textline 各 Starter，
对应端点 [`TextLineController`](../../mica-ai-example/src/main/java/net/dreamlu/mica/ai/example/controller/TextLineController.java) 提供 `POST /textline/classify` 与 `POST /textline/upright`。

本 Starter 的 Bean 装配与配置绑定由 `TextLinePropertiesTest` 覆盖，核心链路由
[`TextLineIntegrationTest`](../../mica-ai-core/mica-ai-textline/src/test/java/net/dreamlu/mica/ai/textline/TextLineIntegrationTest.java) 用真模型覆盖。
