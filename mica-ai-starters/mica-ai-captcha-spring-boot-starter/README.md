# mica-ai-captcha-spring-boot-starter

> [ddddocr](https://github.com/sml2h3/ddddocr) 通用验证码识别 Spring Boot Starter，基于 [mica-ai-captcha](../../mica-ai-core/mica-ai-captcha/README.md) 核心模块。**MIT 可商用**。
>
> 注入 `CaptchaEngine` Bean 即可用。**模型外置**：`model-path` / `charset-path` 必填，模型与字符集不随 jar 分发。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-captcha-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `ddddocr.onnx`（12.98 MB）**已随仓库分发**，但**不打包进 jar**，需指向本地文件。

---

## 2. 配置项（`mica.ai.captcha` 前缀）

```yaml
mica:
  ai:
    captcha:
      enabled: true
      model-path: model-tools/captcha/models/ddddocr.onnx     # 必填，支持 classpath:
      charset-path: model-tools/captcha/models/charset.txt   # 必填，支持 classpath:
      input-height: 64                    # 必须等于模型输入高（NCHW 倒数第二维）
      normalization: DIV_255              # DIV_255 / MEAN_0_5（对齐 ddddocr Python 原版）
      interpolation: BILINEAR             # NEAREST / BILINEAR / BICUBIC
      png-fix: true                       # 透明 PNG 的透明像素按黑色合成
      onnx:
        device: CPU                       # CPU / GPU（枚举）
        intra-op-num-threads: 0           # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `CaptchaEngine` Bean |
| `model-path` | **必填** | 模型路径，支持 `classpath:`；**不随 jar 分发**，未配置时启动即失败并给出提示 |
| `charset-path` | **必填** | 字符集文件路径，支持 `classpath:`；必须与模型配套，否则识别结果是乱码 |
| `input-height` | `64` | 模型输入高（NCHW 倒数第二维）；**必须等于模型输入高**，不一致启动即抛 `MicaAiException` |
| `normalization` | `DIV_255` | `DIV_255` / `MEAN_0_5`。本模型（动态量化）下两者实测等价，换非量化模型时按训练配置设置 |
| `interpolation` | `BILINEAR` | 缩放插值：`NEAREST` / `BILINEAR` / `BICUBIC` |
| `png-fix` | `true` | 是否把带透明通道的 PNG 的透明像素按黑色合成 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> **模型外置是本 Starter 的硬性要求**：`model-path` 与 `charset-path` 不填，
> 容器启动时即抛 `MicaAiException` 并提示正确路径，不会拖到第一次识别才暴露。

### 切换到外置模型

同族的其它 CRNN + CTC 验证码模型**只改配置即可切换**，无需改代码：

```yaml
mica:
  ai:
    captcha:
      # 指向你自行放置的模型与配套字符集
      model-path: /data/models/my_captcha.onnx
      charset-path: /data/models/my_charset.txt
      input-height: 64
      normalization: MEAN_0_5   # 若该模型训练时用的是 (x-0.5)/0.5
```

---

## 3. 使用示例

### 3.1 REST 端点：上传验证码图片

```java
@RestController
@RequiredArgsConstructor
public class CaptchaController {

    private final CaptchaEngine captchaEngine;

    /** 只要文本 */
    @PostMapping(value = "/captcha/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> recognize(@RequestPart("file") MultipartFile file) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text", captchaEngine.recognize(file.getBytes()));
        return body;
    }

    /** 要置信度，便于「识别失败走人工」 */
    @PostMapping(value = "/captcha/recognize-detail", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> recognizeDetail(@RequestPart("file") MultipartFile file) throws IOException {
        CaptchaResult r = captchaEngine.recognizeDetailed(file.getBytes());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text", r.getText());
        body.put("confidence", r.getConfidence());
        body.put("charConfidences", r.getCharConfidences());
        return body;
    }
}
```

### 3.2 业务用法：低置信度走人工复核

```java
@Service
@RequiredArgsConstructor
public class CaptchaService {

    private final CaptchaEngine captchaEngine;

    /**
     * 置信度低于 0.8 时返回 null，交给上层决定是否走人工 / 重新获取验证码。
     */
    public String solveOrNull(byte[] captchaBytes) {
        CaptchaResult r = captchaEngine.recognizeDetailed(captchaBytes);
        return r.isConfident(0.8f) ? r.getText() : null;
    }
}
```

### 3.3 公开 API（`CaptchaEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `recognize(byte[])` / `recognizeBase64(String)` / `recognizePath(String)` | `String` | 只返回识别文本 |
| `recognizeDetailed(byte[])` / `recognizeDetailedBase64(String)` / `recognizeDetailedPath(String)` / `recognizeDetailed(BufferedImage)` | `CaptchaResult` | 带平均置信度与逐字符置信度 |
| `getConfig()` | `CaptchaConfig` | 构造时传入的配置 |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `CaptchaEngine` Bean：

```java
@Configuration
public class MyCaptchaConfig {

    @Bean(destroyMethod = "close")
    public CaptchaEngine myEngine(CaptchaProperties props) {
        CaptchaConfig cfg = CaptchaConfig.builder()
            .modelPath(props.getModelPath())
            .charsetPath(props.getCharsetPath())
            .inputHeight(props.getInputHeight())
            .normalization(CaptchaNormalization.MEAN_0_5)
            .build();
        return CaptchaEngine.create(cfg);
    }
}
```

只要返回 `CaptchaEngine` 即可被 Starter 识别为自定义实现，**业务代码零改动**。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype / plate / layout / matting / textline / captcha 各 Starter，
对应端点 [`CaptchaController`](../../mica-ai-example/src/main/java/net/dreamlu/mica/ai/example/controller/CaptchaController.java) 提供 `POST /captcha/recognize`。

本 Starter 的 Bean 装配与配置绑定由 `CaptchaPropertiesTest` 覆盖，核心链路由
[`CaptchaIntegrationTest`](../../mica-ai-core/mica-ai-captcha/src/test/java/net/dreamlu/mica/ai/captcha/CaptchaIntegrationTest.java) 用真模型覆盖。
