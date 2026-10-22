# mica-ai-captcha

> [ddddocr（带带弟弟 OCR）](https://github.com/sml2h3/ddddocr) 通用验证码识别的 Java SDK：纯 ONNX Runtime，**零 Python**，**零 Spring**，MIT 可商用。
> **模型外置配置**：模型与字符集都不打包进 jar，由 `model-path` / `charset-path` 指向本地文件。

## 1. 模型规格

| 文件 | 来源 | License | 大小 | 说明 |
|------|------|---------|------|------|
| `ddddocr.onnx` | [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) `common.onnx`（动态量化衍生版） | MIT | 12.98 MB | 通用验证码识别，CRNN + CTC，已入库 |
| `charset.txt` | 同上（字符集索引映射表） | MIT | 24.4 KB | 8209 字符 + 1 blank = 8210 类，已入库 |

模型资产：[`model-tools/captcha/models/`](../../model-tools/captcha/README.md)（12.98 MB，<50MB 约定内，随仓库分发）。

> ⚠️ **本仓库的 `ddddocr.onnx` 是动态量化衍生版**，`sha256` 与上游 float32 原文件不同
> （`b8f2ad9c…` vs 上游 `33b5cd35…`），但**行为同源**：同一 CRNN + CTC 架构、同一 8210 类字符集、
> 同一输入高度 64，已由集成测试用真模型逐字符比对确认。
> 换成上游未量化的 float32 文件时，`model-path` 指过去即可，**Java 代码零改动**。

### 模型可插拔：换模型不改 Java 代码

契约参数全部显式化（`modelPath` / `charsetPath` / `inputHeight` / `normalization` / `interpolation`），
因此凡「CRNN + CTC、输出类别数与字符集一一对应」的验证码 ONNX，都**只改配置即可切换**：

| 配置项 | 默认值 | 换模型时可能要改 |
|--------|--------|------------------|
| `model-path` | 必填 | ✅ 必改 |
| `charset-path` | 必填 | ✅ 必改（必须与模型配套） |
| `input-height` | `64` | 模型输入高度不同时改 |
| `normalization` | `DIV_255` | 模型训练时用的是 `(x-0.5)/0.5` 则改 `MEAN_0_5` |
| `interpolation` | `BILINEAR` | 视模型训练时的缩放方式调整 |

外置模型配置示例：

```yaml
mica:
  ai:
    captcha:
      model-path: /data/models/my_captcha.onnx
      charset-path: /data/models/my_charset.txt
```

### 模型 I/O（实测于 2026-10-22）

**输入（1 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `input1` | `[1, 1, 64, image_width]` | `float32` | 单通道，**高度固定 64**，宽度按原图比例动态；灰度化后归一化 |

**输出（1 个）**

| 节点 | 运行时 shape | dtype | 说明 |
|------|-------------|-------|------|
| `387` | `[23, 1, 8210]` | `float32` | **原始 logits**（未 softmax，值域约 `[-7, 34]`），`23` = CTC 时间步 |

> ⚠️ **输出节点的声明形状是错的**：ONNX 元信息里写的是 `[1, seqlen]`，
> 但运行时实际返回 `[seqlen, 1, num_classes]`（实测 `[23, 1, 8210]`）。
> 这是动态量化模型（`DynamicQuantizeLinear` + `ConvInteger`）的常见问题。
> 因此本模块**不读声明形状**，一律按运行时数组维度归一化，并同时兼容
> `[seq][1][cls]` / `[1][seq][cls]` / `[seq][cls]` 三种布局。
>
> ⚠️ **输出节点名不可依赖**：`387` 是自动编号，换一次导出就会变。
> 本模块按「唯一输出」定位，多输出时再按名称提示回落。

### 预处理：与 ddddocr Python 原版严格对齐

```
原图
 └─ 透明通道处理（可选）：alpha=0 → 黑，0<alpha<255 → 与黑按 alpha 混合
 └─ 等比缩放到高 64：宽 = (int)(w * 64 / h)，最小 1 像素
 └─ 灰度化：L = 0.299R + 0.587G + 0.114B（同 PIL Image.convert('L')）
 └─ 归一化：DIV_255（默认）或 MEAN_0_5
 └─ CHW float32：[1, 1, 64, W]
```

### 归一化：`DIV_255` 与 `MEAN_0_5` 实测等价

ddddocr Python 原版对官方模型用的是 `(x - 0.5) / 0.5`，本模块默认 `DIV_255`。
**实测在同一张验证码图上两者给出完全相同的 argmax 与解码结果**：

| 归一化 | 解码结果 | 平均置信度 |
|--------|---------|-----------|
| `DIV_255`（`x/255`） | `SHTP` | 0.99960 |
| `MEAN_0_5`（`(x-0.5)/0.5`） | `SHTP` | 0.99943 |

原因：图头的 `DynamicQuantizeLinear` 按**实际张量的 min/max** 动态量化到 uint8，
对输入的绝对尺度不敏感。换成**非动态量化**的同族模型时两者差异会放大，
此时应按该模型训练时的归一化方式显式设置 `normalization`。

### 字符集契约

`charset.txt` 是**索引映射表**，不是普通文本：

- 每个字符占一个索引，**换行符不占索引**
- 索引 `0` 固定为 CTC 的 **blank**（空串），由 `CharsetManager` 补上
- 模型输出的第 `i` 类 ⇄ 字符集的第 `i` 个字符
- 共 `8209 + 1 = 8210` 类，与模型输出维度**严格相等**

> ⚠️ **不要去重**：字符集是索引映射表，出现重复字符时若去重，后续所有索引都会错位，
> 识别结果变成乱码。`CharsetManager` 刻意保留重复字符，并有单元测试守住这条。
>
> 字符集大小与模型类别数不一致时，模块在**首次推理时直接抛异常**，而不是输出乱码。

## 2. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`CaptchaEngine`](src/main/java/net/dreamlu/mica/ai/captcha/CaptchaEngine.java) | `AutoCloseable`，对外提供 `recognize*`（只要文本）与 `recognizeDetailed*`（带置信度）两组接口 |
| 配置 | [`CaptchaConfig`](src/main/java/net/dreamlu/mica/ai/captcha/config/CaptchaConfig.java) | Builder 模式：模型路径、字符集路径、输入高度、归一化、插值、透明处理、ONNX 参数 |
| 检测 | [`CaptchaDetector`](src/main/java/net/dreamlu/mica/ai/captcha/detection/CaptchaDetector.java) | 预处理 + ONNX 推理 + softmax + CTC 贪心解码；输出形状按运行时维度归一化 |
| 字符集 | [`CharsetManager`](src/main/java/net/dreamlu/mica/ai/captcha/charset/CharsetManager.java) | 加载外部字符集文件，建立「索引 → 字符」映射（**不去重**） |
| 图像工具 | [`CaptchaImageUtils`](src/main/java/net/dreamlu/mica/ai/captcha/util/CaptchaImageUtils.java) | byte[] / base64 / 路径解码、alpha 合成、等比缩放、灰度化、归一化 |
| 归一化枚举 | [`CaptchaNormalization`](src/main/java/net/dreamlu/mica/ai/captcha/config/CaptchaNormalization.java) | `DIV_255` / `MEAN_0_5` |
| 插值枚举 | [`CaptchaInterpolation`](src/main/java/net/dreamlu/mica/ai/captcha/config/CaptchaInterpolation.java) | `NEAREST` / `BILINEAR` / `BICUBIC` |
| 结果 | [`CaptchaResult`](src/main/java/net/dreamlu/mica/ai/captcha/model/CaptchaResult.java) | 文本 + 平均置信度 + 逐字符置信度 |

### 处理流程

```
byte[] / base64 / 文件路径
      │
      ▼
[CaptchaEngine]
   ├─ 解码为 BufferedImage（ImageIO）
   ├─ [CaptchaDetector.recognize]
   │     ├─ alpha 合成（可选）
   │     ├─ 等比缩放到高 64
   │     ├─ 灰度化 + 归一化 → [1,1,64,W]
   │     ├─ 输入 input1 → ONNX → 输出 [23,1,8210] 原始 logits
   │     ├─ 按运行时维度归一化为 [23][8210]
   │     ├─ 逐时刻 softmax + argmax
   │     └─ CTC 贪心解码（折叠重复、跳过 blank）
   ▼
CaptchaResult{text, confidence, charConfidences}
```

> 本模块只用 JDK 的 `javax.imageio` / `java.awt`，**不依赖 OpenCV 原生库**，
> 也**不依赖 Spring**（核心模块零 Spring 依赖）。

## 3. 快速开始

### 纯 Java（零 Spring）

```java
CaptchaConfig config = CaptchaConfig.builder()
    .modelPath("model-tools/captcha/models/ddddocr.onnx")
    .charsetPath("model-tools/captcha/models/charset.txt")
    .build();
try (CaptchaEngine engine = CaptchaEngine.create(config)) {
    // 1) 字节
    String text = engine.recognize(Files.readAllBytes(Paths.get("captcha.png")));

    // 2) base64（自动剥离 data:image/png;base64, 前缀）
    String text2 = engine.recognizeBase64(base64);

    // 3) 文件路径
    String text3 = engine.recognizePath("captcha.png");

    // 4) 要置信度 → 二次校验
    CaptchaResult result = engine.recognizeDetailed(Files.readAllBytes(Paths.get("captcha.png")));
    if (!result.isConfident(0.8f)) {
        // 置信度太低，视为识别失败
    }
}
```

### Spring Boot Starter

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-captcha-spring-boot-starter</artifactId>
</dependency>
```

```yaml
mica:
  ai:
    captcha:
      enabled: true
      model-path: model-tools/captcha/models/ddddocr.onnx    # 必填
      charset-path: model-tools/captcha/models/charset.txt  # 必填
      input-height: 64
      normalization: DIV_255
      interpolation: BILINEAR
      png-fix: true
      onnx:
        device: cpu                # cpu / gpu（⚠️ device 在 onnx 下，不是顶层）
```

```java
@RestController
@RequiredArgsConstructor
public class DemoController {
    private final CaptchaEngine captchaEngine;

    @PostMapping(value = "/captcha", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> captcha(@RequestPart("file") MultipartFile file) throws IOException {
        CaptchaResult r = captchaEngine.recognizeDetailed(file.getBytes());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text", r.getText());
        body.put("confidence", r.getConfidence());
        return body;
    }
}
```

### 配置项（`mica.ai.captcha` 前缀，见 `CaptchaProperties`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `CaptchaEngine` Bean |
| `model-path` | **必填** | 模型路径，支持 `classpath:`；**不随 jar 分发**，未配置时启动即失败并给出提示 |
| `charset-path` | **必填** | 字符集文件路径，支持 `classpath:`；必须与模型配套 |
| `input-height` | `64` | 模型输入高（NCHW 倒数第二维）；**必须等于模型输入高**，不一致启动即抛 `MicaAiException` |
| `normalization` | `DIV_255` | `DIV_255` / `MEAN_0_5`（见「归一化」一节） |
| `interpolation` | `BILINEAR` | 缩放插值：`NEAREST` / `BILINEAR` / `BICUBIC` |
| `png-fix` | `true` | 是否把带透明通道的 PNG 的透明像素按黑色合成 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | 0 = ORT 默认 |
| `onnx.inter-op-num-threads` | `0` | 0 = ORT 默认 |

### 公开 API（`CaptchaEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `recognize(byte[])` | `String` | 字节入参，只返回识别文本 |
| `recognizeBase64(String)` | `String` | base64 入参，自动剥离 data URI 前缀 |
| `recognizePath(String)` | `String` | 文件路径入参 |
| `recognizeDetailed(byte[])` | `CaptchaResult` | 字节入参，带平均置信度与逐字符置信度 |
| `recognizeDetailedBase64(String)` | `CaptchaResult` | base64 入参 |
| `recognizeDetailedPath(String)` | `CaptchaResult` | 文件路径入参 |
| `recognizeDetailed(BufferedImage)` | `CaptchaResult` | `BufferedImage` 入参 |
| `getConfig()` | `CaptchaConfig` | 构造时传入的配置 |

## 4. 已知行为与边界

- **模型与字符集必须成对**：字符集类别数与模型输出类别数不一致时，模块在首次推理时**直接抛异常**。
  换模型务必同时换字符集。
- **模型是外置的**：jar 里没有模型。部署时需自行把 `ddddocr.onnx` 与 `charset.txt` 放到应用可读位置，
  再配 `model-path` / `charset-path`。未配置时启动即失败（而不是等到第一次识别才报错）。
- **单行文字假设**：模型面向「文字占据图片主体」的单行验证码。多行 / 混排场景不在其擅长范围。
- **宽高自适应**：输入高度固定 64、宽度按原图比例。极扁或极窄的图会被拉伸，识别率下降。
- **归一化差异在本模型上不敏感**（动态量化），但换非量化模型时需按训练配置显式设置 `normalization`。
- **透明 PNG**：默认 `png-fix=true` 把透明像素按黑色合成，与 PIL `convert('L')` 在常见场景下一致；
  需要严格「忽略 alpha」时置 `false`。
- **置信度只统计已输出字符**（不含 blank 时间步），可直接用于「低于阈值判失败」的二次校验。

## 5. License

- 代码：Apache License 2.0
- 模型：[sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) **MIT License**，**可商用** ✅
  （MIT 要求保留版权声明与许可文本，随分发附带原始 LICENSE 即可）
