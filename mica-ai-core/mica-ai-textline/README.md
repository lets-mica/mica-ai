# mica-ai-textline

> [PP-LCNet](https://github.com/PaddlePaddle/PaddleX) 文本行方向分类（0° / 180°）的 Java SDK：纯 ONNX Runtime + openpnp/opencv，零 Python，Apache-2.0 可商用。
> 内置 `PP-LCNet_x1_0_textline_ori`（6.46MB）开箱可用；**模型可插拔**，轻量版 `PP-LCNet_x0_25_textline_ori`（约 0.96MB）改一个 `model-path` 即可切换。
>
> ⚠️ 输入是**已经定位并裁剪好的单行文本图**（不是整页文档）。它的职责只有一个：判断这一行是正的还是倒的，并把它转正。

## 1. 模型规格

| 文件 | 来源 | License | 大小 | 说明 |
|------|------|---------|------|------|
| `PP-LCNet_x1_0_textline_ori.onnx` | [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX)（PaddleOCR 文本行方向分类，paddle2onnx 转换产物） | Apache 2.0 | 6.46 MB | 默认模型，已直接入库 |
| `PP-LCNet_x0_25_textline_ori.onnx` | 同上 | Apache 2.0 | ~0.96 MB | 轻量版，**不入库**（按需自行下载），契约一致 |

入库模型：[`model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx`](../../model-tools/textline/README.md)（6,777,816 bytes，<50MB 约定内，随仓库分发）。
完整性指纹：`sha256 = 38aa97cd4be591e0ad304e659f07ba30d946f27a63315433f6659c69c8778345`。

### 模型可插拔：切换模型不需要改 Java 代码

本模块的模型契约是**参数化**的（`modelPath` / `inputWidth` / `inputHeight` / `mean` / `std` / `channelOrder` / `outputIsProbability`），
因此凡同任务的 PP-LCNet 文本行方向分类 ONNX，**只改配置即可切换**：

| 模型 | 大小 | 输入 | 输出 | 归一化 | 取哪个输出 |
|------|------|------|------|--------|-----------|
| `PP-LCNet_x1_0_textline_ori` | 6.46 MB | `x [N,3,80,160]` | `fetch_name_0 [N,2]`（裸 logits） | `(0.485,0.456,0.406)/(0.229,0.224,0.225)` | 唯一输出 |
| `PP-LCNet_x0_25_textline_ori` | ~0.96 MB | 同 | 同 | 同 | 同 |

外置模型配置示例：

```yaml
mica:
  ai:
    textline:
      model-path: /data/models/PP-LCNet_x0_25_textline_ori.onnx   # 只需改这一项
```

### 模型 I/O（PaddleX 导出形态，2026-09-21 实测确认）

**输入（1 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `[N, 3, 80, 160]` | `float32` | 动态 batch；resize 到 **160×80** 后 CHW，按 `(x/255 - mean) / std` 归一化 |

> ⚠️ **H=80、W=160，别写反**。官方 `inference.yml` 写的是 `ResizeImage: {size: [160, 80]}` ——
> PaddlePaddle 的 `size` 是 **(宽, 高)**，对应 NCHW 就是 `[N, 3, 80, 160]`。
> 文本行是窄长条，宽高写反会被拉成竖排，**方向判定随之失真却不会报错**。
> 本模块在构造期比对模型输入形状与 `input-width` / `input-height`，不一致直接 `MicaAiException` 快速失败。

**输出（1 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `fetch_name_0` | `[N, 2]` | `float32` | **原始 logits（未 softmax）**，索引 0 = `0_degree`、索引 1 = `180_degree` |

> ⚠️ **输出名不可依赖**：`fetch_name_0` 是 Paddle2ONNX 的自动命名。定位策略为「先按名称提示
> （`fetch_name_0` / `logits` / `output` / `prob` / `softmax`）精确匹配，再子串兜底，
> 最后回落到『唯一的 2 维 float 输出』结构判定」。输入名同理（`x` / `input` / `images` / `image`），
> 且校验「有且仅有 1 个输入」，否则启动失败。

### 类别语义：**索引 1 = 180 度**，不可按字典序猜测

类别顺序唯一权威来源是官方推理包的 `inference.yml`：

```yaml
PostProcess:
  Topk:
    topk: 1
    label_list:
    - 0_degree      # 索引 0
    - 180_degree    # 索引 1
```

写反的后果比不判更严重——**会把本来正确的行转成倒的**。因此本模块把该语义固化在
`TextLineOrientation` 枚举（`DEGREE_0(0, "0_degree")` / `DEGREE_180(180, "180_degree")`），
并由集成测试 `classIndexSemanticsShouldMatchOfficialLabelList` 兜底。

### 输出语义：裸 logits，需要 softmax

实测该 ONNX 直接输出 logits（如 `[+1.0000, +0.0000]`），**未内置 softmax**，故 `outputIsProbability` 默认 `false`，
由本模块做 softmax 后再当概率用。若换到已经内置 softmax 的导出，把该项置 `true` 即可，避免对概率再做一次 softmax。

### ⚠️ 真实数据的置信度远低于合成图（重要，直接影响阈值选型）

实测（2026-09-21）同一模型在两类输入上的表现差异显著：

| 输入 | logits | p(180) | 说明 |
|------|--------|--------|------|
| 合成图 `flipped.png`（干净、大字号、无噪） | 饱和为 `[+1.0000, +0.0000]` | **0.7311** | 2 类 softmax 在 `1:0` 下的数学上限 |
| 真实扫描件 `real_flipped.jpg`（官方样例） | 未饱和 | **0.5772** | 已逼近默认阈值 `0.5` |
| 同上，把阈值调到 `1.0` | — | 0.2689（被判为 0 度） | 阈值确实生效 |

**含义**：默认 `upside-down-threshold: 0.5` 在真实扫描件上没有多少余量。
如果「误旋转一张本来正确的行」的代价高于「漏掉一张倒置行」（多数 OCR 场景正是如此），
把阈值调到 `0.6 ~ 0.7` 让拿不准的行保持不动是更稳妥的策略。

集成测试 `realWorldSampleShouldBeClassifiedAndScoreShouldBeRealistic` 用 `0.55 ~ 0.72` 这个
「真实数据可达区间」做断言，而不是照抄合成图上的 0.73 —— 否则测试无法反映真实场景的退化风险。

### 预处理参数：PaddleX 默认归一化

`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`、`scale=1/255`，即 ImageNet 统计量。

**通道顺序**：PaddleX 的 `NormalizeImage` 在 OpenCV 原生 **BGR** 上逐通道施加，不做 RGB 转换，
故本模块默认 `channel-order: BGR` 以严格复刻官方行为。
但实测本任务对通道顺序**不敏感** —— 因为模型判定的是文字的结构朝向而非颜色：

| 通道序 | p(180) upright | p(180) flipped | 结论 |
|--------|---------------|----------------|------|
| BGR | 0.2689 | 0.7311 | 0°/180° |
| RGB | 0.2689 | 0.7311 | 0°/180°（一致） |

可以认为「BGR / RGB 结论一致」是模型的鲁棒性而非巧合，但**默认值仍按官方约定取 BGR**；
该配置项的价值在于换模型时的可调空间。复现脚本见
[`model-tools/textline/scripts/probe_contract.py`](../../model-tools/textline/scripts/probe_contract.py)。

## 2. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`TextLineEngine`](src/main/java/net/dreamlu/mica/ai/textline/TextLineEngine.java) | `AutoCloseable`，对外提供 `classify*`（只判定）/ `rotateIfUpsideDown` / `uprightBytes`（判定+转正） |
| 配置 | [`TextLineConfig`](src/main/java/net/dreamlu/mica/ai/textline/config/TextLineConfig.java) | Builder 模式：模型路径、输入宽高、通道序、mean/std、插值、倒置阈值、输出是否概率、ONNX 参数 |
| 检测器 | [`TextLineDetector`](src/main/java/net/dreamlu/mica/ai/textline/detection/TextLineDetector.java) | resize + 归一化 + ONNX 推理 + softmax + 阈值判定；解析并校验输入输出名与形状 |
| 图像工具 | [`TextLineImageUtils`](src/main/java/net/dreamlu/mica/ai/textline/util/TextLineImageUtils.java) | byte[] ↔ Mat、resize + 归一化的 **BGR Mat → CHW float**、`rotate180`、PNG 编码 |
| 方向枚举 | [`TextLineOrientation`](src/main/java/net/dreamlu/mica/ai/textline/model/TextLineOrientation.java) | `DEGREE_0` / `DEGREE_180`，带 `angle`（0/180）与 `label`（官方标签名） |
| 结果 | [`TextLineOrientationResult`](src/main/java/net/dreamlu/mica/ai/textline/model/TextLineOrientationResult.java) | `orientation` / `score` / `rawLogits`；**不持有 Mat，无需 close** |
| 通道序枚举 | [`TextLineChannelOrder`](src/main/java/net/dreamlu/mica/ai/textline/config/TextLineChannelOrder.java) | `BGR` / `RGB` |
| 插值枚举 | [`TextLineInterpolation`](src/main/java/net/dreamlu/mica/ai/textline/config/TextLineInterpolation.java) | `LINEAR` / `NEAREST` / `CUBIC` |

### 处理流程

```
Path / Bytes / Mat（单行文本图）
      │
      ▼
[TextLineEngine]
   ├─ Imgcodecs.imdecode → BGR Mat
   ├─ [TextLineDetector.classify]
   │     ├─ Imgproc.resize → 160×80（Size 是 (宽,高)！）
   │     ├─ (BGR → RGB 可选) + (x/255-mean)/std + CHW float
   │     ├─ 输入 x [1,3,80,160] → ONNX → 取 fetch_name_0 [1,2]
   │     ├─ softmax（outputIsProbability=false 时）
   │     └─ argmax + upsideDownThreshold 保护 → TextLineOrientation
   ▼
TextLineOrientationResult
   ├─ classify*          → 只判定，不产出图像
   ├─ uprightBytes       → 倒置则 rotate180 + PNG；正常则**原样返回输入字节**
   └─ rotateIfUpsideDown → 倒置则返回新 Mat；正常则**返回入参本身**
```

## 3. 快速开始

### 纯 Java（零 Spring）

```java
TextLineConfig config = TextLineConfig.builder()
    .modelPath("model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx")
    .build();
try (TextLineEngine engine = TextLineEngine.create(config)) {
    // 1) 只判定方向
    TextLineOrientationResult r = engine.classifyBytes(lineBytes);
    System.out.println(r.getOrientation() + " score=" + r.getScore()
        + " 转正需旋转 " + r.angle() + " 度");

    // 2) 一把梭：倒置则转正输出 PNG，正常则原样返回输入字节
    byte[] upright = engine.uprightBytes(lineBytes);
    Files.write(Paths.get("line-upright.png"), upright);

    // 3) 拿到 Mat 自行续接（注意：方向正常时返回的就是入参本身）
    Mat fixed = engine.rotateIfUpsideDown(bgr);
}
```

### Spring Boot Starter

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-textline-spring-boot-starter</artifactId>
</dependency>
```

```yaml
mica:
  ai:
    textline:
      enabled: true
      model-path: model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx
      input-width: 160             # 必须等于模型输入宽（NCHW 最后一维），否则启动即失败
      input-height: 80             # 必须等于模型输入高
      channel-order: BGR
      upside-down-threshold: 0.5   # 真实扫描件余量小，误旋转代价高时调到 0.6~0.7
      onnx:
        device: cpu                # cpu / gpu（⚠️ device 在 onnx 下，不是顶层）
```

```java
@RestController
@RequiredArgsConstructor
public class DemoController {
    private final TextLineEngine textlineEngine;

    /** 判定方向，返回 JSON；转正则返回 PNG */
    @PostMapping(value = "/textline/upright", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] upright(@RequestParam MultipartFile file) throws IOException {
        return textlineEngine.uprightBytes(file.getBytes());
    }
}
```

### 配置项（`mica.ai.textline` 前缀，见 `TextLineProperties`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `PP-LCNet_x1_0_textline_ori` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/textline/PP-LCNet_x1_0_textline_ori.onnx` | 模型路径，支持 `classpath:`；**指向本地文件即可接入外置模型** |
| `input-width` | `160` | 模型输入宽（NCHW 最后一维）；**与模型不一致启动即抛 `MicaAiException`** |
| `input-height` | `80` | 模型输入高（NCHW 倒数第二维）；同上 |
| `channel-order` | `BGR` | `BGR`（官方 PaddleX 约定）/ `RGB`；本任务实测不敏感 |
| `mean` / `std` | `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | 归一化参数（PaddleX `NormalizeImage` 默认值） |
| `interpolation` | `LINEAR` | 缩放到模型尺寸的插值：`LINEAR` / `NEAREST` / `CUBIC` |
| `upside-down-threshold` | `0.5` | 判定为「倒置」的最低概率；**调高可让拿不准的行保持不动** |
| `output-is-probability` | `false` | 模型输出是否已是 softmax 后的概率；官方导出为裸 logits |
| `onnx.device` | `cpu` | `cpu` / `gpu`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | 0 = ORT 默认 |
| `onnx.inter-op-num-threads` | `0` | 0 = ORT 默认 |

### 公开 API

| 方法 | 返回 | 说明 |
|------|------|------|
| `classify(Mat)` / `classifyBytes(byte[])` / `classifyPath(String)` | `TextLineOrientationResult` | 只判定方向；输入为 `null` / 空图时返回 `null` |
| `rotateIfUpsideDown(Mat)` | `Mat` | 倒置则返回旋转 180° 的**新 Mat**；方向正常时返回**入参本身** |
| `uprightBytes(byte[])` | `byte[]` | 倒置则旋转 180° 输出 PNG；方向正常时**原样返回输入字节**（不重编码，无画质损失） |
| `modelInputWidth()` / `modelInputHeight()` / `classCount()` | `int` | 模型自检：应为 160 / 80 / 2 |

`TextLineOrientationResult` 的字段：`orientation`（枚举）、`score`（softmax 概率）、`rawLogits`（原始 logits，便于排查阈值问题或换模型对照）。

## 4. 已知行为与边界

- **只管 0° / 180°，不管其它角度**：90° / 270° 的文本行会被强行归到这两类之一（模型设计如此）。整体页面旋转 / 倾斜纠正请用 [`mica-ai-layout`](../mica-ai-layout/README.md) 或专门的文档方向分类模型。
- **输入必须是单行**：喂整页文档会得到没有意义的结果。本模块不做文本检测，文本行的定位请先用 `mica-ppocr` 等能力完成。
- **真实扫描件置信度低**：合成图饱和到 0.7311，真实样例仅 0.5772（见 §1）。**不要**用 0.73 这类合成图数字去设阈值。
- **阈值是单向保护**：低于阈值时一律按「方向正常」处理，绝不主动旋转。它只会抑制误旋转，不会提升召回。
- **`uprightBytes` 可能返回同一个字节数组引用**（方向正常时，实测 `isSameAs`）。需要长期持有时自行复制。
- **`rotateIfUpsideDown` 可能返回入参本身**（方向正常时）。若用 try-with-resources 风格释放，注意不要重复 release 入参。
- **判定与转正是两次独立推理**：`uprightBytes` 内部只推理一次（转正后再判需要自己再调一次 `classifyBytes`），不做隐式复检。
- **`score` 是判定方向的置信度，不是「图像质量」**：两张都是 0.7311 的图，可能一张是清晰的合同正文、一张是干净的收据。

## 5. License

- 代码：Apache License 2.0
- 模型：PP-LCNet 文本行方向分类（PaddlePaddle/PaddleX，Apache License 2.0），**可商用** ✅
