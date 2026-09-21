# mica-ai-matting

> [U²-Net](https://github.com/xuebinqin/U-2-Net)（显著性目标检测）通用抠图的 Java SDK：纯 ONNX Runtime + openpnp/opencv，零 Python，Apache-2.0 可商用。
> 内置 `u2netp`（4.36MB）开箱可用；**模型可插拔**，`u2net` / `u2net_human_seg` 等同族模型改一个 `model-path` 即可切换。

## 1. 模型规格

| 文件 | 来源 | License | 大小 | 说明 |
|------|------|---------|------|------|
| `u2netp.onnx` | [danielgatis/rembg release v0.0.0](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx)（原始权重出自 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)） | Apache 2.0 | 4.36 MB | U²-Net 轻量版，CPU/移动端友好，已直接入库 |
| `u2net.onnx` | 同上 release | Apache 2.0 | 168 MB | 通用显著性完整版，细碎边缘更好；**超 50MB 不入库** ⚠️ |
| `u2net_human_seg.onnx` | 同上 release | Apache 2.0 | 168 MB | **人体分割专用**，对「无人图」更保守；**超 50MB 不入库** ⚠️ |

入库模型：[`model-tools/matting/models/u2netp.onnx`](../../model-tools/matting/README.md)（4.36 MB，<50MB 约定内，随仓库分发）。
另两个 168MB 模型按 AGENTS.md §6.2 不入库，需自行下载后指向 `model-path`。

### 模型可插拔：切换模型不需要改 Java 代码

本模块的模型契约是**参数化**的（`modelPath` / `inputSize` / `mean` / `std` / `outputSelect`），
因此凡 rembg 导出的 U²-Net 族 ONNX，**只改配置即可切换**。实测（2026-09-21）三者契约完全一致：

| 模型 | 输入 | 输出 | 归一化 | 取哪个输出 |
|------|------|------|--------|-----------|
| `u2netp` | `input.1 [1,3,320,320]` | 7 个 `[1,1,320,320]`，名 `1959..1965` | `(0.485,0.456,0.406)/(0.229,0.224,0.225)` | 首个（d0） |
| `u2net` | 同 | 同（名同为 `1959..1965`） | 同 | 同 |
| `u2net_human_seg` | 同 | 同（名同为 `1959..1965`） | 同 | 同 |

外置模型配置示例：

```yaml
mica:
  ai:
    matting:
      model-path: /data/models/u2net_human_seg.onnx   # 只需改这一项
```

**选型建议**：
- 通用主体抠图（商品、静物、宠物、风景）→ `u2netp` 或 `u2net`
- 人像 / 人体抠图，且画面里**常常没有人**（需要模型在这种情况下「别乱抠」）→ `u2net_human_seg`

### ⚠️ `u2netp` 的真实身份：实测纠正过一条误导性线索

本机 `u2netp.onnx` 的 `sha256` 与 HuggingFace
[`BritishWerewolf/U-2-Net-Human-Seg`](https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg)
的 `onnx/model.onnx` **逐字节相同**，很容易据此推断「rembg 的 u2netp 就是 human_seg 权重」。

**但行为实测推翻了该推断**（同一张官方人像图 + 一张本地合成静物图，min-max 归一化后逐像素 MAE）：

| 对照 | 人像图 MAE | 静物图 MAE |
|------|-----------|-----------|
| `u2netp` vs **`u2net`** | 0.0085 | **0.0009** ← 几乎重合 |
| `u2netp` vs `u2net_human_seg` | 0.0112 | **0.0400** ← 差 44 倍 |

对「无人图」的保守度（前景占比 person/object 比）：`u2netp` **1.54×**、`u2net` **1.53×**、
`u2net_human_seg` **2.15×**。

**结论**：入库的 `u2netp` 行为上贴近**通用显著性 `u2net`**，而不是 `human_seg`。
文件哈希相同只说明「两处分发的产物来自同一次导出」，**不构成行为同源的证据** ——
判断权重语义必须看行为，不能只看文件名或哈希。复现脚本见
[`model-tools/matting/scripts/probe_model_identity.py`](../../model-tools/matting/scripts/probe_model_identity.py)。

### 模型 I/O（rembg 导出形态，2026-09-21 用 onnxruntime 1.30 实测确认）

**输入（1 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `input.1` | `[1, 3, 320, 320]` | `float32` | BGR→RGB 后 CHW，按 `(x/255 - mean) / std` 归一化；**直接 resize 到 320×320，不做 letterbox** |

**输出（7 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| 索引 0（即 d0） | `[1, 1, 320, 320]` | `float32` | **融合后的显著性掩码，本模块唯一消费的节点** |
| 索引 1..6（d1..d6） | `[1, 1, 320, 320]` | `float32` | 训练期 deep supervision 中间输出，形状与值域同 d0，**推理阶段必须忽略** |

> ⚠️ **输出节点名不可依赖**：该导出由 `torch.onnx.export` 产出，节点名是纯数字（实测 `1959`..`1965`），换一次导出就会变。因此定位 d0 的策略被显式化为可配置的 `output-select`：
> - `AUTO`（默认）：先按名称提示（`d0`/`alpha`/`saliency`/`mask`）精确匹配，再子串兜底；都没有时校验「7 个同形输出」并取首个
> - `FIRST`：不依赖节点名，强制取第 0 个输出（出口节点名恰好撞上提示词时，用它消除歧义）
> - `D0`：只接受精确名为 `d0` 的输出，匹配不到即**启动失败**（对导出有严格约定时的「宁可炸不要错」）
>
> 输入节点名同样按名称提示（`input.1`/`input`/`images`）解析。

### 输出语义：**已经是 Sigmoid，不是裸 logits**

实测 d0 直接落在 `[0, 1]`（合成图角落 `2.5e-5`、主体中心 `0.99999`），**没有负值、没有超出 1 的值** —— 该导出末尾已内置 Sigmoid，**不要**再对 d0 做 sigmoid。

但仍保留 `minMaxNormalize`（默认开启）作为 rembg 的参考行为：它把「该图内的相对显著性」铺满整个 alpha 值域。对低对比度输入差别巨大，实测 `flat.png`（近纯色）：

| `minMaxNormalize` | 最终掩码 max |
|-------------------|-------------|
| `false` | `0.00142710` |
| `true` | `0.673819`（≈ 470× 放大） |

> ⚠️ **min-max 的边界条件**：它只保证 **320×320 的模型输出**被铺满 `[0,1]`。掩码随后要 resize 回原尺寸，而 `INTER_LINEAR` 会对邻域加权平均，把孤立的单像素极大值稀释掉（实测 320×320 的 `1.0` → 240×180 的 `0.67`）。所以**不要在最终掩码上断言 `max == 1`**。

### 预处理参数：ImageNet 均值方差（RGB 顺序）

rembg 官方导出**未在模型内嵌入归一化**，归一化参数由调用方提供。本模块默认 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`（ImageNet 统计量，RGB 顺序）。

实测结论（真实照片 `test_img.jpg` 1920×1080，以 ImageNet-RGB 为基准的掩码 MAE）：

| 预处理 | MAE vs ImageNet-RGB |
|--------|---------------------|
| **通道序写反（BGR 当 RGB 喂）** | `0.0320` |
| 不做归一化 | `0.0287` |
| 均值镜像（`mean/std` 通道序写反，等价 BGR 喂入） | `0.0104` |
| 全用 0.5 | `0.0069` |

即：u2netp 对 `mean/std` 的绝对取值**并不敏感**（这是它能「瞎凑也算对」的原因），但**通道顺序敏感** —— 通道序写反的误差是镜像均值的 3 倍、是全 0.5 的近 5 倍。本模块固定走 ImageNet-RGB 这条参考路径。

> 验证方式：[`model-tools/matting/scripts/probe_preprocess.py`](../../model-tools/matting/scripts/probe_preprocess.py)（对照实验：同一张真实照片，只改 `mean`/`std`/通道序，比较掩码差异）。

## 2. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`MattingEngine`](src/main/java/net/dreamlu/mica/ai/matting/MattingEngine.java) | `AutoCloseable`，对外提供 `alpha` / `matte*` / `cutout*` / `matteBinary*` 四组接口 |
| 配置 | [`MattingConfig`](src/main/java/net/dreamlu/mica/ai/matting/config/MattingConfig.java) | Builder 模式：模型路径、输入边长、mean/std、插值、二值阈值、min-max 开关、底色、ONNX 参数 |
| 检测 | [`MattingDetector`](src/main/java/net/dreamlu/mica/ai/matting/detection/MattingDetector.java) | resize + 归一化 + ONNX 推理 + min-max + 掩码 resize 回原尺寸；输入输出名按提示解析并校验导出特征 |
| 图像工具 | [`MattingImageUtils`](src/main/java/net/dreamlu/mica/ai/matting/util/MattingImageUtils.java) | byte[] ↔ Mat、**BGR Mat → RGB CHW** float(mean/std)、掩码缩放/二值化、alpha 合成、PNG/JPEG 编码 |
| 插值枚举 | [`MattingInterpolation`](src/main/java/net/dreamlu/mica/ai/matting/config/MattingInterpolation.java) | `LINEAR` / `NEAREST` / `CUBIC` |
| 结果 | [`MattingResult`](src/main/java/net/dreamlu/mica/ai/matting/model/MattingResult.java) | `AutoCloseable`，持有原尺寸 `CV_32FC1` alpha + width/height |

### 处理流程

```
Path / Bytes / Mat
      │
      ▼
[MattingEngine]
   ├─ Imgcodecs.imdecode → BGR Mat
   ├─ [MattingDetector.alpha]
   │     ├─ Imgproc.resize → 320×320（直接拉伸，不 letterbox）
   │     ├─ BGR → RGB + (x/255-mean)/std + CHW float
   │     ├─ 输入 input.1 [1,3,320,320] → ONNX → 取索引 0 输出 [1,1,320,320]
   │     ├─ minMaxNormalize（默认开）→ [0,1]
   │     └─ Imgproc.resize 掩码 → 原图尺寸（Size 是 (宽,高)！）
   ▼
CV_32FC1 alpha（原尺寸）
   ├─ cutout*      → BGRA → PNG（透明底）
   ├─ cutoutOnColor* → alpha 合成纯色底 → PNG
   └─ matteBinary* → 阈值 0/255 → 单通道 PNG
```

## 3. 快速开始

### 纯 Java（零 Spring）

```java
MattingConfig config = MattingConfig.builder()
    .modelPath("model-tools/matting/models/u2netp.onnx")
    .build();
try (MattingEngine engine = MattingEngine.create(config)) {
    // 1) 直接要透明底 PNG
    byte[] png = engine.cutoutPath("photo.jpg");
    Files.write(Paths.get("photo-nobg.png"), png);

    // 2) 合成到纯色底
    byte[] onBlue = engine.cutoutOnColorBytes(
        Files.readAllBytes(Paths.get("photo.jpg")), new int[]{0, 128, 255});

    // 3) 只要 alpha 掩码，自行合成
    try (MattingResult r = engine.matteBytes(Files.readAllBytes(Paths.get("photo.jpg")))) {
        Mat alpha = r.getAlpha();   // CV_32FC1, [0,1], 原图尺寸
    }

    // 4) 二值掩码（硬边缘）
    byte[] mask = engine.matteBinaryBytes(Files.readAllBytes(Paths.get("photo.jpg")));
}
```

### Spring Boot Starter

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-matting-spring-boot-starter</artifactId>
</dependency>
```

```yaml
mica:
  ai:
    matting:
      enabled: true
      model-path: model-tools/matting/models/u2netp.onnx
      input-size: 320              # 必须等于模型输入边长，否则启动即失败
      binary-threshold: 0.5
      min-max-normalize: true
      interpolation: LINEAR
      background-color: [255, 255, 255]
      onnx:
        device: cpu                # cpu / gpu（⚠️ device 在 onnx 下，不是顶层）
```

```java
@RestController
@RequiredArgsConstructor
public class DemoController {
    private final MattingEngine mattingEngine;

    @PostMapping(value = "/matting", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] matting(@RequestParam MultipartFile file) throws IOException {
        return mattingEngine.cutoutBytes(file.getBytes());
    }
}
```

### 配置项（`mica.ai.matting` 前缀，见 `MattingProperties`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `u2netp` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/matting/u2netp.onnx` | 模型路径，支持 `classpath:`；**指向本地文件即可接入 168MB 级外置模型** |
| `input-size` | `320` | 模型输入边长；**必须等于模型输入边长**，不一致启动即抛 `MicaAiException` |
| `output-select` | `AUTO` | d0 定位策略：`AUTO` / `FIRST` / `D0`（详见「模型 I/O」一节） |
| `mean` / `std` | `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | 归一化参数（**RGB** 顺序，ImageNet 统计量） |
| `interpolation` | `LINEAR` | 掩码缩放插值：`LINEAR` / `NEAREST` / `CUBIC` |
| `binary-threshold` | `0.5` | 二值掩码阈值（0~1），仅二值输出接口使用 |
| `min-max-normalize` | `true` | 是否把 d0 min-max 拉伸到 `[0,1]`（rembg 参考行为） |
| `background-color` | `[255,255,255]` | 纯色底输出的默认底色（RGB） |
| `onnx.device` | `cpu` | `cpu` / `gpu`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | 0 = ORT 默认 |
| `onnx.inter-op-num-threads` | `0` | 0 = ORT 默认 |

### 公开 API

| 方法 | 返回 | 说明 |
|------|------|------|
| `alpha(Mat)` / `alphaBytes(byte[])` / `alphaPath(String)` | `Mat` | 原尺寸 `CV_32FC1` alpha（`[0,1]`），调用方负责 release |
| `matte(Mat)` / `matteBytes(byte[])` | `MattingResult` | 带尺寸的 alpha 包装，`AutoCloseable` |
| `cutoutBytes(byte[])` / `cutoutPath(String)` | `byte[]` | **透明底 PNG**（BGRA 4 通道） |
| `cutoutOnColorBytes(byte[], int[])` | `byte[]` | 合成到指定纯色底（RGB）后的 PNG |
| `matteBinaryBytes(byte[])` | `byte[]` | 二值掩码 PNG（单通道 `0/255`） |
| `modelInputSize()` / `outputCount()` | `int` | 模型自检：输入边长 / 输出节点数（U²-Net 族应为 320 / 7） |

## 4. 已知行为与边界

- **单主体假设**：U²-Net 是*显著性*检测，不是实例分割。画面里有多个主体时，输出是「所有显著区域」的并集，**无法区分个体**，也无法按类别筛选（要按类别请走通用检测类模型）。
- **`u2netp` 比完整版糊**：4.36MB 对发丝、蕾丝、网纱等细碎边缘会有明显损失。对质量要求高且能接受 168MB 体积时，改 `model-path` 指向 `u2net.onnx` 即可（契约完全相同，Java 代码零改动；超 50MB 约定需自行放置）。
- **`u2net_human_seg` 不是「更好的 u2netp」**：它是人体分割专用权重，对**有人**的画面质量相近，价值在于对**无人**画面更保守（实测 person/object 前景比 2.15× vs 1.54×）。抠商品 / 静物时换它不会有收益，反而可能漏抠。
- **别用文件名判断权重语义**：`u2netp` 与 HF 上名为 `U-2-Net-Human-Seg` 的 onnx 哈希相同，但行为上贴近 `u2net` 而非 `human_seg`（实测见 §1）。判断语义请用 `probe_model_identity.py` 跑行为对照。
- **小图会被放大**：输入小于 320×320 时也会 resize 到 320×320，掩码再缩回，等价于「先放大再缩回」，边缘会略软。这是官方参考行为的固有特性。
- **近纯色输入**：模型输出的极差可能极小（实测 `0.002`）。`minMaxNormalize=true` 会把噪声也一并放大，此时掩码**没有语义**，属于正常现象而非 bug —— 空图 / 纯色图本来就没有可抠的主体。

## 5. License

- 代码：Apache License 2.0
- 模型：U²-Net 族（`u2netp` / `u2net` / `u2net_human_seg`，均 Apache License 2.0），**可商用** ✅
- ⚠️ **不要**使用 `u2net_portrait.onnx`：其训练数据 APDrawing 含 NC 条款，会污染本项目的商用承诺
