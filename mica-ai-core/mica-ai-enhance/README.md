# mica-ai-enhance

**Real-ESRGAN `realesr-general-x4v3` 图像超分**（4 倍放大），纯 **ONNX Runtime + openpnp/opencv** 实现，零 Python、零 PyTorch、零 Spring 依赖，JDK 8 兼容。

模型权重出自 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（**BSD-3-Clause**，可商用），ONNX 为 [microapp.io](https://microapp.io) 提供的官方权重转换版（2.33MB），已入库 `model-tools/enhance/models/`。

## 快速开始

```java
EnhanceConfig config = EnhanceConfig.builder()
    .modelPath("model-tools/enhance/models/realesr-general-x4v3.onnx")
    .build();

try (EnhanceEngine engine = EnhanceEngine.create(config)) {
    // 一张 480×360 的图 → 1920×1440 的 PNG
    byte[] upscaled = engine.upscalePath("photo.jpg");
    Files.write(Paths.get("photo@4x.png"), upscaled);
}
```

`EnhanceEngine` 实现 `AutoCloseable`，**必须** try-with-resources（内部持有 ONNX session 与会话选项）；`upscale(Mat)` 返回的 `Mat` 由调用方 release。

## 核心组件

| 类 | 职责 |
|----|------|
| `EnhanceEngine` | 门面：`upscale(Mat)` / `upscaleBytes(byte[])` / `upscalePath(String)`；负责 tile 切分与拼接 |
| `EnhanceDetector` | ONNX 加载、预处理、输出解码；推理后校验「输出宽高 == 输入宽高 × `scale`」 |
| `EnhanceConfig` | 模型路径 / `scale` / tile 参数 / ORT 参数的 builder + `validate()` |
| `EnhanceImageUtils` | BGR ↔ RGB CHW 张量、8bit clip、tile 裁剪、PNG 编解码 |

## I/O 格式（实测于 2026-10-30）

- **输入** 名称 `input`，形状 `[1,3,H,W]` float32，**H / W 为动态维度**：
  1. BGR → RGB
  2. CHW 排列
  3. 归一化 `/255`（**无 mean / std**）
- **输出** 1 个张量 `output`，形状 `[1,3,4H,4W]` float32：
  - 值域可轻微越界（实测约 `-0.03 ~ 1.05`），转 8bit 前 **clip 到 `[0,1]`** 再 `×255`
  - 通道序为 RGB，落盘前转回 BGR
- **放大倍数**：固定 4×；若换同族模型而 `scale` 配置不符，推理期立即抛异常（不会静默输出错尺寸）

> ⚠️ 输入是动态尺寸，因此**没有**「配置边长 == 模型边长」这类启动期校验；
> 契约校验放在每次推理的「输入→输出尺寸比例」上。
> ⚠️ 预处理只有 `/255`；输入像素必须做 `& 0xff` 无符号处理，否则 >127 的像素会变成负数（本项目已踩过）。

## tile 推理（大图不爆内存）

`EnhanceEngine.upscale` 对大图按 `tileSize` 切块：

1. 步长 `step = tileSize - tileOverlap`
2. 每个 tile 在四周额外取 `tileOverlap / 2` 像素作为上下文，推理后**只回写自己的核心区域**
3. 图像本身不超过一个 tile 时整图推理

这样既避免 12MP 图一次 4 倍推理的内存峰值，又不会在拼接处留下接缝（核心区域之外的上下文参与了卷积感受野）。

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `modelVersion` | `realesr-general-x4v3` | 内置 classpath 资源名，`modelPath` 为空时使用 |
| `modelPath` | 空 | 外置模型路径，优先级高于 `modelVersion` |
| `scale` | `4` | 放大倍数，必须与模型实际倍数一致 |
| `tileSize` | `256` | 单次推理 tile 边长（原图像素）；越小内存越省、越慢 |
| `tileOverlap` | `16` | tile 重叠边宽；`0` 表示不重叠（可能出现接缝） |
| `onnx` | defaults | ORT 会话参数（`intraOpNumThreads` / `device` 等） |

### 同族可插拔

| 模型 | 体积 | 说明 |
|------|------|------|
| `realesr-general-x4v3`（入库） | 2.33MB | 默认，通用 4 倍超分，CPU 友好 |
| `realesr-general-wdn-x4v3` | ~2.3MB | 去噪版（`denoiseStrength` 可调），需自行转换 ONNX 后配 `modelPath` |
| `RealESRGAN_x4plus` | ~64MB | RRDBNet 完整版，精度更高，**超出 50MB 入库上限**，外置 |

> 三者都是 4 倍、`/255`、RGB 契约；`RealESRGAN_x4plus` 体积大、CPU 更慢。

## Spring Boot Starter

```yaml
mica:
  ai:
    enhance:
      enabled: true
      model-path: model-tools/enhance/models/realesr-general-x4v3.onnx
      scale: 4
      tile-size: 256
      tile-overlap: 16
      onnx:
        intra-op-num-threads: 2
```

容器关闭时由 `destroyMethod = "close"` 自动释放 session 与会话选项。

## 测试

```bash
mvn -pl mica-ai-core/mica-ai-enhance -am test
mvn -pl mica-ai-core/mica-ai-enhance -am test -Dmica.ai.enhance.externalModel=/abs/path/model.onnx
```

- 单测（20 个）：配置默认值与校验；BGR→RGB CHW 的通道序与 `/255`；8bit clip；tile 裁剪独立性；PNG 编解码；异常路径
- 集成测试（3 个）：真 ONNX 推理输出恰为 4 倍尺寸、与双三次放大对比 MAE < 6%（不是黑图 / 噪点）、小图走整图路径、空图返回 `null`

## 模型获取

模型随仓库分发于 `model-tools/enhance/models/`（2.33MB，<50MB），License 原文见 `model-tools/enhance/LICENSE-realesrgan.txt`（BSD-3-Clause，需保留版权声明）。
