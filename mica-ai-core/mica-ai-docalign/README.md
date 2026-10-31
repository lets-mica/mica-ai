# mica-ai-docalign

DocAligner **文档角点检测 + 透视矫正**，纯 **ONNX Runtime + openpnp/opencv** 实现，零 Python、零 PyTorch、零 Spring 依赖，JDK 8 兼容。

典型链路：手机拍纸质文档 → **检测 4 个角点** → `getPerspectiveTransform` 拉平成矩形 → 交给 OCR / 版面分析。

## 快速开始

```java
DocAlignConfig config = DocAlignConfig.builder()
    .modelPath("model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx")
    .build();

try (DocAlignEngine engine = DocAlignEngine.create(config)) {
    // 1) 一步到位：角点检测 + 拉平，返回 PNG
    byte[] flat = engine.alignPath("document.jpg");

    // 2) 只要角点，自己做后续处理
    DocAlignResult result = engine.cornersPath("document.jpg");
    DocCorner lt = result.getTopLeft();    // 原图坐标系
    // ...

    // 3) 拉平到指定尺寸（如 A4 比例 2480×3508）
    try (Mat bgr = DocAlignImageUtils.byteArrayToMat(Files.readAllBytes(Path.of("document.jpg")))) {
        Mat warped = engine.warp(bgr, result, 2480, 3508);
        warped.release();
    }
}
```

`DocAlignEngine` 实现 `AutoCloseable`，**必须** try-with-resources（内部持有 ONNX session 与会话选项）；`warp` / `corners(Mat)` 返回的 `Mat` 由调用方 release。

## 核心组件

| 类 | 职责 |
|----|------|
| `DocAlignEngine` | 门面：`corners*` / `warp*` / `alignBytes` / `alignPath` |
| `DocAlignDetector` | ONNX 加载、预处理、4 通道热力图解码；构造期校验「配置宽高 == 模型输入宽高」「输出为 `[1,4,H,W]` float32」 |
| `DocAlignConfig` | 模型路径 / 输入宽高 / 热力图阈值 / ORT 参数的 builder + `validate()` |
| `DocAlignResult` | 4 个角点（顺序固定）+ 原图宽高，另提供 `getTopLeft/getTopRight/getBottomRight/getBottomLeft` |
| `DocCorner` | 单角点：`x` / `y` / `score` |
| `DocAlignImageUtils` | resize、CHW 张量化、热力图缩放、连通域质心定位、透视变换、PNG 编码 |

## I/O 格式（实测于 2026-10-30）

- **输入** 名称 `img`，形状 `[1,3,256,256]`，NCHW float32：
  1. 直接 resize 到 256×256（**不 letterbox**）
  2. BGR → RGB
  3. 归一化 `/255`（**无 mean / std**，与上游 DocAligner 一致）
- **输出** 1 个张量 `heatmap`，形状 `[1,4,128,128]`，float32：
  - 末端算子为 **Sigmoid**，值域天然为 `[0,1]`（不要再套一次 sigmoid）
  - 4 个通道依次为 **左上 → 右上 → 右下 → 左下**（用合成斜拍文档实测确认，写反会把文档镜像/翻转）
- **后处理**（对齐上游 `postprocess`）：每个通道热力图 `INTER_LINEAR` 缩放到原图尺寸 → 按 `heatmapThreshold`（默认 0.3）二值化 → 取**面积最大连通域的质心**作为角点
  - 无连通域时回退到该通道的 argmax 位置（保证不返回空点）
  - `score` 取该热力图的峰值
- **矫正面尺寸**：`suggestOutputSize()` 取「上下边长最大值 × 左右边长最大值」，避免拉伸失真

> ⚠️ 热力图质心会天然向文档内部收缩，实测角点偏差 10~23px（800×600 合成图）。需要绝对精确角点时，可在质心基础上用原图局部边缘做二次细化。
> ⚠️ 预处理归一化只有 `/255`，**不要**套 ImageNet mean/std，否则热力图完全失效。

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `modelVersion` | `lcnet100_h_e_bifpn_256_fp32` | 模型标识，仅用于日志展示，不参与路径解析 |
| `modelPath` | **必填** | 模型路径，支持 `classpath:` 前缀；**模型不随 jar 分发**，未配置时启动即失败并给出提示 |
| `inputWidth` / `inputHeight` | `256` / `256` | 必须等于模型输入宽高，否则构造期抛异常 |
| `heatmapThreshold` | `0.3` | 热力图二值化阈值，取值 `[0,1]` |
| `onnx` | defaults | ORT 会话参数（`intraOpNumThreads` / `device` 等） |

### 同族可插拔

DocAligner 官方每个模型都直发 ONNX，且契约一致（单输入 `img` + 单输出 4 通道热力图），**切换只改 `modelPath`**：

| 模型 | 体积 | 输入 | 精度 | 说明 |
|------|------|------|------|------|
| `lcnet100_h_e_bifpn_256_fp32`（入库） | 4.55MB | 256×256 | 均衡 | 默认，CPU 毫秒级 |
| `lcnet050_p_multi_decoder_l3_d64_256_fp32` | 4.91MB | 256×256 | 略低 | 轻量版 |
| `fastvit_t8_h_e_bifpn_256_fp32` | 13.2MB | 256×256 | 更高 | 外置 |
| `fastvit_sa24_h_e_bifpn_256_fp32` | 83MB | 256×256 | 最高 | 外置，**超出 50MB 入库上限** |

> 外置模型从 [DocsaidLab/DocAligner releases](https://github.com/DocsaidLab/DocAligner/releases) 下载后配 `model-path` 即可；契约不一致时按 AGENTS.md §7 先把差异显式化为配置项。

## Spring Boot Starter

```yaml
mica:
  ai:
    docalign:
      enabled: true
      model-path: model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx
      input-width: 256
      input-height: 256
      heatmap-threshold: 0.3
      onnx:
        intra-op-num-threads: 2
```

容器关闭时由 `destroyMethod = "close"` 自动释放 session 与会话选项。

## 测试

```bash
mvn -pl mica-ai-core/mica-ai-docalign -am test
mvn -pl mica-ai-core/mica-ai-docalign -am test -Dmica.ai.docalign.externalModel=/abs/path/model.onnx
```

- 单测（19 个）：配置校验与 classpath 路径解析；CHW 张量化的通道序与 `/255` 数值；连通域质心定位（含最大连通域优先、低于阈值回退 argmax）；`suggestOutputSize`；透视变换映射；解码 / 编码 / 资源释放
- 集成测试（4 个）：真 ONNX 加载 + 输入宽高 / 输出个数契约、**4 通道角点顺序**（合成斜拍图，30px 容差）、`alignBytes` 拉平后尺寸与 `suggestOutputSize` 一致且画面以纸面白为主、`inputWidth` 不符时快速失败、空图返回 `null`

## 模型获取

模型随仓库分发于 `model-tools/docalign/models/`（4.55MB，<50MB）。来源、License 与 SHA256 见 `model-tools/docalign/README.md`。
