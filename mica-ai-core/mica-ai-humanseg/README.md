# mica-ai-humanseg

PP-HumanSeg V2-Lite **人像分割**，纯 **ONNX Runtime + openpnp/opencv** 实现，零 Python、零 PyTorch、零 Spring 依赖，JDK 8 兼容。

与 `mica-ai-matting`（U²-Net 通用显著性抠图）的区别：

| 维度 | `mica-ai-humanseg` | `mica-ai-matting` |
|------|--------------------|-------------------|
| 目标 | **人像专用语义分割** | 通用显著性目标 |
| 模型 | PP-HumanSeg V2-Lite（192×192，5.88MB） | U²-Net 族（深监督 7 输出） |
| 输出 | `[1,2,H,W]` 已 Softmax（通道 1 = 人像） | `[1,1,H,W]` 归一化显著性 |
| 归一化 | `(x/255 - 0.5) / 0.5` | min-max（u2net 契约） |

两者 I/O 契约不同，**不能**只改 `model-path` 互换，因此拆成独立模块（AGENTS.md §7）。

## 快速开始

```java
HumanSegConfig config = HumanSegConfig.builder()
    .modelPath("model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx")
    .build();

try (HumanSegEngine engine = HumanSegEngine.create(config)) {
    // 1) 直接拿透明底 PNG
    byte[] png = engine.cutoutPath("portrait.jpg");

    // 2) 换底色（RGB）
    byte[] white = engine.cutoutOnColorBytes(Files.readAllBytes(Path.of("portrait.jpg")),
        new int[]{255, 255, 255});

    // 3) 只要 alpha，自己合成
    Mat alpha = engine.alphaPath("portrait.jpg");   // CV_32FC1，[0,1]
    try {
        // ...
    } finally {
        alpha.release();
    }
}
```

`HumanSegEngine` 与 `HumanSegResult` 均实现 `AutoCloseable`，**必须** try-with-resources（内部持有 ONNX session、`Mat` 需主动 release）。

## 核心组件

| 类 | 职责 |
|----|------|
| `HumanSegEngine` | 门面：`alpha*` / `matte*` / `cutout*` / `maskBinaryBytes` |
| `HumanSegDetector` | ONNX 加载、预处理、前景通道解码、掩码回缩；构造期校验「配置宽高 == 模型输入宽高」 |
| `HumanSegConfig` | 模型路径 / 输入宽高 / mean / std / 二值阈值 / 底色 / ORT 参数的 builder + `validate()` |
| `HumanSegResult` | 原尺寸 alpha `Mat` + 宽高，自带 `close()` |
| `HumanSegImageUtils` | resize、RGB CHW 张量化、掩码缩放、二值化、BGRA 合成、PNG 编码 |

## I/O 格式（实测于 2026-10-30）

- **输入** 名称 `x`，形状 `[1,3,192,192]`，NCHW float32：
  1. 直接 resize 到 192×192（**不 letterbox**，与 PaddleSeg 参考实现一致）
  2. BGR → RGB
  3. 归一化 `(x/255 - mean) / std`，`mean = std = [0.5,0.5,0.5]`
- **输出** 1 个张量 `save_infer_model/scale_0.tmp_1`，形状 `[1,2,192,192]`，float32：
  - 图内**已内置 Softmax**，两通道逐像素和恒为 1
  - 通道 0 = 背景，通道 1 = 人像前景（实测人像图通道 1 均值 0.20~0.35，背景图 0.00）
  - 取通道 1 作为 alpha，`INTER_LINEAR` 缩放到原图尺寸
- **二值输出**：`alpha > binaryThreshold(0.5)` → 255，否则 0

> ⚠️ 输出节点名是 Paddle 导出的中间变量名，**每次导出都会变**，因此模块只校验「有且仅有 1 个输出」，不按名字查找。
> ⚠️ 输入节点名固定为 `x`（带 `input` / `images` 等常见别名兜底），若某次导出改成非常规名字会直接 fail-fast。

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `modelVersion` | `human_segmentation_pphumanseg_2023mar` | 内置 classpath 资源名，`modelPath` 为空时使用 |
| `modelPath` | 空 | 外置模型路径，优先级高于 `modelVersion` |
| `inputWidth` / `inputHeight` | `192` / `192` | 必须等于模型输入宽高，否则构造期抛异常 |
| `mean` / `std` | `[0.5,0.5,0.5]` | RGB 3 通道归一化参数 |
| `binaryThreshold` | `0.5` | 仅 `maskBinaryBytes` 使用 |
| `backgroundColor` | `[255,255,255]` | `cutoutOnColorBytes(bytes, null)` 时的默认底色（RGB） |
| `onnx` | defaults | ORT 会话参数（`intraOpNumThreads` / `device` 等） |

### 同族可插拔

契约差异全部显式化为配置项，切换**不改 Java 代码**：

| 模型 | 体积 | 输入 | 配置 |
|------|------|------|------|
| `human_segmentation_pphumanseg_2023mar`（入库） | 5.88MB | 192×192 | 默认即可 |
| Portrait-PP-HumanSegV2-Lite（外置） | 2.3MB | 256×144 | `modelPath` + `inputWidth=256` + `inputHeight=144` |

> 外置模型需自行 `paddle2onnx` 转换（一次性离线动作，不引入 Python 运行时依赖）。

## Spring Boot Starter

```yaml
mica:
  ai:
    humanseg:
      enabled: true
      model-path: model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx
      input-width: 192
      input-height: 192
      binary-threshold: 0.5
      onnx:
        intra-op-num-threads: 2
```

容器关闭时由 `destroyMethod = "close"` 自动释放 session 与会话选项。

## 测试

```bash
mvn -pl mica-ai-core/mica-ai-humanseg -am test
mvn -pl mica-ai-core/mica-ai-humanseg -am test -Dmica.ai.humanseg.externalModel=/abs/path/model.onnx
```

- 单测（25 个）：配置校验、CHW 张量化的通道序与归一化数值、掩码缩放、二值化、BGRA / 纯色合成、PNG 编解码
- 集成测试（5 个）：真 ONNX 加载、输入宽高 / 输出个数契约、主体中心 alpha ≈ 0.996 且四角 < 0.05、透明底 PNG 的 alpha 通道、二值掩码单通道、`inputWidth` 不符时快速失败

## 模型获取

模型随仓库分发于 `model-tools/humanseg/models/`（5.88MB，<50MB）。来源、License 与 SHA256 见 `model-tools/humanseg/README.md`。
