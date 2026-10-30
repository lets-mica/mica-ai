# mica-ai-classify

PaddleClas **PULC 轻量图像分类族**，纯 **ONNX Runtime + openpnp/opencv** 实现，零 Python、零 PyTorch、零 Spring 依赖，JDK 8 兼容。

一个 `ClassifyEngine` 实例绑定「一个 ONNX 模型 + 一份 label_list」；需要同时用多个 PULC 模型
（如整图方向 + 清晰度 + 表格属性）时，分别创建多个实例即可。

与 `mica-ai-textline` 的区别：textline 是 **单行 0° / 180°** 专用能力；classify 是 **N 类 / 多标签通用分类底座**，
输出 `topK` + 全量概率，可承载 PULC 其余模型。

## 快速开始

```java
ClassifyConfig config = ClassifyConfig.builder()
    .modelPath("model-tools/classify/models/text_image_orientation.onnx")
    .labelsPath("model-tools/classify/models/text_image_orientation_labels.txt")
    .build();

try (ClassifyEngine engine = ClassifyEngine.create(config)) {
    ClassifyResult result = engine.classifyPath("scan.jpg");
    System.out.println(result.topLabel() + " " + result.topScore());  // 例如 "0 0.99"

    // 需要前 2 名时：ClassifyConfig.builder().topK(2)...
}
```

多标签（`table_attribute`）：

```java
ClassifyConfig table = ClassifyConfig.builder()
    .modelPath("model-tools/classify/models/table_attribute.onnx")
    .labelsPath("model-tools/classify/models/table_attribute_labels.txt")
    .resizeShort(0)      // 官方配置直接 resize 到 224，无短边缩放 + 中心裁剪
    .multiLabel(true)
    .topK(6)
    .build();

try (ClassifyEngine engine = ClassifyEngine.create(table)) {
    ClassifyResult r = engine.classifyPath("table.jpg");
    float source = r.scoreOf("source");          // 按属性名取概率
    List<ClassifyTopK> hit = r.labelsAbove(0.5f); // 阈值过滤
}
```

`ClassifyEngine` 实现 `AutoCloseable`，**必须** try-with-resources（内部持有 ONNX session 与会话选项）。
`ClassifyResult` 不持有原生内存，**无需 close**，可自由缓存。

## 核心组件

| 类 | 职责 |
|----|------|
| `core.ClassifyEngine` | 门面：`classify(Mat)` / `classifyBytes` / `classifyPath` + `labels()` / `classCount()` |
| `detection.PPClassificationDetector` | ONNX 加载、输入输出节点解析、预处理、激活、Top-K；构造期校验输入宽高与 label 数 |
| `config.ClassifyConfig` | 模型路径 / label_list / 输入宽高 / `resizeShort` / 通道序 / mean / std / 多标签 / `topK` / ORT 参数 |
| `postprocess.ClassifyPostProcessor` | 数值稳定 softmax、逐元素 sigmoid、稳定 Top-K（同分按索引升序） |
| `util.ClassifyImageUtils` | 解码、短边缩放 + 中心裁剪 / 直接 resize、RGB/BGR CHW 张量化 |
| `util.ClassifyLabelUtils` | `label_list` 读取（跳过空行，**保留重复标签**） |
| `model.ClassifyResult` / `ClassifyTopK` | Top-K 列表 + 原始索引顺序全量概率 + 按标签取值 |

## I/O 格式（实测于 2026-10-30）

- **输入** 名称 `x`，形状 `[N,3,224,224]`，NCHW float32：
  1. 短边缩放到 256 → 中心裁剪 224×224（`table_attribute` 直接 resize 到 224）
  2. BGR → **RGB**
  3. 归一化 `(x/255 - mean) / std`，ImageNet 均值 / 方差
- **输出**（已内置激活，不要重复做 softmax）：
  - 单标签模型：`softmax_1.tmp_0 [N,classes]`（`text_image_orientation` / `clarity_assessment` / `person_exists`）
  - 多标签模型：`sigmoid_2.tmp_0 [N,6]`（`table_attribute`）
- **输出节点名不硬编码**：按 `softmax / sigmoid / prob / logits / output` 名称提示匹配，再回落到
  「唯一的 float `[N,classes]` 输出」结构判定。
- **类别语义由 label_list 决定**，不按输出维度猜（AGENTS.md §7）。

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `modelVersion` | `text_image_orientation` | 内置 classpath 资源名，`modelPath` 为空时使用 |
| `modelPath` | 空 | 外置模型路径，优先级高于 `modelVersion` |
| `labelsPath` | **必填** | 与模型强耦合的 label_list；无 classpath 兜底，缺失即抛异常 |
| `inputWidth` / `inputHeight` | `224` / `224` | 必须等于模型输入宽高，否则构造期抛异常 |
| `resizeShort` | `256` | 短边缩放目标；`0` 表示直接 resize（`table_attribute`） |
| `channelOrder` | `RGB` | PaddleClas 约定；与 textline 的 BGR 不同 |
| `mean` / `std` | ImageNet | RGB 3 通道归一化参数 |
| `interpolation` | `LINEAR` | `LINEAR` / `NEAREST` / `CUBIC` |
| `multiLabel` | `false` | 每个输出独立 sigmoid（如表格 6 属性） |
| `outputIsProbability` | `true` | 官方导出已内置 softmax / sigmoid；换到裸 logits 导出时置 `false` |
| `topK` | `1` | 返回前 K 个类别 |
| `onnx` | defaults | ORT 会话参数（`intraOpNumThreads` / `device` 等） |

> `resizeShort > 0` 时必须 `>= max(inputWidth, inputHeight)`，否则中心裁剪会越界，`validate()` 直接拒绝。

## 同族可插拔

| 模型 | 体积 | 预处理 | 配置 |
|------|------|--------|------|
| `text_image_orientation`（入库） | 6.78MB | 256 → 裁剪 224 | 默认 |
| `clarity_assessment`（入库） | 4.14MB | 256 → 裁剪 224 | 默认 |
| `person_exists`（入库） | 6.77MB | 256 → 裁剪 224 | 默认 |
| `table_attribute`（入库） | 6.79MB | 直接 resize 224 | `resizeShort=0` + `multiLabel=true` |
| 其余 PULC 模型（`car_exists` / `safety_helmet` 等） | 7–8MB | 以官方 inference.yml 为准 | 转换 ONNX + 配 `labelsPath`，必要时调关键字 |

> 外置模型需自行 `paddle2onnx` 转换（一次性离线动作），转换命令与 License 见
> [`model-tools/classify/README.md`](../../model-tools/classify/README.md)。

## Spring Boot Starter

```yaml
mica:
  ai:
    classify:
      enabled: true
      model-path: model-tools/classify/models/text_image_orientation.onnx
      labels-path: model-tools/classify/models/text_image_orientation_labels.txt
      top-k: 1
      onnx:
        intra-op-num-threads: 2
```

容器关闭时由 `destroyMethod = "close"` 自动释放 session 与会话选项。

## 测试

```bash
mvn -pl mica-ai-core/mica-ai-classify -am test
mvn -pl mica-ai-core/mica-ai-classify -am test \
  -Dmica.ai.classify.externalModel=/abs/path/model.onnx \
  -Dmica.ai.classify.externalLabels=/abs/path/labels.txt
```

- 单测（36 个）：配置校验、label_list 读取、softmax / sigmoid / Top-K、CHW 张量化的通道序 / 归一化 / 裁剪尺寸 / 解码
- 集成测试（8 个，外置模型项自动跳过）：4 个模型真实推理与类别语义、`table_attribute` 多标签、
  label 数不一致与输入尺寸不符快速失败、null / 空图返回 null
