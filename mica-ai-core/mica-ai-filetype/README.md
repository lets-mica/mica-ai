# mica-ai-filetype

> Google [Magika](https://github.com/google/magika) 文件类型检测的 Java 复刻：纯 ONNX Runtime，**零 Python / 零 OpenCV**，Apache-2.0 可商用。

---

## 1. 模型规格

| 文件 | 来源 | License | 说明 |
|------|------|---------|------|
| `model.onnx` | [google/magika `standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3) | Apache 2.0 | Magika 文件类型检测 ONNX 模型（opset=15） |
| `config.min.json` | [`standard_v3_3/config.min.json`](https://github.com/google/magika/blob/main/assets/models/standard_v3_3/config.min.json) | Apache 2.0 | 超参：`beg_size` / `end_size` / `block_size` / `padding_token` / `target_labels_space` / `thresholds` / `overwrite_map` |
| `content_types_kb.min.json` | [`python/src/magika/config/content_types_kb.min.json`](https://github.com/google/magika/blob/main/python/src/magika/config/content_types_kb.min.json) | Apache 2.0 | 类型元数据：`mime_type` / `group` / `description` / `extensions` / `is_text`，353 个条目 |

模型已直接入库：[`model-tools/filetype/models/`](../../model-tools/filetype/README.md)（无需下载脚本）。

### 模型 I/O

| 项 | 值 |
|----|----|
| 输入节点 | `bytes`，shape `[1, 2048]`（batch 维度为 dynamic=0），dtype **`int32`** |
| 输出节点 | `target_label`，shape `[1, 214]`，dtype `float32`（已 softmax 归一化，无需再激活） |
| 输入含义 | 先从文件头/尾各读 `block_size=4096` 字节（seek 不全读），strip 首尾 ASCII whitespace，取前 `beg_size=1024` + 后 `end_size=1024` 字节转 int 0-255，padding 用 `padding_token=256` 填充至 `beg_size + end_size = 2048` |
| 输出含义 | 每个内容类型（214 种）的得分（已归一化为 0~1 的概率分布）；`argmax` 得到预测 label，`scores[best]` 作为 score |

> **注**：`block_size` (4096) 是读取窗口大小，`features_size = beg_size + end_size = 2048` 才是模型输入张量长度。来源：[`rust/lib/src/config.rs`](https://github.com/google/magika/blob/main/rust/lib/src/config.rs)。

---

## 2. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | **8+** | 推荐 Temurin / Azul Zulu 8、11、17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.18.0 | Maven 自动拉取；GPU 场景换 `onnxruntime_gpu` |

---

## 3. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-filetype</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

---

## 4. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`FiletypeDetector`](src/main/java/net/dreamlu/mica/ai/filetype/detection/FiletypeDetector.java) | 实现 `AutoCloseable`，对外提供 `detectPath` / `detectBytes` / `detectStream` |
| 配置 | [`FiletypeConfig`](src/main/java/net/dreamlu/mica/ai/filetype/config/FiletypeConfig.java) | Builder 模式，含 `modelPath` / `configPath` / `contentTypesPath` / `predictionMode` / `onnx` |
| 模型配置 POJO | [`ModelConfig`](src/main/java/net/dreamlu/mica/ai/filetype/config/ModelConfig.java) | 解析 `config.min.json`（`@JsonProperty` 显式注解 snake_case 字段） |
| 类型知识库 | [`ContentTypeRegistry`](src/main/java/net/dreamlu/mica/ai/filetype/config/ContentTypeRegistry.java) | 加载 `content_types_kb.min.json` + 缺失字段兜底（mime / group / extensions）+ unknown fallback |
| 特征提取 | [`FeaturesExtractor`](src/main/java/net/dreamlu/mica/ai/filetype/feature/FeaturesExtractor.java) | head+tail lstrip/rstrip → 取首尾 → padding 到 `beg_size+end_size=2048` |
| 后处理 | [`PredictionPostProcessor`](src/main/java/net/dreamlu/mica/ai/filetype/postprocess/PredictionPostProcessor.java) | overwrite_map → 按 `PredictionMode` 应用 thresholds → fallback（txt / unknown） |
| 结果对象 | [`FiletypeResult`](src/main/java/net/dreamlu/mica/ai/filetype/model/FiletypeResult.java) | 含 `outputLabel` / `modelLabel` / `score` / `contentType` / `mode` / `isText` |
| 类型元数据 | [`ContentTypeInfo`](src/main/java/net/dreamlu/mica/ai/filetype/model/ContentTypeInfo.java) | 解析 `content_types_kb.min.json` 单条记录 |
| 标签常量 | [`ContentTypeLabel`](src/main/java/net/dreamlu/mica/ai/filetype/model/ContentTypeLabel.java) | 特殊标签：`unknown` / `txt` / `empty` / `directory` / `symlink` / `undefined` |
| 预测模式 | [`PredictionMode`](src/main/java/net/dreamlu/mica/ai/filetype/config/PredictionMode.java) | 枚举：`HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE` / `BEST_GUESS` |

### 处理流程

```
Path / Bytes / Stream
      │
      ▼
[FiletypeDetector.detect]
   ├─ size == 0               → empty（特殊结果，score=1.0）
   ├─ size < min_file_size_for_dl=8 → few-bytes 兜底（UTF-8 解码成功 → txt，否则 unknown）
   └─ 否则
        ├─ [FeaturesExtractor]  head/tail 各取 block_size=4096 → strip ASCII whitespace → beg_size+end_size=2048 int 数组
        ├─ [OnnxModelSession]   喂 int32 tensor → ONNX Runtime → float[214]（已 softmax）
        ├─ argmax → modelLabel，score = scores[best]
        └─ [PredictionPostProcessor]  overwrite_map → 按 PredictionMode 应用 thresholds → 通过 → outputLabel；不通过 → txt / unknown 兜底
      │
      ▼
FiletypeResult{outputLabel, modelLabel, score, contentType, mode, isText}
```

> 注意 `modelLabel` 与 `outputLabel` 的区别：`modelLabel` 是模型 argmax 的直接结果（调试用），
> `outputLabel` 才是经 `overwrite_map` 改写 + 阈值判定后的对外标签，两者可能不同。

---

## 5. 预测模式

| 模式 | 行为 |
|------|------|
| `HIGH_CONFIDENCE` | `score >= thresholds[label]`（缺失则回退 `medium_confidence_threshold=0.5`）→ 信任模型输出；否则按 kb 的 `is_text` 兜底为 `txt` / `unknown` |
| `MEDIUM_CONFIDENCE` | `score >= medium_confidence_threshold=0.5` → 信任模型输出；否则同上兜底 |
| `BEST_GUESS` | 不做阈值检查，永远采用模型 argmax 输出 |

---

## 6. 快速使用

```java
FiletypeConfig config = FiletypeConfig.builder()
    .modelPath("classpath:mica-ai/models/filetype/standard_v3_3/model.onnx")
    .configPath("classpath:mica-ai/models/filetype/standard_v3_3/config.min.json")
    .contentTypesPath("classpath:mica-ai/models/filetype/standard_v3_3/content_types_kb.min.json")
    .predictionMode(PredictionMode.HIGH_CONFIDENCE)
    .build();
try (FiletypeDetector detector = new FiletypeDetector(config)) {
    FiletypeResult r = detector.detectPath(Paths.get("test.pdf"));
    // r.getOutputLabel()              == "pdf"
    // r.getContentType().getMimeType() == "application/pdf"
    // r.getContentType().getGroup()     == "document"
    // r.isText()                         == false
}
```

### 输出字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `outputLabel` | `String` | 最终对外标签（已应用 overwrite_map + 阈值 fallback） |
| `modelLabel`  | `String` | 模型直接 argmax 得到的标签（调试用） |
| `score`       | `float`  | 模型直接输出（已 softmax），`0.0 ~ 1.0` |
| `contentType` | `ContentTypeInfo` | 类型元数据 |
| `mode`        | `PredictionMode` | 当前预测模式 |
| `isText`      | `boolean` | 是否文本类型（`contentType.isText()` 或 fallback 到 `txt`） |

---

## 7. License

- 代码：Apache License 2.0
- 模型：Google Magika `standard_v3_3`，[Apache License 2.0](https://github.com/google/magika/blob/main/LICENSE)，**可商用** ✅

---

## 8. 相关链接

- 启动器：[`mica-ai-starters/mica-ai-filetype-spring-boot-starter/README.md`](../../mica-ai-starters/mica-ai-filetype-spring-boot-starter/README.md)
- 模型资产：[`model-tools/filetype/README.md`](../../model-tools/filetype/README.md)