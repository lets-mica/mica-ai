# mica-ai-layout

> [PaddleOCR PP-DocLayoutV2 / V3](https://github.com/PaddlePaddle/PaddleOCR) 文档版面分析（25 类 + V3 reading order）的 Java SDK：纯 ONNX Runtime + openpnp/opencv，零 Python，Apache-2.0 可商用。

## 1. 模型规格

| 文件 | 来源 | License | 大小 | 说明 |
|------|------|---------|------|------|
| `model.onnx` | [PaddleOCR PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 | 125 MB | RT-DETR 版面检测（25 类）+ 指针网络阅读顺序，已直接入库 |

模型已入库：[`model-tools/layout/models/model.onnx`](../../model-tools/layout/README.md)（125 MB，需注意仓库体积策略）。

### 模型 I/O（PaddleX 导出形态，2026-09-17 用 onnxruntime 实测确认）

**输入（3 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `image` | `[N, 3, 800, 800]` | `float32` | BGR→RGB 后 CHW，按 `(x/255 - mean) / std` 归一化；letterbox 到 800×800，**右下**补 114 |
| `im_shape` | `[N, 2]` | `float32` | `[orig_h, orig_w]`（**2 维，不是 3 维**） |
| `scale_factor` | `[N, 2]` | `float32` | 模型按 `输出坐标 = 原图坐标 / scale_factor` 计算，见下 |

**输出（3 个）**

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `fetch_name_0` | `[300, 7]` | `float32` | **未 NMS**：`[class_id, score, x1, y1, x2, y2, order]`，300 个 query 全输出，需自己按 score 过滤 + NMS |
| `fetch_name_1` | `[1]` | `int32` | 有效检测数（实测恒为 300，无过滤作用） |
| `fetch_name_2` | `[300, 200, 200]` | `int32` | 指针网络原始输出，**本模块不消费**：阅读顺序已由 `fetch_name_0` 第 7 列给出 |

#### 坐标空间（关键约定，写错会整体错位）

实测恒等式：**`输出坐标 = 原图坐标 / scale_factor`**（原图坐标由模型依据 `im_shape` 换算）。由此：

| 喂入 `scale_factor` | `fetch_name_0` 坐标落在 | 备注 |
|--------------------|----------------------|------|
| `1 / letterboxScale`（本模块采用） | **letterbox 画布坐标**，必然落在 `[0,800]` | 天然自检：越界即说明导出约定变了 |
| `1.0` | 原图坐标 | 数值等价但不自检，且居中 padding 时失效 |
| `letterboxScale` | 被放大 `1/r²` 倍（demo 图坐标到 5000+） | 错误写法 |

反算：`orig = (out - pad) / letterboxScale`，本模块 letterbox 为右下补边（padLeft = padTop = 0），再 clip 到原图。

> 验证脚本：[`model-tools/layout/scripts/probe_coord_space.py`](../../model-tools/layout/scripts/probe_coord_space.py)（固定 `image` 张量、只改 aux 输入做对照，300 行坐标误差 < 1px）。

### 25 类标签（V2/V3 共享字典）

| index | code | 含义 |
|-------|------|------|
| 0 | `abstract` | 摘要 |
| 1 | `algorithm` | 算法 |
| 2 | `aside_text` | 侧栏文本 |
| 3 | `chart` | 图表 |
| 4 | `content` | 内容（普通文本块） |
| 5 | `display_formula` | 行间公式 |
| 6 | `doc_title` | 文档主标题 |
| 7 | `figure_title` | 图标题 |
| 8 | `footer` | 页脚 |
| 9 | `footer_image` | 页脚图 |
| 10 | `footnote` | 脚注 |
| 11 | `formula_number` | 公式编号 |
| 12 | `header` | 页眉 |
| 13 | `header_image` | 页眉图 |
| 14 | `image` | 图 |
| 15 | `inline_formula` | 行内公式 |
| 16 | `number` | 页码 |
| 17 | `paragraph_title` | 段落标题 |
| 18 | `reference` | 参考文献 |
| 19 | `reference_content` | 参考文献内容 |
| 20 | `seal` | 印章 |
| 21 | `table` | 表格 |
| 22 | `text` | 文本 |
| 23 | `vertical_text` | 竖排文本 |
| 24 | `vision_footnote` | 视觉脚注 |

> 字典来源：[RapidLayout `write_dict.py`](https://rapidai.github.io/RapidLayout/main/blog/2026/02/10/support-PP-DocLayoutv2-v3/)。V2 / V3 共享。

### V2 vs V3

- **PP-DocLayoutV2**：Paddle2ONNX 转换后 ONNX 与 Paddle 推理结果有 **14.8% 误差**（[Paddle2ONNX #1608](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1608)），不建议生产
- **PP-DocLayoutV3**：同套转换脚本，误差仅 **1.57%**，**生产可用** ✅
- **V3 多**：`fetch_name_0` 第 7 列（order 键）→ 阅读顺序

## 2. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 主引擎 | [`LayoutPipeline`](src/main/java/net/dreamlu/mica/ai/layout/pipeline/LayoutPipeline.java) | `AutoCloseable`，对外提供 `detectPath` / `detectBytes` / `detect(Mat)` |
| 配置 | [`LayoutConfig`](src/main/java/net/dreamlu/mica/ai/layout/config/LayoutConfig.java) | Builder 模式：模型路径、版本、letterbox 边长、阈值、NMS、`maxDetections`、per-class 阈值、均值方差 |
| 检测 | [`LayoutDetector`](src/main/java/net/dreamlu/mica/ai/layout/detection/LayoutDetector.java) | letterbox + 3 输入 ONNX 推理；输入名按提示解析、检测输出按「float32 且最后一维 ≥ 6」推断（新版导出名为 `fetch_name_0`）；校验输入边长 |
| 后处理 | [`LayoutPostProcessor`](src/main/java/net/dreamlu/mica/ai/layout/postprocess/LayoutPostProcessor.java) | 7 列解析 → 反 letterbox → clip → per-class 阈值 → NMS → 整页 `image` 伪框过滤 → 阅读顺序 rank |
| 图像工具 | [`LayoutImageUtils`](src/main/java/net/dreamlu/mica/ai/layout/util/LayoutImageUtils.java) | byte[] ↔ Mat、**BGR Mat → RGB CHW** float(mean/std)（通道转换在此完成，调用方不要先 cvtColor） |
| 标签 | [`LayoutLabel`](src/main/java/net/dreamlu/mica/ai/layout/model/LayoutLabel.java) | 25 类 enum，含 `code` / `index` / `of(String)` / `ofIndex(int)` |
| 结果 | [`LayoutResult`](src/main/java/net/dreamlu/mica/ai/layout/model/LayoutResult.java) | 单个版面区域：label + bbox + score + index + order + readingOrder |

### 处理流程

```
Path / Bytes / Mat
      │
      ▼
[LayoutPipeline.detect]
   ├─ Imgcodecs.imdecode → BGR Mat
   ├─ [LayoutDetector.detect]
   │     ├─ letterbox (maxSide=800, 右下补 114)
   │     ├─ BGR → RGB + (x/255-mean)/std + CHW float
   │     ├─ 输入 image + im_shape[1,2] + scale_factor[1,2]=1/r → ONNX → fetch_name_0 [300,7]
   │     └─ [LayoutPostProcessor.postProcess]
   │           ├─ 7 列解析 [cls, score, x1,y1,x2,y2, order]
   │           ├─ 反 letterbox (out - pad) / r + 取整 + clip 到原图
   │           ├─ per-class 阈值过滤（默认 scoreThreshold=0.4）
   │           ├─ NMS（同类 IoU 0.6 / 异类 0.98）
   │           ├─ 整页 image 伪框过滤 + maxDetections 截断
   │           └─ 阅读顺序 rank：按第 7 列升序（同键高分先读）
   ▼
List<LayoutResult>
```

## 3. 快速开始

### 纯 Java（零 Spring）

```java
LayoutConfig config = LayoutConfig.builder()
    .modelPath("model-tools/layout/models/model.onnx")
    .build();
try (LayoutPipeline pipeline = LayoutPipeline.create(config)) {
    List<LayoutResult> regions = pipeline.detectPath("doc.png");
    for (LayoutResult r : regions) {
        // r.getLabelCode()   == "title"
        // r.getScore()       == 0.93
        // r.getBoundingBox() == [x1, y1, x2, y2]
        // r.getReadingOrder() == 3   // V3 reading order rank, V2 时为 -1
    }
}
```

### Spring Boot Starter

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-layout-spring-boot-starter</artifactId>
</dependency>
```

```yaml
mica:
  ai:
    layout:
      enabled: true
      model-path: model-tools/layout/models/model.onnx
      max-side-length: 800          # 必须等于模型输入边长，否则启动即失败
      score-threshold: 0.4
      layout-nms: true
      nms-threshold: 0.6            # 同类 IoU
      nms-diff-class-threshold: 0.98 # 异类 IoU（0.98 ⇒ 类间几乎不互斥）
      max-detections: 100
      class-score-thresholds:        # per-class 阈值（可选）
        14: 0.5   # image 类更严
        21: 0.5   # table 类更严
      onnx:
        device: cpu                  # cpu / gpu（⚠️ device 在 onnx 下，不是顶层）
```

```java
@RestController
@RequiredArgsConstructor
public class DemoController {
    private final LayoutPipeline pipeline;

    @PostMapping("/layout")
    public List<LayoutResult> detect(@RequestParam MultipartFile file) throws IOException {
        return pipeline.detectBytes(file.getBytes());
    }
}
```

### 配置项（`mica.ai.layout` 前缀，见 `LayoutProperties`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `v3` | 版本标识（日志用） |
| `model-path` | `classpath:mica-ai/models/layout/v3/model.onnx` | 模型路径，必填，支持 `classpath:` |
| `max-side-length` | `800` | letterbox 方形边长；**必须等于模型输入边长**，不一致启动即抛 `MicaAiException` |
| `score-threshold` | `0.4` | 全局置信度阈值 |
| `score-ratio` | `0`（关闭） | **相对阈值系数**：有效阈值 = `max(score-threshold, top1 × score-ratio)`。`0.6` 为实测推荐值，见下方「阈值标定」 |
| `class-score-thresholds` | 空 | per-class 阈值覆盖；key=classId, value=threshold |
| `layout-nms` | `true` | 是否启用 NMS（导出的 ONNX 未内置 NMS） |
| `nms-threshold` | `0.6` | 同类框 IoU 阈值 |
| `nms-diff-class-threshold` | `0.98` | 异类框 IoU 阈值（0.98 ⇒ 类间几乎不互斥，允许版面区域嵌套） |
| `max-detections` | `100` | 单图最多返回版面区域数 |
| `mean` / `std` | `[0.8286,0.8281,0.8282]` / `[0.1889,0.1889,0.1889]` | 归一化参数 |
| `onnx.device` | `cpu` | `cpu` / `gpu`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | 0 = ORT 默认 |
| `onnx.inter-op-num-threads` | `0` | 0 = ORT 默认 |

### 输出字段（`LayoutResult`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `label` | `LayoutLabel` | 25 类枚举 |
| `labelCode` | `String` | 字符串标签，与官方 `label_list` 完全一致 |
| `boundingBox` | `int[4]` | 原图坐标 `[x1, y1, x2, y2]`（已反 letterbox + clip） |
| `score` | `float` | 置信度（模型直接输出，无需再 softmax） |
| `index` | `int` | 模型输出行号（300 个 query 中的下标） |
| `order` | `int` | 按 score 降序过滤后的返回序号，从 0 起 |
| `readingOrder` | `int` | **V3 专属**：对齐 PaddleX `order` 字段的阅读顺序编号，**从 1 开始**且连续；跳过类（见下）与「模型未输出该列」时为 `-1` |

#### 阅读顺序与跳过类（`SKIP_ORDER_LABELS`）

对齐 PaddleX `LayoutAnalysisProcess.update_order_index`，**11 类标签不参与编号**且不占用序号：

```
figure_title, vision_footnote, image, chart, table,
header, header_image, footer, footer_image, footnote, aside_text
```

- 名单内的区域 `readingOrder = -1`（对应 PaddleX 的 `order = None`），其余区域从 **1** 开始连续编号
- 名单可用 `mica.ai.layout.skip-order-labels` 整体覆盖；配空列表表示所有类别都参与编号
- 需要与 PaddleX 的 `order` 字段**逐值一致**时，保持默认名单即可

> ⚠️ **并列 order 键的次序与 PaddleX 不保证逐值一致**：PaddleX 用 `np.argsort(boxes[:, 6])`，
> 其默认实现为 quicksort（**非稳定排序**，已实测 numpy 2.2.6），并列键顺序属实现细节。
> 本模块明确定义为「同键按 score 降序、再按原始下标」，确定性可复现；
> 两者在 **order 键的升序关系**上一致，仅并列键内部的相对次序可能不同。

## 4. 阈值标定

> 数据来源：官方 demo 文档 1654×2339，脚本 [`model-tools/layout/scripts/calibrate_thresholds.py`](../../model-tools/layout/scripts/calibrate_thresholds.py)

### 4.1 为什么需要 `score-ratio`

PP-DocLayoutV3 的分数分布存在**悬崖**：真实内容区集中在 0.69–0.94，随后直接跌到 0.518 / 0.437 的长尾。
单看长尾分数（0.437）会以为「阈值 0.4 刚好合适」，但实测暴露出两个反向问题：

| 输入 | top1 | 真实内容最低分 | 0.4 阈值的结果 |
|------|------|--------------|---------------|
| 整页 1654×2339 | 0.9449 | 0.6938 | 保留 14 个区域，**含 2 个长尾误检**（其中 0.437 那个是跨整页高度 0→2337 的长条，标签却是 `text`） |
| 右半列裁剪 | 0.3653 | 0.3113 | **全部低于 0.4 ⇒ 返回 0 个区域** |
| 上半裁剪 | 0.9468 | 0.7074 | 保留 6 个区域 |
| 1.5× 放大 | 0.9458 | 0.7346 | 保留 15 个区域，**含 3 个长尾误检** |

即：**绝对阈值无法同时兼顾「整页要收紧」与「窄图要放宽」** —— 同一张图换个裁剪方式，合适的阈值就变了。

### 4.2 相对阈值的表现

`score-ratio` 启用后有效阈值为 `max(score-threshold, top1 × score-ratio)`，随 top1 自适应：

| `score-ratio` | 整页 | 右半列裁剪 | 上半裁剪 | 1.5× 放大 |
|--------------|------|-----------|---------|----------|
| 0.5 | 13/15 | 6/6 ✅ | 6/6 ✅ | 12/15 |
| **0.6（推荐）** | **12/15 ✅** | **6/6 ✅** | **6/6 ✅** | **12/15 ✅** |
| 0.7 | 12/15 | 6/6 ✅ | 6/6 ✅ | 12/15 ✅ |

`0.6` 在四种变体下都能完整保留真实内容，同时丢掉长尾误检。整页场景下表为**保留 12 个区域、编号连续 1..12**
（`LayoutIntegrationTest#scoreRatioShouldProduceContiguousReadingOrder` 对此做了回归断言）。

> ⚠️ 默认值为 `0`（关闭）以保持向后兼容；建议在真实业务文档上调一遍后显式打开。

### 4.3 已知上游行为（非本模块 bug）

- **跨页长条 `text`/`doc_title` 误检不会被「整页伪框」规则拦掉**：已核对 PaddleX 源码，
  该规则（面积占比 > 0.82 横 / 0.93 竖）**只作用于 `image` 标签**，其它类一律保留 ⇒ 属于上游行为。
  本模块保持与之一致，不擅自扩大过滤范围；要靠 `score-ratio` 或 per-class 阈值来抑制。
- **合成图 / 程序绘制的纯色块文档表现差**：模型对硬边缘色块会大量输出 `image` 碎片。
  标定与验收**必须用真实文档图**（`demo.png` 是 800×800 合成图，只有 1 个区域、分数 0.437 勉强过阈值）。

## 5. License

- 代码：Apache License 2.0
- 模型：PP-DocLayoutV2 / V3（Apache License 2.0），**可商用** ✅