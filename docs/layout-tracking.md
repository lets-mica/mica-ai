# mica-ai-layout 进度跟踪

> 跟踪 PP-DocLayoutV3 版面分析模块的落地进度、I/O 决策与待办事项。
> 这份文档**只在落地期间使用**，模块稳定后会并入主 README。
> **最后更新**：2026-09-17（真 ONNX 已入库；I/O 形状与**输出坐标空间**均已用 onnxruntime Python 实测闭环，见 2.1）

## 1. 任务来源

`mica-ai-layout` 新模块（首版于 2026-09-17 按「RT-DETR 标准 2 输入 2 输出 / 11 类」假设完成初稿）。

通过查证 [RapidLayout V2/V3 集成文章](https://rapidai.github.io/RapidLayout/main/blog/2026/02/10/support-PP-DocLayoutv2-v3/) 中注入到 `paddlex/inference/models/layout_analysis/predictor.py` 第 103 行附近的代码，确认初稿假设**与 PaddleX 实际导出形态不符**，需要二次落地。

## 2. 真实 ONNX I/O（已用 onnxruntime Python 实测确认）

> 模型：`PP-DocLayoutV3_infer` → paddle2onnx 2.1.0 导出 → constant folding（6440→2818 节点）→ 125MB `model.onnx`，已入库 `model-tools/layout/models/model.onnx`。

### 输入 3 个

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `image` | `[N, 3, 800, 800]` | `float32` | BGR→RGB，CHW 布局，letterbox 到 800×800，归一化 `(x/255-mean)/std` |
| `im_shape` | `[N, 2]` | `float32` | `[orig_h, orig_w]`，原图高宽（**注意：不是 3 维，是 2 维**） |
| `scale_factor` | `[N, 2]` | `float32` | `[h_scale, w_scale]`，**语义已实测确认：`输出坐标 = 原图坐标 / scale_factor`**（详见第 2.1 节） |

### 输出 3 个

| 节点 | shape | dtype | 说明 |
|------|-------|-------|------|
| `fetch_name_0` | `[300, 7]` | `float32` | **未 NMS** 的检测框，每行：`[class_id, score, x1, y1, x2, y2, order]`；坐标空间随 `scale_factor` 变化（见 2.1），第 7 列为阅读顺序键 |
| `fetch_name_1` | `[1]` | `int32` | 有效检测数（实测恒为 300，即所有 query 都输出，需自行按 score 过滤） |
| `fetch_name_2` | `[300, 200, 200]` | `int32` | 指针网络原始输出；**Java 端不消费** —— 阅读顺序已由 `fetch_name_0` 第 7 列给出（见 2.2） |

### 关键修正（vs. 初稿假设）

| 维度 | 初稿/文档假设 | **真实形态** |
|------|-------------|-------------|
| `im_shape` | `[1, 3]` `[h, w, ?]` | **`[1, 2]` `[h, w]`** |
| 输出节点数 | 2（boxes + order_index） | **3**（`fetch_name_0/1/2`） |
| boxes 列序 | `[x1,y1,x2,y2,score,cls]` 6 列 | **`[cls, score, x1, y1, x2, y2, slot]` 7 列** |
| boxes 坐标 | 原图坐标、已 NMS | **未 NMS；空间由 `scale_factor` 决定**（见 2.1：喂 `1/r` ⇒ 画布坐标；喂 `1.0` ⇒ 原图坐标） |
| 有效框数 | N（NMS 后） | 恒为 300（需 score 阈值 + NMS 过滤） |
| order 矩阵 | `[1, N, N]` float | **`[300, 200, 200]` int32**，按 `slot_id` 索引 |
| `slot_id`（col[6]） | 无 | query/slot 编号 0–199，用于关联 order 矩阵 |

### 2.1 输出坐标空间（已实测闭环，2026-09-17 二次验证）

**结论：模型不做任何 letterbox 反算，只做一个纯标量运算 —— `输出坐标 = 原图坐标 / scale_factor`。**
其中「原图坐标」是模型依据 `im_shape` 自行换算出的原图像素坐标；`scale_factor` 是对它的**额外除数**。

| 喂入 `scale_factor` | `fetch_name_0` 坐标落在 | 实测（demo 图 1654×2339，r=0.3420，content=566×800） |
|--------------------|----------------------|--------------------------------------------------|
| `1/r` = 2.9238（= 原图/输入 之比） | **letterbox 画布坐标**（0..566 × 0..800） | x2 max=586.8、y2 max=808.4 ✅ 与 content 尺寸同量级 |
| `1.0` | **原图坐标**（0..1654 × 0..2339） | x2 max=1714.6、y2 max=2363.5 ✅ 与原图同量级 |
| `r` = 0.3420（**Java 现状＝错误**） | 被放大 `1/r²`≈8.55 倍 | x2 max=5013.3、y2 max=6910.7 ❌ 严重越界 |
| `1/r` 且 `im_shape=[800,800]` | 随 `im_shape` 等比缩小 | x2 max=284.0 ⇒ 证实输出还乘了 `im_shape` 比例 |

**代码注释遗留的 2453 / 4255 之谜（已解释）**：那条实测数据是喂 `scale_factor=r`（即 Java 当前写法 `{pre.scale, pre.scale}`）得到的，
被放大了 `1/r²`≈8.55 倍，故出现「远超 800」的假象；**不是**模型输出原图坐标的证据。

**验证方法与证据**（脚本：`model-tools/layout/scripts/probe_coord_space.py`）

- 固定同一张输入张量（`image` 完全不变），只改 `scale_factor`，得到 A(`1/r`) / B(`1.0`) / C(`r`) 三组输出；
- 三组 `slot_id`、`score` 列**逐元素完全一致**，说明检测结果本身与 `scale_factor` 无关，仅坐标被标量缩放；
- 逐行校验 `B.coords == A.coords / r`：**300 行全部吻合，最大偏差 0.87px**（≈ float32 精度级别）；
  （注意：只能比 `[2:6]` 坐标列；`class_id`/`score`/`slot_id` 不参与缩放，误比会得到虚假的 565px 差异。）
- 三张图交叉验证均成立：竖版 1654×2339、本模块测试图 800×800（r=1，三组输出完全相同，作对照）、横版 2339×1654；
- 模型固有的 3.7% 越界（x2=1714.6 vs 1654）在三组中等比出现 ⇒ 证明是模型行为、非映射误差，Java 端必须 clip。

**Java 端应采用的口径（推荐 A：显式 letterbox + 自己反算）**

```java
// 喂入：im_shape = [1,2] = {origH, origW}
//       scale_factor = [1,2] = {1/letterboxScale, 1/letterboxScale}
// 输出 = letterbox 画布坐标 → Java 反算回原图
float invScale = 1f / pre.scale;
origX1 = (outX1 - pre.padLeft) / pre.scale;
origY1 = (outY1 - pre.padTop)  / pre.scale;
// 本模块 letterBox() 采用右下补边（padLeft=padTop=0），故等价于 orig = out * (1/pre.scale)
```

> 为何不选「喂 `1.0` 直接拿原图坐标」：虽然数值等价（已验证），但那等于把 letterbox 反算责任交给模型且无自检；
> 喂 `1/r` 时输出必然落在 `[0, 800]` 画布内，**是一个天然的自检断言**（越界即说明导出/约定变了）；
> 且将来若改为居中 padding，`(out - pad) / scale` 仍是正确公式，而 `scale=1` 方案会失效（模型不知道 pad 偏移）。

**尺寸无关性**：`maxSideLength=800` 与输入张量固定 `[N,3,800,800]` 一致；r=1 的 800×800 图三组输出完全一致 ⇒ 无额外缩放。

### 2.2 阅读顺序：第 7 列就是 order 键（无需解 `fetch_name_2`）✅

查 [PaddleX `layout_analysis/processors.py::LayoutAnalysisProcess.apply`](https://raw.githubusercontent.com/PaddlePaddle/PaddleX/develop/paddlex/inference/models/layout_analysis/processors.py)
得到权威依据 —— 该文件里对输出列数有明确注释与分支：

```python
# boxes.shape[1] == 6 is object detection, 7 is new ordered object detection, 8 is ordered object detection
...
if boxes.shape[1] == 7:
    # Sort boxes by their order
    sorted_idx = np.argsort(boxes[:, 6])
    boxes = boxes[sorted_idx][:, :6]
```

即：**7 列输出 = 「带顺序列的检测输出」，第 7 列即 order 键，按它升序排就是阅读顺序**；
排序后 PaddleX 再顺序赋 `order = idx + 1`。`fetch_name_2`（`[300,200,200]`）是导出时一并暴露的
指针网络原始张量，PaddleX 侧也不消费它 —— **Java 端无需解码**（初稿的 win-accumulation 方案作废）。

副产物（否定掉初稿的 slot 假设）：

- `slot_id`（col[6]）在 300 行里只有 **116 个唯一值**，且 ≤ 273 而 `fetch_name_2` 只有 200×200
  ⇒ col[6] **不是** query 索引、也**不是** order 矩阵下标，它本身就是排序键；
- `fetch_name_2` 三个维度切片里非零区呈块状（如 slice0 非零行 72–124、列 28–76），不符合锦标赛矩阵形态。

**Java 落地**：`LayoutPostProcessor.decodeReadingOrder()` 按 col[6] 升序给 rank（同键则 score 高者先读，
因为模型输出已按 score 降序、`argsort` 对并列键本就按原序稳定取）；col 缺失时全部返回 `-1`。
未实现 PaddleX 的 `SKIP_ORDER_LABELS`（某些标签不参与编号）—— 本模块 rank 是「全体返回框的相对顺序」，
如需与 PaddleX 的 `order` 字段逐值一致需补该名单。

### 实测数据（官方 demo 图 1654×2339，喂 `scale_factor = 1/r`）

```
boxes[0] = cls=22 score=0.9420 box=(80.8, 286.9, 216.0, 498.3)   slot=43   # letterbox 画布坐标
boxes[1] = cls=22 score=0.9310 box=(222.3,  62.5, 357.9, 158.7)  slot=49
boxes[2] = cls=22 score=0.9213 box=(221.8, 288.5, 359.5, 417.9)  slot=75
...
count = [300]                      # 300 个 query 全输出，score<0.4 占绝大多数
order.shape = (300, 200, 200) int32
slot_id 唯一值仅 116 个 / 300 行  # ⚠️ slot 不是 query 唯一 id，step 10 需单独探针
```

### 25 类标签（V2/V3 共享）

字典来源：[RapidLayout write_dict.py](https://rapidai.github.io/RapidLayout/main/blog/2026/02/10/support-PP-DocLayoutv2-v3/)

```
abstract, algorithm, aside_text, chart, content, display_formula,
doc_title, figure_title, footer, footer_image, footnote, formula_number,
header, header_image, image, inline_formula, number, paragraph_title,
reference, reference_content, seal, table, text, vertical_text, vision_footnote
```

### 转换误差

| 模型 | 误差 | 结论 |
|------|------|------|
| PP-DocLayoutV2 | **14.8%** | 已知 Paddle2ONNX 问题（[issue #1608](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1608)），不建议生产 |
| PP-DocLayoutV3 | **1.57%** | **可生产使用** ✅ |

我们使用 V3。

### License

Apache-2.0，PaddleOCR 官方仓库 LICENSE：[github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)。可商用。

## 3. 实施差异清单（vs. 初稿）

| 维度 | 初稿假设 | 真实形态 | 改动 |
|------|---------|---------|------|
| 输入节点数 | 2（image + scale_factor） | **3**（image + im_shape + scale_factor） | ✅ 已加 `im_shapeInputName` 探测 |
| `im_shape` shape | `[1, 3]` | **`[1, 2]`** `[h, w]` | ✅ 已改 `imShapeShape={1,2}`、`imShapeData={origH, origW}` |
| `scale_factor` 语义 | 字面「缩放比」= `min(800/h, 800/w)` | **除数** `输出 = 原图 / scale_factor`，须喂 `1/r` | ✅ 已改 `invScale = 1/pre.scale`（见 2.1） |
| 输出节点数 | 2（boxes + order_index） | **3**（`fetch_name_0/1/2`） | ✅ 检测输出改为按形状推断（float32 且最后一维 ≥6），不依赖节点名 |
| boxes 列序 | `[x1,y1,x2,y2,score,cls]` 6 列 | **`[cls, score, x1, y1, x2, y2, order]` 7 列** | ✅ 已重写 `LayoutPostProcessor` 解析 |
| boxes 坐标 | 原图坐标、已 NMS | **未 NMS、画布坐标** | ✅ 已加 NMS + 反 letterbox + clip |
| 类别数 | 11 类（RT-DETR 通用） | **25 类**（PP-DocLayout 字典） | ✅ 已改 `LayoutLabel` 25 类 |
| 输入尺寸 | 960×960 | **800×800** | ✅ `maxSideLength=800` + 与模型输入边长不一致即失败 |
| NMS | 模型内置 | **Java 端需自行实现**（同类 0.6 / 异类 0.98） | ✅ 已加 `layoutNms` / `nmsThreshold` / `nmsDiffClassThreshold` |
| reading order | `order_index[1,N,N]` + win-accumulation | **col[6] 即 order 键，升序即阅读顺序** | ✅ 已按 2.2 重写，`fetch_name_2` 不消费 |
| `slot_id` | query slot 0–199，用于索引 order 矩阵 | **就是 order 键**（116 唯一值、≤273，非矩阵下标） | ✅ 假设作废 |
| 整页伪框 | 未考虑 | PaddleX 丢弃面积占比 >0.82（横）/0.93（竖）的整页 `image` 框 | ✅ 已实现 |

## 4. 实施步骤

- [x] 1. 建跟踪文档（本文件）
- [x] 2. 改 `LayoutLabel` → 25 类
- [x] 3. 改 `LayoutConfig` → 25 类阈值 / 800 默认 / `numClasses=25`
- [x] 4. WSL 下载 PP-DocLayoutV3 pdiparams + paddle2onnx 转 ONNX（125MB）
- [x] 5. ONNX 入库 `model-tools/layout/models/model.onnx`
- [x] 6. onnxruntime Python 实测确认 I/O 形状与列序
- [x] 6b. 实测闭环输出坐标空间（2.1 节，脚本 `probe_coord_space.py`）
- [x] 6c. 查 PaddleX 源码闭环阅读顺序语义（2.2 节：col[6] 即 order 键）
- [x] 7. 改 `LayoutDetector` → `im_shape [1,2]` + `scale_factor = 1/pre.scale` + 检测输出按形状推断
- [x] 8. 改 `LayoutConfig` → `layoutNms` / `nmsThreshold` / `nmsDiffClassThreshold`
- [x] 9. 重写 `LayoutPostProcessor` → 7 列解析 + 反 letterbox + clip + 阈值 + NMS + 整页框过滤 + 阅读顺序
- [x] 10. `readingOrder` 改为按 col[6] 解码（原 win-accumulation 方案作废）
- [x] 11. 同步 starter `LayoutProperties`、`LayoutAutoConfiguration`
- [x] 12. 重写 `LayoutIntegrationTest`（真模型 + `OpenCV.loadShared()` + 可选外部文档图）
- [x] 13. 更新 `mica-ai-layout/README.md` 与 `model-tools/layout/README.md`
- [x] 14. `mvn test`（layout 25 用例全绿）+ 全量 install
- [x] 15. **修 Java 端 BGR/RGB 双转换 bug**（见下）
- [x] 16. **修 letterbox 张量形状 bug**（见下）
- [x] 17. 删除 `scripts/build_fake_layout_onnx.py`（会覆盖真模型，且真模型测试已取代 fake 测试）

### 4.1 落地时发现并修掉的两个真 bug

| # | 症状 | 根因 | 证据 |
|---|------|------|------|
| 15 | 集成测试在 `demo.png` 上返回**空结果** | `LayoutDetector.letterBox` 先 `cvtColor(BGR2RGB)`，而 `LayoutImageUtils.bgrHwcToRgbChwFloat` 自身再做一次 BGR→RGB 读取 ⇒ 实际喂给模型的是 **BGR** 张量 | 实测同一张图：RGB 输入过 0.4 阈值 **1 个框**，BGR 输入 **0 个框**（分数从 0.4375 掉到 0.328） |
| 16 | 非正方形图上推理抛 `Shape [1, 3, 800, 566], requires 1358400 elements but the buffer has 1920000` | 张量形状用了 **resize 后的内容尺寸**（566×800），而数据是**补边后的 800×800 画布** | 修前：官方 demo 直接报错；修后：14 个区域正常输出。`demo.png` 是 800×800 正方形（内容=画布）掩盖了该 bug ⇒ 这也是加「外部文档图」用例的原因 |

> 15 号 bug 属于「ONNX 预处理契约」典型坑：写错不报错、只静默降精度（此处是静默丢光所有框），
> 与 skill `onnx-model-preprocessing-verify` 记录的 MiniFASNet `/255` 同类。

## 5. 待办 & 风险

| 风险 | 影响 | 处理 |
|------|------|------|
| ~~boxes 坐标是 letterbox 还是原图尺度~~ | ~~坐标反算错误 → 框偏移~~ | ✅ **已闭环**：`输出 = 原图坐标 / scale_factor`；Java 喂 `1/pre.scale` 得画布坐标，再 `(out-pad)/pre.scale` 反算 + clip（2.1 节） |
| `fetch_name_1` count 恒为 300 | 无效框太多拖慢 NMS | 先按 `scoreThreshold` 过滤再 NMS，最后 `maxDetections` 截断 |
| 模型固有 3.7% 越界框（x2 超出原图宽） | 输出坐标非法 | ✅ 反算后已 clip 到 `[0,w]x[0,h]`；越界量与 `scale_factor` 无关，是模型自身行为 |
| ~~`fetch_name_2` order 索引语义不明~~ | ~~readingOrder 可能错误~~ | ✅ 已闭环：col[6] 即 order 键（PaddleX 7 列分支），`fetch_name_2` 不消费（2.2 节） |
| Java 端 OpenCV native 加载 | 集成测试跑不起来 | ✅ 已解决：`nu.pattern.OpenCV.loadShared()`（openpnp jar 自带 4.9.0 原生库）；测试可正常解码图片 |
| WSL `/tmp` 是 tmpfs，VM 停止即清空 | 探针脚本缓存/中间产物丢失 | ✅ 探针缓存改 `~/.cache/mica-ai-layout/`；ONNX 已入库 |
| Windows paddle2onnx wheel 缺 `common.dll` | 无法在 Win 端直接转 | 走 WSL 转换路线（已验证可行） |
| `model.onnx` 125MB 超出 AGENTS.md「入库模型 <50MB」约定 | 仓库体积（clone/push 代价） | ⚠️ **待 owner 决策**：维持入库 / 走 Git LFS / 改下载脚本。当前未 commit，改动留在工作区 |
| `LayoutPostProcessor` 未实现 PaddleX 的 `SKIP_ORDER_LABELS` | 少数标签（如 `figure_title`/`image`）在 PaddleX 侧不参与编号，本模块 rank 是全体返回框的相对顺序 | 语义已在模块 README 说明；如要与 PaddleX `order` 逐值一致需补名单 |
| `demo.png` 是合成图（仅 1 个区域、分数 0.437 勉强过阈值） | 集成测试覆盖度弱 | 已加可选外部文档图用例（`-Dmica.ai.layout.test.image=...`）；官方 demo 图版权不明，未入库 |

## 6. 实施完后需要复核的事（状态）

1. ✅ `mvn -pl mica-ai-core/mica-ai-layout test` 全绿：**25 个用例**（集成 5 + 配置 8 + 后处理 13，含 1 个默认跳过的外部图用例）
2. ✅ `mvn -DskipTests install` 全模块构建通过
3. ✅ 真 ONNX 端到端：官方 demo（1654×2339）**14 个区域**、top1 `text 0.945`；与 Python 参考坐标逐像素吻合（见 4.1）
4. ✅ README 的 I/O 表格、25 类别表与代码默认值一致（坐标空间约定单列一节）
5. ✅ `model-tools/layout/README.md` 已更新为实际文件与 125MB
6. ✅ NMS 后无重叠框，坐标全部落在原图范围内（集成测试 `allSatisfy` 断言）

### 6.1 尚未做的（需 owner 决策或后续排期）

- 仓库体积策略（125MB 模型）
- `SKIP_ORDER_LABELS` 对齐
- 在真实文档集上标定 `scoreThreshold` / per-class 阈值（当前 0.4 沿用官方 `draw_threshold`）
- `mica-ai-example` 增加 layout 示例 + starter 的 `application.yml` 示例（本次未动 example 模块）