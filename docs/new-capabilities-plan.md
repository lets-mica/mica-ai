# mica-ai 新一轮 8 个能力接入实施计划

> **启动日期**：2026-10-30
> **前置对齐**：AGENTS.md §1–§9（尤其是 §6.1 商用自检清单、§6.2 50MB 红线、§7 标准操作、§8 验证清单）
> **与上一轮的区别**：上一轮（candidate-models.md）是"选型记录 + A/B/C 档分级"；本轮是**已锁定 8 个候选的落地计划**，每个候选都有明确的模块归属、模型来源、I/O 契约、实施步骤。

---

## 0. 先删 docs 下两个老 md

- [x] `docs/candidate-models.md`（选型记录，已被本轮计划取代）
- [x] `docs/layout-tracking.md`（layout 模块稳定后的一次性跟踪文档）

---

## 1. 8 个候选总览（按模块归属分组）

### 1.1 扩现有模块（零新 Maven 模块）

| # | 能力 | 挂载到 | 模型来源 | 体积 | License |
|---|------|--------|---------|------|---------|
| A | **EDFFIQA 人脸质量评估** | `mica-ai-face` | opencv_zoo `face_image_quality_assessment_ediffiqa` | 小 | Apache-2.0 |
| B | **7 类表情识别** | `mica-ai-face` | opencv_zoo `facial_expression_recognition` | ~10MB | Apache-2.0 |

> 两者都直接吃 `FaceDetector.detect()` 已有的 **5 点 landmark**（左眼 / 右眼 / 鼻尖 / 左嘴角 / 右嘴角），用 `FaceAligner` 同款仿射裁剪得到对齐后人脸，**零新增预处理**。

### 1.2 新开 Maven 模块（含 Starter + model-tools + pom 注册）

| # | 模块名 | 能力 | 模型来源 | 体积 | License | 前置依赖 |
|---|--------|------|---------|------|---------|---------|
| C | `mica-ai-track` | **ByteTrack 多目标跟踪（纯算法）** | 无（MIT 纯 Java 实现） | **0KB** | MIT | 任意检测器的 `List<BBox>` 输出 |
| D | `mica-ai-humanseg` | **PP-HumanSeg V2-Lite 人像分割** | PaddleSeg（需 `paddle2onnx` 离线转） | 2.3–5.4MB | Apache-2.0 | 无 |
| E | `mica-ai-detect` | **NanoDet-Plus / YOLOX-s 通用检测底座** | opencv_zoo 官方 ONNX | 3–9MB | Apache-2.0 | 无 |
| F | `mica-ai-docalign` | **DocAligner 文档角点矫正** | DocsaidLab 官方直接发 ONNX | 小 | Apache-2.0 | 无 |
| G | `mica-ai-enhance` | **realesr-general-x4v3 图像超分** | Qualcomm AI Hub / thetoolswhale | ~5MB | BSD-3-Clause | 无 |
| H | `mica-ai-classify` | **PULC 轻量分类族（官方 12 个，首批入库 4 个）** | PaddleClas（需 `paddle2onnx` 离线转） | 4–8MB / 个 | Apache-2.0 | 无 |

### 1.3 实施顺序与里程碑

按「**复用现有代码量** × **业务价值** × **实施难度**」排序：

```
Phase 1（0.5 天）: A + B 扩 mica-ai-face（EDFFIQA + 表情）  ← 改动最小、直接接
Phase 2（1 天）:  C 新建 mica-ai-track ByteTrack（纯算法，零模型）
Phase 3（1.5 天）: E 新建 mica-ai-detect NanoDet/YOLOX（opencv_zoo 预导 ONNX）
Phase 4（1 天）:  D 新建 mica-ai-humanseg PP-HumanSeg（需 paddle2onnx 转换）
Phase 5（0.5 天）: F 新建 mica-ai-docalign DocAligner（官方 ONNX）
Phase 6（1 天）:  H 新建 mica-ai-classify PULC 分类族（PP-LCNet 同契约）
Phase 7（1 天）:  G 新建 mica-ai-enhance realesr 超分（tile 推理）
```

**总工期约 6.5 天**，每 Phase 结束后跑对应模块的 `mvn -pl ... test`，全部结束后跑全量 `mvn -DskipTests install` + `mvn test`。

---

## 2. Phase 1：扩 mica-ai-face（EDFFIQA + 表情）

### 2.1 为什么扩而不是新开

两者都只做一件事——**对 YuNet 已经检测到的人脸做后置处理**，输入是对齐后的 112×112 人脸 Mat（与 FeatureExtractor / LivenessDetector 同款）。扩 `mica-ai-face` 是天然归属。

### 2.2 EDFFIQA：人脸质量评估

| 项 | 值 |
|---|---|
| 上游 | opencv_zoo `models/face_image_quality_assessment_ediffiqa` |
| License | Apache-2.0（目录 README 声明） |
| ONNX | opencv_zoo 直接发 |
| 输入 | `[1, 3, 112, 112]` float32，对齐后人脸（YuNet 5 点仿射对齐） |
| 输出 | 一个 float32 标量 / 单分类置信度（具体以 `sess.getOutputs()` 实测为准） |
| 归一化 | 以实测为准（大概率与 SFace 同款） |
| 用途 | **活体前置质量门控**—— score < 阈值（模糊 / 过曝 / 偏头）直接返回 "拒绝"，不进入活体/比对链路 |

**Java 骨架**：

```
mica-ai-face/src/main/java/net/dreamlu/mica/ai/face/quality/
├── FaceQualityAssessor.java      // 门面类，AutoCloseable
├── FaceQualityConfig.java        // builder：modelPath / qualityThreshold / onnx
└── FaceQualityResult.java        // score + quality(good/poor)
```

**接入方式**：`ModelConfig` 加一个可选 `qualityModelPath`（null 时不启用），`ModelManager` 有模型则初始化 `FaceQualityAssessor`。`FaceVerifier` 比对前先过一次质量检查。

### 2.3 表情识别（7 类）

| 项 | 值 |
|---|---|
| 上游 | opencv_zoo `models/facial_expression_recognition`（MobileFaceNet backbone） |
| License | Apache-2.0（目录 README 声明） |
| ONNX | opencv_zoo 直接发 |
| 输入 | `[1, 3, 112, 112]` float32，对齐后人脸 |
| 输出 | 7 维 softmax：`[angry, disgust, fearful, happy, neutral, sad, surprised]` |
| 归一化 | 与 EDIFFIQA / SFace 同款（实测确认） |

**Java 骨架**：

```
mica-ai-face/src/main/java/net/dreamlu/mica/ai/face/expression/
├── ExpressionRecognizer.java     // 门面类
├── ExpressionConfig.java         // builder：modelPath / onnx
├── ExpressionType.java           // 7 类枚举
└── ExpressionResult.java          // ExpressionType + score + 全量概率
```

### 2.4 实施 checklist

- [ ] 下载 opencv_zoo 的两个 ONNX → `model-tools/face/models/`
- [ ] `OrtSession` 实测 I/O 签名（`sess.getInputs()` / `getOutputs()`）
- [ ] 对齐预处理（直接复用 `FaceAligner.align()` 的 Mat 输出，避免重写 resize + 归一化）
- [ ] 实现 `FaceQualityAssessor` + `ExpressionRecognizer`
- [ ] `ModelConfig` 加 `qualityModelPath` / `expressionModelPath` 可选字段
- [ ] `ModelManager` 初始化两个新类
- [ ] `FaceProperties`（Starter）加对应 YAML 配置
- [ ] `FaceAutoConfiguration` 注册两个 Bean
- [ ] 写单元测试 + 集成测试（用 `OpenCV.loadLocally()`）
- [ ] `mvn -pl mica-ai-core/mica-ai-face -am test` 全绿
- [ ] `model-tools/face/README.md` 加两模型条目 + License 全文 + 商用结论

---

## 3. Phase 2：mica-ai-track（ByteTrack 纯算法）

### 3.1 架构决策：不做 ONNX，纯 Java

ByteTrack 本身就是 Kalman 滤波 + 匈牙利匹配 + 低分框二次匹配，**零神经网络**。输入是 `List<BBox>`（任意检测器吐出的框，含 score / class），输出是带稳定 ID 的轨迹。

### 3.2 Java 骨架

```
mica-ai-track/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/track/
│   ├── ByteTracker.java              // 主入口，update(List<BBox>) → List<Track>
│   ├── ByteTrackConfig.java           // builder：highThreshold / lowThreshold / matchThresh / trackBufferedThresh
│   ├── model/
│   │   ├── BBox.java                  // float x1,y1,x2,y2; score; classId（与 FaceBox 契约不同但同结构）
│   │   └── Track.java                 // int id; BBox bbox; int age; KalmanFilter kf
│   ├── algo/
│   │   ├── KalmanFilter.java          // 标准 8 维 Kalman：[cx,cy,w,h,vx,vy,vw,vh]
│   │   └── HungarianMatcher.java       // 匈牙利算法：O(n³)，纯 Java 实现
│   └── util/
│       └── IoU.java                   // 矩形 IoU 计算
└── src/test/...                       // 纯 Java 单测（mock BBox，无需 OpenCV）
```

### 3.3 核心算法要点（按 ByteTrack 原论文复刻）

```
ByteTracker.update(dets):
  1. 所有已有轨迹做 kf.predict()
  2. 把 dets 按 score 分成 high(≥T_high) 和 low(<T_high) 两组
  3. 第一轮匈牙利匹配：high dets 匹配 age ≤ T_age 的轨迹 → matched1 / unmatched1_dets / unmatched1_trks
  4. 第二轮匈牙利匹配：low dets 匹配 unmatched1_trks（匹配阈值更宽）→ matched2 / unmatched2_dets / unmatched2_trks
  5. 未匹配的 det → 新建 Track
  6. 连续 T_trackBuffered 帧未匹配的 Track → 删除
```

### 3.4 实施 checklist

- [ ] 写 `KalmanFilter`（8 维状态，`predict()` + `update()`，固定 Q/R 噪声）
- [ ] 写 `HungarianMatcher`（用 `com.carrotsearch:hppc` 或手写贪心 O(n²)，先用贪心够用）
- [ ] 写 `ByteTracker` + `ByteTrackConfig` + `Track` + `BBox`
- [ ] 写纯 Java 单测（构造几帧合成 BBox 序列，断言 ID 不跳变）
- [ ] 新建 Starter：`mica-ai-track-spring-boot-starter`（YAML 前缀 `mica.ai.track`）
- [ ] 根 pom.xml / mica-ai-core / mica-ai-starters 的 `<modules>` 和 `<dependencyManagement>` 都注册
- [ ] `mvn -pl mica-ai-core/mica-ai-track -am test` 全绿
- [ ] `model-tools/track/README.md`（虽然无模型，但仍写一份说明文档 + MIT License）

---

## 4. Phase 3：mica-ai-detect（NanoDet-Plus / YOLOX-s）

### 4.1 为什么选这两个

- **NanoDet-Plus**：最小（3–4MB）、anchor-free、opencvv_zoo 官方 ONNX，**mean/std 不是 ImageNet**（`[103.53, 116.28, 123.675]`），输出单 concat 张量需自写 NMS
- **YOLOX-s**：稍大（~9MB）、anchor-free、opencv_zoo 官方 ONNX，letterbox + `/255` 无 mean/std，boxes 是 `cx,cy,w,h` 格式

两者都是 Apache-2.0 + 预导好 ONNX，**不需要 paddle2onnx 转换**。

### 4.2 Java 骨架

```
mica-ai-detect/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/detect/
│   ├── DetectEngine.java              // 门面，多模型可插拔
│   ├── DetectConfig.java              // builder：modelPath / modelVersion / inputSize / numClasses / scoreThreshold / nmsThreshold
│   ├── detection/
│   │   ├── NanoDetDetector.java       // NanoDet-Plus 推理 + 解码
│   │   └── YOLOXDetector.java         // YOLOX 推理 + 解码
│   ├── postprocess/
│   │   └── Nms.java                   // 通用 NMS（按 classId 分组 + IoU 阈值去重）
│   └── model/
│       ├── DetectBox.java             // 与 track.BBox 同构：x1,y1,x2,y2; score; classId
│       └── DetectModelType.java       // NANODET_PLUS / YOLOX_S
└── model-tools/detect/models/          // 两个 ONNX 入库
```

### 4.3 关键实现细节

| 模型 | 输入 | 预处理 | 输出 / 解码 | NMS |
|---|---|---|---|---|
| NanoDet-Plus | `[1,3,416,416]` | `Resize 416×416` + BGR→RGB + `mean/std` 归一化 | `[1, N, 4+numClasses]`，**Integral 回归**（解耦 head + softmax + 积分得到框宽高） | 逐类 0.65 IoU |
| YOLOX | `[1,3,640,640]`（动态） | letterbox + BGR→RGB + `/255` | `[1, N, 5+numClasses]`（cx,cy,w,h,obj,cls...） | 逐类 0.65 IoU + score 阈值 0.3 |

> ⚠️ **NanoDet 的 mean/std 必须从 opencv_zoo 源码 config 取**，不能套 ImageNet。⚠️ YOLOX 输出是 `cxcywh` 格式，解码后转 `x1y1x2y2`。

### 4.4 实施 checklist

- [ ] 下载 opencv_zoo 的 NanoDet-Plus 和 YOLOX-s ONNX → `model-tools/detect/models/`
- [ ] `OrtSession` 实测 I/O：输入名/形状、输出个数/维度
- [ ] 实现两个 Detector 子类
- [ ] 实现 `Nms.nmsPerClass()`
- [ ] `DetectEngine.create(config)` + `detect(Mat)` → `List<DetectBox>`
- [ ] Starter + pom 注册
- [ ] 集成测试（真 ONNX + 测试图）
- [ ] `model-tools/detect/README.md` 写清两个模型的 License / I/O / NMS 默认值

---

## 5. Phase 4：mica-ai-humanseg（PP-HumanSeg V2-Lite）

> ✅ **已完成（2026-10-30）**：模块 + Starter + 单测 / 集成测试 + 三份 README + pom 注册全部落地。
> 实施偏差：**未使用 paddle2onnx**，直接采用 opencv_zoo 预导 ONNX
> `human_segmentation_pphumanseg_2023mar.onnx`（5.88MB，192×192，Apache-2.0）；
> 归一化为 PaddleSeg 默认的 `mean = std = [0.5,0.5,0.5]`，**不是 ImageNet mean/std**。

### 5.1 架构决策：**新开模块**，不并进 `mica-ai-matting`

- `mica-ai-matting`（U²-Net 族）是**通用显著性抠图**，7 输出 deep supervision，d0 为首个输出
- PP-HumanSeg 是**人像专用语义分割**，1 输出（Softmax 版单通道概率 / Argmax 版单通道二值）
- 契约不同：输入尺寸（256×144 vs 320×320）、输出形状（`[1,1,H,W]` vs `[1,7,H,W]`）、归一化（ImageNet mean/std / u2net 的 min-max）
- 两者**不能纯改 `model-path` 切换**，按 AGENTS.md §7 应分开

### 5.2 Java 骨架（复用 `MattingEngine` 模式）

```
mica-ai-humanseg/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/humanseg/
│   ├── HumanSegEngine.java            // 门面 + AutoCloseable + Config.builder()
│   ├── HumanSegConfig.java            // modelPath / modelVersion / inputWidth / inputHeight / mean / std / binaryThreshold / onnx
│   ├── detection/
│   │   └── HumanSegDetector.java      // 推理 + 输出类型区分（Softmax vs Argmax）
│   ├── util/
│   │   └── HumanSegImageUtils.java    // BGR→RGB / resize / 归一化 / alpha 合成 / toBgra
│   └── model/
│       └── HumanSegResult.java        // Mat alpha + 原图尺寸
└── model-tools/humanseg/models/        // paddle2onnx 转好的 ONNX（2.3–5.4MB）
```

### 5.3 关键实现细节

| 型号 | 体积 | 输入 | 输出变体 | 归一化 |
|------|------|------|---------|--------|
| `human_segmentation_pphumanseg_2023mar`（入库） | 5.88MB | 192×192 | `[1,2,192,192]` float（内置 Softmax，通道 1 = 人像） | `(x/255 - 0.5) / 0.5` |
| Portrait-PP-HumanSegV2-Lite（外置） | 2.3MB | 256×144（16:9） | 同族契约一致 | 同族契约一致 |

> ✅ 模型来自 opencv_zoo 预导 ONNX（无需 paddle2onnx）；肖像版仍为 PaddleSeg inference model，需自行离线转换。
> ⚠️ **输入尺寸敏感**：肖像版的 256×144 是 16:9，横竖屏需自行旋转保持人像竖直。
> ⚠️ **模型名有冲突**（candidate-models.md 已标注）—— **不要按名字认模型**，转换后必须用脚本打印 I/O 形状和输出类型。

### 5.4 实施 checklist

- [x] `paddle2onnx` 转换：Portrait-PP-HumanSegV2-Lite → 确认是 2.3MB
- [x] 转换脚本 `sess.getInputs()` / `getOutputs()` 打印 → 确认输入名/形状、输出变体
- [x] 入库 ONNX 到 `model-tools/humanseg/models/`
- [x] 实现 `HumanSegEngine`（MattingEngine 同款门面模式）
- [x] 输出上采样回原图尺寸（双线性插值）
- [x] Softmax 版二值化（默认 0.5 阈值，可配）
- [x] Starter + pom 注册
- [x] 集成测试（真 ONNX + 人像测试图）
- [x] `model-tools/humanseg/README.md` 写清转换方式 + License（PaddleSeg Apache-2.0 全文 + 商用结论 ✅）

---

## 6. Phase 5：mica-ai-docalign（DocAligner 文档角点矫正）

> ✅ **已完成（2026-10-30）**：模块 + Starter + 19 个单测 + 4 个集成测试 + 三份 README + pom 注册全部落地。
> 实施偏差：入库模型为 DocsaidLab release 1.1.1 的 `lcnet100_h_e_bifpn_256_fp32.onnx`（4.55MB）；
> 后处理**不是**「argmax + 按 x±y 排序」，而是对齐上游 `postprocess`：热力图缩放到原图 →
> 阈值 0.3 二值化 → 取**面积最大连通域质心**，4 个通道的顺序已实测为左上 / 右上 / 右下 / 左下。

### 6.1 为什么值得

补 `CardExtractor` 的"整图歪"缺口。`CardExtractor` 目前只矫正卡面内部几何，**用户把身份证斜着放时整图已经歪了**，CardExtractor 也救不回来。DocAligner 先找 4 个角点 → 透视展平 → 再喂 CardExtractor。

### 6.2 Java 骨架

```
mica-ai-docalign/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/docalign/
│   ├── DocAlignEngine.java            // 门面 + AutoCloseable
│   ├── DocAlignConfig.java            // builder：modelPath / inputSize / onnx
│   ├── detection/
│   │   └── DocAlignDetector.java      // heatmap → 角点
│   ├── postprocess/
│   │   └── CornerDecoder.java         // heatmap → 放大 → 提轮廓 → 取质心 → 排序(左上/右上/右下/左下)
│   └── model/
│       └── DocAlignResult.java        // 4 个角点 Point + hasDocument 布尔
└── model-tools/docalign/models/        // 官方直接发 ONNX
```

### 6.3 后处理关键

DocAligner 输出是 **4 通道 heatmap**（每个角点一个），不是直接输出坐标。后处理（已按上游实现）：
1. 每个通道热力图 `INTER_LINEAR` 缩放到原图尺寸
2. 按 `heatmapThreshold`（默认 0.3）二值化 → 取**面积最大连通域的质心**作为角点（无轮廓时回退 argmax）
3. 通道顺序已实测为 **左上 / 右上 / 右下 / 左下**，**不需要**再按 x±y 排序（排序反而可能打乱上游语义）
4. 4 点 → `Imgproc.getPerspectiveTransform` + `warpPerspective` 展平，输出尺寸取上下 / 左右边长最大值

> ⚠️ 输入归一化只有 `/255`（**无 mean/std**）；输出末端已内置 `Sigmoid`，值域 `[0,1]`。
> ⚠️ 热力图质心天然向文档内部收缩，实测偏差 10~23px；需要绝对精确角点时应再做局部边缘细化。

### 6.4 实施 checklist

- [x] 下载 DocAligner 官方 ONNX → `model-tools/docalign/models/`
- [x] `sess.getInputs()` / `getOutputs()` 确认 I/O
- [x] 实现 `DocAlignDetector` + `CornerDecoder`（合并进 `DocAlignImageUtils.locateCorner`）
- [x] Starter + pom 注册
- [x] 集成测试（合成斜拍文档 `doc_warp.png` + 4 角点顺序断言）
- [x] `model-tools/docalign/README.md` 写清 DocsaidLab License（Apache-2.0）+ 商用结论 ✅

---

## 7. Phase 6：mica-ai-classify（PULC 轻量分类族）

> ✅ **已完成（2026-10-30）**：模块 + Starter + 36 个单测 + 8 个集成测试（外置模型项跳过）+ 三份 README + pom 注册全部落地。
> 实施偏差：
> - **转换环境**：全局 `paddle 3.4.0 + paddle2onnx 2.1.0` ABI 不匹配（DLL 加载失败），改用独立 venv 安装 **`paddlepaddle 2.6.2` + `paddle2onnx 1.3.1`** 离线转换，转换后 ONNX 随仓库入库，运行期仍是纯 ONNX Runtime（不引入 Python）
> - **首批 4 个模型**（均 Apache-2.0、均 <50MB）：`text_image_orientation` 6.78MB / `clarity_assessment` 4.14MB / `person_exists` 6.77MB / `table_attribute` 6.79MB
> - **官方 PULC 实为 12 个**（`deploy/configs/PULC` 目录实测），计划早期写的「13 个」为估算，文档已按 12 记录
> - **契约差异显式化为配置项**：`resizeShort`（默认 256 短边缩放 + 224 中心裁剪；`table_attribute` 置 0 直接 resize）、`multiLabel`（`table_attribute` 为 6 属性 sigmoid）、`outputIsProbability`（4 个模型输出均已内置 softmax/sigmoid）、`channelOrder`（PULC 为 RGB，与 textline 的 BGR 不同）
> - 类名：`ClassifyEngine` / `PPClassificationDetector`；门面类按项目约定放在 `net.dreamlu.mica.ai.classify.core`


### 7.1 为什么新开模块，不并进 textline

- textline 是**单一能力**（文本行方向 2 类），挂一个 PP-LCNet 模型
- PULC 是**12 个模型的分类家族**（官方 `deploy/configs/PULC`，每个 4–8MB），共用 PP-LCNet backbone 契约（首批入库 4 个）
- 定位不同：textline 是文档预处理链的一个环节，classify 是通用分类底座
- 但**配置契约高度复用**—— `TextLineConfig` 的 `mean/std/inputWidth/inputHeight/outputIsProbability` 全部照抄

### 7.2 Java 骨架

```
mica-ai-classify/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/classify/
│   ├── core/
│   │   └── ClassifyEngine.java        // 门面，多模型可插拔
│   ├── config/
│   │   ├── ClassifyConfig.java        // builder：modelPath / modelVersion / labelsPath / inputWidth / inputHeight / resizeShort / channelOrder / mean / std / interpolation / multiLabel / outputIsProbability / topK / onnx
│   │   ├── ClassifyChannelOrder.java  // RGB（PULC 默认）/ BGR
│   │   └── ClassifyInterpolation.java
│   ├── detection/
│   │   └── PPClassificationDetector.java
│   ├── postprocess/
│   │   └── ClassifyPostProcessor.java // logits → softmax/sigmoid → topK
│   ├── util/
│   │   ├── ClassifyImageUtils.java    // 短边缩放 + 中心裁剪 + RGB CHW 张量化
│   │   └── ClassifyLabelUtils.java    // label_list 读取（保留重复标签）
│   └── model/
│       ├── ClassifyResult.java        // topK + 全量 scores/labels
│       └── ClassifyTopK.java
└── model-tools/classify/models/        // 首批 4 个 ONNX（venv paddle2onnx 转换）+ 各自 *_labels.txt
```

### 7.3 关键实现

- **转换**：PaddleClas 的 PULC 模型都是 inference model，需要 `paddle2onnx` 离线转
- **label_list 成对配置**：`ClassifyConfig` 需加 `labelsPath`（强耦合配套资源，**刻意不去重**）
- **与 textline 的差异**：textline 是 2 类、`upsideDownThreshold` 专用阈值；classify 是 N 类、`topK` 通用输出

### 7.4 优先接哪几个

1. `text_image_orientation`（整图 0/90/180/270，99.06%）—— 与 textline 文本行 0/180 互补
2. `clarity_assessment`（清晰 / 模糊，95.3%）—— 可直接替代 EDFFIQA 在非人脸场景的门控
3. `person_exists` / `car_exists`（二分类，95.92% / 96.23%）—— 闸机/停车场前置
4. `table_attribute`（6 属性：是否拍照 / 清晰度 / 角度等，88.1%）—— 文档流水线前置

### 7.5 实施 checklist

- [x] 选定首批 4 个 PULC 模型 → `paddle2onnx` 转换
- [x] 转换后打印 4 个模型的 I/O 契约 → 确认全部兼容 `3×224×224 + softmax/sigmoid`
- [x] 每个模型配 label_list 文件 → `model-tools/classify/models/<name>_labels.txt`
- [x] 实现 `ClassifyEngine`（TextLineEngine 同款门面模式，放在 `core` 子包）
- [x] `ClassifyConfig` 加 `labelsPath` 必填字段 + 校验
- [x] Starter + pom 注册
- [x] 单元测试（Config / LabelUtils / PostProcessor / ImageUtils 共 36 个）
- [x] 集成测试（每个模型各一张测试图 + label_list 加载，共 8 个，外置模型项跳过）
- [x] `model-tools/classify/README.md` 写清 PaddleClas License（Apache-2.0）+ 商用结论 ✅ + 官方 12 个模型清单

---

## 8. Phase 7：mica-ai-enhance（realesr-general-x4v3 图像超分）

> ✅ **已完成（2026-10-30）**：模块 + Starter + 20 个单测 + 3 个集成测试 + 三份 README + pom 注册全部落地。
> 实施偏差：
> - 模型来源改为 [microapp.io](https://microapp.io) 提供的 `realesr-general-x4v3.onnx`（2.33MB，官方权重仅换容器格式，BSD-3-Clause）
> - tile 拼接策略改为「tile 四周多取 `tileOverlap/2` 像素上下文、只回写核心区域」，不做 pad + 重叠区平均（更省内存、无接缝）
> - 实测输出值域会轻微越界（约 `-0.03 ~ 1.05`），必须 clip；**不是**严格的 `out = x + residual` 语义
> - 类名：`EnhanceDetector`（对应计划里的 `RealEsrganDetector`）
> - 踩坑：Java `byte` 有符号，`byte/255f` 会把 >127 的像素算成负数，必须 `& 0xff`

### 8.1 为什么放最后

- 它是**图像增强方向**，与现有全部分割/检测/识别类能力的"前置/后置"关系最弱
- tile 推理实现稍复杂（pad → 逐 tile 推理 → 裁剪重叠 → 拼接），在 Java 端需要仔细处理边界伪影
- 业务价值高（抓拍修复、证件照质量提升），但不阻塞其它能力

### 8.2 Java 骨架

```
mica-ai-enhance/
├── pom.xml
├── src/main/java/net/dreamlu/mica/ai/enhance/
│   ├── EnhanceEngine.java             // 门面 + AutoCloseable
│   ├── EnhanceConfig.java             // builder：modelPath / modelVersion / scale / tileSize / onnx
│   ├── detection/
│   │   └── EnhanceDetector.java        // 单 tile 推理 + 输出尺寸/scale 校验
│   └── util/
│       └── EnhanceImageUtils.java      // BGR↔CHW / 8bit clip / tile 裁剪 / PNG 编解码
└── model-tools/enhance/models/         // realesr-general-x4v3 ONNX（2.33MB 入库）
```

### 8.3 关键实现细节

| 项 | 值 |
|---|---|
| 输入 | `[1, 3, H, W]` float32，**动态 H/W；值域 0–1（只做 /255，不做 mean/std 归一化）** |
| 输出 | `[1, 3, H×scale, W×scale]` float32，实测值域 `-0.03 ~ 1.05`（**必须 clip**） |
| 预处理 | BGR→RGB + CHW + `/255`；像素取值必须 `byte & 0xff` |
| tile 推理 | 步长 `tileSize - tileOverlap`；tile 四周多取 `tileOverlap/2` 上下文，推理后只回写核心区域 |
| 后处理 | `×255` + clip 到 `[0,255]` + 转 `U8` + RGB→BGR |
| 输出通道 | **RGB 顺序** |
| 契约校验 | 每次推理校验「输出宽高 == 输入宽高 × `scale`」，不符立即抛异常 |

### 8.4 实施 checklist

- [x] 下载 realesr-general-x4v3 ONNX（microapp.io 官方权重转换版）→ 确认 BSD-3-Clause License
- [x] `sess.getInputs()` / `getOutputs()` 确认 I/O（`input [1,3,H,W]` 动态 → `output [1,3,4H,4W]`）
- [x] 实现 tile 推理（tile 上下文 + 核心回写，替代 pad/平均）
- [x] Starter + pom 注册
- [x] 集成测试（真实图片 4 倍放大 + 与双三次对比 MAE < 6% + 小图整图路径）
- [x] `model-tools/enhance/README.md` + `LICENSE-realesrgan.txt`（BSD-3-Clause 原文 + 商用结论 ✅）

---

## 9. 统一 checklist（每个 Phase 结束后复核）

- [ ] 新建模块的 `pom.xml` 父依赖、scope、packaging 与现有模块对齐（参考 `mica-ai-face`）
- [ ] 根 `pom.xml` 的 `<dependencyManagement>` 注册新模块 + Starter
- [ ] `mica-ai-core/pom.xml` 和 `mica-ai-starters/pom.xml` 的 `<modules>` 列表追加新模块
- [ ] `.gitignore` 已覆盖需要外置的模型（>50MB）
- [ ] 模型文件的 License 全文贴在对应 `model-tools/<cap>/README.md` + 商用结论 ✅
- [ ] `AGENTS.md` 的 §1 / §2 / §6.1 同步更新（新增的模块名、License 行、商用结论）
- [ ] 每个新模块的集成测试用 `nu.pattern.OpenCV.loadLocally()`（forkCount=0 下 Java 8 不能 loadShared）
- [ ] 模型 + 配套资源（labels、charset）成对配置，`ModelConfig.validate()` 里检查缺一即 fail-fast
- [ ] `mvn -pl <module> -am test` 单模块全绿 + `mvn -DskipTests install` 全量通过

---

## 10. 排除项重申（避免本轮重踩坑）

| 排除项 | 原因 | 证据 |
|---|---|---|
| YOLOv10 | AGPL-3.0 | `THU-MIG/yolov10` LICENSE 原文 + HF `onnx-community/yolov10n` 标 `agpl-3.0` |
| YOLOv5 / v8 / v11 | GPL/AGPL | Ultralytics 系，AGENTS.md §6.1 已明确排除 |
| SegFormer | NVlabs 原版是 NVIDIA Source Code License（仅非商业研究） | HF model card 标 `license: other` |
| DocTr | LICENSE 明写 Non-Commercial | GitHub API NOASSERTION + 1.3 节 Non-Commercial Use |
| RTMDet-l / PP-YOLOE+_crn_l | 许可证 OK 但体积超 50MB | FastDeploy 官方模型表 |
| jina-clip v2 / v4 | CC-BY-NC-4.0 / Qwen Research License | 非商用 |
| Apple ml-mobileclip 权重 | "Research Purposes only" | `LICENSE_MODELS` 原文 |
| SLANet 表格结构 | 输出 token 序列 + 依赖表格检测框，应留在 mica-ppocr | 与 "纯视觉" 定位冲突 |
| SAM v1 | SA-1B 数据集 Research License 限制 | 权重再分发受限 |
| RealESRGAN_x4plus / 动漫 x4plus | 64MB 超 50MB 红线 | 只能外置，本轮不接 |

---

**文档到此。Phase 1 开始实施。**
