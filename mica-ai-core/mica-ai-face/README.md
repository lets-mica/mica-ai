# mica-ai-face

> OpenCV Zoo 人脸识别推理：**YuNet 检测 + 5 关键点对齐 + SFace 128d Embedding + MiniFASNetV2 活体 + 头像 / 证件卡片提取**，纯 ONNX Runtime，**零 Python / 零 PyTorch**。

完整复现 **YuNet 检测 → 5 关键点仿射对齐 → SFace 推理 → L2 归一化** 全链路，输出可直接写入 Milvus / pgvector / Qdrant 等向量数据库做人脸库与 1:N 检索。

> **本模块只做"图片 → 向量 / 框 / 活体分"**：人脸库与 1:N 检索是业务领域，不在本模块范围内。

---

## ✅ License 一句话总结

整套链路（Java 代码 + ONNX Runtime + YuNet / SFace / MiniFASNetV2 模型）**全部 Apache-2.0 / MIT（可商用）**，可以闭源分发、商用 SaaS、卖盒子，无须邮件申请。

| 组件 | 模型 | License | 商用 |
|------|------|---------|------|
| 检测 | YuNet `face_detection_yunet_2023mar.onnx` | Apache 2.0 | ✅ |
| 特征 | SFace `face_recognition_sface_2021dec.onnx` | Apache 2.0 | ✅ |
| 活体 | MiniFASNetV2 `2.7_80x80_MiniFASNetV2.onnx` | MIT（minivision Silent-Face-Anti-Spoofing） | ✅ |

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | **8+** | 推荐 Temurin / Azul Zulu 8、11、17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.18.0 | Maven 自动拉取；GPU 场景换 `onnxruntime_gpu` |
| OpenCV | 4.9.0（openpnp） | Maven 自动拉取对应系统 / 架构的原生库 |

---

## 2. 模型准备

使用 [OpenCV Zoo](https://github.com/opencv/opencv_zoo) 人脸模型（Apache-2.0）+ [minivision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) 活体（MIT）：

```
models/
├── face_detection_yunet_2023mar.onnx        # YuNet 检测（320x320 RGB）
├── face_recognition_sface_2021dec.onnx      # SFace 识别（112x112 RGB → 128d）
└── 2.7_80x80_MiniFASNetV2.onnx              # MiniFASNetV2 活体（80x80 RGB → 3 类）
```

模型已直接入库：[`model-tools/face/models/`](../../model-tools/face/README.md)（无需下载脚本）。

---

## 3. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

---

## 4. 快速使用

### 4.1 零代码跑通检测 → 对齐 → 特征

```java
ModelConfig config = ModelConfig.builder()
    .detectionModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
    .recognitionModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
    .build();

try (ModelManager manager = ModelManager.create(config)) {
    FaceDetector detector = new FaceDetector(manager);
    FaceAligner aligner  = new FaceAligner();
    FeatureExtractor extractor = new FeatureExtractor(manager);

    BufferedImage img = ImageIO.read(new File("group.jpg"));
    List<FaceBox> boxes = detector.detect(img);
    for (FaceBox box : boxes) {
        try (Mat aligned = aligner.align(img, box)) {
            float[] feature = extractor.extract(aligned);   // 128d，已 L2 归一化
            // 入库 / 检索交给你自己的向量库（Milvus / pgvector）
        }
    }
}
```

### 4.2 只检测，不要向量

```java
List<FaceBox> boxes = detector.detect(img);
for (FaceBox box : boxes) {
    System.out.printf("score=%.3f, [%.0f,%.0f,%.0f,%.0f]%n",
        box.getScore(), box.getX1(), box.getY1(), box.getX2(), box.getY2());
}
```

### 4.3 1:1 比对（两向量算余弦相似度）

```java
float score = FaceVerifier.cosineSimilarity(embA, embB);
boolean samePerson = score > 0.35f;   // SFace 经验阈值
```

> 1:1 比对不需要"库"，是单纯的数学运算；本模块保留这个方法用于无库场景。

### 4.4 活体（默认关闭）

```java
LivenessDetector liveness = new LivenessDetector(manager);
LivenessResult lr = liveness.detect(aligned);
// lr.getScore() ∈ [0,1]，> 0.5 通常视为真人
```

> MiniFASNetV2 活体模型**默认关闭**：未配置路径时跳过加载；启用请设置 `mica.ai.face.liveness.enabled=true`。

### 4.5 头像提取

```java
AvatarOptions opts = AvatarOptions.builder().size(256).build();
AvatarExtractor avatarExtractor = new AvatarExtractor(opts);
AvatarResult avatar = avatarExtractor.extract(img, boxes.get(0));
// avatar.getImage() 已是正方形头像 BufferedImage
```

### 4.6 证件卡片提取

```java
CardOptions cardOpts = CardOptions.builder()
    .outputWidth(1011)
    .outputHeight(638)
    .build();
CardExtractor cardExtractor = new CardExtractor(cardOpts);
CardResult card = cardExtractor.extract(idCardImage);
// card.getImage() 透视矫正 + USM/CLAHE 增强后的证件卡面
```

---

## 5. 核心组件

| 组件 | 类 | 职责 |
|------|----|------|
| 配置 | [`ModelConfig`](src/main/java/net/dreamlu/mica/ai/face/config/ModelConfig.java) | Builder 模式：det / rec / live 三模型路径 + 阈值 |
| 模型会话 | [`ModelManager`](src/main/java/net/dreamlu/mica/ai/face/model/ModelManager.java) | 实现 `AutoCloseable`，托管三个 ONNX Session |
| 检测 | [`FaceDetector`](src/main/java/net/dreamlu/mica/ai/face/detection/FaceDetector.java) | YuNet 推理 + NMS，返回 `FaceBox`（含 5 关键点） |
| 对齐 | [`FaceAligner`](src/main/java/net/dreamlu/mica/ai/face/alignment/FaceAligner.java) | 5 关键点仿射到 112×112 |
| 特征 | [`FeatureExtractor`](src/main/java/net/dreamlu/mica/ai/face/recognition/FeatureExtractor.java) | SFace 推理 + L2 归一化 |
| 活体 | [`LivenessDetector`](src/main/java/net/dreamlu/mica/ai/face/liveness/LivenessDetector.java) | MiniFASNetV2 推理 + softmax（3 类） |
| 比对 | [`FaceVerifier`](src/main/java/net/dreamlu/mica/ai/face/verification/FaceVerifier.java) | 检测 → 对齐 → 特征 → 余弦相似度 一体化门面；提供静态 `cosineSimilarity` |
| 头像 | [`AvatarExtractor`](src/main/java/net/dreamlu/mica/ai/face/avatar/AvatarExtractor.java) | 自动摆正 / 分块兜底 / 两阶段重采样 |
| 卡片 | [`CardExtractor`](src/main/java/net/dreamlu/mica/ai/face/card/CardExtractor.java) | 掩膜 + 四边形拟合 + 透视矫正 + USM/CLAHE |
| 图像工具 | [`ImageUtils`](src/main/java/net/dreamlu/mica/ai/face/util/ImageUtils.java) | Letterbox / RGB→Float32 / 仿射 |

### 处理流程

```
图片 → [FaceDetector] YuNet ONNX → 人脸框 + 5 关键点（已 NMS）
     → [FaceAligner] 5 点仿射 → 112x112 RGB
     ├→ [FeatureExtractor] SFace ONNX → 128d Float32 → L2 归一化
     ├→ [LivenessDetector] MiniFASNetV2 ONNX → 3 类 softmax（可选）
     ├→ [AvatarExtractor] 摆正 / 分块 → 正方形头像
     └→ [CardExtractor] 透视矫正 + 增强 → 证件卡面

[FaceVerifier] = 检测 + 对齐 + 特征 + cosineSimilarity（1:1 比对门面）
```

---

## 6. 模型 I/O

| 任务 | 模型 | 输入 | 输出 |
|------|------|------|------|
| 检测 | YuNet | `input` `[1,3,320,320]` float32（RGB / 0~1） | `output` `[1,6300,15]`（`[x, y, w, h, ...]` 候选框 + 5 关键点 + 置信度） |
| 特征 | SFace | `input` `[1,3,112,112]` float32（对齐后） | `output` `[1,128]` float32（已 L2 归一化） |
| 活体 | MiniFASNetV2 | `input` `[1,3,80,80]` float32（`(-1, 1)` 归一化） | `output` `[1,3]` float32 softmax（real / replay / print） |

---

## 7. 人脸库 / 1:N 检索怎么办？

**本模块不提供**，因为人脸库是业务领域（用户体系、权限、生命周期），不是 AI 模块该管的。

### 推荐路径

| 规模 | 推荐存储 | 说明 |
|------|---------|------|
| < 1 万人脸 | 内存 `Map<String, float[]>` | 简单暴力遍历，无需额外组件 |
| 1 万 ~ 100 万 | pgvector / OpenSearch | 与业务库同库运维 |
| > 100 万 | Milvus / Qdrant / Pinecone | 专业 ANN 索引，毫秒级返回 |

### 与 Milvus 协作示例（伪代码）

```java
FaceEmbedding emb = extractor.extract(aligned);   // 128d
milvusClient.insert("face_gallery", List.of(
    InsertParam.Field.of("user_id", "alice"),
    InsertParam.Field.of("embedding", Floats.asList(emb.getVector()))
));
```

---

## 8. 注意事项

- **图像格式**：支持 JPEG / PNG / BMP（通过 JDK 自带 `ImageIO` / OpenCV Mat）。
- **Embedding 维度**：固定 **128d**，已 L2 归一化（点积 = 余弦相似度）。
- **GPU 加速**：替换 `onnxruntime` 依赖为 `onnxruntime_gpu` 并启用 CUDA provider 即可。
- **批量输入**：本模块按"一次一图"调用，向量库侧的批量写入由调用方控制。
- **活体**：默认关闭（无模型路径时跳过加载），商业落地前请按 `AGENTS.md` §6.1 自查模型许可。

---

## 9. 模型替换备忘

要替换为其他 ONNX 人脸检测 / 识别模型（如 MobileFaceNet、ArcFace R50 自训版）：

1. **检测端**：实现 `FaceDetector` 接口约定（输入任意 `BufferedImage` / `Mat`，输出 `List<FaceBox>` + 5 关键点）。
2. **识别端**：实现 `FeatureExtractor` 接口约定（输入已对齐的 112×112 图，输出 `float[128]` 已 L2-normalized 向量）。
3. **接入**：`ModelManager` 已抽象各算子，通过 Starter 的 `@ConditionalOnMissingBean` 替换默认实现即可。

模型规格 / Starter 详见：

- 启动器：[`mica-ai-starters/mica-ai-face-spring-boot-starter/README.md`](../../mica-ai-starters/mica-ai-face-spring-boot-starter/README.md)
- 模型资产：[`model-tools/face/README.md`](../../model-tools/face/README.md)