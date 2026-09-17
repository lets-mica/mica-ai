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
├── face_detection_yunet_2023mar.onnx        # YuNet 检测（640x640 BGR，0~255 不归一化）
├── face_recognition_sface_2021dec.onnx      # SFace 识别（112x112 RGB → 128d）
└── 2.7_80x80_MiniFASNetV2.onnx              # MiniFASNetV2 活体（80x80 BGR → 3 类）
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
nu.pattern.OpenCV.loadLocally();   // 非 Spring 环境需先加载 OpenCV 原生库

ModelConfig config = ModelConfig.builder()
    .detectionModelPath("models/face_detection_yunet_2023mar.onnx")
    .recognitionModelPath("models/face_recognition_sface_2021dec.onnx")
    .build();

try (ModelManager manager = ModelManager.create(config)) {
    FaceDetector detector = new FaceDetector(manager);
    FaceAligner aligner  = new FaceAligner();
    FeatureExtractor extractor = new FeatureExtractor(manager);

    Mat img = Imgcodecs.imread("group.jpg");   // 字节流用 ImageUtils.byteArrayToMat(bytes)
    List<FaceBox> boxes = detector.detect(img);
    for (FaceBox box : boxes) {
        try (Mat aligned = aligner.align(img, box)) {
            float[] feature = extractor.extract(aligned);   // 128d，已 L2 归一化
            // 入库 / 检索交给你自己的向量库（Milvus / pgvector）
        }
    }
    img.release();
}
```

> 全部入口都是 OpenCV `Mat`（人脸框 `FaceBox` 用像素坐标），没有 `BufferedImage` 重载；
> `ModelConfig` 的模型路径是 `String`，支持文件路径与 `classpath:` 前缀（Starter 场景）。

### 4.2 只检测，不要向量

```java
for (FaceBox box : detector.detect(img)) {
    System.out.printf("score=%.3f, [%.0f,%.0f,%.0f,%.0f]%n",
        box.getScore(), box.getX1(), box.getY1(), box.getX2(), box.getY2());
}
```

### 4.3 1:1 比对（两向量算余弦相似度）

```java
// 已有两张特征向量（已 L2 归一化 → 点积即余弦相似度）
float score = FeatureExtractor.compare(embA, embB);
boolean samePerson = score > 0.35f;   // SFace 经验阈值
```

> 也可以直接走门面（内部自动完成 检测 → 对齐 → 特征）：
> `new FaceVerifier(manager).verify(probeMat, refMat)` 返回 `VerifyResult`；
> `verifyThreshold` 默认 `0.35`，可用 `ModelConfig.builder().verifyThreshold(...)` 调整。

### 4.4 活体（默认关闭）

```java
LivenessDetector liveness = new LivenessDetector(manager);
LivenessResult lr = liveness.check(mat, box);   // 需要原图 + 人脸框
// lr.getLiveScore() ∈ [0,1]，> 0.85（默认阈值）视为真人；lr.getAttackType() ∈ real / print / replay
```

> MiniFASNetV2 活体模型**默认关闭**：未配置路径时跳过加载；启用请设置 `mica.ai.face.liveness.enabled=true`。
> 活体输入由本类按人脸框外扩 2.7 倍裁剪到 80x80，**不要**直接传对齐后的 112x112 人脸。

### 4.5 头像提取

```java
AvatarOptions opts = AvatarOptions.builder()
    .size(256)          // 输出边长
    .faceScale(1.6)     // 窗口边长 = 检测框长边 × faceScale
    .autoOrient(true)   // 人脸横躺（90/180/270）时自动摆正
    .deRotate(true)     // 按眼线做残余倾角精修
    .build();
AvatarExtractor avatarExtractor = new AvatarExtractor(new FaceDetector(manager), opts);
AvatarResult avatar = avatarExtractor.extract(img);   // 也可 extract(img, opts)
// avatar.getImage() 是 size × size 的 Mat；avatar.getBox() / getWindowQuad() 给出裁剪窗口
```

> 窗口边长恒等于 `检测框长边 × faceScale`（与是否触发自动摆正无关），窗口中心在人脸框中心，
> `verticalOffset` 可上下平移。提取完成后请 `avatar.release()` 释放 `Mat`。
>
> `autoOrient` 只在 0/90/180/270 四个基数角里投票（裁剪区边长 = 2 倍人脸长边，正确角需比原始角高
> `0.03` 才改判），专治「证件照 / 卡面在画面里横躺」；残余倾角由 `deRotate` 按眼线精修。
> 大图小脸走全图检测会漏检，`tileDetect`（默认开）按 `tileSize` 分块 + NMS 兜底。

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
| 活体 | [`LivenessDetector`](src/main/java/net/dreamlu/mica/ai/face/liveness/LivenessDetector.java) | MiniFASNetV2 推理 + softmax（print / real / replay） |
| 比对 | [`FaceVerifier`](src/main/java/net/dreamlu/mica/ai/face/verification/FaceVerifier.java) | 检测 → 对齐 → 特征 → 余弦相似度 一体化门面；相似度计算在静态 `FeatureExtractor.compare(a, b)` |
| 头像 | [`AvatarExtractor`](src/main/java/net/dreamlu/mica/ai/face/avatar/AvatarExtractor.java) | 自动摆正 / 分块兜底 / 两阶段重采样 |
| 卡片 | [`CardExtractor`](src/main/java/net/dreamlu/mica/ai/face/card/CardExtractor.java) | 掩膜 + 四边形拟合 + 透视矫正 + USM/CLAHE |
| 图像工具 | [`ImageUtils`](src/main/java/net/dreamlu/mica/ai/face/util/ImageUtils.java) | 字节流解码 / 外扩裁剪 / BGR→CHW float / 编码输出 / 清晰度 / 批量 release |

### 处理流程

```
图片 → [FaceDetector] YuNet ONNX → 人脸框 + 5 关键点（已 NMS）
     → [FaceAligner] 5 点仿射 → 112x112 RGB
     ├→ [FeatureExtractor] SFace ONNX → 128d Float32 → L2 归一化
     ├→ [LivenessDetector] MiniFASNetV2 ONNX → 3 类 softmax（可选）
     ├→ [AvatarExtractor] 摆正 / 分块 → 正方形头像
     └→ [CardExtractor] 透视矫正 + 增强 → 证件卡面

[FaceVerifier] = 检测 + 对齐 + 特征 + 余弦相似度（1:1 比对门面）
```

---

## 6. 模型 I/O

| 任务 | 模型 | 输入 | 输出 |
|------|------|------|------|
| 检测 | YuNet | `input` `[1,3,640,640]` float32，**BGR 顺序 + 原始 0~255**（等比缩放后右下补 0，不做均值 / 方差归一化） | 12 个输出 = `cls / obj / bbox / kps` × stride `8/16/32`：`cls`、`obj` `[1,N,1]`（导出时已过 sigmoid），`bbox` `[1,N,4]`，`kps` `[1,N,10]`；分值取 `sqrt(cls × obj)`，框按 anchor-free 解码 |
| 特征 | SFace | `input` `[1,3,112,112]` float32，**RGB 顺序 + 原始 0~255**（对齐后人脸） | `output` `[1,128]` float32（归一化在模型内，`FeatureExtractor` 再补一次 L2） |
| 活体 | MiniFASNetV2 | `input` `[1,3,80,80]` float32，**BGR 顺序 + 原始 0~255**（人脸框外扩 2.7 倍后裁剪） | `output` `[1,3]` float32 logits（`LivenessDetector` 内部 softmax：0=print / 1=real / 2=replay） |

> 三个模型的预处理都是「仅缩放，不归一化」，唯一差异是 YuNet / MiniFASNetV2 用 BGR、SFace 用 RGB。

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

- **图像格式**：解码交给 OpenCV（jpg / png / bmp / webp 等），统一为 BGR 三通道 `Mat`：
  文件用 `Imgcodecs.imread`，字节流用 `ImageUtils.byteArrayToMat(bytes)`；输出用 `ImageUtils.matToBytes(mat, "jpg", 90)`。
- **资源释放**：所有推理入口返回的 `Mat`（含 `AvatarResult` / `CardResult`）由调用方 `release()`；
  `ModelManager` 是 `AutoCloseable`，请用 try-with-resources。
- **Embedding 维度**：固定 **128d**，已 L2 归一化（点积 = 余弦相似度）。
- **GPU 加速**：替换 `onnxruntime` 依赖为 `onnxruntime_gpu` 并启用 CUDA provider 即可。
- **批量输入**：本模块按"一次一图"调用，向量库侧的批量写入由调用方控制。
- **活体**：默认关闭（无模型路径时跳过加载），商业落地前请按 `AGENTS.md` §6.1 自查模型许可。

---

## 9. 模型替换备忘

要替换为其他 ONNX 人脸检测 / 识别模型（如 MobileFaceNet、ArcFace R50 自训版）：

1. **检测端**：实现 `FaceDetector` 接口约定（输入 OpenCV `Mat`，输出 `List<FaceBox>` + 5 关键点）。
2. **识别端**：实现 `FeatureExtractor` 接口约定（输入已对齐的 112×112 图，输出 `float[128]` 已 L2-normalized 向量）。
3. **接入**：`ModelManager` 已抽象各算子，通过 Starter 的 `@ConditionalOnMissingBean` 替换默认实现即可。

模型规格 / Starter 详见：

- 启动器：[`mica-ai-starters/mica-ai-face-spring-boot-starter/README.md`](../../mica-ai-starters/mica-ai-face-spring-boot-starter/README.md)
- 模型资产：[`model-tools/face/README.md`](../../model-tools/face/README.md)