# mica-ai-face

> InsightFace 人脸识别推理（纯 ONNX Runtime），基于 buffalo_l 模型集。
>
> **只做一件事：把图片转成 512 维 L2 归一化的 Embedding 向量。人脸库与 1:N 检索不在本模块范围内。**

零 OpenCV / 零 Python / 零 PyTorch 依赖。完整复现 **RetinaFace 检测 → 5 关键点对齐 → ArcFace 推理 → L2 归一化** 全链路，输出可直接写入向量数据库（Milvus / pgvector / Qdrant 等）做入库与检索。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Temurin / Azul Zulu 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.26.0 | 通过 Maven 自动拉取 |

---

## 2. 模型准备

使用 buffalo_l 模型集（精度优先），目录结构：

```
models/
├── det_10g.onnx       # RetinaFace 检测（输入 640x640，输出框 + 5 关键点）
└── w600k_r50.onnx     # ArcFace 识别（输入 112x112，输出 512d Embedding）
```

下载 / 转换方式见 [`model-tools/face/`](../../model-tools/face/README.md)。

---

## 3. 快速使用

### 3.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 3.2 一张图 → 所有人脸向量

```java
try (FaceEngine engine = FaceEngine.builder()
    .detModelPath(Path.of("models/det_10g.onnx"))
    .recModelPath(Path.of("models/w600k_r50.onnx"))
    .build()) {

    // 文件路径
    List<FaceEmbedding> faces = engine.extract(Path.of("group.jpg"));

    // 或 BufferedImage
    BufferedImage img = ImageIO.read(new File("portrait.jpg"));
    List<FaceEmbedding> faces2 = engine.extract(img);

    for (FaceEmbedding fe : faces) {
        System.out.println("向量维度 = " + fe.dimension());    // 512
        System.out.println("L2 范数 = " + computeNorm(fe.getVector()));  // ≈ 1.0
    }
}
```

### 3.3 只检测，不要向量

```java
List<FaceBox> boxes = engine.detect(image);
for (FaceBox box : boxes) {
    System.out.printf("score=%.3f, [%.0f,%.0f,%.0f,%.0f]%n",
        box.getScore(), box.getX1(), box.getY1(), box.getX2(), box.getY2());
}
```

---

## 4. 人脸库 / 1:N 检索怎么办？

**本模块不提供**，因为人脸库是业务领域（用户体系、权限、生命周期），不是 AI 模块该管的。

> 1:N 检索的本质就是"向量最近邻"，使用向量数据库是更专业的做法。

### 推荐路径

| 规模 | 推荐存储 | 说明 |
|------|---------|------|
| < 1 万人脸 | 内存 `Map<String, float[]>` | 简单暴力遍历，无需额外组件 |
| 1 万 ~ 100 万 | pgvector / OpenSearch | 与业务库同库运维 |
| > 100 万 | Milvus / Qdrant / Pinecone | 专业 ANN 索引，毫秒级返回 |

### 与 Milvus 协作示例（伪代码）

```java
try (FaceEngine engine = FaceEngine.builder()
    .detModelPath(Path.of("models/det_10g.onnx"))
    .recModelPath(Path.of("models/w600k_r50.onnx"))
    .build()) {

    // 入库
    FaceEmbedding emb = engine.extract(portrait).get(0);
    milvusClient.insert("face_gallery", List.of(
        InsertParam.Field.of("user_id", "alice"),
        InsertParam.Field.of("embedding", Floats.asList(emb.getVector()))
    ));

    // 检索
    FaceEmbedding unknown = engine.extract(testImage).get(0);
    SearchResult hit = milvusClient.search(SearchParam.builder()
        .collectionName("face_gallery")
        .topK(5)
        .vectors(List.of(Floats.asList(unknown.getVector())))
        .build());
    // hit.getResults() 即为 top5 user_id + 相似度
}
```

### 1:1 比对（两向量算相似度）

如果你确实只有两个 embedding 想比较（无库），用 `ArcFaceRecognizer.cosineSimilarity()`：

```java
float score = ArcFaceRecognizer.cosineSimilarity(embA.getVector(), embB.getVector());
boolean samePerson = score > 0.45f;  // buffalo_l 经验阈值
```

> 注意：1:1 比对不需要"库"，是单纯的数学运算，因此**不属于人脸库职责**，本模块保留这个工具方法。

---

## 5. 核心组件

| 组件 | 类 | 说明 |
|------|-----|------|
| **门面** | `FaceEngine` | 主入口，编排 检测 → 对齐 → 识别 |
| **检测** | `RetinaFaceDetector` | RetinaFace（det_10g.onnx）ONNX 推理，含 NMS |
| **识别** | `ArcFaceRecognizer` | ArcFace（w600k_r50.onnx）ONNX 推理 + 5 关键点仿射对齐 + L2 归一化 |
| **图像工具** | `ImageUtils` | 读取 / Letterbox / BGR→Float32，零 OpenCV |

### 处理流程

```
图片 → Letterbox → BGR Float32 [1,3,640,640]
     → RetinaFace ONNX → 人脸框 + 5 关键点
     → 按关键点仿射对齐 → 112x112 BGR
     → ArcFace ONNX → 512d Float32
     → L2 归一化 → 输出 FaceEmbedding.vector
```

---

## 6. 注意事项

- **图像格式**：支持 JPEG / PNG / GIF / BMP（通过 JDK 自带 `ImageIO`）。
- **Letterbox**：检测输入会自动按长宽比缩放并灰边填充（与 InsightFace Python 版完全一致）。
- **Embedding 维度**：固定 512，已 L2 归一化（点积 = 余弦相似度）。
- **GPU 加速**：替换 `onnxruntime` 依赖为 `onnxruntime_gpu` 并启用 CUDA provider 即可。
- **批量输入**：本模块按"一次一图"调用，向量库侧的批量写入由调用方控制。
