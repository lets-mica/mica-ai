# mica-ai-face

> OpenCV Zoo（YuNet + SFace）人脸识别推理，纯 ONNX Runtime，**Apache-2.0 可商用**。
>
> **只做一件事：把图片转成 512 维 L2 归一化的 Embedding 向量。人脸库与 1:N 检索不在本模块范围内。**

零 OpenCV Java 绑定 / 零 Python / 零 PyTorch 依赖。完整复现 **YuNet 检测 → 5 关键点对齐 → SFace 推理 → L2 归一化** 全链路，输出可直接写入向量数据库（Milvus / pgvector / Qdrant 等）做入库与检索。

---

## ✅ License 一句话总结

整套链路（Java 代码 + ONNX Runtime + YuNet 模型 + SFace 模型）**全部 Apache-2.0**，可以闭源分发、商用 SaaS、卖盒子，无须邮件申请。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Temurin / Azul Zulu 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.26.0 | 通过 Maven 自动拉取 |

---

## 2. 模型准备

使用 [OpenCV Zoo](https://github.com/opencv/opencv_zoo) 的人脸模型（Apache-2.0），目录结构：

```
models/
├── face_detection_yunet_2023mar.onnx        # YuNet 检测（320x320 RGB，~340 KB）
└── face_recognition_sface_2021dec.onnx      # SFace 识别（112x112 RGB，~89 MB）
```

下载 / 转换方式见 [`model-tools/face/`](../../model-tools/face/README.md)：

```bash
cd model-tools/face
python download.py        # 从 OpenCV Zoo GitHub raw 下载 YuNet + SFace
python convert.py         # 拷贝/链接到 model/out/
```

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
    .detModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
    .recModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
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

## 4. 接入自定义模型（可插拔架构）

`FaceDetector` 与 `FaceRecognizer` 都是**接口**，默认实现 YuNet + SFace。要换其他模型集（如自家训练的识别模型）：

```java
FaceDetector   myDetector   = new MyCustomDetector(config);   // 实现 FaceDetector
FaceRecognizer myRecognizer = new MyCustomRecognizer(config); // 实现 FaceRecognizer

try (FaceEngine engine = FaceEngine.builder()
    .config(config)
    .detector(myDetector)
    .recognizer(myRecognizer)
    .build()) {
    // ...
}
```

> 未来 mica-ai-face 增加新的内置实现，只需在
> [FaceEngine.createDefault* 工厂](src/main/java/net/dreamlu/mica/ai/face/FaceEngine.java) 加 `case`，
> 业务代码零改动。

---

## 5. 人脸库 / 1:N 检索怎么办？

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
    .detModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
    .recModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
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

如果你确实只有两个 embedding 想比较（无库），用 `FaceRecognizer.cosineSimilarity()`：

```java
float score = FaceRecognizer.cosineSimilarity(embA.getVector(), embB.getVector());
boolean samePerson = score > 0.5f;  // SFace 经验阈值（OpenCV Zoo 推荐 0.363 / 1:1）
```

> 注意：1:1 比对不需要"库"，是单纯的数学运算，因此**不属于人脸库职责**，本模块保留这个工具方法。

---

## 6. 核心组件

| 组件 | 类型 | 说明 |
|------|-----|------|
| **门面** | `FaceEngine` | 主入口，编排 检测 → 对齐 → 识别 |
| **检测接口** | `FaceDetector` | `detect(BufferedImage) -> List<FaceBox>` |
| **识别接口** | `FaceRecognizer` | `extract(BufferedImage) -> FaceEmbedding` + 静态 l2Normalize / cosineSimilarity |
| **默认检测** | `YuNetDetector` | YuNet（face_detection_yunet_2023mar.onnx）ONNX 推理 |
| **默认识别** | `SFaceRecognizer` | SFace（face_recognition_sface_2021dec.onnx）ONNX 推理 + L2 归一化 |
| **图像工具** | `ImageUtils` | 读取 / Letterbox / RGB→Float32 / 5 关键点仿射对齐，零 OpenCV |

### 处理流程

```
图片 → Letterbox → RGB Float32 [1,3,320,320]
     → YuNet ONNX → 人脸框 + 5 关键点（已 NMS）
     → 按关键点仿射对齐 → 112x112 BGR
     → SFace ONNX → 512d Float32
     → L2 归一化 → 输出 FaceEmbedding.vector
```

---

## 7. 注意事项

- **图像格式**：支持 JPEG / PNG / GIF / BMP（通过 JDK 自带 `ImageIO`）。
- **Letterbox**：检测输入会自动按长宽比缩放并灰边填充。
- **Embedding 维度**：固定 512，已 L2 归一化（点积 = 余弦相似度）。
- **GPU 加速**：替换 `onnxruntime` 依赖为 `onnxruntime_gpu` 并启用 CUDA provider 即可。
- **批量输入**：本模块按"一次一图"调用，向量库侧的批量写入由调用方控制。

---

## 8. 模型替换备忘

要替换为其他 ONNX 人脸检测 / 识别模型（如 MobileFaceNet、ArcFace R50 自训版），参考：

1. **检测端**：实现 `FaceDetector` 接口（推荐继承 `ImageUtils.letterbox` + 自行 ONNX 推理）
2. **识别端**：实现 `FaceRecognizer` 接口（约定输入为已对齐的 112x112 BGR 图，返回 512d L2-normalized 向量）
3. **注册**：`FaceEngine.builder().detector(...).recognizer(...).build()` 注入

或更彻底地：fork 后改 `FaceEngine.createDefault*` 的 switch。