# mica-ai-face-spring-boot-starter

> InsightFace 人脸识别 Spring Boot Starter，基于 [mica-ai-face](../mica-ai-core/mica-ai-face/README.md) 核心模块。
>
> **只暴露 `FaceEngine` Bean，把图片转成 512 维 Embedding。人脸库与 1:N 检索不在本 Starter 范围内。**

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`，建议 JDK 17+。

---

## 2. 配置项

```yaml
mica:
  ai:
    face:
      det-model-path: models/det_10g.onnx       # 必填
      rec-model-path: models/w600k_r50.onnx     # 必填
      det-score-threshold: 0.5
      det-nms-threshold: 0.4
      intra-op-num-threads: 1
      inter-op-num-threads: 1
```

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `mica.ai.face.det-model-path` | Path | — | RetinaFace 检测模型路径（必填） |
| `mica.ai.face.rec-model-path` | Path | — | ArcFace 识别模型路径（必填） |
| `mica.ai.face.det-score-threshold` | float | `0.5` | 检测置信度阈值 |
| `mica.ai.face.det-nms-threshold` | float | `0.4` | NMS IoU 阈值 |
| `mica.ai.face.intra-op-num-threads` | int | `1` | ONNX 内部线程 |
| `mica.ai.face.inter-op-num-threads` | int | `1` | ONNX 交互线程 |

> `det-model-path` 和 `rec-model-path` 至少配置一项以上，本 Starter 才会装配 `FaceEngine` Bean。

---

## 3. 使用示例

### 3.1 控制器示例：上传图片 → 返回 512 维向量

```java
@RestController
@RequiredArgsConstructor
public class FaceController {

    private final FaceEngine faceEngine;

    @PostMapping("/face/embed")
    public List<float[]> embed(@RequestParam("file") MultipartFile file) throws IOException {
        BufferedImage image = ImageIO.read(file.getInputStream());
        return faceEngine.extract(image).stream()
            .map(FaceEmbedding::getVector)
            .toList();
    }

    @PostMapping("/face/detect")
    public List<FaceBox> detect(@RequestParam("file") MultipartFile file) throws IOException {
        return faceEngine.detect(ImageIO.read(file.getInputStream()));
    }
}
```

### 3.2 与向量数据库协作（伪代码）

```java
@Service
@RequiredArgsConstructor
public class FaceService {

    private final FaceEngine faceEngine;
    private final MilvusClient milvus;

    public void enroll(String userId, MultipartFile portrait) throws IOException {
        BufferedImage image = ImageIO.read(portrait.getInputStream());
        FaceEmbedding emb = faceEngine.extract(image).get(0);
        milvus.insert("face_gallery", userId, emb.getVector());
    }

    public List<Match> recognize(MultipartFile probe, int topK) throws IOException {
        BufferedImage image = ImageIO.read(probe.getInputStream());
        FaceEmbedding emb = faceEngine.extract(image).get(0);
        return milvus.search("face_gallery", emb.getVector(), topK);
    }
}
```

> 1:N 检索由向量数据库负责，**Starter 不内置**任何 `FaceDatabase` 实现。
