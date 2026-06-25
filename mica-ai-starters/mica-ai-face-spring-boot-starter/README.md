# mica-ai-face-spring-boot-starter

> OpenCV Zoo（YuNet + SFace）人脸识别 Spring Boot Starter，基于 [mica-ai-face](../mica-ai-core/mica-ai-face/README.md) 核心模块。**Apache-2.0 可商用**。
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
      det-model-path: models/face_detection_yunet_2023mar.onnx        # 必填
      rec-model-path: models/face_recognition_sface_2021dec.onnx      # 必填
      model-type: YUNET_SFACE                                         # 默认，可选
      det-score-threshold: 0.6
      det-nms-threshold: 0.3
      intra-op-num-threads: 1
      inter-op-num-threads: 1
```

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `mica.ai.face.det-model-path` | Path | — | YuNet 检测模型路径（必填） |
| `mica.ai.face.rec-model-path` | Path | — | SFace 识别模型路径（必填） |
| `mica.ai.face.model-type` | enum | `YUNET_SFACE` | 模型实现（当前仅 OpenCV Zoo） |
| `mica.ai.face.det-score-threshold` | float | `0.6` | 检测置信度阈值 |
| `mica.ai.face.det-nms-threshold` | float | `0.3` | NMS IoU 阈值（YuNet 内部已 NMS） |
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

---

## 4. 自定义 detector / recognizer

Starter 通过 `ObjectProvider` 优先使用用户声明的 Bean，没声明则走默认 YuNet + SFace：

```java
@Configuration
public class MyFaceConfig {

    /**
     * 用自家训练的检测器替换默认 YuNet。
     */
    @Bean
    public FaceDetector myDetector(FaceProperties props) {
        return new MyTrainedDetector(props.getDetModelPath());
    }

    /**
     * 用自家训练的识别器替换默认 SFace。
     */
    @Bean
    public FaceRecognizer myRecognizer(FaceProperties props) {
        return new MyTrainedRecognizer(props.getRecModelPath());
    }
}
```

只要 `FaceDetector` / `FaceRecognizer` 是接口实现，Starter 就会自动注入到 `FaceEngine`，**零业务代码改动**。