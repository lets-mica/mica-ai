# mica-ai-face-spring-boot-starter

> OpenCV Zoo 人脸识别 Spring Boot Starter，基于 [mica-ai-face](../../mica-ai-core/mica-ai-face/README.md) 核心模块。**Apache-2.0 / MIT 可商用**。

零配置即可注入 `FaceDetector` / `FaceAligner` / `FeatureExtractor` / `LivenessDetector` / `FaceVerifier` / `AvatarExtractor` / `CardExtractor` 等引擎 Bean。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。

---

## 2. 配置项（`mica.ai.face` 前缀）

```yaml
mica:
  ai:
    face:
      enabled: true                       # 总开关，默认 true
      detection:
        model-path: classpath:models/face_detection_yunet_2023mar.onnx   # 必填
        threshold: 0.9                    # 检测置信度阈值
        nms-threshold: 0.3                # NMS IoU 阈值
      recognition:
        model-path: classpath:models/face_recognition_sface_2021dec.onnx # 必填
      liveness:
        enabled: false                    # 默认 false
        model-path: classpath:models/2.7_80x80_MiniFASNetV2.onnx        # 启用必填
      verify:
        threshold: 0.35                   # 1:1 比对阈值
      avatar:
        size: 256
      card:
        output-width: 1011
        output-height: 638
      onnx:
        intra-op-num-threads: 0           # 0 = ORT 默认
        inter-op-num-threads: 0
        graph-optimization-level: ORT_ENABLE_ALL
```

| 配置 | 默认 | 说明 |
|------|------|------|
| `mica.ai.face.enabled` | `true` | 总开关 |
| `mica.ai.face.detection.model-path` | — | YuNet 路径（必填，支持 `classpath:`） |
| `mica.ai.face.detection.threshold` | `0.6` | 检测置信度阈值 |
| `mica.ai.face.detection.nms-threshold` | `0.3` | NMS IoU 阈值 |
| `mica.ai.face.recognition.model-path` | — | SFace 路径（必填） |
| `mica.ai.face.liveness.enabled` | `false` | 活体开关（开启必填 `model-path`） |
| `mica.ai.face.liveness.model-path` | — | MiniFASNetV2 路径 |
| `mica.ai.face.verify.threshold` | `0.35` | 1:1 比对阈值 |
| `mica.ai.face.avatar.size` | `256` | 头像边长（像素） |
| `mica.ai.face.card.output-width` | `1011` | 卡片输出宽 |
| `mica.ai.face.card.output-height` | `638` | 卡片输出高 |
| `mica.ai.face.onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `mica.ai.face.onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> `enabled=false` 时不装配任何 face Bean；`detection.model-path` / `recognition.model-path` 缺失时启动会 **fail-fast**。

---

## 3. 使用示例

### 3.1 注入并使用

```java
@Service
@RequiredArgsConstructor
public class FaceEnrollService {

    private final FaceDetector detector;              // YuNet ONNX
    private final FaceAligner aligner;                // 5 关键点仿射
    private final FeatureExtractor extractor;         // SFace 128d
    private final LivenessDetector liveness;          // MiniFASNetV2（启用后可用）
    private final FaceVerifier verifier;              // 1:1 比对门面
    private final AvatarExtractor avatarExtractor;    // 头像提取
    private final CardExtractor cardExtractor;        // 证件卡片提取

    public float[] enroll(BufferedImage image) {
        FaceBox box = detector.detect(image).get(0);
        try (Mat aligned = aligner.align(image, box)) {
            return extractor.extract(aligned);        // 128d L2 归一化
        }
    }

    public boolean verify(BufferedImage a, BufferedImage b) {
        float[] fa = enroll(a), fb = enroll(b);
        return verifier.cosineSimilarity(fa, fb) > 0.35f;
    }
}
```

### 3.2 REST 端点：上传图片 → 128d 向量

```java
@RestController
@RequiredArgsConstructor
public class FaceController {

    private final FaceDetector detector;
    private final FaceAligner aligner;
    private final FeatureExtractor extractor;

    @PostMapping("/face/detect")
    public List<FaceBox> detect(@RequestParam("file") MultipartFile file) throws IOException {
        BufferedImage img = ImageIO.read(file.getInputStream());
        return detector.detect(img);
    }

    @PostMapping("/face/extract")
    public List<float[]> extract(@RequestParam("file") MultipartFile file) throws IOException {
        BufferedImage img = ImageIO.read(file.getInputStream());
        List<float[]> out = new ArrayList<>();
        for (FaceBox box : detector.detect(img)) {
            try (Mat aligned = aligner.align(img, box)) {
                out.add(extractor.extract(aligned));
            }
        }
        return out;
    }
}
```

### 3.3 与 Milvus 协作（伪代码）

```java
@Service
@RequiredArgsConstructor
public class FaceGalleryService {
    private final FeatureExtractor extractor;
    private final MilvusClient milvus;

    public void enroll(String userId, BufferedImage portrait) {
        FaceBox box = detector.detect(portrait).get(0);
        try (Mat aligned = aligner.align(portrait, box)) {
            float[] emb = extractor.extract(aligned);
            milvus.insert("face_gallery", userId, emb);
        }
    }
}
```

---

## 4. 自定义实现

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 Bean，没声明则走默认实现：

```java
@Configuration
public class MyFaceConfig {

    @Bean
    public FaceDetector myDetector(FaceProperties props) {
        return new MyTrainedDetector(props.getDetection().getModelPath());
    }

    @Bean
    public FeatureExtractor myExtractor(FaceProperties props) {
        return new MyTrainedExtractor(props.getRecognition().getModelPath());
    }
}
```

只要替换的 Bean 类型一致，**业务代码零改动**。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype Starter，提供 `/face/detect`、`/face/extract` 等 REST 端点。