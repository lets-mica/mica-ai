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
      model:                              # ⚠️ 模型路径统一挂在 model 下
        detection:
          path: classpath:models/face_detection_yunet_2023mar.onnx       # 必填
        recognition:
          path: classpath:models/face_recognition_sface_2021dec.onnx     # 必填
        liveness:
          path: classpath:models/2.7_80x80_MiniFASNetV2.onnx             # liveness.enabled=true 时必填
      detection:
        threshold: 0.9                    # 检测置信度阈值
        nms-threshold: 0.3                # NMS IoU 阈值
      liveness:
        enabled: true                     # 活体开关，默认 true；关闭请显式设 false
        threshold: 0.85                   # 活体判定阈值
        crop-scale: 2.7                   # 人脸框外扩比例
      verify:
        threshold: 0.35                   # 1:1 比对阈值
        strategy: LARGEST_AREA            # 多脸选脸：LARGEST_AREA / LARGEST_SCORE / REJECT
      avatar:
        size: 256
      card:
        output-width: 1011
        output-height: 638
      onnx:
        device: CPU                       # CPU / GPU（枚举）
        intra-op-num-threads: 0           # 0 = ORT 默认
        inter-op-num-threads: 0
        graph-optimization-level: ENABLE_ALL   # DISABLE_ALL / ENABLE_BASIC / ENABLE_EXTENDED / ENABLE_ALL
```

| 配置 | 默认 | 说明 |
|------|------|------|
| `mica.ai.face.enabled` | `true` | 总开关 |
| `mica.ai.face.model.detection.path` | — | YuNet 路径（必填，支持 `classpath:`） |
| `mica.ai.face.model.recognition.path` | — | SFace 路径（必填） |
| `mica.ai.face.model.liveness.path` | — | MiniFASNetV2 路径（`liveness.enabled=true` 时必填） |
| `mica.ai.face.detection.threshold` | `0.9` | 检测置信度阈值 |
| `mica.ai.face.detection.nms-threshold` | `0.3` | NMS IoU 阈值 |
| `mica.ai.face.liveness.enabled` | `true` | 活体开关（开启时 `model.liveness.path` 必填，否则启动失败） |
| `mica.ai.face.liveness.threshold` | `0.85` | 活体判定阈值（低于该值判为攻击） |
| `mica.ai.face.liveness.crop-scale` | `2.7` | 人脸框外扩比例 |
| `mica.ai.face.verify.threshold` | `0.35` | 1:1 比对阈值 |
| `mica.ai.face.verify.strategy` | `LARGEST_AREA` | 单图多脸时的选脸策略：`LARGEST_AREA`（面积最大）/ `LARGEST_SCORE`（置信度最高）/ `REJECT`（多脸直接抛 `MicaAiException`） |
| `mica.ai.face.avatar.size` | `256` | 头像边长（像素） |
| `mica.ai.face.card.output-width` | `1011` | 卡片输出宽 |
| `mica.ai.face.card.output-height` | `638` | 卡片输出高 |
| `mica.ai.face.onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `mica.ai.face.onnx.inter-op-num-threads` | `0` | ORT 交互线程 |
| `mica.ai.face.onnx.device` | `CPU` | `CPU` / `GPU`（枚举） |

> `enabled=false` 时不装配任何 face Bean；`model.detection.path` / `model.recognition.path` 缺失时启动会 **fail-fast**。
> ⚠️ 活体默认**开启**，因此只配检测 / 识别路径会因活体模型缺失而启动失败；不需要活体请显式设 `mica.ai.face.liveness.enabled=false`。

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

    public float[] enroll(Mat image) {
        FaceBox box = detector.detect(image).get(0);   // 入口统一是 OpenCV Mat
        Mat aligned = null;
        try {
            aligned = aligner.align(image, box);
            return extractor.extract(aligned);         // 128d L2 归一化
        } finally {
            ImageUtils.releaseAll(aligned);
        }
    }

    public boolean verify(Mat probe, Mat reference) {
        // 相似度计算在静态工具方法上（点积即余弦，向量已 L2 归一化）
        return FeatureExtractor.compare(enroll(probe), enroll(reference)) > 0.35f;
    }
}
```

> ⚠️ 全部入口都是 OpenCV `Mat`，**没有 `BufferedImage` 重载**；字节流请先 `ImageUtils.byteArrayToMat(bytes)` 解码。
> 需要 1:1 比对时更推荐直接用 `FaceVerifier#verify(probe, reference)`，它内部已完成「检测 → 选脸 → 对齐 → 特征」全链路。

### 3.2 REST 端点：上传图片 → 128d 向量

```java
@RestController
@RequiredArgsConstructor
public class FaceController {

    private final FaceDetector detector;
    private final FaceAligner aligner;
    private final FeatureExtractor extractor;

    @PostMapping(value = "/face/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public List<FaceBox> detect(@RequestParam("file") MultipartFile file) throws IOException {
        Mat img = ImageUtils.byteArrayToMat(file.getBytes());
        try {
            return detector.detect(img);
        } finally {
            ImageUtils.releaseAll(img);
        }
    }

    @PostMapping(value = "/face/extract", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public List<float[]> extract(@RequestParam("file") MultipartFile file) throws IOException {
        Mat img = ImageUtils.byteArrayToMat(file.getBytes());
        try {
            List<float[]> out = new ArrayList<>();
            for (FaceBox box : detector.detect(img)) {
                Mat aligned = null;
                try {
                    aligned = aligner.align(img, box);
                    out.add(extractor.extract(aligned));
                } finally {
                    ImageUtils.releaseAll(aligned);
                }
            }
            return out;
        } finally {
            ImageUtils.releaseAll(img);
        }
    }
}
```

### 3.3 与 Milvus 协作（伪代码）

```java
@Service
@RequiredArgsConstructor
public class FaceGalleryService {
    private final FaceDetector detector;
    private final FaceAligner aligner;
    private final FeatureExtractor extractor;
    private final MilvusClient milvus;

    public void enroll(String userId, Mat portrait) {
        FaceBox box = detector.detect(portrait).get(0);
        Mat aligned = null;
        try {
            aligned = aligner.align(portrait, box);
            float[] emb = extractor.extract(aligned);
            milvus.insert("face_gallery", userId, emb);
        } finally {
            ImageUtils.releaseAll(aligned);
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
        return new MyTrainedDetector(props.getModel().getDetection().getPath());
    }

    @Bean
    public FeatureExtractor myExtractor(FaceProperties props) {
        return new MyTrainedExtractor(props.getModel().getRecognition().getPath());
    }
}
```

只要替换的 Bean 类型一致，**业务代码零改动**。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype Starter，提供 `/face/detect`、`/face/extract` 等 REST 端点。