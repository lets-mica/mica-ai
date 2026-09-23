# mica-ai-plate-spring-boot-starter

> [HyperLPR3](https://github.com/szad670401/HyperLPR) 中国车牌识别 Spring Boot Starter，基于 [mica-ai-plate](../../mica-ai-core/mica-ai-plate/README.md) 核心模块。**Apache-2.0 可商用**。

零配置即可注入 `PlatePipeline` Bean，对外提供 `recognizePath` / `recognizeBytes` / `recognize(Mat)` 三个 API。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-plate-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。

---

## 2. 配置项（`mica.ai.plate` 前缀）

```yaml
mica:
  ai:
    plate:
      enabled: true                                          # 默认 true
      model-version: 20230229                                # 模型版本（日志用）
      detection-model-path: classpath:models/y5fu_320x_sim.onnx
      recognition-model-path: classpath:models/rpv3_mdict_160_r3.onnx
      classification-model-path: classpath:models/litemodel_cls_96x_r1.onnx
      detection-input-size: 320                              # 320 / 640
      detection-confidence-threshold: 0.25
      detection-nms-threshold: 0.5
      recognition-input-height: 48                          # HyperLPR3 默认
      recognition-input-width: 160                          # HyperLPR3 默认
      classification-input-size: 96                         # HyperLPR3 默认
      max-plates: 5
      onnx:
        device: CPU                                         # CPU / GPU（枚举）
        intra-op-num-threads: 0                             # 0 = ORT 默认
        inter-op-num-threads: 0
        cuda-device-id: 0
```

| 配置 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 是否启用自动装配 |
| `model-version` | `20230229` | 版本标识（日志用） |
| `detection-model-path` / `recognition-model-path` / `classification-model-path` | 无 | 三模型路径，**必填**，支持 `classpath:` |
| `detection-input-size` | `320` | 检测输入边长（仅 `320` / `640`） |
| `detection-confidence-threshold` | `0.25` | 检测置信度阈值 |
| `detection-nms-threshold` | `0.5` | NMS IoU 阈值 |
| `max-plates` | `5` | 单图最多返回车牌数 |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> `enabled=false` 时不装配 `PlatePipeline`；三个模型路径必填，缺失启动会 **fail-fast**。

---

## 3. 使用示例

### 3.1 REST 端点：上传图片 → 返回车牌

```java
@RestController
@RequiredArgsConstructor
public class PlateController {

    private final PlatePipeline pipeline;

    @PostMapping("/plate/recognize")
    public List<PlateResult> recognize(@RequestParam("file") MultipartFile file) throws IOException {
        return pipeline.recognizeBytes(file.getBytes());
    }
}
```

### 3.2 业务用法

```java
@Service
@RequiredArgsConstructor
public class GateService {

    private final PlatePipeline pipeline;

    public String recognizeFirstPlate(byte[] imageBytes) {
        List<PlateResult> results = pipeline.recognizeBytes(imageBytes);
        if (results.isEmpty()) {
            return null;
        }
        PlateResult r = results.get(0);
        return r.getPlateCode();           // e.g. "津B6H920"
    }
}
```

### 3.3 输出字段（`PlateResult`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `plateCode` | `String` | 车牌号（双层牌为上下两行拼接） |
| `plateType` | `PlateType` | 10 类枚举，详见 [`mica-ai-plate/README.md`](../../mica-ai-core/mica-ai-plate/README.md) |
| `detectionConfidence` | `float` | 检测框得分 |
| `recognitionConfidence` | `float` | CTC 解码字符概率均值 |
| `boundingBox` | `int[4]` | 原图坐标 `[x1, y1, x2, y2]` |
| `landmarks` | `int[4][2]` | 4 角点（左上 / 右上 / 右下 / 左下） |

---

## 4. 自定义 Pipeline

```java
@Configuration
public class MyPlateConfig {

    @Bean
    public PlatePipeline myPipeline(PlateProperties props) {
        return PlatePipeline.create(props.toPlateConfig());
    }
}
```

只要返回 `PlatePipeline` 即可被 Starter 识别为自定义实现，**业务代码零改动**。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)。