# mica-ai-filetype-spring-boot-starter

> Google [Magika](https://github.com/google/magika) 文件类型识别 Spring Boot Starter，基于 [mica-ai-filetype](../mica-ai-core/mica-ai-filetype/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 注入 `FiletypeDetector` Bean，对外提供 `detectPath` / `detectBytes` / `detectStream` 三个 API。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-filetype-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`，建议 JDK 8+（与父项目一致）。

---

## 2. 配置项

```yaml
mica:
  ai:
    filetype:
      enabled: true                    # 默认 true
      model-version: standard_v3_3     # 模型版本（固定，备用未来升级）
      model-path: classpath:mica-ai/models/filetype/standard_v3_3/model.onnx
      config-path: classpath:mica-ai/models/filetype/standard_v3_3/config.min.json
      content-types-path: classpath:mica-ai/models/filetype/standard_v3_3/content_types_kb.min.json
      prediction-mode: HIGH_CONFIDENCE # HIGH_CONFIDENCE / MEDIUM_CONFIDENCE / BEST_GUESS
      onnx:
        intra-op-num-threads: 0        # 0 = ORT 自适应
        inter-op-num-threads: 0
        cuda-device-id: 0
```

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `mica.ai.filetype.enabled` | boolean | `true` | 总开关 |
| `mica.ai.filetype.model-version` | String | `standard_v3_3` | 模型版本 |
| `mica.ai.filetype.model-path` | String | classpath | `model.onnx` 路径（classpath / 绝对路径） |
| `mica.ai.filetype.config-path` | String | classpath | `config.min.json` 路径 |
| `mica.ai.filetype.content-types-path` | String | classpath | `content_types_kb.min.json` 路径 |
| `mica.ai.filetype.prediction-mode` | String | `HIGH_CONFIDENCE` | 三种预测模式 |
| `mica.ai.filetype.onnx.intra-op-num-threads` | int | `0` | ONNX 内部线程 |
| `mica.ai.filetype.onnx.inter-op-num-threads` | int | `0` | ONNX 交互线程 |
| `mica.ai.filetype.onnx.cuda-device-id` | int | `0` | CUDA 设备 id |

> 三个路径必须都能读到；`enabled=false` 时不装配 `FiletypeDetector` Bean。

---

## 3. 使用示例

### 3.1 控制器示例：上传文件 → 返回类型

```java
@RestController
@RequiredArgsConstructor
public class FiletypeController {

    private final FiletypeDetector detector;

    @PostMapping("/filetype/detect")
    public FiletypeResult detect(@RequestParam("file") MultipartFile file) throws IOException {
        try (InputStream in = file.getInputStream()) {
            return detector.detectStream(in);
        }
    }

    @PostMapping(value = "/filetype/detect-bytes", consumes = MediaType.APPLICATION_OCTET_STREAM_VALUE)
    public FiletypeResult detectBytes(@RequestParam("file") MultipartFile file) throws IOException {
        return detector.detectBytes(file.getBytes());
    }
}
```

### 3.2 文件入库预审（伪代码）

```java
@Service
@RequiredArgsConstructor
public class UploadAuditService {

    private final FiletypeDetector detector;

    public void audit(MultipartFile file) throws IOException {
        FiletypeResult r = detector.detectBytes(file.getBytes());
        if (!"application/pdf".equals(r.getContentType().getMimeType())) {
            throw new IllegalArgumentException("仅允许 PDF 上传，检测到：" + r.getOutputLabel());
        }
    }
}
```

---

## 4. 自定义 Detector

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `FiletypeDetector` Bean：

```java
@Configuration
public class MyFiletypeConfig {

    @Bean
    public FiletypeDetector myDetector(FiletypeProperties props) {
        return FiletypeDetector.create(FiletypeConfig.builder()
            .modelPath("models/my-fine-tuned.onnx")
            .configPath("models/my-config.min.json")
            .contentTypesPath("models/my-kb.min.json")
            .build());
    }
}
```

只要 `FiletypeDetector` 是接口实现（或自定义类），Starter 就会用你的 Bean 替换默认实现，**业务代码无需改动**。