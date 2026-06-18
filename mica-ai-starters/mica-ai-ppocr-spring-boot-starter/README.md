# mica-ai-ppocr-spring-boot-starter

> Spring Boot Starter for PP-OCRv6 文字识别，一行配置即可接入。

基于 [mica-ai-ppocr](../../mica-ai-core/mica-ai-ppocr/README.md) 核心引擎，提供 `@ConfigurationProperties` 自动配置与 `PPOcrV6Engine` Bean 自动注入。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Spring Boot | 4.1.0+ | 自动配置依赖 |
| Maven | 3.6+ | 编译 / 打包 |

---

## 2. 快速使用

### 2.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-ppocr-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2.2 配置文件

```yaml
mica:
  ai:
    ppocr:
      detection-model-path: models/PP-OCRv6_tiny_det_onnx/inference.onnx
      recognition-model-path: models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx
      dict-path: models/rec_char_dict.txt
      onnx-provider: cpu
```

### 2.3 注入使用

```java
import net.dreamlu.mica.ai.ppocr.core.PPOcrV6Engine;
import net.dreamlu.mica.ai.ppocr.core.PPOcrV6Result;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OcrService {

    private final PPOcrV6Engine engine;

    public OcrService(PPOcrV6Engine engine) {
        this.engine = engine;
    }

    public void recognize(String imagePath) {
        Mat img = Imgcodecs.imread(imagePath);
        List<PPOcrV6Result> results = engine.run(img);
        for (PPOcrV6Result r : results) {
            System.out.printf("%s (%.3f)%n", r.text(), r.score());
        }
    }
}
```

---

## 3. 配置属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mica.ai.ppocr.detection-model-path` | String | — | 检测模型 ONNX 路径（必填） |
| `mica.ai.ppocr.recognition-model-path` | String | — | 识别模型 ONNX 路径（必填） |
| `mica.ai.ppocr.dict-path` | String | — | 字符字典路径（必填） |
| `mica.ai.ppocr.onnx-provider` | String | `cpu` | ONNX 执行提供者 |

### AutoConfiguration 条件

- `mica.ai.ppocr.enabled` 不为 `false`（默认启用）
- classpath 存在 `PPOcrV6Engine` 类
