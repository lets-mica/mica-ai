# mica-ai-intent-spring-boot-starter

> Spring Boot Starter for BERT 中文意图识别，一行配置即可接入。

基于 [mica-ai-intent](../../mica-ai-core/mica-ai-intent/README.md) 核心引擎，提供 `@ConfigurationProperties` 自动配置与 `BertIntent` Bean 自动注入。

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
    <artifactId>mica-ai-intent-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2.2 配置文件

```yaml
mica:
  ai:
    intent:
      model-path: model/bert_intent.onnx
      vocab-path: model/vocab.txt
      labels:
        - weather
        - music
        - chat
        - news
      max-length: 128
```

### 2.3 注入使用

```java
import net.dreamlu.mica.ai.intent.BertIntent;
import net.dreamlu.mica.ai.intent.config.IntentResult;
import org.springframework.stereotype.Service;

@Service
public class IntentService {

    private final BertIntent intent;

    public IntentService(BertIntent intent) {
        this.intent = intent;
    }

    public String classify(String text) {
        IntentResult result = intent.predict(text);
        return result.intent();
    }
}
```

---

## 3. 配置属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mica.ai.intent.model-path` | String | — | ONNX 模型文件路径（必填） |
| `mica.ai.intent.vocab-path` | String | — | BERT 词表文件路径（必填） |
| `mica.ai.intent.labels` | List\<String\> | — | 意图标签列表（必填） |
| `mica.ai.intent.max-length` | int | `128` | 最大序列长度 |
| `mica.ai.intent.intra-op-num-threads` | int | `1` | ONNX 内部线程数 |
| `mica.ai.intent.inter-op-num-threads` | int | `1` | ONNX 交互线程数 |

### AutoConfiguration 条件

- `mica.ai.intent.enabled` 不为 `false`（默认启用）
- classpath 存在 `BertIntent` 类
