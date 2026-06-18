# mica-ai-speaker

> ERes2Net 声纹识别推理的 **Java 17** 实现，基于 ONNX Runtime + JTransforms FFT。

零 PyTorch / 零 Python 依赖。完整复现 80 维 log-Mel FBank 特征提取（HTK 标准 Mel 滤波器组）→ ONNX 推理 → 余弦相似度验证全链路。

支持声纹验证（Speaker Verification）和说话人识别（Speaker Identification）。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.26.0 | 通过 Maven 自动拉取 |
| JTransforms | 3.1 | 通过 Maven 自动拉取（FFT 运算） |

---

## 2. 模型准备

将 ERes2Net ONNX 模型放到 `models/` 下：

```
models/
└── eres2net.onnx          # ERes2Net 声道识别模型
```

模型需从官方渠道下载并导出为 ONNX 格式。

---

## 3. 快速使用

### 3.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-speaker</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 3.2 基本用法

```java
import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;

public class Demo {
    public static void main(String[] args) throws Exception {
        // 1. 创建声纹验证器
        try (SpeakerVerifier verifier = new SpeakerVerifier("models/eres2net.onnx")) {

            // 2. 提取声纹嵌入向量
            float[] embedding1 = verifier.extractEmbedding("speaker1.wav");
            float[] embedding2 = verifier.extractEmbedding("speaker2.wav");

            // 3. 计算相似度
            double similarity = verifier.similarity(embedding1, embedding2);
            System.out.printf("声纹相似度: %.4f%n", similarity);

            // 4. 声纹验证（判断是否同一人）
            boolean samePerson = similarity > 0.7;  // 阈值可按场景调整
            System.out.println("同一人: " + samePerson);
        }
    }
}
```

### 3.3 声纹验证（快速接口）

```java
// 直接比较两段音频
double similarity = verifier.verify("enroll.wav", "test.wav");
boolean samePerson = similarity > 0.7;
```

### 3.4 说话人识别

```java
// 注册说话人
Map<String, float[]> speakers = Map.of(
    "张三", verifier.extractEmbedding("zhangsan.wav"),
    "李四", verifier.extractEmbedding("lisi.wav"),
    "王五", verifier.extractEmbedding("wangwu.wav")
);

// 识别未知音频
float[] unknown = verifier.extractEmbedding("unknown.wav");
String bestMatch = speakers.entrySet().stream()
    .max(Comparator.comparingDouble(
        e -> verifier.similarity(unknown, e.getValue())))
    .map(Map.Entry::getKey)
    .orElse("未知");
System.out.println("识别结果: " + bestMatch);
```

---

## 4. 核心组件

| 组件 | 类 | 说明 |
|------|-----|------|
| **验证器** | `SpeakerVerifier` | 主入口，编排 FBank 提取 + ONNX 推理 + 相似度计算 |
| **特征提取** | `FBankExtractor` | 80 维 log-Mel FBank 特征，HTK 标准 Mel 滤波器组 + FFT |

### 处理流程

```
音频 → 预加重 → 分帧 → 汉明窗 → FFT → Mel 滤波 → Log → 80维 FBank → ONNX → 256维 Embedding
```

---

## 5. 注意事项

- **音频格式**：支持 WAV（PCM 16/32bit），自动重采样到 16kHz 单声道。
- **相似度阈值**：推荐 0.5~0.8，具体阈值需根据业务场景调整（越高越严格）。
- **Embedding 存储**：每个 256 维向量约 1KB，可存入数据库用于大规模检索。
- **GPU 加速**：替换 `onnxruntime` 为 `onnxruntime_gpu` 并指定 CUDA provider 即可。
