# mica-ai-tts

> Kokoro-82M 纯 ONNXRuntime 语音合成（TTS）推理的 **Java 17** 实现，移植自 [`hexgrad/kokoro`](https://github.com/hexgrad/kokoro) 与 [`thewh1teagle/kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx)。

零 PyTorch / 零 Python 依赖。完整复现分句 → G2P → ONNX 推理 → 静音裁剪 → WAV 编码全链路逻辑。

支持中 / 英双语、103 个音色（zf/zm/af/bf_*），内置可插拔 G2P 接口，可选用 `houbb/pinyin` 实现多音字智能消歧。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.20+ | 通过 Maven 自动拉取 |

无其他强制依赖。G2P 默认使用内置简化实现（覆盖 ~80 个常用汉字）；如需多音字消歧与中文分词，引入 `houbb/pinyin` 即可。

---

## 2. 模型准备

### 2.1 下载 ONNX 模型

Kokoro-82M 模型（v1.1，dynamic shape 版本，约 310MB）：

```bash
# ModelScope 镜像（推荐）
pip install modelscope
modelscope download --model KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX
```

或从 Hugging Face 下载：

```bash
# 原始仓库
# https://huggingface.co/hexgrad/Kokoro-82M
```

### 2.2 目录结构

模型下载完成后组织为如下结构：

```
model/
├── model_dynamic.onnx          # 主 ONNX 模型（动态 shape，约 310MB）
├── config.json                 # 词表 + 模型配置
└── voices/                     # 音色目录（103 个 .bin 文件）
    ├── zf_001.bin              # 中文女声 1
    ├── zf_002.bin              # 中文女声 2
    ├── zm_010.bin              # 中文男声
    ├── af_heart.bin            # 英文女声
    ├── bf_emma.bin             # 英文女声 2
    └── ...                     # 共 103 个音色
```

每个 `.bin` 文件为 raw float32 binary，shape = `[510, 256]`，总大小 522240 字节。

---

## 3. 快速使用

### 3.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 3.2 基本用法

```java
import net.dreamlu.mica.ai.tts.KokoroTts;
import net.dreamlu.mica.ai.tts.KokoroTtsConfig;
import net.dreamlu.mica.ai.tts.TtsResult;

public class Demo {
    public static void main(String[] args) throws Exception {
        // 1. 配置引擎
        KokoroTtsConfig config = KokoroTtsConfig.builder()
            .modelPath("model/model_dynamic.onnx")
            .voicesDir("model/voices")
            .configPath("model/config.json")
            .defaultVoice("zf_001")    // 中文女声
            .defaultSpeed(1.0f)        // 语速 0.5 ~ 2.0
            .onnxProvider("cpu")       // 可选: "cpu", "cuda"
            .build();

        // 2. 创建引擎并合成
        try (KokoroTts tts = new KokoroTts(config)) {
            // 方式 1：直接输入文本（使用默认 G2P）
            TtsResult result = tts.synthesize("你好世界，欢迎使用 Kokoro TTS。");
            System.out.printf("生成音频: %.2fs, %d 采样点%n",
                result.duration(), result.audio().length);

            // 3. 保存为 WAV 文件（24kHz, 16-bit PCM）
            tts.saveWav(result, "output.wav");

            // 方式 2：使用预生成音素（更高质量，可对接外部 G2P）
            // TtsResult result2 = tts.synthesizeFromPhonemes(
            //     "ㄋㄧ3 ㄏㄠ3 ㄕ4 ㄐㄧㄝ4", "zf_001", 1.0f);

            // 4. 列出所有可用音色
            System.out.println("可用音色: " + tts.listVoices());
        }
    }
}
```

### 3.3 切换音色

```java
// 使用不同音色合成
TtsResult r1 = tts.synthesize("Hello, this is a test.", "af_heart", 1.0f);  // 英文女声
TtsResult r2 = tts.synthesize("中文男声测试", "zm_010", 1.0f);                 // 中文男声
TtsResult r3 = tts.synthesize("语速更快", "zf_001", 1.5f);                     // 1.5 倍速
```

---

## 4. G2P 可插拔架构

mica-ai-tts 通过 `G2P` 接口让用户自由选择文本前端实现：

| 实现 | 覆盖度 | 多音字 | 分词 | 繁简体 | 依赖 |
|------|--------|--------|------|--------|------|
| `ChineseG2P`（默认） | ~80 常用汉字 | ❌ | ❌ | ❌ | 零 |
| `HoubbPinyinG2P` | 全字符 | ✅ | ✅ | ✅ | `com.github.houbb:pinyin:0.4.0` |
| 自定义 `G2P` | 自由 | - | - | - | - |

### 4.1 默认实现（零依赖）

```java
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath(...)
    .voicesDir(...)
    .configPath(...)
    .build();
// 自动使用 ChineseG2P 简化实现
```

### 4.2 高质量实现（houbb/pinyin）

引入依赖：

```xml
<dependency>
    <groupId>com.github.houbb</groupId>
    <artifactId>pinyin</artifactId>
    <version>0.4.0</version>
</dependency>
```

注入 G2P：

```java
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath(...)
    .voicesDir(...)
    .configPath(...)
    .g2p(new HoubbPinyinG2P())  // 注入高质量 G2P
    .build();
```

`HoubbPinyinG2P` 特性：
- **多音字智能消歧**：通过分词识别语境，例如 `重庆火锅` → `chóng qìng huǒ guō`（非 `zhòng qìng`）
- **中文分词**：`重量级` 不会被错误地切分为 `zhòng liàng jí`
- **繁简体支持**：`奮鬥` → `fèn dòu`
- **反射加载**：mica-ai-tts 对 houbb/pinyin 无强依赖，调用时检查 `isAvailable()`

### 4.3 自定义 G2P（Lambda）

```java
G2P customG2p = text -> {
    // 调用外部 G2P 工具（espeak-ng、misaki、Python 进程等）
    return callMyExternalG2P(text);
};

KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath(...)
    .voicesDir(...)
    .configPath(...)
    .g2p(customG2p)
    .build();
```

### 4.4 跳过 G2P，直接用音素

如果已有外部 G2P（如 misaki、Python espeak）生成的音素，可直接调用 `synthesizeFromPhonemes`：

```java
// misaki zh 输出的注音符号
String phonemes = "ㄋㄧ3 ㄏㄠ3 ㄕ4 ㄐㄧㄝ4 ㄏㄨㄛ2 ㄕ4";
TtsResult result = tts.synthesizeFromPhonemes(phonemes, "zf_001", 1.0f);
```

---

## 5. 核心组件

| 组件 | 类 | 说明 |
|------|-----|------|
| **引擎** | `KokoroTts` | 主入口，编排全流程 |
| **配置** | `KokoroTtsConfig` | Builder 风格配置，支持 G2P 注入 |
| **结果** | `TtsResult` | record(audio, sampleRate, duration) |
| **ONNX 推理** | `KokoroEngine` | 封装 `session.run()`，支持 CPU/CUDA |
| **词表** | `Vocab` | 从 config.json 解析字符→token 映射 |
| **音色管理** | `VoiceManager` | 加载 raw float32 [510,256] 音色文件 |
| **G2P 接口** | `G2P` | 函数式接口，可注入任意实现 |
| **默认 G2P** | `ChineseG2P` | 简化实现，~80 常用汉字 |
| **高质量 G2P** | `HoubbPinyinG2P` | 基于 houbb/pinyin，多音字消歧 |
| **文本前端** | `TextFrontend` | 分句（≤510 音素/批）+ 静音裁剪 |
| **拼音→注音** | `ChineseG2P.pinyinToBopomofo()` | 公开静态方法，供 HoubbPinyinG2P 复用 |

### 数据结构

```java
record TtsResult(
    float[] audio,       // PCM 音频采样点（[-1, 1]）
    int sampleRate,      // 采样率（固定 24000）
    double duration      // 音频时长（秒）
);
```

### 模型 I/O 格式

| 类型 | 名称 | 形状 | 说明 |
|------|------|------|------|
| Input | `input_ids` | `[1, seq_len]` int64 | token IDs，首尾 padding 0 |
| Input | `ref_s` | `[1, 256]` float32 | 音色 style 向量（按 token 数索引） |
| Input | `speed` | `[1]` float32 | 语速（0.5 ~ 2.0） |
| Output | `audio` | `[audio_len]` float32 | PCM 音频 |
| Output | `pred_dur` | `[pred_dur_len]` int64 | 音素持续帧数（用于调试） |

---

## 6. 注意事项

- **采样率**：固定 24kHz 单声道，`saveWav()` 输出 PCM 16-bit 格式。
- **最大音素长度**：单批 ≤ 510 音素，超长文本自动按标点分句。
- **G2P 选择**：默认 `ChineseG2P` 仅覆盖 ~80 常用汉字，生产环境强烈推荐注入 `HoubbPinyinG2P` 或外部 G2P。
- **音色文件**：每个 .bin 是 raw float32 二进制（[510, 256] = 522240 字节），不可直接打开查看。
- **GPU 加速**：替换 `onnxruntime` 为 `onnxruntime_gpu` 并设置 `onnxProvider("cuda")` 即可。
- **Surefire 警告**：Windows 沙箱环境下 `mvn test` 末尾可能报 "Error occurred in starting fork"，但实际测试全部通过。

---

## 7. 致谢

- [Kokoro TTS](https://github.com/hexgrad/kokoro) — StyleTTS2 架构的 82M 参数轻量 TTS 模型
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — Python ONNX 推理参考实现
- [houbb/pinyin](https://github.com/houbb/pinyin) — 高性能 Java 中文拼音库
