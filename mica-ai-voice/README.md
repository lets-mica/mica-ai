# mica-ai-voice

> SenseVoice 纯 ONNXRuntime 语音识别推理的 **Java 17** 实现，移植自 [`SenseVoice-ONNX`](https://github.com/FunASR/SenseVoice) 推理引擎。

零 PyTorch / 零 Python 依赖。完整复现 Mel 特征提取 + LFR、SentencePiece 分词、CTC Greedy 解码、热词雷达（Radar）、结果整合、中文 ITN 等全链路逻辑。

支持中 / 英 / 日 / 韩 / 粤语多语种识别，内置 Trie 树加速热词召回，支持长音频自动分段拼接。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.20+ | 通过 Maven 自动拉取 |

无其他外部依赖。FFmpeg / PyTorch / Python 均不需要。

---

## 2. 模型准备

### 2.1 下载原始模型

从 ModelScope 下载官方 SenseVoiceSmall 模型：

```bash
pip install modelscope
modelscope download --model iic/SenseVoiceSmall
```

模型默认缓存到 `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall`。

### 2.2 导出 ONNX

克隆 [SenseVoice-ONNX](https://github.com) 项目，依次运行导出脚本：

```bash
# 安装依赖
pip install -r requirements.txt

# 导出（按顺序执行）
python 01-Export-Encoder.py    # 导出编码器 ONNX
python 02-Export-CTC.py        # 导出 CTC 解码器 ONNX
python 03-Prepare-Assets.py    # 提取 Tokenizer 与配置文件
python 04-Quantize-Models.py   # （可选）量化为 fp16 / int8 / int4
```

导出完成后，`model/` 目录下包含：

```
model/
├── SenseVoice-Encoder.fp32.onnx      # 编码器（~900MB）
├── SenseVoice-Encoder.fp32.onnx.data # 编码器外部权重
├── SenseVoice-CTC.fp32.onnx          # CTC 解码器
├── SenseVoice-CTC.fp32.onnx.data     # 解码器外部权重
├── Tokenizer.bpe.model               # SentencePiece 分词模型
└── tokens.json                       # 词表
```

### 2.3 模型精度

运行 `04-Quantize-Models.py` 可生成多种精度版本：

| 精度 | 后缀 | 说明 |
|------|------|------|
| **fp32** | `.fp32.onnx` | 全精度基准，精度最高 |
| **fp16** | `.fp16.onnx` | 半精度，适合 GPU / DML 加速 |
| **int8** | `.int8.onnx` | 8-bit 量化，体积减半 |
| **int4** | `.int4.onnx` | 4-bit 量化，体积约 130MB |

Java 端构造 `SenseVoiceConfig` 时指定对应路径即可，引擎自动适配精度。

---

## 3. 快速使用

### 3.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-voice</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 3.2 基本用法

```java
import net.dreamlu.mica.ai.voice.*;
import java.util.List;

public class Demo {
    public static void main(String[] args) {
        // 1. 配置引擎
        SenseVoiceConfig config = new SenseVoiceConfig()
            .encoderPath("model/SenseVoice-Encoder.fp32.onnx")
            .decoderPath("model/SenseVoice-CTC.fp32.onnx")
            .tokenizerPath("model/Tokenizer.bpe.model")
            .onnxProvider("cpu")       // 可选: "cpu", "cuda", "dml"
            .topK(5)                   // 热词搜索深度
            .itn(true)                 // 中文数字规范化
            .hotwords(List.of("Fun-ASR-Nano", "热词", "SenseVoice"));

        // 2. 创建引擎并识别
        try (SenseVoice voice = new SenseVoice(config)) {
            // 方式一：从 WAV 文件识别
            TranscriptionResult result = voice.recognizeFile("audio.wav");

            // 方式二：从 PCM float[] 识别（需自行解码音频）
            // float[] audio = ...;
            // TranscriptionResult result = voice.recognize(audio);

            // 3. 输出结果
            System.out.println("识别文本: " + result.text());
            System.out.println("检测到热词: " + result.hotwords());
            System.out.printf("总耗时: %.3fs%n", result.timings().total());

            // 4. 逐字时间戳
            for (RecognitionResult r : result.results()) {
                String type = r.hotword() ? "🔥 HOTWORD" : "  Greedy";
                System.out.printf("  %5.2fs | %-8s | %s%n", r.start(), r.text(), type);
            }
        }
    }
}
```

### 3.3 热词更新

引擎创建后可动态更新热词列表，无需重建实例：

```java
voice.updateHotwords(List.of("新热词A", "新热词B"));
```

### 3.4 长音频

引擎内置自动分段与拼接逻辑，默认 40 秒一段、5 秒重叠，直接传入长音频即可：

```java
// 支持任意长度的 WAV 文件
TranscriptionResult result = voice.recognizeFile("long-audio.wav");
System.out.println(result.text());
```

---

## 4. 核心组件

| 组件 | 类 | 说明 |
|------|-----|------|
| **引擎** | `SenseVoice` | 主入口，编排全流程 |
| **配置** | `SenseVoiceConfig` | Builder 风格配置 |
| **特征提取** | `MelExtractor` | STFT + Log Mel + LFR 7帧拼接 → 560维 |
| **编码器** | `SenseVoiceEncoder` | ONNX 推理，输出 (1, T+4, 512) |
| **解码器** | `SenseVoiceDecoder` | CTC Top-K 推理 + Greedy collapse |
| **分词器** | `SentencePieceTokenizer` | 手写 Protobuf 解析 `.model` 文件，零外部依赖 |
| **热词雷达** | `HotwordRadar` | Trie 树 + DFS，CTC 帧级热词锚点 |
| **结果整合** | `ResultIntegrator` | 双指针合并 Greedy + 热词 |
| **中文 ITN** | `ChineseItn` | 中文数字 → 阿拉伯数字 |

### 结果数据结构

```java
record TranscriptionResult(
    String text,                    // 最终识别文本
    List<RecognitionResult> results, // 逐字时间戳列表
    List<String> hotwords,          // 检测到的热词
    Timings timings                 // 各阶段耗时
);

record RecognitionResult(
    String text,    // 文本片段
    double start,   // 起始时间（秒）
    boolean hotword // 是否由热词雷达识别
);

record Timings(
    double frontend,  // 特征提取耗时
    double encoder,   // 编码器推理耗时
    double decoder,   // 解码器推理耗时
    double radar,     // 热词扫描耗时
    double integrate, // 结果整合耗时
    double total      // 总耗时
);
```

---

## 5. 注意事项

- **音频格式**：`recognizeFile()` 支持 WAV（PCM 16/32bit）。MP3 等其他格式需先转换为 WAV。
- **采样率**：引擎内部自动重采样到 16kHz 单声道，输入任意采样率均可。
- **SentencePiece**：分词器为纯 Java 实现，通过手写 Protobuf 二进制解析 `.model` 文件，零外部依赖。
- **FFT**：内置 Bluestein 算法 + Cooley-Tukey 基 2 FFT，支持任意大小的 STFT 窗口（默认 nFft=400）。
- **热词雷达**：不依赖 Beam Search，通过 CTC 后验概率 Top-K 帧级扫描实现极速召回，开销 < 1ms。

---

## 6. 致谢

- [SenseVoiceSmall](https://github.com/FunASR/SenseVoice) — 阿里达摩院开源的多语种语音识别模型
