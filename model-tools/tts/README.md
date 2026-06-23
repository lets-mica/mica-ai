# tts 模型工具

> 对应 mica-ai-tts：**Kokoro-82M**（纯 ONNX Runtime 语音合成，中英双语）。

## 模型规格

| 项目 | Kokoro-82M |
|------|------------|
| 来源 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| 参数量 | 82M |
| 支持语言 | 中英双语 |
| 音频采样率 | 24 kHz |
| 架构 | 单 ONNX（StyleTTS 2 + ISTFTNet） |
| G2P | ✅ 需要（ChineseG2P、HoubbPinyinG2P） |
| 国内下载 | ModelScope |
| License | Apache 2.0 |

## 默认中文音色

Kokoro 仓库提供大量音色，中文效果实测：

| 音色 | 类型 | 说明 |
|------|------|------|
| `zf_001` | 女声 | **中文效果最佳（推荐默认）** |
| `zm_001` | 男声 | **中文效果最佳（推荐默认）** |
| `af_*` | 女声（英文） | 英文更自然 |
| `am_*` | 男声（英文） | 英文更自然 |

> 中文请使用 `zf_001` / `zm_001`，不要用英文音色合成中文，会很生硬。

## 快速使用

### 安装依赖

```bash
pip install -r ../requirements.txt
# 可选：convert.py --quantize 时需要 onnxruntime
pip install onnxruntime
```

### 端到端

```bash
python download.py                # 从 ModelScope 下载
python convert.py                 # 整理目录结构
python convert.py --quantize      # 可选：动态量化 ONNX（INT8，约 60% 体积）
```

## 下载源

```bash
# 默认：ModelScope（国内 CDN，免登录，速度 5-10 MB/s）
python download.py --source modelscope

# 备选：HuggingFace
python download.py --source huggingface
```

## 最终产物

```
model-tools/tts/model/out/
├── model_dynamic.onnx        # 推理主模型（FP32，约 330 MB）
├── config.json               # vocab/tokenizer 配置
└── voices/
    ├── zf_001.bin             # 女声（中文推荐）
    ├── zm_001.bin             # 男声（中文推荐）
    ├── af.bin                 # 英文女声
    ├── am.bin                 # 英文男声
    └── ... (其它音色)
```

## Java 侧配置

```yaml
mica:
  ai:
    tts:
      model-path: <abs>/model/out/model_dynamic.onnx
      voices-dir: <abs>/model/out/voices
      config-path: <abs>/model/out/config.json
      default-voice: zf_001       # 中文推荐 zf_001 / zm_001
      default-speed: 1.0          # 0.5 ~ 2.0
      onnx-provider: cpu          # cpu | coreml | cuda | ...
```

## G2P 选择

mica-ai-tts 通过 `KokoroTtsConfig.Builder#g2p(G2P)` 注入 G2P 实现：

| 实现 | 字典规模 | 多音字消歧 | 额外依赖 |
|------|----------|------------|----------|
| `ChineseG2P`（默认） | ~2000 字 + 常用词组 | 有限（基于词组表） | ❌ 零依赖 |
| `HoubbPinyinG2P` | 7 万+ 字 | ✅ 基于分词智能消歧 | `com.github.houbb:pinyin:0.4.0` |

```java
// 默认（零依赖）
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath(...)
    .voicesDir(...)
    .build();

// 高质量（依赖 houbb/pinyin，自动反射加载）
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath(...)
    .voicesDir(...)
    .g2p(new HoubbPinyinG2P())
    .build();
```

`ChineseG2P` 的字典文件：
- `src/main/resources/tts/chinese-char-pinyin.txt` — 单字（~2000）
- `src/main/resources/tts/chinese-word-pinyin.txt` — 词组（~500，处理多音字消歧）

如需扩展，可直接编辑这两个文件，每行 `字=pinyin` 或 `词语=pinyin1 pinyin2`。
