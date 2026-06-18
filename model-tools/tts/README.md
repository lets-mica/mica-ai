# tts 模型工具

> 对应 mica-ai-tts（Kokoro-82M 语音合成）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 中英双语 TTS，103 个音色 |
| ModelScope | [`KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX`](https://www.modelscope.cn/models/KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX) |
| 原始仓库 | [hexgrad/kokoro](https://github.com/hexgrad/kokoro) |
| 体积 | ~310MB（dynamic ONNX） |

> Kokoro 上游已经提供 ONNX，`convert.py` 主要是**把 voices 目录整理出来**。

## 快速使用

```bash
pip install -r ../requirements.txt
python download.py
python convert.py
```

最终产物：

```
model-tools/tts/model/
├── model_dynamic.onnx
├── config.json
└── voices/
    ├── af.bin
    ├── am.bin
    ├── ...
    └── zf_001.bin
```

## Java 侧配置

```yaml
mica:
  ai:
    tts:
      model-path: <abs>/model/model_dynamic.onnx
      voices-dir: <abs>/model/voices
      config-path: <abs>/model/config.json
```
