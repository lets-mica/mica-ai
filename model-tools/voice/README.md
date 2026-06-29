# voice 模型工具

> 对应 mica-ai-voice（SenseVoice 语音识别）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 多语种 ASR（中/英/日/韩/粤语） |
| 原始仓库 | [FunASR/SenseVoice](https://github.com/FunASR/SenseVoice) |
| ModelScope | [`iic/SenseVoiceSmall`](https://www.modelscope.cn/models/iic/SenseVoiceSmall) |
| 上游 ONNX 化 | [SenseVoice-ONNX](https://github.com/HaujetZhao/SenseVoice-ONNX) 项目，按 Encoder / CTC 分别导出 |

> `convert.py` 内部调用 SenseVoice-ONNX 项目的 4 个导出脚本，因此依赖较多。
> 如果 mica-ai-voice 后续内置 ONNX 文件分发，可把 `convert.py` 简化为"目录整理"。

## 快速使用

### 安装依赖

```bash
# 公共依赖
pip install -r ../requirements.txt
# 私有依赖
pip install -r requirements.txt

# funasr 路径需要 funasr（如果只用 upstream 路径则可跳过）：
pip install funasr
```

### 端到端

```bash
python download.py

# 路径 A（推荐）：克隆 [SenseVoice-ONNX](https://github.com/lovemefan/SenseVoice-ONNX) 并执行其脚本
python convert.py --method upstream

# 路径 B（自动）：直接用 funasr 切 encoder + CTC head
python convert.py --method funasr --top-k 4

# 路径 C（占位）：打印操作指南（无 GPU/funasr 时回退）
python convert.py --method manual
```

`convert.py` 详细文档见 [convert.py](convert.py) 顶部 docstring，**强烈建议看一遍**。

最终产物：

```
model-tools/voice/model/out/
├── sensevoice_encoder.onnx
├── sensevoice_ctc.onnx
├── tokens.txt
├── config.json
└── am.mvn
```

## Java 侧配置

```yaml
mica:
  ai:
    voice:
      encoder-path: <abs>/model/out/sensevoice_encoder.onnx
      ctc-path:      <abs>/model/out/sensevoice_ctc.onnx
      tokens-path:   <abs>/model/out/tokens.txt
      config-path:   <abs>/model/out/config.json
      mvn-path:      <abs>/model/out/am.mvn
```
