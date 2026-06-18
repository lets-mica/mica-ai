# Model Tools

> mica-ai 的 **Python 端模型工具链**：下载、转换、（按需）训练。
> 与 Java 业务代码同级、互不依赖、互不污染。

本目录把每个 AI 能力的"模型生命周期"封装成 3 类脚本：

| 脚本 | 用途 |
|------|------|
| `download.py` | 从 **ModelScope（国内镜像）** 或 HuggingFace 下载原始模型权重 |
| `convert.py`  | 把原始 PyTorch / PaddlePaddle 模型导出为 mica-ai 使用的 **ONNX** |
| `train.py`    | （仅 `intent/`）在预训练模型上做领域微调 |

> 📌 **为什么用 ModelScope**：国内下载速度远优于 HuggingFace，且能覆盖 PaddleOCR、SenseVoice 等国内生态模型。

---

## 📁 目录结构

```
model-tools/
├── README.md                   # 本文件
├── Makefile                    # 一键命令：make download / convert / train-intent
├── requirements.txt            # 公共依赖（modelscope, tqdm, ...）
├── .gitignore                  # 忽略 model/、output/、__pycache__/
│
├── common/                     # 跨能力复用的工具
│   ├── downloader.py           # 基于 modelscope.snapshot_download 的下载器
│   ├── onnx_utils.py           # ONNX 检查 / 简化 / 量化
│   └── progress.py             # 进度条
│
├── ppocr/                      # 对应 mica-ai-ppocr
│   ├── README.md
│   ├── download.py             # 下载 PaddleOCR v6 模型（det + rec）
│   ├── convert.py              # 解压并整理为 mica-ai-ppocr 期望的目录
│   └── requirements.txt
│
├── tts/                        # 对应 mica-ai-tts
│   ├── README.md
│   ├── download.py             # Kokoro-82M ONNX（已在 ONNX 格式）
│   ├── convert.py              # 整理 voices 目录
│   └── requirements.txt
│
├── voice/                      # 对应 mica-ai-voice
│   ├── README.md
│   ├── download.py             # SenseVoiceSmall
│   ├── convert.py              # 导出 ONNX encoder + CTC（基于 SenseVoice-ONNX）
│   └── requirements.txt
│
├── speaker/                    # 对应 mica-ai-speaker
│   ├── README.md
│   ├── download.py             # ERes2Net / ERes2NetV2
│   ├── convert.py              # 导出 ONNX
│   └── requirements.txt
│
└── intent/                     # 对应 mica-ai-intent（完整生命周期）
    ├── README.md
    ├── download.py             # 下载 chinese-bert-wwm-ext
    ├── train.py                # 微调脚本
    ├── convert.py              # 导出 ONNX
    ├── data/                   # 训练数据样例
    └── configs/base.yaml
```

---

## 🚀 快速开始

### 1. 安装 Python 依赖

```bash
# 建议 Python 3.10+
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 安装公共依赖
pip install -r model-tools/requirements.txt

# 安装某个能力的私有依赖（按需）
pip install -r model-tools/intent/requirements.txt
```

### 2. 下载所有模型

```bash
# 方式一：用顶层 Makefile
make -C model-tools download

# 方式二：单独下载某个能力
cd model-tools/intent && python download.py
```

> 默认下载到 `<能力>/model/` 目录（例如 `model-tools/intent/model/chinese-bert-wwm-ext/`）。
> 设置环境变量 `MICA_MODELS_DIR` 可改变根目录。

### 3. 转换 ONNX

```bash
make -C model-tools convert          # 全部能力
cd model-tools/intent && python convert.py   # 单个能力
```

> 部分模型（如 Kokoro）官方已经提供 ONNX，`convert.py` 主要是 **整理目录**。
> 真正需要 PyTorch→ONNX 转换的是 `voice/` 和 `intent/`。

### 4. 训练（仅 intent）

参见 [model-tools/intent/README.md](intent/README.md) 与
[docs/意图识别模型微调与ONNX导出.md](../docs/意图识别模型微调与ONNX导出.md)。

---

## 🔧 设计原则

- **零侵入**：所有脚本放在 `model-tools/` 下，不修改 Java 模块的 `pom.xml`
- **能力维度切分**：每个能力一个子目录，与 Java 的 `mica-ai-core/mica-ai-xxx/` 一一对应
- **国内优先**：默认走 ModelScope，可通过 `--source huggingface` 切换
- **可重入**：`download.py` 重复执行会跳过已下载文件（依赖 modelscope 内置缓存）
- **版本对齐**：与根 `pom.xml` 的 `<revision>` 同号，写在每个 `download.py` 顶部

---

## ❓ 与 Java 端如何对接

转换完成后，每个能力的最终模型目录大致如下（请参考各子目录的 README）：

| 能力 | 产物 | 对应 Java 配置前缀 |
|------|------|-------------------|
| ppocr | `model/det/inference.onnx` + `model/rec/inference.onnx` + `rec_char_dict.txt` | `mica.ai.ppocr` |
| tts | `model/model_dynamic.onnx` + `model/voices/*.bin` + `model/config.json` | `mica.ai.tts` |
| voice | `model/sensevoice_encoder.onnx` + tokenizer 等 | `mica.ai.voice` |
| speaker | `model/eres2net.onnx` | `mica.ai.speaker` |
| intent | `model/bert_intent.onnx` + `vocab.txt` + `labels.json` | `mica.ai.intent` |

把目录路径配到 Spring Boot 的 `application.yml` 即可使用。
