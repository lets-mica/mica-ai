# Changelog

### [2026.06.01] - 2026-09-15

- 🔪 **精简版本**：抽离音频（TTS / ASR / 声纹）和 OCR / 意图识别能力到独立 mica-* 项目
- 🎭 `mica-ai-face` —— OpenCV Zoo YuNet 检测 + SFace 512d 向量（保留）
- `mica-ai-face-spring-boot-starter` —— 一行 YAML 接入 Spring Boot
- `mica-ai-common` 公共模块：`MicaAiException` 统一异常、`OrtProviders` Provider 管理
- `model-tools/` Python 端仅保留 face 能力脚本（download / convert）+ `scripts/publish.py` / `package.py`

---

### [2026.06.01] - 2026-06-28

- 🎉 **首次发布（v0.1）**，五大 AI 能力一站集成
- 🎤 `mica-ai-tts` —— Kokoro-82M 语音合成，支持中英双语、103 个音色、可插拔 G2P
- 🎧 `mica-ai-voice` —— SenseVoice 语音识别，多语种 + Trie 树热词雷达
- 📷 `mica-ai-ppocr` —— PP-OCRv6 文字识别，tiny / small / medium 三档
- 👤 `mica-ai-speaker` —— ERes2Net 声纹识别，256 维 Embedding
- 🧠 `mica-ai-intent` —— BERT 中文意图识别，HuggingFace 词表兼容
- 5 个 Spring Boot Starter：`*-spring-boot-starter`，一行配置接入
- `mica-ai-common` 公共模块
- `model-tools/` Python 端模型工具链（下载 / 转换 / 训练），默认走 ModelScope 国内镜像
- `docs/websocket实时识别.md` —— 浏览器 / App 实时语音识别完整方案
- `docs/意图识别模型微调与ONNX导出.md` —— BERT 意图分类微调 + ONNX 导出指南
