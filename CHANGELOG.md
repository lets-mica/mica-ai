# Changelog

### [2026.06.01] - 2026-06-28

- 🎉 **首次发布（v0.1）**，五大 AI 能力一站集成
- 🎤 `mica-ai-tts` —— Kokoro-82M 语音合成，支持中英双语、103 个音色、可插拔 G2P
- 🎧 `mica-ai-voice` —— SenseVoice 语音识别，多语种 + Trie 树热词雷达
- 📷 `mica-ai-ppocr` —— PP-OCRv6 文字识别，tiny / small / medium 三档
- 👤 `mica-ai-speaker` —— ERes2Net 声纹识别，256 维 Embedding
- 🧠 `mica-ai-intent` —— BERT 中文意图识别，HuggingFace 词表兼容
- 5 个 Spring Boot Starter：`*-spring-boot-starter`，一行配置接入
- `mica-ai-common` 公共模块：`MicaAiException` 统一异常、`OrtProviders` Provider 管理、`AudioUtils` 音频工具
- `model-tools/` Python 端模型工具链（下载 / 转换 / 训练），默认走 ModelScope 国内镜像
- `docs/websocket实时识别.md` —— 浏览器 / App 实时语音识别完整方案
- `docs/意图识别模型微调与ONNX导出.md` —— BERT 意图分类微调 + ONNX 导出指南
