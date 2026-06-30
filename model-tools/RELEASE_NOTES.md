# mica-ai 模型发布说明

> 与 `model-tools/models/` 一同发布；给 GitHub Release 的 description 用。

## 发布物（mica-ai 2026.06.01）

| zip | 大小 | 适合场景 |
|-----|------|---------|
| `mica-ai-models-face-2026.06.01.zip`         | 38 MB   | **必发** — 人脸检测/识别（OpenCV Zoo） |
| `mica-ai-models-tts-2026.06.01.zip`          | 362 MB  | 按需 — Kokoro TTS（中英 103 音色） |
| `mica-ai-models-voice-2026.06.01.zip`        | 1.8 GB  | 按需 — SenseVoice ASR（多语种） |
| `mica-ai-models-speaker-2026.06.01.zip`      | 69 MB   | 按需 — 声纹识别（ERes2NetV2） |
| `mica-ai-models-ppocr-tiny-2026.06.01.zip`   | 6 MB    | 按需 — OCR 最小档 |
| `mica-ai-models-ppocr-small-2026.06.01.zip`  | 30 MB   | **推荐** — OCR 平衡档 |
| `mica-ai-models-ppocr-medium-2026.06.01.zip` | 133 MB  | 按需 — OCR 高精度档 |

> 全部 Apache-2.0，可商用。详见各能力 README。

## 如何使用

1. 下载需要的 zip，解压到本地某个目录（例如 `~/mica-ai-models/`）
2. 在 `application.yml` 里把路径配到 `mica.ai.<cap>`（参考 `models/README.md`）
3. 启动 Spring Boot 应用即可

## 与 mica-ai Java 端的版本对齐

- mica-ai Java: `${revision}=2026.06.01`（见根 `pom.xml`）
- 模型 zip 文件名也带 `2026.06.01` 标签，方便对应

## 已知边界

- `SenseVoice-Encoder.fp32.onnx` 含外部 data 文件（`.data`），ONNX Runtime 会自动加载，**不要拆分 zip 内的文件**。
- ppocr 字典按 spec 区分（`rec_char_dict_{tiny,small,medium}.txt`），**不要混用**。
- tts `model_dynamic.onnx` 是动态 shape 版本，**不要**替换为 `model_static.onnx`（Java 端只支持 dynamic shape）。
