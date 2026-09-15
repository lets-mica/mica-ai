# mica-ai 模型发布说明

> 与 `model-tools/models/` 一同发布；给 GitHub Release 的 description 用。

## 发布物（mica-ai 2026.06.01）

| zip | 大小 | 适合场景 |
|-----|------|---------|
| `mica-ai-models-face-2026.06.01.zip` | 38 MB | **必发** — 人脸检测/识别（OpenCV Zoo） |

> Apache-2.0，可商用。详见 [mica-ai-face/README.md](../../mica-ai-core/mica-ai-face/README.md)。

## 如何使用

1. 下载 zip，解压到本地某个目录（例如 `~/mica-ai-models/`）
2. 在 `application.yml` 里把路径配到 `mica.ai.face`（参考 `models/README.md`）
3. 启动 Spring Boot 应用即可

## 与 mica-ai Java 端的版本对齐

- mica-ai Java: `${revision}=2026.06.01`（见根 `pom.xml`）
- 模型 zip 文件名也带 `2026.06.01` 标签，方便对应
