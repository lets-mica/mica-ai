# mica-ai 预编译模型包

> 给 GitHub Release 打包用的最终 ONNX 产物。附 `manifest.json` 含 SHA256 / 大小 / 来源 / License。

## 目录结构

```
models/
├── manifest.json              # 总清单（机读：含每项 SHA256 / size / source）
├── manifest.csv               # 总清单（人读：Excel / Numbers 直接打开）
├── README.md                  # 本文件
│
└── face/                      # OpenCV Zoo (Apache-2.0)  ——  38 MB
    ├── face_detection_yunet_2023mar.onnx          # 228 KB
    └── face_recognition_sface_2021dec.onnx        # 37 MB
```

## GitHub Release 分卷

`make -C model-tools package` 在 `model-tools/release/` 下生成 1 个 zip：

| 文件 | 大小 | 内容 |
|------|------|------|
| `mica-ai-models-face-2026.06.01.zip` | 38 MB | face/ |

> zip 内顶层目录前缀 `mica-ai/`（解压后 `mica-ai/face/...` 这样的路径）。

## 如何重新生成

```bash
# 1. 下载 + 转换 face
make -C model-tools download
make -C model-tools convert

# 2. 整理到本目录
python model-tools/scripts/publish.py        # 拷贝 + 生成 manifest.json
python model-tools/scripts/publish.py --verify  # 校验 manifest

# 3. 分卷打包 zip
python model-tools/scripts/package.py
```

## Java 端 application.yml 配置示例

解压 zip 后得到 `mica-ai/face/...`，把目录路径配到 `mica.ai.face` 即可：

```yaml
mica:
  ai:
    face:
      det-model-path: mica-ai/face/face_detection_yunet_2023mar.onnx
      rec-model-path: mica-ai/face/face_recognition_sface_2021dec.onnx
```

## License 声明

| 能力 | 上游模型 | License | 商用 |
|------|---------|---------|------|
| `face` | OpenCV Zoo YuNet + SFace | Apache-2.0 | ✅ |

> mica-ai 的「依赖模型必须可商用」原则：详见 `../../AGENTS.md` §6.1。
