# Model Tools

> mica-ai 的模型资产目录。**所有模型（<50MB）已直接入库**，放在各能力子目录的 `models/` 下，无需下载脚本。

## 📁 目录结构

```
model-tools/
├── README.md                   # 本文件
├── Makefile                    # 仅剩 make smoke（离线冒烟）
├── requirements.txt            # 可选依赖（onnx，用于结构校验）
├── .gitignore
├── face/                       # 对应 mica-ai-face
│   ├── README.md
│   └── models/                 # YuNet 检测 + SFace 识别
├── filetype/                   # 对应 mica-ai-filetype
│   ├── README.md
│   └── models/                 # Magika model.onnx + config + kb
├── plate/                      # 对应 mica-ai-plate
│   ├── README.md
│   └── models/                 # HyperLPR3 检测 / 识别 / 分类
└── scripts/
    └── smoke_test.py           # 离线冒烟：校验目录与 ONNX 完整性
```

## 🧩 模型清单与 License

| 能力 | 模型 | 来源 | License |
|------|------|------|---------|
| face | `face_detection_yunet_2023mar.onnx` + `face_recognition_sface_2021dec.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 ✅ |
| filetype | `model.onnx` + `config.min.json` + `content_types_kb.min.json` | [google/magika standard_v3_3](https://github.com/google/magika) | Apache-2.0 ✅ |
| plate | `y5fu_320x_sim.onnx` / `y5fu_640x_sim.onnx` / `rpv3_mdict_160_r3.onnx` / `litemodel_cls_96x_r1.onnx` | [szad670401/HyperLPR](https://github.com/szad670401/HyperLPR) v20230229 | Apache-2.0 ✅ |

> 活体（`2.7_80x80_MiniFASNetV2.onnx`，MIT）当前**不在仓库内**，按需自行下载。

## 🚀 使用

Java 端直接按路径引用（支持 `classpath:`），以 plate 为例：

```yaml
mica:
  ai:
    plate:
      detection-model-path: model-tools/plate/models/y5fu_320x_sim.onnx
      recognition-model-path: model-tools/plate/models/rpv3_mdict_160_r3.onnx
      classification-model-path: model-tools/plate/models/litemodel_cls_96x_r1.onnx
```

模型规格 / I/O 格式见各能力主 README：

- [`mica-ai-core/mica-ai-face/README.md`](../mica-ai-core/mica-ai-face/README.md)
- [`mica-ai-core/mica-ai-filetype/README.md`](../mica-ai-core/mica-ai-filetype/README.md)
- [`mica-ai-core/mica-ai-plate/README.md`](../mica-ai-core/mica-ai-plate/README.md)

## ✅ 冒烟测试

```bash
make -C model-tools smoke
```

不连外网，校验：目录结构完整、各能力 README 与模型文件齐全、ONNX 结构合法（缺 `onnx` 包时跳过校验项）。

## 🔄 替换 / 升级模型

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 用新模型文件覆盖对应 `models/` 目录
3. `make -C model-tools smoke` 确认 ONNX 结构合法
4. 更新对应能力主 README 的模型规格表