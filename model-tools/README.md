# Model Tools

> mica-ai 的模型资产目录。**绝大多数模型（<50MB）已直接入库**，放在各能力子目录的 `models/` 下，无需下载脚本。
>
> ⚠️ **唯一例外**：`layout/models/model.onnx` 为 **125MB**，超出单文件 <50MB 约定，**不随仓库分发**（已在根 `.gitignore` 显式排除）。获取方式见 [`layout/README.md`](layout/README.md) 的「分发状态」小节。

## 📁 目录结构

```
model-tools/
├── README.md                   # 本文件
├── face/                       # 对应 mica-ai-face
│   ├── README.md
│   └── models/                 # YuNet 检测 + SFace 识别 + MiniFASNetV2 活体
├── filetype/                   # 对应 mica-ai-filetype
│   ├── README.md
│   └── models/                 # Magika model.onnx + config + kb
├── plate/                      # 对应 mica-ai-plate
│   ├── README.md
│   └── models/                 # HyperLPR3 检测 / 识别 / 分类
├── layout/                     # 对应 mica-ai-layout
│   ├── README.md
│   ├── models/                 # PP-DocLayoutV3（125MB，⚠️ 不随仓库分发，见下）
│   └── scripts/                # ONNX 契约探针（probe_coord_space.py 等）
├── matting/                    # 对应 mica-ai-matting
│   ├── README.md
│   ├── models/                 # u2netp 抠图（4.36MB，随仓库分发）
│   └── scripts/                # 契约探针（probe_preprocess.py / probe_model_identity.py）
└── textline/                   # 对应 mica-ai-textline
    ├── README.md
    ├── textline_rot180_demo.jpg # PaddleX 官方 180° 样例图
    ├── models/                 # PP-LCNet_x1_0_textline_ori（6.46MB，随仓库分发）
    └── scripts/                # 契约探针（probe_contract.py）
```

## 🧩 模型清单与 License

| 能力 | 模型 | 来源 | License |
|------|------|------|---------|
| face | `face_detection_yunet_2023mar.onnx` + `face_recognition_sface_2021dec.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 ✅ |
| face 活体 | `2.7_80x80_MiniFASNetV2.onnx` | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | MIT ✅ |
| filetype | `model.onnx` + `config.min.json` + `content_types_kb.min.json` | [google/magika standard_v3_3](https://github.com/google/magika) | Apache-2.0 ✅ |
| plate | `y5fu_320x_sim.onnx` / `y5fu_640x_sim.onnx` / `rpv3_mdict_160_r3.onnx` / `litemodel_cls_96x_r1.onnx` | [szad670401/HyperLPR](https://github.com/szad670401/HyperLPR) v20230229 | Apache-2.0 ✅ |
| layout | `model.onnx`（125MB，⚠️ 不随仓库分发） | [PaddleOCR PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 ✅ |
| matting | `u2netp.onnx`（4.36MB，随仓库分发） | [danielgatis/rembg](https://github.com/danielgatis/rembg) 打包 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | Apache-2.0 ✅ |
| textline | `PP-LCNet_x1_0_textline_ori.onnx`（6.46MB，随仓库分发） | [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX) 文本行方向分类（paddle2onnx 转换产物） | Apache-2.0 ✅ |

> 活体模型默认**关闭**（`mica.ai.face.liveness.enabled=true` 显式启用），但 `2.7_80x80_MiniFASNetV2.onnx`（MIT）已随仓库分发，启用时无须额外下载。
>
> layout 模型未随仓库分发，因此在示例配置里 `mica.ai.layout.enabled=false`；本地放好 `model.onnx` 后置 `true` 即可。

**网盘下载：**
我用夸克网盘给你分享了「mica-ai」，点击链接或复制整段内容，打开「夸克APP」即可获取。
链接：https://pan.quark.cn/s/56cdf019c2c1
提取码：yjAh

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
- [`mica-ai-core/mica-ai-layout/README.md`](../mica-ai-core/mica-ai-layout/README.md)
- [`mica-ai-core/mica-ai-matting/README.md`](../mica-ai-core/mica-ai-matting/README.md)
- [`mica-ai-core/mica-ai-textline/README.md`](../mica-ai-core/mica-ai-textline/README.md)

模型完整性由各能力模块的集成测试覆盖（加载真 ONNX 跑一遍完整推理，全程离线、不连外网）：

```bash
mvn -pl mica-ai-core/mica-ai-layout -am test    # 以 layout 为例，各能力同理
```

> ⚠️ 历史上的 `model-tools/scripts/smoke_test.py`（`make -C model-tools smoke`）已在提交
> `a0a8a6f refactor: 删除 model-tools Python 工具链（模型已直接入库）` 中移除，全仓引用已清理完毕。
> **不要再引入该命令或等价的新冒烟脚本**——模型的唯一自检入口是各能力的集成测试。

## 🔄 替换 / 升级模型

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 用新模型文件覆盖对应 `models/` 目录
3. 跑对应能力模块的集成测试确认可加载且推理结果正常（`mvn -pl mica-ai-core/<cap> -am test`）
4. 更新对应能力主 README 的模型规格表