# Model Tools

> mica-ai 的模型资产目录。**绝大多数模型（<50MB）已直接入库**，放在各能力子目录的 `models/` 下，无需下载脚本。
>
> ⚠️ **唯一例外**：`layout/models/model.onnx` 为 **125MB**，超出单文件 <50MB 约定，**不随仓库分发**（已在根 `.gitignore` 显式排除）。获取方式见 [`layout/README.md`](layout/README.md) 的「分发状态」小节。
>
> 📌 **模型进 Git 仓库 ≠ 模型进 jar**：模型**不打包进 jar**，各能力的 `model-path` **必填**，须显式指向本目录下的文件（如 `model-tools/matting/models/u2netp.onnx`）；未配置时启动即 fail-fast 并给出提示。用户自己把模型放进 classpath 时，路径写 `classpath:` 前缀即可。

## 📁 目录结构

```
model-tools/
├── README.md                     # 本文件
├── face/                         # 对应 mica-ai-face
│   ├── README.md
│   └── models/                   # YuNet + SFace + MiniFASNetV2 + eDifFIQA + 表情识别
├── filetype/                     # 对应 mica-ai-filetype
│   ├── README.md
│   └── models/                   # Magika model.onnx + config.min.json + content_types_kb.min.json
├── plate/                        # 对应 mica-ai-plate
│   ├── README.md
│   └── models/                   # HyperLPR3 检测 / 识别 / 分类
├── layout/                       # 对应 mica-ai-layout
│   ├── README.md
│   ├── models/                   # PP-DocLayoutV3（125MB，⚠️ 不随仓库分发，见下）
│   └── scripts/                  # ONNX 契约探针（probe_coord_space.py 等）
├── matting/                      # 对应 mica-ai-matting
│   ├── README.md
│   ├── models/                   # u2netp 抠图（4.36MB，随仓库分发）
│   └── scripts/                  # 契约探针（probe_preprocess.py / probe_model_identity.py）
├── humanseg/                     # 对应 mica-ai-humanseg
│   ├── README.md
│   └── models/                   # PP-HumanSeg V2-Lite（5.88MB，随仓库分发）
├── detect/                       # 对应 mica-ai-detect
│   ├── README.md
│   └── models/                   # NanoDet-Plus 3.62MB + YOLOX-s 34.2MB
├── docalign/                     # 对应 mica-ai-docalign
│   ├── README.md
│   └── models/                   # DocAligner lcnet100（4.55MB，随仓库分发）
├── enhance/                      # 对应 mica-ai-enhance
│   ├── README.md
│   ├── LICENSE-realesrgan.txt    # BSD-3-Clause 许可证原文
│   └── models/                   # Real-ESRGAN x4v3（2.33MB，随仓库分发）
├── textline/                     # 对应 mica-ai-textline
│   ├── README.md
│   ├── textline_rot180_demo.jpg  # PaddleX 官方 180° 样例图
│   ├── models/                   # PP-LCNet_x1_0_textline_ori（6.46MB，随仓库分发）
│   └── scripts/                  # 契约探针（probe_contract.py）
├── classify/                     # 对应 mica-ai-classify
│   ├── README.md
│   └── models/                   # PULC 4 个模型（4.14–6.79MB）+ 各自 label_list
└── captcha/                      # 对应 mica-ai-captcha
    ├── README.md
    └── models/                   # ddddocr CRNN（12.98MB，随仓库分发；不打包进 jar）
```

> `mica-ai-track` 是**纯算法零模型**能力，没有对应的模型目录。

## 🧩 模型清单与 License

| 能力 | 模型 | 来源 | License |
|------|------|------|---------|
| face 检测 | `face_detection_yunet_2023mar.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 ✅ |
| face 特征 | `face_recognition_sface_2021dec.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 ✅ |
| face 活体 | `2.7_80x80_MiniFASNetV2.onnx` | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | MIT ✅ |
| face 质量评估 | `ediffiqa_tiny_jun2024.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) `face_image_quality_assessment_ediffiqa` | **CC-BY-4.0** ✅（可商用，**分发需保留署名与许可声明**，见 [NOTICE](../NOTICE)） |
| face 表情识别 | `facial_expression_recognition_mobilefacenet_2022july.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) `facial_expression_recognition` | Apache-2.0 ✅ |
| filetype | `model.onnx` + `config.min.json` + `content_types_kb.min.json` | [google/magika standard_v3_3](https://github.com/google/magika) | Apache-2.0 ✅ |
| plate | `y5fu_320x_sim.onnx` / `y5fu_640x_sim.onnx` / `rpv3_mdict_160_r3.onnx` / `litemodel_cls_96x_r1.onnx` | [szad670401/HyperLPR](https://github.com/szad670401/HyperLPR) v20230229 | Apache-2.0 ✅ |
| layout | `model.onnx`（125MB，⚠️ 不随仓库分发） | [PaddleOCR PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 ✅ |
| matting | `u2netp.onnx`（4.36MB，随仓库分发） | [danielgatis/rembg](https://github.com/danielgatis/rembg) 打包 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | Apache-2.0 ✅ |
| humanseg | `human_segmentation_pphumanseg_2023mar.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) `human_segmentation_pphumanseg_2023mar` | Apache-2.0 ✅ |
| detect | `object_detection_nanodet_2022nov.onnx` + `object_detection_yolox_2022nov.onnx` | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) `object_detection_*` | Apache-2.0 ✅ |
| docalign | `lcnet100_h_e_bifpn_256_fp32.onnx` | [DocsaidLab/DocAligner](https://github.com/DocsaidLab/DocAligner) | Apache-2.0 ✅ |
| enhance | `realesr-general-x4v3.onnx` | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（microapp.io 转换的官方权重） | BSD-3-Clause ✅ |
| textline | `PP-LCNet_x1_0_textline_ori.onnx`（6.46MB，随仓库分发） | [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX) 文本行方向分类（paddle2onnx 转换产物） | Apache-2.0 ✅ |
| classify | `text_image_orientation.onnx` / `clarity_assessment.onnx` / `person_exists.onnx` / `table_attribute.onnx` + 各自 label_list | [PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas) PULC（paddle2onnx 转换产物） | Apache-2.0 ✅ |
| captcha | `ddddocr.onnx` + `charset.txt`（12.98MB，随仓库分发；**不打包进 jar**） | [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) `common.onnx`（动态量化衍生版）+ 字符集 | MIT ✅ |

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
- [`mica-ai-core/mica-ai-humanseg/README.md`](../mica-ai-core/mica-ai-humanseg/README.md)
- [`mica-ai-core/mica-ai-detect/README.md`](../mica-ai-core/mica-ai-detect/README.md)
- [`mica-ai-core/mica-ai-docalign/README.md`](../mica-ai-core/mica-ai-docalign/README.md)
- [`mica-ai-core/mica-ai-enhance/README.md`](../mica-ai-core/mica-ai-enhance/README.md)
- [`mica-ai-core/mica-ai-textline/README.md`](../mica-ai-core/mica-ai-textline/README.md)
- [`mica-ai-core/mica-ai-classify/README.md`](../mica-ai-core/mica-ai-classify/README.md)
- [`mica-ai-core/mica-ai-captcha/README.md`](../mica-ai-core/mica-ai-captcha/README.md)

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
