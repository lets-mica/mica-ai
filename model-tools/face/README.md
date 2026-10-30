# face 模型目录

OpenCV Zoo 人脸模型（Apache-2.0，可商用），**已直接入库**，无需下载脚本。

## 模型清单（`models/`）

| 文件 | 说明 |
|------|------|
| `face_detection_yunet_2023mar.onnx` | YuNet 人脸检测（320×320 RGB → 框 + 5 关键点） |
| `face_recognition_sface_2021dec.onnx` | SFace 人脸特征（112×112 RGB → **128d** L2 归一化向量） |
| `2.7_80x80_MiniFASNetV2.onnx` | MiniFASNetV2 静态活体检测（80×80 RGB → 真人/攻击二分类） |
| `ediffiqa_tiny_jun2024.onnx` | EDFFIQA 人脸图像质量评估（112×112 RGB → 1 个质量分） |
| `facial_expression_recognition_mobilefacenet_2022july.onnx` | 人脸表情识别（112×112 RGB → RAF-DB 7 类概率，argmax） |

- 来源：<https://github.com/opencv/opencv_zoo>（`models/face_detection_yunet` / `models/face_recognition_sface` /
  `models/face_image_quality_assessment_ediffiqa` / `models/facial_expression_recognition`）
- 活体来源：<https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>（`2.7_80x80_MiniFASNetV2`）
- License：Apache License 2.0，**可商用** ✅（活体模型为 MIT，**可商用** ✅）
- ⚠️ **EDFFIQA 为 CC-BY-4.0**（非 Apache）：允许商用，但**分发时必须保留署名与许可声明**（引用
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_image_quality_assessment_ediffiqa>）；
  该例外已由项目 owner 显式批准，见 `AGENTS.md` §6.1
- 表情模型来源：<https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition>（Apache-2.0）
- 两个新模型的预处理契约一致：BGR→RGB、`112×112`、`(x/255 - 0.5) / 0.5`（mean = std = 0.5）

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    face:
      detection:
        model-path: model-tools/face/models/face_detection_yunet_2023mar.onnx
      recognition:
        model-path: model-tools/face/models/face_recognition_sface_2021dec.onnx
```

模型规格 / I/O 格式见 [`mica-ai-core/mica-ai-face/README.md`](../../mica-ai-core/mica-ai-face/README.md)。

## 替换 / 升级

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 覆盖 `models/` 下的对应文件
3. 跑 face 模块集成测试确认可加载且推理正常：`mvn -pl mica-ai-core/mica-ai-face -am test`
4. 更新 `mica-ai-core/mica-ai-face/README.md` 的模型规格表