# face 模型目录

OpenCV Zoo 人脸模型（Apache-2.0，可商用），**已直接入库**，无需下载脚本。

## 模型清单（`models/`）

| 文件 | 说明 |
|------|------|
| `face_detection_yunet_2023mar.onnx` | YuNet 人脸检测（320×320 RGB → 框 + 5 关键点） |
| `face_recognition_sface_2021dec.onnx` | SFace 人脸特征（112×112 RGB → **128d** L2 归一化向量） |
| `2.7_80x80_MiniFASNetV2.onnx` | MiniFASNetV2 静态活体检测（80×80 RGB → 真人/攻击二分类） |

- 来源：<https://github.com/opencv/opencv_zoo>（`models/face_detection_yunet` / `models/face_recognition_sface`）
- 活体来源：<https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>（`2.7_80x80_MiniFASNetV2`）
- License：Apache License 2.0，**可商用** ✅（活体模型为 MIT，**可商用** ✅）

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
3. `make -C model-tools smoke` 确认 ONNX 结构合法
4. 更新 `mica-ai-core/mica-ai-face/README.md` 的模型规格表