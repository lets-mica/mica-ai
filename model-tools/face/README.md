# face 模型目录

OpenCV Zoo 人脸模型（Apache-2.0，可商用），**已直接入库**，无需下载脚本。

## 模型清单（models/）

| 文件 | 大小 | 说明 |
|------|------|------|
| `face_detection_yunet_2023mar.onnx` | ≈0.23 MB | YuNet 人脸检测 |
| `face_recognition_sface_2021dec.onnx` | ≈37 MB | SFace 512d 人脸特征 |

- 来源：<https://github.com/opencv/opencv_zoo>（`models/face_detection_yunet` / `models/face_recognition_sface`）
- License：Apache License 2.0，**可商用** ✅

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    face:
      det-model-path: model-tools/face/models/face_detection_yunet_2023mar.onnx
      rec-model-path: model-tools/face/models/face_recognition_sface_2021dec.onnx
```

模型规格 / I/O 格式见 [`mica-ai-core/mica-ai-face/README.md`](../../mica-ai-core/mica-ai-face/README.md)。

## 替换 / 升级

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 覆盖 `models/` 下的对应文件
3. `make -C model-tools smoke` 确认 ONNX 结构合法
4. 更新 `mica-ai-core/mica-ai-face/README.md` 的模型规格表
