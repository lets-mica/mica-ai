# plate 模型目录

HyperLPR3 v20230229（Apache-2.0，可商用）中国车牌识别模型，**已直接入库**，无需下载脚本。

## 模型清单（`models/`）

| 文件 | 说明 |
|------|------|
| `y5fu_320x_sim.onnx` | 车牌检测（YOLOv5 多任务，320 输入） |
| `y5fu_640x_sim.onnx` | 车牌检测（YOLOv5 多任务，640 输入，大图可选） |
| `rpv3_mdict_160_r3.onnx` | CRNN + SVTR 车牌字符识别（77 token CTC） |
| `litemodel_cls_96x_r1.onnx` | 车牌颜色分类（3 类：黄 / 蓝 / 绿） |

- 来源：<https://github.com/szad670401/HyperLPR>（官方在线包 `20230229.zip` 解压）
- License：Apache License 2.0，**可商用** ✅

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    plate:
      detection-model-path: model-tools/plate/models/y5fu_320x_sim.onnx
      recognition-model-path: model-tools/plate/models/rpv3_mdict_160_r3.onnx
      classification-model-path: model-tools/plate/models/litemodel_cls_96x_r1.onnx
```

模型规格 / I/O 格式 / 判型规则见 [`mica-ai-core/mica-ai-plate/README.md`](../../mica-ai-core/mica-ai-plate/README.md)。