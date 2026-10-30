# model-tools/humanseg

`mica-ai-humanseg`（人像分割）的模型资产目录。

| 文件 | 体积 | SHA256 | 输入 | 输出 |
|------|------|--------|------|------|
| `models/human_segmentation_pphumanseg_2023mar.onnx` | 5.88MB | `552D8A984054E59B5D773D24B9B12022B22046CEB2BBC4C9AAEACEB36A9DDF24` | `x [1,3,192,192]` float32 | `save_infer_model/scale_0.tmp_1 [1,2,192,192]` float32（内置 Softmax，通道 1 = 人像） |

## 来源与许可

- 上游：[opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) 的 `models/human_segmentation_pphumanseg`（PP-HumanSeg V2-Lite 的 ONNX 导出）
- 模型权重来源：PaddleSeg / PaddlePaddle（PP-HumanSeg V2-Lite）
- License：**Apache-2.0**（可商用、可分发的宽松协议）
  - opencv_zoo 仓库根 `LICENSE`：Apache License 2.0
  - PaddleSeg 权重：Apache License 2.0
- 分发要求：保留 upstream README 与 LICENSE 中的署名 / 来源声明（本文件即承担该职责）

## 契约要点

- 预处理：直接 resize 到 192×192 → BGR→RGB → `(x/255 - 0.5) / 0.5` → CHW
- 后处理：输出为 2 通道概率（逐像素和 = 1），取通道 1 作为人像 alpha，`INTER_LINEAR` 缩放回原图
- 二值阈值默认 0.5

## 同族可替换模型

| 模型 | 体积 | 输入 | 获取方式 |
|------|------|------|---------|
| Portrait-PP-HumanSegV2-Lite | 2.3MB | 256×144 | PaddleSeg 官方 inference model，用 `paddle2onnx` 离线转换后配 `model-path` + `input-width/height` |

> 外置模型不入库；契约（单输出、2 通道 Softmax、通道 1 = 前景）必须一致，否则需在 `HumanSegConfig` 里把差异显式化为配置项。
