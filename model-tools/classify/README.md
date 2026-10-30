# model-tools/classify

`mica-ai-classify`（PaddleClas PULC 轻量图像分类）的模型资产目录。

## 1. 入库模型（4 个）

| 文件 | 体积 | SHA256 | 输入 | 输出（已内置激活） | label_list |
|------|------|--------|------|-------------------|-----------|
| `models/text_image_orientation.onnx` | 6.78MB | `4B35B8FF98AAC76A48B980CB0BFF968E435ABB8D8B3A2CB4EBD971AEA78B3A3C` | `x [N,3,224,224]` float32 | `softmax_1.tmp_0 [N,4]` | `0 / 90 / 180 / 270` |
| `models/clarity_assessment.onnx` | 4.14MB | `3A116158CD10DC2248E419B13F4B0D0CD9F5EDC027B149A1E133582F30242CF2` | `x [N,3,224,224]` float32 | `softmax_1.tmp_0 [N,2]` | `clear / blur` |
| `models/person_exists.onnx` | 6.77MB | `E847E8E151B2B6276EC4A9F31C025CBD1F46A2164CD34E78668AD3F8FEA88335` | `x [N,3,224,224]` float32 | `softmax_1.tmp_0 [N,2]` | `nobody / someone` |
| `models/table_attribute.onnx` | 6.79MB | `59A5C5D0525AA85B3E8DA5A4D244C70B4222E47DDE04B65AD021FC3C21504F39` | `x [N,3,224,224]` float32 | `sigmoid_2.tmp_0 [N,6]`（多标签） | `source / number / color / clarity / obstruction / angle` |

对应 label_list 文件位于 `models/<name>_labels.txt`，一行一个标签，**行序即类别索引**。
label_list 与模型是**强耦合配套资源**：条目数与模型类别数不一致时，`mica-ai-classify` 在构造期（形状已知时）或首次推理时直接抛异常。

> ⚠️ label_list **刻意不做去重**。去重会让后续索引整体错位，属于静默错误。

## 2. 预处理契约（实测于 2026-10-30）

PaddleClas 官方 Eval/Infer 配置：

```
DecodeImage:     {to_rgb: True}
ResizeImage:     {resize_short: 256}
CropImage:       {size: 224}
NormalizeImage:  {mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], scale: 1/255}
ToCHWImage
```

即「**RGB** → 短边缩放到 256 → 中心裁剪 224×224 → ImageNet 归一化」。
在 `mica-ai-classify` 中对应默认配置 `resizeShort = 256`、`channelOrder = RGB`。

**例外：`table_attribute`** 的官方配置直接 `ResizeImage: {size: [224,224]}`，
没有短边缩放与中心裁剪，需把 `resizeShort` 置 `0`。

> ⚠️ PULC 输出节点名（`softmax_1.tmp_0` / `sigmoid_2.tmp_0`）由 paddle2onnx 从 Paddle 算子名派生，
> **每次导出可能变化**；模块按「名称提示 → float[N,classes] 结构兜底」定位，不硬编码。

## 3. 来源与许可

- 上游仓库：[PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- 权重托管：`https://paddleclas.bj.bcebos.com/models/PULC/<model>/<model>_infer.tar`
- License：**Apache-2.0**
  - PaddleClas 仓库根 [LICENSE](https://github.com/PaddlePaddle/PaddleClas/blob/develop/LICENSE)：Apache License 2.0
  - 结论：**可商用、可分发、可修改**
- 分发要求：保留 upstream LICENSE 与来源声明（本文件即承担该职责）

## 4. Paddle inference → ONNX 转换

PULC 只发布 Paddle inference 模型（`inference.pdmodel` / `inference.pdiparams`），没有官方预导 ONNX。
转换是一次性离线动作，**不引入任何 Python 运行时依赖**（Java 端只读 ONNX）。

```bash
paddle2onnx \
  --model_dir ./text_image_orientation \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file text_image_orientation.onnx \
  --opset_version 11 \
  --enable_onnx_checker True
```

> `paddle2onnx` 遇到 `hard_swish` 等算子时会把 opset 自动提升到 14，属正常行为。
> 实测可用组合：`paddlepaddle 2.6.2` + `paddle2onnx 1.3.1`（`paddle 3.4.0` 与 `paddle2onnx 2.1.0`
> 存在 ABI 不兼容，导入即 `DLL load failed`，勿用）。

## 5. PULC 模型清单

PaddleClas `deploy/configs/PULC` 当前共 **12** 个模型（早期计划中的「13 个」为估算，以官方目录为准）：

| 模型 | 说明 | 入库 |
|------|------|------|
| `car_exists` | 是否存在车辆 | — |
| `person_exists` | 是否存在人物 | ✅ |
| `person_attribute` | 行人属性 | — |
| `safety_helmet` | 是否佩戴安全帽 | — |
| `traffic_sign` | 交通标志分类 | — |
| `vehicle_attribute` | 车辆属性 | — |
| `text_image_orientation` | 整图方向 0° / 90° / 180° / 270° | ✅ |
| `textline_orientation` | 文本行方向 0° / 180°（与 `mica-ai-textline` 场景重叠，本模块不重复入库） | — |
| `language_classification` | 语种分类 | — |
| `table_attribute` | 表格 6 属性（多标签） | ✅ |
| `clarity_assessment` | 清晰 / 模糊 | ✅ |
| `code_exists` | 是否存在二维码 / 条码 | — |

后续模型按同一套契约接入：转换 ONNX → 配一份 `*_labels.txt` → 在 `ClassifyConfig` 配
`modelPath` / `labelsPath`，必要时调 `resizeShort` / `multiLabel` / `topK`，**不需要改 Java 代码**。
