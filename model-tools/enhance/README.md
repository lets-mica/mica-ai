# model-tools/enhance

`mica-ai-enhance`（图像超分）的模型资产目录。

| 文件 | 体积 | SHA256 | 输入 | 输出 |
|------|------|--------|------|------|
| `models/realesr-general-x4v3.onnx` | 2.33MB | `08E4262E0D77133EA429AFA68DA7D87037B30383941624F9418C7A65C3592DE5` | `input [1,3,H,W]` float32（动态 H/W） | `output [1,3,4H,4W]` float32 |
| `LICENSE-realesrgan.txt` | 4KB | — | — | BSD-3-Clause 全文 + 转换说明 |

## 来源与许可

- 权重上游：[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-general-x4v3`（SRVGGNetCompact，64 features / 32 convs / 4×）
- 论文：Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCV Workshops, 2021
- ONNX 转换：[microapp.io](https://microapp.io) 的 `scripts/build-upscaler-model.mjs` 产出（`https://microapp.io/models/realesr-general-x4v3.onnx`）
  - **权重未改动，仅换容器格式**；其 fp16 数值即上游发布值
- License：**BSD-3-Clause**（`Copyright (c) 2021, Xintao Wang`）—— 允许商用、分发、修改
  - 分发义务：必须保留版权声明、许可证文本与免责声明 → 本目录的 `LICENSE-realesrgan.txt` 即承担该职责，**不要删除**

## 契约要点

- 预处理：BGR→RGB → CHW → `/255`（**无 mean / std**）
- 输出 4× 尺寸，浮点值域可轻微越界（约 `-0.03 ~ 1.05`），落盘前 clip 到 `[0,1]`
- 输入 H/W 是**动态维度**：可整图推理，也可按 tile 切块（`mica-ai-enhance` 默认 256 + 16 重叠）
- 修改像素时必须 `byte & 0xff`，Java 的 `byte` 有符号，直接除 255 会产生负数

## 同族可替换模型

| 模型 | 体积 | 入库 | 说明 |
|------|------|------|------|
| `realesr-general-x4v3` | 2.33MB | ✅ | 默认；通用 4× 超分 |
| `realesr-general-wdn-x4v3` | ~2.3MB | ❌ 外置 | 去噪版，可调 `denoiseStrength`；需自行转换 ONNX |
| `RealESRGAN_x4plus` | ~64MB | ❌ 外置 | RRDBNet 完整版，精度更高、CPU 更慢，**超出 50MB 入库上限**（AGENTS.md §6.2） |

> 三者契约一致（4×、`/255`、RGB），切换时只改 `model-path` + 确认 `scale=4`，**不要改 Java 代码**。
