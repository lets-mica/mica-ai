# layout 模型目录

[PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0，可商用）文档版面分析模型。PaddleX `PP-DocLayoutV3_infer` 经 paddle2onnx 2.1.0 转换 + constant folding 得到 125MB `model.onnx`，**本地已就绪但暂未提交仓库**（见下方「入库状态」）。

## 模型清单（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.onnx` | 125 MB | PP-DocLayoutV3 版面检测（25 类 RT-DETR + 指针网络阅读顺序），3 输入 3 输出 |

- 来源：<https://github.com/PaddlePaddle/PaddleOCR>（官方 `pp_layout` / `PaddleDetection`）
- License：Apache License 2.0，**可商用** ✅
- 转换误差：V3 相对 Paddle 原版 **1.57%**（V2 为 14.8%，不可生产）

### 入库状态（⚠️ 待决策）

`model.onnx` **125 MB**，超出 AGENTS.md §6.2 的「<50MB 入库」约定，**当前未提交**，
已在根 `.gitignore` 显式排除（`model-tools/layout/models/model.onnx`）以防误提交。

三个可选方案，待 owner 决定：

| 方案 | 说明 |
|------|------|
| Git LFS | `.gitattributes` 加 `*.onnx filter=lfs`，克隆者需装 LFS |
| 下载脚本 | 恢复 `scripts/download_model.py`（ModelScope/BOS），README 补 sha256 |
| 维持入库 | 打破 <50MB 约定，从 `.gitignore` 移除排除行 |

> 决策后：更新本文件 + `model-tools/README.md` + `.gitignore`，并跑一次
> `mica-ai-core/mica-ai-layout` 的 `LayoutIntegrationTest` 确认真模型可加载。
> 注意 `LayoutIntegrationTest` 在模型缺失时会**跳过**而非失败（用 `Assumptions`）。

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    layout:
      model-path: model-tools/layout/models/model.onnx
```

模型规格 / I/O 格式 / 坐标空间约定 / 后处理详见 [`mica-ai-core/mica-ai-layout/README.md`](../../mica-ai-core/mica-ai-layout/README.md)，
落地过程与实测记录见 [`docs/layout-tracking.md`](../../docs/layout-tracking.md)。

## 脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `probe_coord_space.py` | **核心**：固定 `image` 张量、只改 `im_shape`/`scale_factor`，定位输出坐标空间（结论：`输出坐标 = 原图坐标 / scale_factor`）；自带判定结论输出 |
| `probe_real_onnx.py` | 打印 ONNX 输入 / 输出元信息（名字、shape、dtype），换模型或怀疑导出约定变化时先跑 |
| `probe_real_image.py` / `probe_real_image2.py` / `probe_order.py` | ⚠️ 早期探查脚本，**已被 `probe_coord_space.py` 取代**，仅留作过程记录。`probe_real_image.py` 里 `cxcywh` 解码假设是**错的**（col2-5 实为 `x1y1x2y2`），勿照抄 |

运行前提：WSL / Linux 下已装 `onnxruntime`、`numpy`、`Pillow`；模型路径默认取同级 `../models/model.onnx`
（可用环境变量 `LAYOUT_ONNX` 覆盖，仅 `probe_coord_space.py` 支持）。demo 图缓存到 `~/.cache/mica-ai-layout/`
（WSL 的 `/tmp` 是 tmpfs，不要放中间产物）。
