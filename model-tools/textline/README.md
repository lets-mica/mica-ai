# textline 模型目录

[PP-LCNet](https://github.com/PaddlePaddle/PaddleX)（Apache-2.0，可商用）文本行方向分类（0° / 180°）模型。
**随仓库分发**的是 `PP-LCNet_x1_0_textline_ori`（6.46 MB）；轻量版 `PP-LCNet_x0_25_textline_ori`（约 0.96 MB）**不入库**，按需自行下载并指向 `model-path`。

## 模型清单（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `PP-LCNet_x1_0_textline_ori.onnx` | 6.46 MB (6,777,816 bytes) | PP-LCNet x1_0 骨干，1 输入 1 输出，2 类（`0_degree` / `180_degree`） |

- 来源：[PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX) 的文本行方向分类子模型
  （PaddleOCR 系；本仓库的 ONNX 为官方推理包中 `inference.onnx`，由 paddle2onnx 转换产出）
- License：Apache License 2.0，**可商用** ✅
- 完整性指纹：
  `sha256 = 38aa97cd4be591e0ad304e659f07ba30d946f27a63315433f6659c69c8778345`
- 分发状态：**已入库**，随仓库分发（6.46 MB，低于 AGENTS.md §6.2 的 50MB 上限）

## 契约（官方 `inference.yml`，实测 2026-09-21）

```yaml
PreProcess:
  transform_ops:
  - ResizeImage:
      size: [160, 80]            # ⚠️ PaddlePaddle 的 size 是 (宽, 高)
  - NormalizeImage:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
      scale: 1/255
  - ToCHWImage
PostProcess:
  Topk:
    topk: 1
    label_list:
    - 0_degree                   # 索引 0
    - 180_degree                 # 索引 1
```

| 项 | 实测值 |
|----|--------|
| 输入 | `x`：`[N, 3, 80, 160]` float32（动态 batch） |
| 输出 | `fetch_name_0`：`[N, 2]` float32，**原始 logits（未 softmax）** |
| 类别语义 | 索引 0 = `0_degree`，索引 1 = `180_degree` |

> ⚠️ 两个最容易踩的坑：
> 1. **H/W 是 80×160，不是 160×80** —— 官方 `size: [160, 80]` 是 (宽, 高)，对应 NCHW 就是 `[N,3,80,160]`。
> 2. **输出名 `fetch_name_0` 不可依赖**（Paddle2ONNX 自动命名），且**输出是裸 logits**，必须自己 softmax。

## 切换模型

`mica-ai-textline` 的模型契约是**可插拔**的：凡同任务的 PP-LCNet 文本行方向分类 ONNX，
**Java 代码一行不用改**，只改 `model-path`。

| 模型 | 大小 | 入库 | 适用场景 | 获取 |
|------|------|------|---------|------|
| `PP-LCNet_x1_0_textline_ori` | 6.46 MB | ✅ 是 | 默认；精度优先 | 已在 `models/` |
| `PP-LCNet_x0_25_textline_ori` | ~0.96 MB | ❌ 否 | 吞吐 / 包体敏感场景（移动端、嵌入式） | PaddleX 同目录下载后转 ONNX，或取官方推理包内 `inference.onnx` |

配置示例（外置轻量版模型）：

```yaml
mica:
  ai:
    textline:
      model-path: /data/models/PP-LCNet_x0_25_textline_ori.onnx   # 指向本地文件即可
```

> 两个模型 I/O 契约（输入 `x [N,3,80,160]`、输出 `[N,2]`、归一化、类别语义）一致，
> 因此切换时**只需改 `model-path` 一项**。

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    textline:
      model-path: model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx
```

模型规格 / I/O 契约 / 类别语义 / 真实数据置信度实测结论详见
[`mica-ai-core/mica-ai-textline/README.md`](../../mica-ai-core/mica-ai-textline/README.md)。

## 样图

`textline_rot180_demo.jpg` —— PaddleX 官方 180° 样例图（48×136），既作为人工核验样张，
也复制到集成测试资源目录（`real_flipped.jpg`）用于锁住**真实数据置信度区间**（实测 `p≈0.5772`，
远低于合成图的 `0.7311`）。

## 脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `probe_contract.py` | **只读**契约探针：打印 ONNX I/O 签名 + `sha256`，校验输入 `[N,3,80,160]` / 输出 `[N,2]`、类别语义（索引 0 = 0 度）、通道顺序敏感性（BGR vs RGB）、以及退化输入（缩小 / 模糊 / JPEG q30）下的鲁棒性 |

```bash
python model-tools/textline/scripts/probe_contract.py \
    model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx
# 可用 TEXTLINE_SAMPLE=/path/to/line.jpg 指定真实样图，默认自动找 textline_rot180_demo.jpg
```

运行前提：`numpy` + `onnxruntime` + `opencv-python`。

> ⚠️ 该脚本是**只读核对工具**，不参与构建，也不是模型的「自检入口」。
> 模型的唯一自检入口是集成测试：`./mvnc.sh -pl mica-ai-core/mica-ai-textline test`
> （加载真 ONNX 跑完整链路，并断言 I/O 契约：1 输入 / 160×80 / 2 类输出）。
>
> 想要用**外置模型**跑同一套集成测试（含全部尺寸 / 语义 / 阈值断言）：
>
> ```bash
> ./mvnc.sh -pl mica-ai-core/mica-ai-textline test \
>     -Dmica.ai.textline.externalModel=/path/to/PP-LCNet_x0_25_textline_ori.onnx
> ```
>
> 不传该参数时，这项测试自动跳过（外置模型不入库，不能假设它存在）。
