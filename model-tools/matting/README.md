# matting 模型目录

[U²-Net](https://github.com/xuebinqin/U-2-Net)（Apache-2.0，可商用）通用抠图（显著性目标检测）模型。**随仓库分发**的是轻量版 `u2netp`（4.36 MB）；同族的完整版与人体分割版体积约 168 MB，**不入库**，按需外置接入（详见下文「切换模型」）。

## 模型清单（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `u2netp.onnx` | 4.36 MB (4,574,861 bytes) | U²-Net 轻量版，1 输入 7 输出（d0..d6） |

- 来源：[danielgatis/rembg release v0.0.0](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx)
  （原始权重出自 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)，rembg 重打包为 ONNX）
- License：Apache License 2.0，**可商用** ✅
- 完整性指纹：
  `sha256 = 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`
- 分发状态：**已入库**，随仓库分发（4.36 MB，远低于 AGENTS.md §6.2 的 50MB 上限）

## 切换模型

`mica-ai-matting` 的模型契约是**可插拔**的：凡 rembg 导出的 U²-Net 族 ONNX，
**Java 代码一行不用改**，只改 `model-path`（必要时调 `output-select`）。

实测（2026-09-21，onnxruntime 1.30）三者契约完全一致：
输入 `input.1 [1,3,320,320]` float32、**7 个** `[1,1,320,320]` 输出、d0 为首个输出、
输出名为数字（`1959..1965`）、归一化 `(0.485,0.456,0.406)/(0.229,0.224,0.225)`。

| 模型 | 大小 | 入库 | 适用场景 | 获取 |
|------|------|------|---------|------|
| `u2netp` | 4.36 MB | ✅ 是 | 默认；CPU 单图毫秒级，通用显著性 | 已在 `models/` |
| `u2net` | 168 MB | ❌ 否 | 通用显著性完整版，细碎结构（发丝 / 蕾丝）边缘明显更好 | [rembg release](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx) |
| `u2net_human_seg` | 168 MB | ❌ 否 | **人体分割专用**：对「画面里没有人」的输入明显更保守 | [rembg release](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx) |

配置示例（外置 168MB 模型）：

```yaml
mica:
  ai:
    matting:
      model-path: /data/models/u2net_human_seg.onnx   # 指向本地文件即可
```

> **为什么 168MB 不入库**：AGENTS.md §6.2 规定入库模型单文件须 <50MB。
> 超限模型沿用与 layout（125MB）相同的处理：不入库、集成测试在模型缺失时跳过、
> 由使用者自行下载并配 `model-path`。

### `u2netp` vs `u2net_human_seg`：实测对比

> ⚠️ 这里有一条**容易被误导**的线索，已实测澄清，勿再凭文件哈希下结论。

本机 `u2netp.onnx` 的 `sha256` 与 HuggingFace
[`BritishWerewolf/U-2-Net-Human-Seg`](https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg)
的 `onnx/model.onnx` **完全相同**，容易推断「rembg 的 u2netp 就是 human_seg 权重」。
**但行为实测推翻了这一推断**（同一张人像图 / 一张无人静物图，min-max 归一化后逐像素对比）：

| 对照 | 人像图 MAE | 静物图 MAE |
|------|-----------|-----------|
| `u2netp` vs **`u2net`** | 0.0085 | **0.0009** ← 几乎逐像素重合 |
| `u2netp` vs `u2net_human_seg` | 0.0112 | **0.0400** ← 差 44 倍 |

对「无人图」的保守度（前景占比 person / object 比）：

| 模型 | 人像前景占比 | 静物前景占比 | 比值 |
|------|------------|------------|------|
| `u2netp` | 20.38% | 13.27% | 1.54× |
| `u2net` | 20.26% | 13.25% | 1.53× |
| `u2net_human_seg` | 20.15% | 9.35% | **2.15×** |

**结论**：入库的 `u2netp` 行为上贴近**通用显著性 `u2net`**，不是 `human_seg`。
哈希相同只说明「HF 上那个仓库的来源标注与 rembg 的产物重叠」，**不构成行为同源的证据**。
`u2net_human_seg` 的独立价值在于「无人图更保守」，需要这一特性时才值得换。

复现脚本：`scripts/probe_model_identity.py`（见下文）。

> ⚠️ **不要**换成 `u2net_portrait.onnx`：该权重在 APDrawing 数据集上训练，
> 数据集许可含 **NC（非商用）**，会污染本项目的 Apache-2.0 商用承诺（详见 AGENTS.md §6.1）。

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    matting:
      model-path: model-tools/matting/models/u2netp.onnx
```

模型规格 / I/O 契约 / 输出语义 / 预处理实测结论详见
[`mica-ai-core/mica-ai-matting/README.md`](../../mica-ai-core/mica-ai-matting/README.md)。

## 脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `probe_preprocess.py` | **核心**：打印 ONNX 输入输出签名；并用同一张真实照片做预处理对照（只改 `mean`/`std`/通道序），输出掩码 MAE —— 用于核验「通道顺序」这一隐性约定（结论见模块 README「预处理参数」） |
| `probe_model_identity.py` | 多模型**身份 / 行为**对照：打印各自 I/O 签名、前景占比，并两两计算掩码 MAE 与相关系数 —— 用于核验「某个权重到底是通用显著性还是人体分割」这类只看文件名 / 哈希会误判的问题 |

```bash
python model-tools/matting/scripts/probe_preprocess.py \
    model-tools/matting/models/u2netp.onnx \
    mica-ai-core/mica-ai-plate/src/test/resources/images/test_img.jpg

# 身份对照：两个模型 + 若干图片
python model-tools/matting/scripts/probe_model_identity.py \
    model-tools/matting/models/u2netp.onnx \
    /path/to/u2net_human_seg.onnx \
    person.jpg object.jpg
```

运行前提：`numpy` + `onnxruntime` + `Pillow`（缺 Pillow 时退化为 `opencv-python`）。

> ⚠️ 两个脚本都是**只读核对工具**，不参与构建，也不是模型的「自检入口」。
> 模型的唯一自检入口是集成测试：`./mvnc.sh -pl mica-ai-core/mica-ai-matting test`
> （加载真 ONNX 跑完整链路，并断言 I/O 契约：1 输入 / 7 输出 / 320 输入边长）。
>
> 想要用**外置模型**跑同一套集成测试（含全部尺寸 / 语义 / 策略断言）：
>
> ```bash
> ./mvnc.sh -pl mica-ai-core/mica-ai-matting test \
>     -Dmica.ai.matting.externalModel=/path/to/u2net_human_seg.onnx
> ```
>
> 不传该参数时，这项测试自动跳过（外置模型不入库，不能假设它存在）。
