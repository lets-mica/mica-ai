# model-tools/docalign

`mica-ai-docalign`（文档角点检测 / 透视矫正）的模型资产目录。

| 文件 | 体积 | SHA256 | 输入 | 输出 |
|------|------|--------|------|------|
| `models/lcnet100_h_e_bifpn_256_fp32.onnx` | 4.55MB | `F4117B786E3A18470F3865C93F3C2BD69D9B998EDD60F385574A5C665E79594E` | `img [1,3,256,256]` float32 | `heatmap [1,4,128,128]` float32（内置 Sigmoid，通道序 = 左上 / 右上 / 右下 / 左下） |

## 来源与许可

- 上游：[DocsaidLab/DocAligner](https://github.com/DocsaidLab/DocAligner) release `1.1.1`
  - 下载地址：`https://github.com/DocsaidLab/DocAligner/releases/download/1.1.1/lcnet100_h_e_bifpn_256_fp32.onnx`
- License：**Apache-2.0**（可商用、可分发的宽松协议），见上游仓库根 `LICENSE`
- 分发要求：保留上游项目名称与来源声明（本文件即承担该职责）

## 契约要点

- 预处理：直接 resize 到 256×256 → BGR→RGB → `/255`（**无 mean / std**）→ CHW
- 后处理：4 通道热力图逐个缩放到原图 → 阈值 0.3 二值化 → 取**面积最大连通域质心**
- 角点通道顺序：**左上 → 右上 → 右下 → 左下**（已用合成斜拍文档实测，勿凭直觉写成顺时针/逆时针的另一种）
- 末端算子已内置 `Sigmoid`，输出值域 `[0,1]`

## 同族可替换模型（同一 release）

| 模型 | 体积 | 入库 | 说明 |
|------|------|------|------|
| `lcnet100_h_e_bifpn_256_fp32` | 4.55MB | ✅ | 默认，精度 / 速度均衡 |
| `lcnet050_p_multi_decoder_l3_d64_256_fp32` | 4.91MB | ✅（可选） | 轻量版 |
| `fastvit_t8_h_e_bifpn_256_fp32` | 13.2MB | ❌ 外置 | 精度更高 |
| `fastvit_sa24_h_e_bifpn_256_fp32` | 83MB | ❌ 外置 | 精度最高，**超出 50MB 入库上限**（AGENTS.md §6.2） |

> 四个模型 I/O 契约一致（单输入 `img`、单输出 4 通道热力图、输入 256×256），
> 切换时只改配置的 `model-path`，**不要改 Java 代码**。
