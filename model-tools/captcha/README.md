# captcha 模型目录

[ddddocr（带带弟弟 OCR）](https://github.com/sml2h3/ddddocr) 通用验证码识别模型。模型**不随 jar 分发**，由使用者通过 `model-path` / `charset-path` 指向本地文件。

## 模型清单（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `ddddocr.onnx` | 12.98 MB (13,606,051 bytes) | 通用验证码识别 ONNX（CRNN + CTC），**动态量化**版本 |
| `charset.txt` | 24.4 KB (24,407 bytes) | 字符集（UTF-8 单行），8209 个字符 + 1 个 blank = **8210 类** |

- 来源：[sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) 的 `common.onnx`（OCR 主模型），
  权重由 [dddd_trainer](https://github.com/sml2h3/ddddocr) 训练
- License：**MIT License**，**可商用** ✅
- 完整性指纹：
  - `ddddocr.onnx` → `sha256 = b8f2ad9cbc1f2e3922a6cb9459e30824e7e2467f3fb4fd61420640e34ea0bf68`
  - `charset.txt` → `sha256 = 2ad1d606078ede388943ef28b2cab3ce2ee50f81d42ff275572a69a814d136f0`
- 分发状态：**已入库**，随仓库分发（12.98 MB，低于 AGENTS.md §6.2 的 50 MB 上限）

## ⚠️ 本仓库的 `ddddocr.onnx` 是**动态量化衍生版**，不是上游原文件

上游 `common.onnx` 的 `sha256` 为 `33b5cd351ee94e73a6bf8fa18c415ed8b819b3ffd342e267c30d8ad8334e34e8`
（见 [rustcha 的第三方声明](https://github.com/henrique-coder/rustcha/blob/prod/THIRD_PARTY_NOTICES.md)，
其取自 ddddocr commit `c40f56f9`），**与本仓库文件的 `b8f2ad9c…` 不同**。

本文件的图头是 `DynamicQuantizeLinear` + `ConvInteger`（动态量化），而上游原文件是 float32 权重。
**这正是「文件名 / 哈希相同 ⇒ 行为同源」不可靠的又一个例子**：本文件的字节与上游不同，
但**行为同源**（同一 CRNN + CTC 架构、同一 8210 类字符集、同一输入高度 64），
已由集成测试用真模型逐字符比对确认（见下方「实测结论」）。

> 若要换成上游未量化的 float32 原文件：把 `model-path` 指过去即可，
> **Java 代码零改动**（量化与否只影响精度与体积，不影响 I/O 契约）。
> 换成非动态量化的模型时，`normalization` 建议显式设为 `MEAN_0_5`（见模块 README「归一化」）。

## 字符集契约

`charset.txt` 是**索引映射表**，不是普通文本：

- 每个字符占一个索引，**换行符不占索引**
- 索引 `0` 固定为 CTC 的 **blank**（空串），由 `CharsetManager` 补上
- 模型输出的第 `i` 类 ⇄ 字符集的第 `i` 个字符
- 共 `8209 + 1 = 8210` 类，与模型输出维度**严格相等**（不一致时识别结果是乱码，模块会直接抛异常）

> ⚠️ **不要去重**。字符集是索引映射表，出现重复字符时若去重，后续所有索引都会错位。
> 模块的 `CharsetManager` 刻意保留重复字符，并有单元测试守住这条。

## 使用

模型为**外置配置**，不打包进 jar，需自行指向文件：

```yaml
mica:
  ai:
    captcha:
      enabled: true
      model-path: model-tools/captcha/models/ddddocr.onnx
      charset-path: model-tools/captcha/models/charset.txt
      input-height: 64
```

部署到其它机器时，把 `models/` 下的两个文件拷到应用可读的位置，改这两个配置项即可。

## 实测结论（2026-10-22，onnxruntime 1.18.0 / onnxruntime-python 1.30.0）

| 项 | 实测值 |
|----|--------|
| 输入节点 | `input1`，`[1, 1, 64, image_width]`（高度固定 64，宽度按原图比例动态） |
| 输出节点 | `387`，**运行时**形状 `[23, 1, 8210]`，**原始 logits**（未 softmax，值域约 `[-7, 34]`） |
| 输出声明形状 | `[1, seqlen]` —— **元信息是错的**，不可用于推断类别数 |
| 预处理 | 等比缩放到高 64（宽 = `w * 64 / h` 截断）→ 灰度化（`0.299R+0.587G+0.114B`）→ 归一化 |
| 归一化 | `DIV_255` 与 `MEAN_0_5` 给出**相同**的 argmax 与解码结果（动态量化模型对此不敏感） |
| 样例图 | `captcha_shtp.png`（100×36）→ 识别 `SHTP`，平均置信度 **0.9996** |
| CTC 时间步 | 23 步，argmax 序列为 `[0,0,0,0, 6386, 0,0,0,0, 5554, 0,0,0,0, 5046, 0,0,0,0, 3930, 0,0]` |

模型规格 / I/O 契约 / API 说明详见
[`mica-ai-core/mica-ai-captcha/README.md`](../../mica-ai-core/mica-ai-captcha/README.md)。
