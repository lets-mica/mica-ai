# mica-ai 候选小模型选型记录

> 记录 mica-ai 后续可接入的小模型候选、**已核实的 License / 体积 / ONNX 可得性**、以及每个候选的脆弱点。
> 这份文档是**选型阶段的依据与避坑清单**，不是落地计划 —— 任何模型真正接入前，仍须按 AGENTS.md §6.1 走一遍「商用自检清单」，
> 并按 §7「切换同族模型」的标准动作先核实**真实 I/O 契约**。
> **最后更新**：2026-10-22（首版：完成 12 个候选的协议核实与三档分级）

---

## 1. 背景与范围

mica-ai 当前已落地 7 个能力（face / filetype / plate / layout / matting / textline / captcha），
通用文档 OCR 与语音已抽离到 [mica-ppocr](https://gitee.com/dreamlu/mica-ppocr) / [mica-voice](https://gitee.com/dreamlu/mica-voice)。

本记录面向的问题：**除了现有 7 个能力，还有哪些「小模型」值得接入？**

范围限定：

- ✅ 目标部署形态与现有能力一致 —— **纯 ONNX Runtime、零 Python 进程、零 Spring 依赖、Java 8+**
- ✅ 依赖模型 License **允许下游商业使用**（AGENTS.md §6.1 硬性约束）
- ✅ 单文件 **< 50MB** 方可入库；超限者只能走 `model-path` 外置配置（参照 layout / matting `u2net` 的既有做法）
- ❌ 通用文档 OCR / 通用语音 —— 已在 mica-ppocr / mica-voice，本仓库不重复
- ❌ 依赖 Torch / Paddle / Python 进程 —— 违反「零 Python 进程」原则

---

## 2. 筛选判据

四道筛子，**按顺序**过，任一条不过即降档或淘汰：

| # | 判据 | 说明 |
|---|------|------|
| 1 | **协议可商用** | 权重本身的 License（不是代码仓库的 License）必须允许「商业使用 / 分发 / 修改」 |
| 2 | **能进 ONNX** | 有现成 ONNX 产物，或可一次性离线转换（转换不违反「零 Python 进程」——那是运行时约束） |
| 3 | **体积可入库** | 单文件 < 50MB 可随仓库分发；超限则必须外置，且会牺牲「克隆即用」体验 |
| 4 | **补真实空白** | 填补 mica-ai 现有能力的缺口，而非与 matting / mica-ppocr 等已有能力重复 |

> 判据 1 与 3 是硬约束；判据 4 是**价值判断**，也是本记录里最需要被质疑的一环。

---

## 3. 候选总表

| 档位 | 候选 | 体积 | License（已核实） | 补的空白 | 判据 3 结论 |
|------|------|------|------------------|-----------|------------|
| **A** | DINOv2-small 图像向量 | q8 ~24MB<br>fp32 ~88MB | Apache-2.0（Meta） | 除人脸外没有通用图像向量 | q8 ✅ / fp32 ❌ 外置 |
| **A** | MoveNet SinglePose Lightning | fp16 4.8MB / int8 2.9MB | Apache-2.0（代码+权重双层） | 人体姿态 17 关键点 | ✅ |
| **A** | YoutuReID 行人重识别 | fp32 ~107MB | Apache-2.0（opencv_zoo） | 跨摄像头「找人」 | ❌ **必须外置** |
| **B** | PP-HumanSeg v2（肖像版） | 3.6–5.4MB | Apache-2.0（PaddleSeg） | 人像分割（与 matting 部分重叠） | ✅ |
| **B** | MP-PalmDet + MP-HandPose | 双模型，小 | Apache-2.0（MediaPipe） | 21 手部关键点 / 手势 | ✅ |
| **B** | PP-LCNet 分类底座（PaddleClas） | 12MB 起 | Apache-2.0（PaddleClas） | 挂载行业分类模型的底座 | ✅ |
| **C** | open_nsfw 内容审核 | ~22.5MB | BSD-2-Clause（Yahoo） | 图片合规筛查 | ✅ |
| **C** | Depth Anything V2 Small | fp32 ~100MB<br>int8 ~25MB | Apache-2.0 | 单目深度估计 | int8 ✅ / fp32 ❌ 外置 |
| **C** | SlimSAM-77 | int8 ~13MB（2 文件） | Apache-2.0 | 交互式圈选分割 | ✅ |
| **C** | MODNet | fp32 25.9MB | Apache-2.0 | 发丝级人像 alpha | ✅ |
| — | WeChatQRCode 二维码检测 | 小 | Apache-2.0（zoo 目录声明） | 复杂背景二维码定位 | ⚠️ **Caffe 格式**，见 6.3 |
| — | GFPGAN 人脸修复 | ~340MB | Apache-2.0 但含灰区 | 老照片修复 | ❌ 淘汰，见 6.2 |

---

## 4. A 档：三个真正补空白的能力

### 4.1 DINOv2-small —— 通用图像向量

**为什么值得**：mica-ai 现在只有人脸 128d 向量，**没有通用图像向量**。这是当前能力矩阵里最明显的结构空白。

| 项 | 值 |
|---|---|
| 上游 | `facebook/dinov2-small`（Meta），ONNX 导出 `Xenova/dinov2-small` |
| License | **Apache-2.0** |
| 参数量 | 22.1M（ViT-S/14） |
| 输出 | 384-d `[CLS]` 向量，L2 归一化后余弦相似度 |
| 体积 | q8 ~24MB（int8） / fp32 ~88MB |

**能力映射**：图片去重（近重复检测）、以图搜图、素材聚类、图库归类、few-shot 分类（给几个样例，用最近邻打标）。

**脆弱点**：

- ⚠️ **只有图像塔，没有文本塔**。所以能做「以图搜图」，**做不了「以文搜图」**。后者需要 CLIP / MobileCLIP，但体积直接冲到 90MB+，性价比反而更差。
- ⚠️ **q8 是 int8 量化**，与 fp32 存在可见差异。若业务要求高精度比对，24MB 的 int8 未必够用，而 fp32 88MB 又超了入库线 —— 这是个**两难**，必须实测（见 8.1）。

### 4.2 MoveNet SinglePose —— 人体姿态

**为什么值得**：健身教练、跌倒检测、行为分析、舞蹈教学 —— 纯 Java 生态目前几乎没有轻量姿态方案可用。

| 项 | 值 |
|---|---|
| 上游 | `google/movenet`（TensorFlow），ONNX 导出 `Xenova/movenet-singlepose-lightning` |
| License | **Apache-2.0**（tfjs-models 代码 + MoveNet.SinglePose model card **双层**均已核实） |
| 输出 | `[1, 1, 17, 3]`，每关键点 `(x, y, score)`，坐标已归一化到 `[0,1]` |
| 关键点 | 鼻 / 双眼 / 双耳 / 双肩 / 双肘 / 双腕 / 双胯 / 双膝 / 双踝 |
| 体积 | 官方 TFLite：Lightning fp16 4.8MB / int8 2.9MB；Thunder fp16 12.6MB / int8 7.1MB |
| 性能 | Pixel 5 CPU 4 线程：Lightning int8 52ms、fp16 60ms；Raspberry Pi 4 CPU：int8 95ms |

**脆弱点**：

- ⚠️ **TFLite 与 ONNX 的输入契约完全不同**。官方 TFLite 是 **uint8 输入**；Xenova 的 ONNX 导出是 **float32 + 192×192**。**照 TFLite 文档写归一化必然错**。
  这与本项目 `textline` 的「H=80 / W=160 别写反」、`matting` 的「通道顺序敏感」是**同一类坑** —— 必须以 ONNX 实测签名为准。
- ⚠️ **MultiPose 版不要碰**。SinglePose 的 model card 明确写 Apache-2.0；**MultiPose 的 model card 抽取文本里没有 license 行**（代码 Apache-2.0，但权重条款缺失）。按 §6.1「允许下游商用」的口径，只能接 SinglePose。
- ⚠️ model card 声明「多人物场景不在设计范围内」—— 单人场景（健身、单摄像头）完全够用，**多人场景需另找方案**。

### 4.3 YoutuReID —— 行人重识别

**为什么值得**：768d 全身外观向量（而非人脸），视频找人、零售动线分析、智慧安防。

| 项 | 值 |
|---|---|
| 上游 | 腾讯优图 `ReID-Team/ReID_extra_testdata`，opencv_zoo 收录 |
| License | **Apache-2.0**（opencv_zoo `models/person_reid_youtureid/LICENSE`） |
| 输入 | `[1, 3, 256, 128]` float32，取值 `0–1`（**高在前的瘦长比例**） |
| 输出 | `[1, 768]` L2 归一化向量，余弦相似度即匹配分 |
| 体积 | fp32 ~107MB；zoo 同时提供 `person_reid_youtu_2021nov_int8.onnx` |
| 性能 | INTEL CPU 35.81ms / RPI 521.98ms / JETSON-GPU 90.07ms |

**脆弱点**：

- ⚠️ **fp32 ~107MB 远超 50MB 入库线** —— 必须走 `model-path` 外置配置。这意味着**它做不到「克隆即用」**，
  集成测试也要像 `LayoutIntegrationTest` 那样在模型缺失时用 `Assumptions` 整类跳过。
- ⚠️ **int8 版是唯一能入库的路径，但重识别精度对量化敏感**。是否可接受，必须实测（见 8.3）。
- ⚠️ 前提是**已有行人检测**（bbox 裁剪）—— 仓库里没有通用目标检测模型，这一段得由使用方自己接。

---

## 5. B 档：能接，但要先划清边界

### 5.1 PP-HumanSeg v2 —— 人像分割

**为什么值得**：**极小**。肖像分割版只要 3.6–5.4MB，mIoU 96.63%，比 u2netp 的通用显著性分割更贴人像。

| 型号 | 体积 | 输入 | mIoU |
|------|------|------|------|
| Portrait-PP-HumanSegV2-Lite | 3.6MB | 256×144 | 96.63% |
| PP-HumanSegV2-Lite | 12MB | 192×192 | 92.52% |
| PP-HumanSegV2-Mobile | 29MB | 192×192 | 93.13% |
| PP-HumanSegV1-Lite | 543KB | 192×192 | 86.2% |

**边界问题（必须先答）**：它和 `mica-ai-matting`（U²-Net `u2netp`）**功能重叠**。

- 严格说它是「matting 的第二模型族」，**不是新能力**
- 但契约与 u2net **不同**：u2net 是 1 输入 / 7 输出 `[1,1,320,320]`（含 deep supervision）；HumanSeg 是 1 输入 / 1 输出 logits `[1,1,192,192]`
- ⇒ **不能像 `u2net_human_seg` 那样纯改 `model-path` 切换**，得按 AGENTS.md §7「切换同族模型」把差异显式化为配置项（`output-select` / `input-size` / `mean` / `std`）
- ⇒ 另开一个 `mica-ai-humanseg` 模块，还是并进 `mica-ai-matting` 作为可选引擎，**是个需要 owner 拍板的架构决策**

⚠️ **命名陷阱**：FastDeploy 的模型清单与 HuggingFace 模型卡对「PP-HumanSegV2-Lite」的体积/尺寸标注**不一致**（3.6MB/256×144/96.63 vs 12MB/192×192/92.52 vs 5.4MB/256×144/96.63）。
**不要按名字认模型** —— 这与本项目 `matting` 踩过的 `u2netp` 哈希相同但行为不同是同一类问题。接入时须以**实际文件的 I/O 契约**为准。

### 5.2 MP-PalmDet + MP-HandPose —— 手部关键点

**为什么值得**：21 个手部关键点（腕 + 4 指 × 5 关节），含 z 坐标；手势识别、手语、AR/VR 交互。

| 项 | MPPalmDet | MPHandPose |
|---|---|---|
| 输入 | 192×192 RGB，`[0,1]`，**NHWC 布局** | 掌检测裁剪出的手部区域 |
| 输出 | 框偏移 + 7 个掌关键点 + 置信度 | 21 关键点 `(x, y, z)` |
| 量化 | int8bq（block-wise，权重 only） | int8（Intel Neural Compressor，FreiHAND 标定） |

**脆弱点**：

- ⚠️ **两阶段流水线**，不是单模型 —— PalmDet 定位 → 裁剪 → HandPose 估计，比现有单段能力复杂一档
- ⚠️ **输入是 NHWC 布局**，与仓库内其它模型（NCHW）相反，写错不报错只静默降精度
- ⚠️ **anchor 数量在两个资料源里冲突**（opencv_zoo `palm_detection_mediapipe` README 记 **2016** = 896 fine + 1120 coarse；某索引页记 2376）—— 接入前须以**实际模型输出张量形状**为准
- ⚠️ 掌检测的 anchor 网格是**硬编码常量数组**（`loadAnchors()`），Java 端要复刻 2016 个 anchor 的生成规则

### 5.3 PP-LCNet 分类底座（PaddleClas）

**为什么值得**：ImageNet-1000 通用分类本身商业价值很低，**但 PaddleClas 的价值是让你挂载行业分类模型**（垃圾分类、商品识别、场景审核、工业质检）。
而且与现有 `textline` 的 `PP-LCNet_x1_0_textline_ori` 是**同族骨干**，预处理约定可复用。

| 模型 | 体积 | Top-1 | Top-5 |
|------|------|-------|-------|
| PPLCNet_x1_0 | 12MB | 71.32% | 90.03% |
| EfficientNetB0_small | 18MB | 75.8% | — |
| MobileNetV2_x0_25 | 5.9MB | 53.2% | 76.5% |
| SqueezeNet1_1 | 4.8MB | 60.1% | 81.9% |
| ShuffleNetV2_x0_25 | 2.4MB | 49.9% | 73.79% |
| MobileNetV1_x0_25 | 1.9MB | 51.4% | 75.5% |

**标准预处理契约**（与 PaddleClas `inference_cls.yaml` 一致，接入时照抄别自创）：

```
ResizeImage: resize_short = 256
CropImage:   size = 224
NormalizeImage: scale = 1/255, mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]
ToCHWImage
```

**脆弱点**：

- ⚠️ **必须配 label 列表文件**（`imagenet1k_label_list.txt`，1000 类）。这与 `mica-ai-captcha` 的 charset 是同一类**强耦合配套资源** ——
  按 AGENTS.md §7 的既有范式，`model-path` 与 `labels-path` 应**成对配置、缺一即 fail-fast**，且**刻意不去重**（去重会让索引整体错位）。
- ⚠️ ImageNet 类别体系的商业意义有限，**不建议作为独立能力售卖**；定位应是「行业分类模型的底座 + 现成挂载点」。

---

## 6. C 档与淘汰清单

### 6.1 C 档（场景明确才做）

| 候选 | 关键参数 | 适用场景 | 脆弱点 |
|------|---------|---------|--------|
| **open_nsfw** | ResNet-50 thin 1by2，5 类（drawings/hentai/neutral/porn/sexy），ONNX ~22.5MB，**BSD-2-Clause** | 图片合规预筛 | ⚠️ **只识别色情写实照片**，不识别暴力、卡通、素描；官方明确「不保证准确率」；⚠️ 原始是 Caffe，ONNX 为第三方转换产物（权重 BSD-2-Clause 转换后不变，但须核对转换件的来源与完整性） |
| **Depth Anything V2 Small** | DINOv2 ViT-S + DPT，24.8M 参数；输入 `[1,3,H,W]`、长边 518、**两边向上取整到 14 的倍数**、ImageNet 归一化；输出 `predicted_depth [1,H,W]`（相对逆深度） | 景深虚化、AR 遮挡判断、3D 重建前置 | ⚠️ **fp32 ~100MB 超入库线**，只能走 int8 ~25MB；⚠️ 输出是**相对深度**，仅单图内可比，不是米制；⚠️ 14 倍数对齐是最容易写错的预处理点 |
| **SlimSAM-77** | Apache-2.0；int8：vision encoder 8.47MB + prompt/decoder 4.68MB；encoder 输入 `[1,3,1024,1024]`，decoder 吃 `image_embeddings [1,256,64,64]` + `input_points/input_labels` | 交互式圈选分割、自动抠图、标注辅助 | ⚠️ **双模型两阶段**（encoder 一次 + decoder 多次），比现有单段能力复杂；⚠️ encoder 在 CPU 上约 388ms；⚠️ 输入点/标签的坐标语义（图像像素 vs 归一化）须实测 |
| **MODNet** | trimap-free 人像抠图；fp32 25.9MB / fp16 12.38MB / q8 ~7MB；输入 `[1,3,H,W]` 动态分辨率，输出 `[1,1,H,W]` alpha ∈ [0,1] | 发丝级人像 alpha、证件照换底 | ⚠️ **与 matting 重叠更大**（u2netp 也是抠图）；MODNet 的增量价值是「软 alpha 边缘」而非「主体定位」；⚠️ 与 `u2netp` 契约不同，不能改 `model-path` 直接换 |

### 6.2 明确不建议

| 候选 | 淘汰理由 |
|------|---------|
| **GFPGAN**（人脸修复） | ① ONNX ~340MB，远超入库线；② **License 灰区** —— TencentARC/GFPGAN 仓库主体 Apache-2.0，但 LICENSE 文件额外列明第三方组件，GitHub 识别为 NOASSERTION；③ **FFHQ 数据集（汇编许可 CC-BY-NC-SA）如何传导到权重**是未决问题，按 §6.1「允许下游商用」口径不可接受；④ 附带的 MLX 复现实测 fp16 精度崩坏（余弦相似度 ≈ −0.3） |
| **YOLOv5 / v8 / v11** | AGPL-3.0，AGENTS.md §6.1 已明确排除 |
| **RMBG-1.4** | 需单独授权，已排除 |
| **u2net_portrait** | APDrawing 数据集带 NC 限制，已排除 |
| **OpenPose** | LICENSE 标题为「ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY」，商用需另购授权 |
| **通用 ImageNet 分类当独立能力卖** | 商业价值低（见 5.3，应定位为底座而非能力） |

### 6.3 特殊项：WeChatQRCode 二维码检测

| 项 | 值 |
|---|---|
| License | opencv_zoo 该目录声明 **Apache-2.0**（权重来自 `opencv/opencv_3rdparty:wechat_qrcode_20210119`） |
| 定位 | 复杂背景 / 倾斜 / 小尺寸二维码的**检测与解析** |
| 性能 | INTEL CPU 7.04ms / RPI 37.68ms（100×100 输入） |

**为什么只列为「特殊项」而非 A/B 档**：

1. ⚠️ **是 Caffe 格式**（`detect_2021nov.caffemodel` + `sr_2021nov.caffemodel`），**不是 ONNX** —— 需要一次性离线转换才能进 ORT 体系（转换不违反「零 Python 进程」，那是运行时约束，但多一道工序）
2. ⚠️ **openpnp/opencv 4.9.0 是否含 `wechat_qrcode` contrib 模块需实测** —— `mica-ai-common` 依赖的是 openpnp 打包版，contrib 模块不一定在
3. ⚠️ **更重要的**：**二维码/条码「解码」根本不需要模型** —— [ZXing](https://github.com/zxing/zxing)（Apache-2.0）是纯 Java 实现，零模型、零依赖。
   只有「在杂乱的图里把歪的、小的、被遮挡的二维码**定位**出来」才需要模型。
   ⇒ **先问清需求是「解码」还是「检测+解码」**：前者用 ZXing 即可，**不该为一个模型新增一个模块**。

---

## 7. 协议核实证据汇总

> 全部于 2026-10-22 核实。**核实的是权重条款，不是代码仓库条款** —— 这是 §6.1 的要求。

| 候选 | 权重 License | 依据（原始来源） |
|------|------------|-----------------|
| DINOv2-small | Apache-2.0 | [facebook/dinov2](https://github.com/facebookresearch/dinov2) 权重 + [Xenova/dinov2-small](https://huggingface.co/sefaburak/dinov2-small-onnx) 模型卡 `license: apache-2.0` |
| MoveNet SinglePose | Apache-2.0 | [MoveNet.SinglePose Model Card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf) 明确写 Apache-2.0；[tfjs-models LICENSE](https://github.com/tensorflow/tfjs-models/blob/master/LICENSE) 亦为 Apache-2.0（**代码 + 权重双层**） |
| YoutuReID | Apache-2.0 | [opencv_zoo/models/person_reid_youtureid/LICENSE](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid) |
| PP-HumanSeg v2 | Apache-2.0 | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 仓库；HF 模型卡 [unography/PP-HumanSegV2-Lite](https://huggingface.co/unography/PP-HumanSegV2-Lite) `license: apache-2.0` |
| MP-PalmDet / MP-HandPose | Apache-2.0 | [opencv_zoo/models/palm_detection_mediapipe/README.md](https://github.com/opencv/opencv_zoo/tree/main/models/palm_detection_mediapipe) 声明 Apache-2.0；上游 MediaPipe 代码同为 Apache-2.0 |
| PP-LCNet / PaddleClas | Apache-2.0 | [PaddleClas README](https://github.com/PaddlePaddle/PaddleClas) 「本项目的发布受 Apache 2.0 license 许可认证」 |
| open_nsfw | BSD-2-Clause | [yahoo/open_nsfw](https://github.com/yahoo/open_nsfw) LICENSE（BSD 2-Clause, Yahoo Inc.） |
| Depth Anything V2 Small | Apache-2.0 | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)；ONNX 导出 `onnx-community/depth-anything-v2-small` 同为 apache-2.0 |
| SlimSAM-77 | Apache-2.0 | [czg1225/SlimSAM](https://github.com/czg1225/SlimSAM) LICENSE（Apache-2.0，对 SAM ViT-B 的结构化剪枝）；ONNX 导出 `Xenova/slimsam-77-uniform` |
| MODNet | Apache-2.0 | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) LICENSE；HF [Xenova/modnet](https://huggingface.co/Xenova/modnet) 同为 apache-2.0 |
| WeChatQRCode | Apache-2.0 | [opencv_zoo/models/qrcode_wechatqrcode/README.md](https://github.com/opencv/opencv_zoo/tree/main/models/qrcode_wechatqrcode) 声明 Apache-2.0 |
| GFPGAN | ⚠️ 灰区 | 主体 Apache-2.0，但 LICENSE 另列第三方组件，GitHub 识别 **NOASSERTION**；FFHQ 数据集许可传导未决 → **不接入** |

---

## 8. 下一步：三个低成本的验证动作

> 这三项都**只读契约 + 跑少量样本，不动 Java 代码**，目的是把选型从「看着不错」变成「有数据支撑」。
> 每项都刻意配了可判定的**衡量指标** —— 没有指标的探针等于没跑。

### 8.1 DINOv2：q8 与 fp32 的差距

- 动作：同批图片分别跑 int8 与 fp32 导出，比 384d 向量
- 样本：50 组「同图 / 异图」配对
- 指标：**可分性 =（两组余弦分布间隔）/（组内标准差）**
- 判据：若 int8 的可分性显著低于 fp32，则 24MB 方案不可用，fp32 88MB 只能外置 —— 这会改变整个 A 档的结论

### 8.2 MoveNet：ONNX 导出的真实签名

- 动作：`sess.getInputs()` / `getOutputs()` 直接打印，**不读文档**
- 指标：① 输入是否 float32；② 尺寸是否 192×192；③ 预处理归一化形式（`[0,1]` 还是 `[-1,1]`）；④ **ONNX 文件实际 MB**
- ⚠️ 对照点：官方 TFLite 是 uint8 输入 —— 若 ONNX 侧与文档不一致，**以 ONNX 实测为准**（这正是 textline 那个 bug 的同类）

### 8.3 YoutuReID：int8 的精度衰减

- 动作：同一组行人跨帧 / 跨相机比对，fp32 vs int8 各跑一遍
- 指标：mAP 掉幅
- 判据：掉幅 **< 5%** → int8 可入库（体验大幅改善）；否则维持 fp32 外置（并接受「不能克隆即用」的代价）

---

## 9. 尚未核实的候选（同族扩展）

以下方向**有价值但本轮未逐个核实协议**，接入前须单独查（opencv_zoo 仓库级是 Apache-2.0，但**每个模型目录各带自己的 LICENSE**，不可沿用仓库级结论）：

| 候选 | 潜在价值 | 待核事项 |
|------|---------|---------|
| OpenCV Zoo FER（Progressive Teacher 表情识别） | 表情 / 情绪分析，可并入 `mica-ai-face` | 目录 LICENSE、ONNX 版本、输入契约 |
| face_landmark_106（106 点人脸关键点） | 现有 YuNet 只给 5 点；106 点可支撑精细对齐、表情、头姿估计 | 目录 LICENSE、体积、ONNX 可得性 |
| PP-LCNet_x1_0_table_cls（表格分类） | 有线表 / 无线表判别 | ⚠️ 属通用文档 OCR 范畴，**应放 mica-ppocr 而非本仓库** |
| PaddleX 场景分类 / 行业分类模型 | 可作为 5.3 的「行业挂载点」实例 | 各模型 LICENSE、是否已有 ONNX 产物 |

---

## 10. 决策记录与待办

### 10.1 已完成的核查

- [x] 盘点仓库现有 7 个能力 + 已抽离的 OCR / 语音，划清「不重复」边界
- [x] 核实 12 个候选的**权重** License（逐个查原始来源，非代码仓库条款）
- [x] 核实各候选的**体积**，并对照 50MB 入库线分档
- [x] 核实各候选的 **ONNX 可得性**，标出 TFLite-only / Caffe-only / 需转换的
- [x] 逐个标注**脆弱点**（含 3 处资料源互相冲突的命名 / 尺寸问题）

### 10.2 待办

- [ ] 跑 8.1 DINOv2 q8 vs fp32 可分性实测
- [ ] 跑 8.2 MoveNet ONNX 真实签名探针
- [ ] 跑 8.3 YoutuReID int8 精度衰减实测
- [ ] owner 拍板：**PP-HumanSeg 是并入 `mica-ai-matting` 还是另开模块**（架构决策，见 5.1）
- [ ] owner 拍板：**二维码需求是「解码」还是「检测+解码」** —— 若是前者，ZXing 即可，不新增模块（见 6.3）
- [ ] owner 拍板：**是否接受 YoutuReID 走外置**（失去「克隆即用」体验，换 107MB fp32 精度）
- [ ] A 档三项探针跑完后再定**接入顺序与排期**

### 10.3 接入任一能力时的既有范式（回顾）

参照本仓库已落地的 7 个能力，任一新能力都应满足：

1. **模块 + Starter + `model-tools/<cap>/README.md` 三处齐全**，根 README / AGENTS.md §1 §2 §6.1 同步
2. **模型与强耦合配套资源成对配置**（如 captcha 的 charset、分类的 labels），缺一即 fail-fast，**刻意不去重**
3. **I/O 契约以真模型实测为准**，不读元信息、不信文档（`textline` / `captcha` 都踩过「声明形状是错的」）
4. **集成测试用 `nu.pattern.OpenCV.loadLocally()`**（`forkCount=0` 下多模块共用 JVM，`loadShared()` 在 Java 8 会抛 `UnsatisfiedLinkError`）
5. **模型 + 配套资源可外置**，集成测试支持 `-Dmica.ai.<cap>.externalModel=...`，不传时自动跳过
6. **§6.1 商用自检清单逐条勾选**（模型仓库地址、官方 LICENSE 全文/链接、一句话商用结论）

---

## 附：与本项目历史教训的对应关系

| 历史坑 | 本记录中的对应处理 |
|--------|------------------|
| `matting`：`u2netp` 与 HF 上 `U-2-Net-Human-Seg` 哈希逐字节相同但行为不同 | 5.1 显式标注 PP-HumanSeg 的命名/尺寸冲突，**要求以实际文件契约为准，不按名字认模型** |
| `textline`：H=80 / W=160 宽高写反 | 4.2 标注 MoveNet 的 TFLite/ONNX 预处理差异；8.2 要求以 `getInputs()` 实测为准 |
| `captcha`：动态量化模型声明形状是错的 | 8.1–8.3 三项探针全部要求**实测输出维度**而非读元信息 |
| `captcha`：字符集是索引映射表，刻意不去重 | 5.3 标注 ImageNet label 列表同属强耦合配套资源，**去重会让索引整体错位** |
| `layout` / `matting u2net`：超 50MB 模型只能外置 | 4.3 YoutuReID fp32、6.1 Depth Anything fp32 均标注「必须外置 + 测试整类跳过」 |
| `face` / `plate` / `layout`：Java 8 下 `loadShared()` 抛 `UnsatisfiedLinkError` | 10.3 第 4 条列入接入 checklist |
