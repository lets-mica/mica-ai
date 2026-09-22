# Changelog

本项目的所有显著变更都会记录在此文件中。版本遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [Unreleased]

### ✨ 新增能力

- ✂️ `mica-ai-matting` —— U²-Net `u2netp` 通用抠图（显著性目标检测）：`MattingEngine` 提供原尺寸 alpha 掩码、透明底 PNG（BGRA）、纯色底合成、二值掩码四组输出
  - 模型 `u2netp.onnx` **4.36 MB**（Apache-2.0，[danielgatis/rembg](https://github.com/danielgatis/rembg) 打包 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)），**随仓库分发**
  - I/O 实测（onnxruntime 1.30）：输入 `input.1 [1,3,320,320]` float32；输出 **7 个** `[1,1,320,320]`，索引 0 即 d0，索引 1..6 为 deep supervision 中间输出，推理须忽略
  - d0 **已内置 Sigmoid**（实测落在 `[0,1]`），不要重复 sigmoid；`minMaxNormalize`（默认开）保留 rembg 参考行为，对低对比度输入实测放大 ~470×
  - 预处理 ImageNet 归一化（**RGB** 顺序）；实测通道顺序敏感、`mean/std` 绝对值不敏感（对照实验见 `model-tools/matting/scripts/probe_preprocess.py`）
  - 输出节点名不可依赖（数字名 `1959`..`1965`），定位策略显式化为可配置的 `output-select`（`AUTO` / `FIRST` / `D0`），默认 `AUTO` 为先名称提示、再校验「7 个同形输出」取首个
- 🔌 `mica-ai-matting-spring-boot-starter`（`mica.ai.matting` 前缀）：`@Bean(destroyMethod = "close")` + `@ConditionalOnMissingBean`；`MattingPropertiesTest` 覆盖配置绑定、默认值一致性，并通过 `MattingAutoConfiguration#toConfig` 校验**真实装配链**（不重抄 builder，漏装配即失败）
- 🔤 `mica-ai-textline` —— PP-LCNet 文本行方向分类（0° / 180°）：`TextLineEngine` 提供「只判定」（`classify*` → `TextLineOrientationResult`）与「判定+转正」（`uprightBytes` / `rotateIfUpsideDown`）两组能力，典型用途是 OCR 流水线前置转正
  - 模型 `PP-LCNet_x1_0_textline_ori.onnx` **6.46 MB**（Apache-2.0，[PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX) 文本行方向分类，paddle2onnx 转换产物），**随仓库分发**
  - I/O 实测（2026-09-21）：输入 `x [N,3,80,160]` float32；输出 `fetch_name_0 [N,2]` float32 **原始 logits**（未 softmax）；类别顺序来自官方 `inference.yml` 的 `label_list: [0_degree, 180_degree]`
  - ⚠️ **H=80 / W=160，别写反**：官方 `ResizeImage: {size: [160, 80]}` 是 PaddlePaddle 的 **(宽, 高)**；构造期比对模型输入形状与 `input-width` / `input-height`，不一致直接 `MicaAiException` 快速失败
  - ⚠️ **真实数据置信度远低于合成图**：合成文本行饱和到 `logits=[+1.0, +0.0]` → `p=0.7311`；官方真实样例倒置态仅 `p≈0.5772`，已逼近默认阈值 `0.5`。默认保留 `0.5`，但 README / yml 均记录「误旋转代价高时调到 0.6~0.7 让拿不准的行保持不动」
  - 通道顺序默认 `BGR`（严格复刻 PaddleX 在 OpenCV 原生 BGR 上逐通道归一化的行为）；实测本任务判定结构朝向而非颜色，**BGR / RGB 结论一致**
  - 输出名 `fetch_name_0` 是 Paddle2ONNX 自动命名，**不可硬编码**：先按名称提示（`fetch_name_0`/`logits`/`output`/`prob`/`softmax`）匹配，再回落「唯一的 2 维 float 输出」结构判定
  - `uprightBytes` 在方向正常时**原样返回输入字节**（不重编码、零画质损失），实测 `isSameAs` 为真；`rotateIfUpsideDown` 同理返回**入参本身**
- 🔌 `mica-ai-textline-spring-boot-starter`（`mica.ai.textline` 前缀）：`@Bean(destroyMethod = "close")` + `@ConditionalOnMissingBean`；`TextLinePropertiesTest` 覆盖配置绑定与默认值一致性，并通过 `TextLineAutoConfiguration#toConfig` 校验**真实装配链**（已用「删掉一行 `.upsideDownThreshold(...)` 即测试失败」验证过该测试确实有牙）

### 🔧 模型可插拔

- **切换 U²-Net 族模型无需改 Java 代码**：`u2net`（168MB，通用显著性完整版）与 `u2net_human_seg`（168MB，人体分割）经实测与 `u2netp` **契约完全一致**（同 `input.1 [1,3,320,320]` 输入、7 个 `[1,1,320,320]` 输出、同名 `1959`..`1965`、同归一化），仅需改 `model-path`
- 两个 168MB 模型按 AGENTS.md §6.2 **不入库**（超 50MB），由使用者自行下载后指向本地路径；集成测试支持 `-Dmica.ai.matting.externalModel=<path>` 用外置模型跑同一套断言（不传则该项自动跳过）
- ⚠️ **实测纠正一条误导性线索**：本机 `u2netp.onnx` 的 `sha256` 与 HuggingFace `BritishWerewolf/U-2-Net-Human-Seg` 的 onnx **逐字节相同**，但行为实测显示其贴近**通用显著性 `u2net`**（静物图 MAE `0.0009`）而非 `human_seg`（MAE `0.0400`，差 44 倍）；对「无人图」的保守度 `u2netp` 1.54× vs `human_seg` 2.15×。新增 `model-tools/matting/scripts/probe_model_identity.py` 用于多模型行为对照 —— **权重语义须看行为，不能只看文件名或哈希**

- **切换 PP-LCNet 文本行方向模型同样无需改 Java 代码**：轻量版 `PP-LCNet_x0_25_textline_ori`（约 0.96MB，未入库）与内置 `x1_0` 版 I/O 契约一致（同 `x [N,3,80,160]` 输入、同 `[N,2]` 输出、同归一化、同类别语义），仅需改 `model-path`
  - 新增 `model-tools/textline/scripts/probe_contract.py`：只读契约探针，打印 I/O 签名与 `sha256`，并核验**类别语义**（索引 0 = `0_degree`）、**通道顺序敏感性**（BGR vs RGB）、**退化输入鲁棒性**（缩小 / 模糊 / JPEG q30）—— 换导出 / 排查判定异常时先跑它

### 🧪 测试

- `MattingIntegrationTest`：15 项真模型集成测试，覆盖 I/O 契约（1 输入 / 7 输出 / 320 边长）、掩码尺寸回填（含 640×120 极端长宽比，防 OpenCV `Size(宽,高)` 写反）、alpha 语义、min-max 边界、BGRA/纯色底/二值三种输出形态、缺文件与空输入兜底、`output-select` 策略实际生效（`D0` 在数字命名导出上必须快速失败）、外置模型可加载
- `TextLineIntegrationTest`：16 项真模型集成测试，覆盖 I/O 契约（1 输入 / 160×80 / 2 类）、**宽高写反必须快速失败**、类别索引语义（索引 1 = 180°，写反即「把正的转成倒的」）、前景极性无影响、窄长行（700×60，宽高比 11.7:1）仍判定正确、**真实数据置信度区间断言**（`isBetween(0.55, 0.72)`，而非照抄合成图的 0.7311）、通道顺序不改变结论、转正后复检闭环、`uprightBytes` 在方向正常时零拷贝返回（`isSameAs`）、阈值确实生效（阈值 1.0 时不判倒置）、路径入参与缺失文件兜底、null / 空图返回 null、外置模型可加载

---

## [1.0.0] - 2026-09-19

🎉 **首个正式版发布** —— mica-ai 1.0.0 正式发布，提供四个可商用、零 Python、纯 ONNX Runtime 的 Java 8+ AI 能力：

### ✨ 新增能力

- 🎭 `mica-ai-face` —— OpenCV Zoo 人脸识别：YuNet 检测（640×640 BGR anchor-free + 5 关键点）+ SFace 128d Embedding（L2 归一化）+ MiniFASNetV2 活体（real/print/replay 三分类）+ 头像提取（自动摆正 / 分块兜底 / 残角精修）+ 证件卡片提取（掩膜 + 四边形拟合 + 透视矫正 + USM/CLAHE）
- 📄 `mica-ai-filetype` —— Google Magika `standard_v3_3` Java 复刻，214 类文件类型识别；包含 HIGH_CONFIDENCE / MEDIUM_CONFIDENCE / BEST_GUESS 三种预测模式
- 🚗 `mica-ai-plate` —— HyperLPR3 v20230229 中国车牌识别：检测（320/640）+ CRNN 字符识别（77 token CTC）+ 颜色分类（黄/蓝/绿）+ 10 类判型（含港澳 / 学 / WJ 等规则）
- 📐 `mica-ai-layout` —— PaddleOCR PP-DocLayoutV3 文档版面分析（25 类 + V3 reading order）；模型 125MB 不随仓库分发，需本地放模型

### 🔌 Spring Boot Starter

- `mica-ai-face-spring-boot-starter`（`mica.ai.face` 前缀）
- `mica-ai-filetype-spring-boot-starter`（`mica.ai.filetype` 前缀）
- `mica-ai-plate-spring-boot-starter`（`mica.ai.plate` 前缀）
- `mica-ai-layout-spring-boot-starter`（`mica.ai.layout` 前缀）

全部基于 `mica-auto` 生成 `META-INF/spring.factories` / `AutoConfiguration.imports`，零配置即可注入引擎 Bean，`@ConditionalOnMissingBean` 支持自定义实现替换。

### 🧱 基础模块

- `mica-ai-common` —— 跨能力共享基础设施：`OnnxModelSession`（含 `classpath:` 前缀资源解析）、`OrtSessionFactory`（统一构造 OrtSessionOptions）、`OrtProviders`（CPU/CoreML/CUDA 自动选择与回退）、`MicaAiException` / `ErrorCode` 统一异常体系

### 📦 模型与构建

- 模型资产 `model-tools/` 全部直接入库（YuNet + SFace + MiniFASNetV2 + Magika standard_v3_3 + HyperLPR3 v20230229）
- 唯一例外：PP-DocLayoutV3 模型 125MB 超出「<50MB 入库」约定，已在 `.gitignore` 显式排除；`LayoutIntegrationTest` 在模型缺失时用 `Assumptions` 整类跳过
- Maven 顶层 `${revision}=1.0.0`，JDK 8+ / Spring Boot 2.7.x / ONNX Runtime 1.18.0 / OpenCV 4.9.0（openpnp）

### ✅ License 一句话

代码 Apache-2.0；所有依赖模型 **全部可商用**：YuNet + SFace（Apache-2.0）、MiniFASNetV2（MIT）、Magika `standard_v3_3`（Apache-2.0）、HyperLPR3 v20230229（Apache-2.0）、PP-DocLayoutV3（Apache-2.0）。

---

## 历史

### [2026.06.01] - 2026-09-15

集成开发期快照：

- `mica-ai-face` 主线能力就位（YuNet + SFace + MiniFASNetV2 + 头像 / 卡片）
- `mica-ai-filetype` 完成 Magika 复刻
- `mica-ai-face-spring-boot-starter` / `mica-ai-filetype-spring-boot-starter` 自动装配落地
- `mica-ai-common` 抽出 ONNX 通用基础设施 + `MicaAiException`
- 移除 `model-tools/` Python 工具链（模型已直接入库）
- 移除 `javax.annotation` 依赖，统一 ONNX 配置到 `mica-ai-common`