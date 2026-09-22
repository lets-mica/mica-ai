# AGENTS.md

> 给 AI 编码 Agent（Claude Code / Cursor / Copilot / TRAE 等）阅读的工作约定。
> 本文件描述 mica-ai 项目的协作规则，**Agent 在生成/修改代码前必须先通读本文件**。

---

## 1. 项目一句话

**mica-ai**（精简版）提供 **OpenCV Zoo 人脸识别**（YuNet 检测 + SFace 128d 向量 + MiniFASNetV2 活体）+ 头像/证件卡片提取 + **Google Magika 文件类型识别**（standard_v3_3 214 类）+ **HyperLPR3 中国车牌识别** + **PP-DocLayoutV3 文档版面分析**（含阅读顺序）+ **U²-Net 族通用抠图**（`u2netp` 入库，`u2net` / `u2net_human_seg` 可外置切换）+ **PP-LCNet 文本行方向分类**（0° / 180°，`x1_0` 入库，`x0_25` 可外置切换）能力，封装成 **零 Python、零 PyTorch、纯 ONNX Runtime** 的 Java 8+ SDK，并提供对应的 Spring Boot Starter。

音频（TTS / ASR / 声纹）和 OCR / 意图识别能力已抽离到独立的 mica-* 项目，本仓库不再包含。

- 主语言：Java 8+
- 构建：Maven（`pom.xml` 顶层 `${revision}=1.0.0`）
- 推理运行时：[ONNX Runtime 1.18.0](https://onnxruntime.ai/)（CPU/CUDA）
- OpenCV：[openpnp/opencv 4.9.0](https://github.com/openpnp/openpnp-vision)（封装原生库，跨平台）
- Spring Boot：2.7.x
- 协议：Apache License 2.0

---

## 2. 仓库地图

```
mica-ai/
├── pom.xml                         # 顶层 BOM（revision / spring-boot / onnxruntime）
├── mica-ai-common/                 # ONNX 通用基础设施（OnnxModelSession / OrtSessionFactory / 统一异常）
├── mica-ai-core/                   # 核心引擎（零 Spring，纯 Java 8）
│   ├── mica-ai-face/               # 🎭 OpenCV Zoo（人脸检测 + 128d 向量 + 活体 + 头像 + 卡片，Apache-2.0）
│   ├── mica-ai-filetype/           # 📄 Google Magika（214 类文件类型识别，Apache-2.0）
│   ├── mica-ai-plate/              # 🚗 HyperLPR3（中国车牌检测 + 识别 + 判型，Apache-2.0）
│   ├── mica-ai-layout/             # 📐 PP-DocLayoutV3（文档版面分析 + 阅读顺序，Apache-2.0）
│   ├── mica-ai-matting/            # ✂️ U²-Net 族（通用抠图，Apache-2.0）
│   └── mica-ai-textline/           # 🔤 PP-LCNet（文本行 0°/180° 方向分类 + 转正，Apache-2.0）
├── mica-ai-starters/               # Spring Boot Starter（自动注入 Bean）
│   ├── mica-ai-face-spring-boot-starter/
│   ├── mica-ai-filetype-spring-boot-starter/
│   ├── mica-ai-plate-spring-boot-starter/
│   ├── mica-ai-layout-spring-boot-starter/
│   ├── mica-ai-matting-spring-boot-starter/
│   └── mica-ai-textline-spring-boot-starter/
├── mica-ai-example/                # Spring Boot 集成示例
├── docs/                           # 落地跟踪文档（模块稳定后并入 README）
└── model-tools/                    # 模型资产（随仓库分发，单文件上限见 §6.2）
    ├── face/models/                #   YuNet + SFace（Apache-2.0）
    ├── filetype/models/            #   Magika standard_v3_3（Apache-2.0）
    ├── plate/models/               #   HyperLPR3 v20230229（Apache-2.0）
    ├── layout/models/              #   PP-DocLayoutV3（Apache-2.0，⚠️ 125MB 不随仓库分发）
    ├── matting/models/             #   U²-Net u2netp（Apache-2.0，4.36MB；u2net / u2net_human_seg 168MB 不随仓库分发）
    └── textline/models/            #   PP-LCNet_x1_0_textline_ori（Apache-2.0，6.46MB；x0_25 轻量版不随仓库分发）
```

> **重要**：
> - `mica-ai-core/mica-ai-<cap>/README.md` 是各能力"模型规格 / 核心组件 / I/O 格式"的事实来源
>   （`face` / `filetype` / `plate` / `layout` / `matting` / `textline` 各一份）
>
> Agent 在对应能力内做修改前**先读**对应 README。

---

## 3. 常用命令

### 3.1 Java 构建 / 测试

```bash
mvn -q -DskipTests clean install   # 编译所有模块
mvn test                            # 跑全部单元测试
mvn -pl mica-ai-core/mica-ai-face -am test    # 仅测 face
```

JDK：**8+**（推荐 Temurin / Azul Zulu 8/11/17；源码兼容到 JDK 17）。Surefire 已配 `-Djdk.net.URLClassPath.disableClassPathURLCheck=true` + `forkCount=0`，规避 Windows 跨盘符 fork 问题。

### 3.2 模型资产

模型已直接入库（各能力子目录 `models/`），**无需下载 / 转换脚本**；替换模型时直接覆盖对应文件。

模型完整性由**各能力模块的集成测试**覆盖（加载真 ONNX 跑一遍完整推理，兼作 I/O 契约回归）：

```bash
mvn -pl mica-ai-core/mica-ai-layout -am test    # 以 layout 为例，其它能力同理
```

> ⚠️ 历史上的 `make -C model-tools smoke` / `model-tools/scripts/smoke_test.py` 已在提交
> `a0a8a6f refactor: 删除 model-tools Python 工具链（模型已直接入库）` 中删除。
> **该命令不存在，不要再去找，也不要再写进任何文档。**
>
> 例外：`model-tools/layout/models/model.onnx`（125MB）**不随仓库分发**，其
> `LayoutIntegrationTest` 在模型缺失时用 `Assumptions` 整类跳过，不阻塞新克隆仓库的 `mvn test`。

---

## 4. 编码约定

- **JDK 8 兼容**：不依赖 JDK 9+ 语法，**不要**使用 `var`（保留给业务代码）、text block、records、sealed 等。
  - 源码风格保持 JDK 8 兼容：`new ArrayList<>()`、`@Override`、普通 for 循环，避免 stream 过度抽象。
- **构建器模式**：所有引擎配置走 `FaceConfig.builder()...build()` 风格的 builder；Lombok `@Builder(toBuilder=true)` 仅用于 immutable 参数对象（`AvatarOptions` / `CardOptions`），不用于可变 Java Bean。
- **资源管理**：引擎主类实现 `AutoCloseable`，**统一 try-with-resources**，禁止 finalize。
- **日志**：仅用 SLF4J（`@Slf4j`），禁 `System.out.println`（测试 main 例外）。
- **空值语义**：使用 `@Nullable` 显式标注可空参数；Java 8 无 `org.jspecify`，按需引入 `javax.annotation.Nullable` 或 `org.springframework.lang.Nullable`。
- **异常**：业务异常继承 `MicaAiException`（在 `mica-ai-common`），禁止直接抛 `RuntimeException`。
- **Lombok**：可使用 `@Getter / @Setter / @RequiredArgsConstructor / @Slf4j / @Data / @Builder`（视场景选择）。
- **缩进 / 命名**：4 空格缩进，类名 `UpperCamelCase`，包名全小写（`net.dreamlu.mica.ai.face`）。
- **Maven 坐标**：`net.dreamlu:mica-ai-face`，版本用 `${revision}` 占位。
- **注释**：**默认不加任何代码注释**（项目根 README 与各子 README 是事实来源），除非被显式要求。
- 提交信息：`<scope>: <verb> <object>`，例如 `face: tighten det score threshold`。
- 一个 PR 一个能力；**不要**主动 `git commit` / `push` / `merge`。

---

## 5. 架构与设计原则

1. **零 Spring 依赖**：核心模块不引任何 `spring-*`，确保能在非 Spring 环境直接用。
2. **零 Python 进程**：Java 端不能 spawn Python / subprocess，模型全部在 JVM 内 ONNX 推理。
3. **ONNX 一致性优先**：默认 CPU bit-exact，需要 GPU 时再切 `onnxruntime_gpu`。
4. **国内友好**：模型工具链默认 ModelScope；Spring Boot Starter 全部走 `@ConfigurationProperties(prefix = "mica.ai.face")`。
5. **不改 BOM 不引新依赖**：新增能力 / 新增三方库时，**先在根 `pom.xml` 评审**，避免子模块 `pom.xml` 散落版本号。

---

## 6. ⚠️ 硬性约束（Agent 必须遵守）

### 6.1 🟥 依赖模型必须可商用（Non-negotiable）

> **mica-ai 的目标是"让 Java 工程师在商业产品里直接落地 AI 能力"，因此任何被本项目收录 / 推荐的 AI 模型（"依赖模型"）的许可证必须允许商业使用，禁止使用纯研究 / 教学 / 非商业许可证（如 `CC BY-NC-*`、`Research Use Only`、`NonCommercial`、`NoDerivatives` 等）。**

- **新增 / 替换能力依赖的模型**时（含预训练权重、词表、配置、ONNX 产物），**必须**先在 PR 描述里贴出：
  1. 模型原始仓库 / 权重托管地址
  2. 官方 LICENSE 全文或链接
  3. 一句话结论：「可商用 / 不可商用」
- **仅接受以下协议（或同等宽松）**：Apache License 2.0 / MIT / BSD / ISC / MulanPSL-2.0 / 其它 OSI-approved 商业友好的协议。
- **当前能力的依赖模型 License**：

  | 能力 | 模型 | License | 商用 |
  |------|------|---------|------|
  | `mica-ai-face` 检测 / 特征 | YuNet + SFace（OpenCV Zoo） | Apache 2.0 | ✅ |
  | `mica-ai-face` 活体 | MiniFASNetV2（minivision Silent-Face-Anti-Spoofing） | MIT | ✅ |
  | `mica-ai-filetype` 检测 | Google Magika `standard_v3_3`（model.onnx + config.min.json + content_types_kb.min.json） | Apache 2.0 | ✅ |
  | `mica-ai-plate` 检测 / 识别 / 判型 | HyperLPR3 v20230229（`y5fu_*` 检测 + `rpv3_mdict_*` 识别 + `litemodel_cls_*` 判型） | Apache 2.0 | ✅ |
  | `mica-ai-layout` 版面检测 | PP-DocLayoutV3（PaddleOCR，paddle2onnx 转换产物） | Apache 2.0 | ✅ |
  | `mica-ai-matting` 抠图 | U²-Net 族 `u2netp`（入库 4.36MB）/ `u2net` / `u2net_human_seg`（外置 168MB）（xuebinqin/U-2-Net，danielgatis/rembg 打包 ONNX），三者 I/O 契约一致 | Apache 2.0 | ✅ |
  | `mica-ai-textline` 方向分类 | PP-LCNet_x1_0_textline_ori（入库 6.46MB）/ PP-LCNet_x0_25_textline_ori（外置 ~0.96MB）（PaddlePaddle/PaddleX，paddle2onnx 转换产物），两者 I/O 契约一致 | Apache 2.0 | ✅ |

- **禁止**：
  - 直接搬运 GPL / AGPL / LGPL 模型权重并以"商用"名义打包
  - 在 README / 文档里推荐带 `NC`（NonCommercial）字样的权重
    （已明确排除：`u2net_portrait`（APDrawing 数据集 NC）、YOLOv5/v8/v11 全系（GPL/AGPL）、`RMBG-1.4`（需授权）、各类 NSFW 模型）
  - 把"我可以自己用"当作"我可以让别人商用"的依据 —— 默认按"允许下游商用"的口径来评估
- **Agent 自检清单**（在替换 `model-tools/<cap>/models/` 下任何模型 / 写 README 之前过一遍）：
  - [ ] 我要引入 / 推荐的模型 License 是什么？
  - [ ] 该 License 是否允许「商业使用」「分发」「修改」？
  - [ ] 是否需要在 README / NOTICE 里保留署名 / 来源声明？
  - [ ] 是否有 `MODEL_LICENSE` 之类的二级文件需要一并随模型分发？

### 6.2 其它硬性约束

- **不要创建无意义文件**：没有用户显式要求时，不要新建 `*.md`、空目录、占位脚本。
- **不要**提交 `model/` / `output/` / `*.bin` / `target/` 等产物；`.gitignore` 已配置。**例外**：`model-tools/<cap>/models/` 下的入库模型按 owner 要求随仓库分发，**单文件须 <50MB**。
- **超出 50MB 的模型不入库**：当前唯一例外是 `model-tools/layout/models/model.onnx`（125MB，PP-DocLayoutV3），已在根 `.gitignore` 显式排除。新增超限模型沿用同一套配套措施：对应模块的集成测试在文件缺失时整类跳过、example 里该能力 `enabled: false`、模块 README 记录获取方式。**不要**擅自引入 Git LFS。
- **不要**在 Java 端引 `torch*` / `paddle*` / `python*` 依赖，破坏「零 Python 进程」原则。
- **不要**直接修改 `pom.xml` 中 `mica-auto` 插件配置（它负责生成 `@AutoConfiguration.imports`）。
- **不要**把训练数据 commit 到仓库；**推理模型**仅在 `model-tools/<cap>/models/`（单文件 <50MB，超限者不入库）随仓库分发，其它位置禁止。
- **不要**主动 commit / push / merge 任何东西；改动完等用户确认。
- **不要**引入与「商业可用」冲突的依赖（含 GPL 类 Java 库）；如有疑虑，宁可不加。

> 历史记录：2026-06-01 起 mica-ai 由 Java 17 + Spring Boot 4.1 降级为 Java 8 + Spring Boot 2.7，以贴近 mica-face 生态。降级会破坏此前 Java 17 / Spring Boot 4 的兼容性，需经项目 owner 显式批准。

---

## 7. 常见改动场景的"标准操作"

| 场景 | 必读 | 标准动作 |
|------|------|---------|
| 新增一个 ONNX 输入节点 | `mica-ai-face/.../FaceDetector.java` 或 `FeatureExtractor.java` 或 `LivenessDetector.java` | 在检测器 / 提取器 / 活体类的推理段加常量 → 更新 mica-ai-face/README.md「I/O 格式」 |
| 替换底层模型 | `model-tools/<cap>/models/` | 先按 §6.1 自检 License → 覆盖模型文件 → 跑对应能力模块的集成测试（`mvn -pl mica-ai-core/<cap> -am test`）确认可加载且输出合理 → 更新对应 README 模型规格表 |
| **切换同族模型（不改代码）** | 该能力 README 的「模型规格 / 可插拔」一节 | 若新模型与现有模型**契约一致**（输入名/形状、输出个数与语义、归一化、取值域），则只改配置的 `model-path` 即可，**不要改 Java 代码**；契约有任何差异时，先把差异显式化为配置项（参考 `mica-ai-matting` 的 `output-select` / `input-size` / `mean` / `std`），再切换。**超 50MB 模型不入库**（§6.2），由使用者配 `model-path` 外置 |
| 用外置模型跑集成测试 | 该能力集成测试的 `externalModel` 系统属性 | `mvn -pl mica-ai-core/<cap> -am test -Dmica.ai.<cap>.externalModel=<绝对路径>`；不传时该项测试自动跳过（外置模型不入库，不能假设存在） |
| 判断权重语义（不是看文件名） | 该能力 README 的实测章节 | **文件哈希 / 文件名不能作为行为同源的证据**（`mica-ai-matting` 已踩过：`u2netp` 与 HF 上名为 `U-2-Net-Human-Seg` 的 onnx 哈希逐字节相同，行为却贴近 `u2net`）。必须用行为对照脚本（如 `model-tools/matting/scripts/probe_model_identity.py`）跑同一组图片比 MAE，再下结论 |
| 新增 Spring Boot 配置项 | Starter `FaceProperties.java` / `FiletypeProperties.java` + 对应 `*AutoConfiguration.java` | 用 `mica-auto` 生成 import → 跑 `mvn install` → 子 README 加示例 |
| 性能调优 | `mica-ai-face/onnx/OrtSessionFactory.java` + `mica-ai-face/onnx/OrtSessionOptions.java` | 优先调整 `intraOpNumThreads` / `interOpNumThreads` / `device` (cpu/gpu) |
| 新增头像提取参数 | `mica-ai-face/avatar/AvatarOptions.java` | 在 Builder 加字段 → `AvatarOptions.validate()` 加范围校验 → Starter `FaceProperties.Avatar` 同步 → mica-ai-face/README.md「头像提取」一节 |
| 新增分类/方向能力（文本行方向这类 N 类分类） | `mica-ai-textline/detection/TextLineDetector.java` | **先拿到官方 `inference.yml` 确认 `label_list` 顺序与预处理契约，不要凭 ONNX 输出维度猜类别语义**（写反 = 正的判成倒的）→ 校验「配置的输入宽高 == 模型形状」并快速失败 → 复核 `Imgproc.resize` 的 `Size` 是 **(宽, 高)** → 集成测试断言类别语义 + 真实数据置信度区间 |

---

## 8. 验证清单（改完跑一遍）

- [ ] `mvn -DskipTests install` 通过
- [ ] `mvn test` 全绿（face / filetype / plate / layout / matting / textline / example 七个模块）
- [ ] 替换了模型：对应能力模块的**集成测试**通过（加载真 ONNX 跑一遍完整推理）
- [ ] 新增能力：模块 README + Starter + `model-tools/<cap>/README.md` 三处齐全，且能力清单（根 README / 本文件 §1 §2 §6.1）同步
- [ ] 改了 Starter：在 `application.yml` 加示例，且至少 1 个 `@Autowired` 使用点
- [ ] 改了 README：标题层级、代码块语言、链接自检
- [ ] 新增/替换模型：**§6.1 商用自检清单全部勾选**
- [ ] 未触发「§6.2 硬性约束」任何一条

---

## 9. 给 Agent 的额外提示

- **回复语言**：与用户最新消息保持一致（默认中文）。代码 / 注释 / 标识符一律英文，文档若用中文则与 `README.md` 风格保持一致。
- **不要解释基础概念**（如"什么是 ONNX"），用户是资深 Java 工程师。
- **少而准**：能用一段代码说清就别写长篇说明；优先给「最小可运行示例 + 关键配置项」。
- **出错时优先复现**：先写一个失败测试 / 复现脚本，再改实现。
- **遇到不确定的模型 / 协议**（尤其是「依赖模型必须可商用」这条），**先停下问**，别凭印象判断。

---

> 最后更新：随本仓库 `CHANGELOG.md` 一同维护；任何破坏「可商用 / 零 Python / 零 Spring」三条原则之一的改动，需在 PR 描述里显式声明。
