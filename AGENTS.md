# AGENTS.md

> 给 AI 编码 Agent（Claude Code / Cursor / Copilot / TRAE 等）阅读的工作约定。
> 本文件描述 mica-ai 项目的协作规则，**Agent 在生成/修改代码前必须先通读本文件**。

---

## 1. 项目一句话

**mica-ai**（精简版）提供 **OpenCV Zoo 人脸识别**（YuNet 检测 + SFace 128d 向量 + MiniFASNetV2 活体）+ 头像/证件卡片提取能力，封装成 **零 Python、零 PyTorch、纯 ONNX Runtime** 的 Java 8+ SDK，并提供对应的 Spring Boot Starter。

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
│   └── mica-ai-face/               # 🎭 OpenCV Zoo（人脸检测 + 128d 向量 + 活体 + 头像 + 卡片，Apache-2.0）
├── mica-ai-starters/               # Spring Boot Starter（自动注入 Bean）
│   └── mica-ai-face-spring-boot-starter/
├── mica-ai-example/                # Spring Boot 集成示例
├── model-tools/                    # Python 端：download / convert / smoke
└── CHANGELOG.md
```

> **重要**：`mica-ai-core/mica-ai-face/README.md` 是该能力的"模型规格 / 核心组件 / I/O 格式"事实来源，Agent 在该能力内做修改前**先读**。

---

## 3. 常用命令

### 3.1 Java 构建 / 测试

```bash
mvn -q -DskipTests clean install   # 编译所有模块
mvn test                            # 跑全部单元测试
mvn -pl mica-ai-core/mica-ai-face -am test    # 仅测 face
```

JDK：**8+**（推荐 Temurin / Azul Zulu 8/11/17；源码兼容到 JDK 17）。Surefire 已配 `-Djdk.net.URLClassPath.disableClassPathURLCheck=true` + `forkCount=0`，规避 Windows 跨盘符 fork 问题。

### 3.2 模型工具链（Python）

```bash
make -C model-tools smoke       # 离线冒烟（不下载任何模型）
make -C model-tools download    # 下载 face 模型（默认 ModelScope）
make -C model-tools convert     # 把原始模型转换为 ONNX 产物
make -C model-tools publish     # 整理 out/ → models/ 并生成 manifest
make -C model-tools package     # 把 models/face 打包成 zip（用于 GitHub Release）
```

下载源可用 `--source modelscope|huggingface` 切换；`MICA_MODELS_DIR` 环境变量改变模型根目录。

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

- **禁止**：
  - 直接搬运 GPL / AGPL / LGPL 模型权重并以"商用"名义打包
  - 在 README / 文档里推荐带 `NC`（NonCommercial）字样的权重
  - 把"我可以自己用"当作"我可以让别人商用"的依据 —— 默认按"允许下游商用"的口径来评估
- **Agent 自检清单**（在写任何 `download.py` / `convert.py` / README 之前过一遍）：
  - [ ] 我要引入 / 推荐的模型 License 是什么？
  - [ ] 该 License 是否允许「商业使用」「分发」「修改」？
  - [ ] 是否需要在 README / NOTICE 里保留署名 / 来源声明？
  - [ ] 是否有 `MODEL_LICENSE` 之类的二级文件需要一并随模型分发？

### 6.2 其它硬性约束

- **不要创建无意义文件**：没有用户显式要求时，不要新建 `*.md`、空目录、占位脚本。
- **不要**提交 `model/` / `output/` / `*.onnx` / `*.bin` / `target/` 等产物；`.gitignore` 已配置。
- **不要**在 Java 端引 `torch*` / `paddle*` / `python*` 依赖，破坏「零 Python 进程」原则。
- **不要**直接修改 `pom.xml` 中 `mica-auto` 插件配置（它负责生成 `@AutoConfiguration.imports`）。
- **不要**把模型 / 训练数据 commit 到仓库。
- **不要**主动 commit / push / merge 任何东西；改动完等用户确认。
- **不要**引入与「商业可用」冲突的依赖（含 GPL 类 Java 库）；如有疑虑，宁可不加。

> 历史记录：2026-06-01 起 mica-ai 由 Java 17 + Spring Boot 4.1 降级为 Java 8 + Spring Boot 2.7，以贴近 mica-face 生态。降级会破坏此前 Java 17 / Spring Boot 4 的兼容性，需经项目 owner 显式批准。

---

## 7. 常见改动场景的"标准操作"

| 场景 | 必读 | 标准动作 |
|------|------|---------|
| 新增一个 ONNX 输入节点 | `mica-ai-face/.../FaceDetector.java` 或 `FeatureExtractor.java` 或 `LivenessDetector.java` | 在检测器 / 提取器 / 活体类的推理段加常量 → 更新 mica-ai-face/README.md「I/O 格式」 |
| 替换底层模型 | `model-tools/face/download.py` + `convert.py` | 先按 §6.1 自检 License → 改 `MODEL_*` 常量 → 重跑 smoke test → 更新 README 模型规格表 |
| 新增 Spring Boot 配置项 | Starter `FaceProperties.java` + `FaceAutoConfiguration.java` | 用 `mica-auto` 生成 import → 跑 `mvn install` → mica-ai-face/README.md 加示例 |
| 性能调优 | `mica-ai-face/onnx/OrtSessionFactory.java` + `mica-ai-face/onnx/OrtSessionOptions.java` | 优先调整 `intraOpNumThreads` / `interOpNumThreads` / `device` (cpu/gpu) |
| 新增头像提取参数 | `mica-ai-face/avatar/AvatarOptions.java` | 在 Builder 加字段 → `AvatarOptions.validate()` 加范围校验 → Starter `FaceProperties.Avatar` 同步 → mica-ai-face/README.md「头像提取」一节 |

---

## 8. 验证清单（改完跑一遍）

- [ ] `mvn -DskipTests install` 通过
- [ ] `mvn test` 通过（14 个用例：face 12 + example 2）
- [ ] `make -C model-tools smoke` 通过
- [ ] 改了模型脚本：跑 `download.py` + `convert.py`，产物可被 Java 端加载
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
