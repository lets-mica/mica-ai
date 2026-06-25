# AGENTS.md

> 给 AI 编码 Agent（Claude Code / Cursor / Copilot / TRAE 等）阅读的工作约定。
> 本文件描述 mica-ai 项目的协作规则，**Agent 在生成/修改代码前必须先通读本文件**。

---

## 1. 项目一句话

**mica-ai** 是面向 Java 生态的 AI 模型全家桶：把 Kokoro TTS / SenseVoice ASR / PP-OCRv6 / ERes2Net 声纹 / BERT 意图 / OpenCV Zoo 人脸 共 6 大能力，全部封装成 **零 Python、零 PyTorch、纯 ONNX Runtime** 的 Java 17+ SDK，并提供对应的 Spring Boot Starter。

- 主语言：Java 17+
- 构建：Maven（`pom.xml` 顶层 `${revision}=2026.06.01`）
- 推理运行时：[ONNX Runtime 1.26.0](https://onnxruntime.ai/)（CPU/CUDA/DML/CoreML）
- Java CV：[openpnp/opencv 4.9.0-0](https://github.com/openpnp/opencv)（自带跨平台原生库）
- Python 工具链：仅在 `model-tools/` 下，**不进 Java 运行时**
- 协议：Apache License 2.0

---

## 2. 仓库地图

```
mica-ai/
├── pom.xml                         # 顶层 BOM（revision / spring-boot / onnxruntime / opencv）
├── mica-ai-common/                 # ONNX Provider、统一异常、音频工具
├── mica-ai-core/                   # 核心引擎（零 Spring，纯 Java 17）
│   ├── mica-ai-ppocr/              # 📷 PP-OCRv6（det + rec）
│   ├── mica-ai-tts/                # 🎤 Kokoro-82M（可插拔 G2P）
│   ├── mica-ai-voice/              # 🎧 SenseVoice（多语种 + 热词雷达）
│   ├── mica-ai-speaker/            # 👤 ERes2Net（声纹 Embedding）
│   ├── mica-ai-intent/             # 🧠 BERT 中文意图分类
│   └── mica-ai-face/               │ 🎭 OpenCV Zoo（人脸检测 + 512d 向量，Apache-2.0）
├── mica-ai-starters/               # Spring Boot Starter（自动注入 Bean）
│   └── mica-ai-*-spring-boot-starter/
├── model-tools/                    # Python 端：download / convert / train
│   ├── common/                     # modelscope 下载器、onnx_utils、progress
│   ├── ppocr/  tts/  voice/  speaker/  intent/  face/
│   ├── scripts/smoke_test.py       # 离线冒烟测试
│   └── Makefile                    # make download / convert / train-intent
└── docs/
    ├── websocket实时识别.md
    └── 意图识别模型微调与ONNX导出.md
```

> **重要**：每个 `mica-ai-core/mica-ai-xxx/` 目录下都有自己的 `README.md`，Agent 在该能力内做修改时**先读**对应 README 的「模型规格 / 核心组件 / I/O 格式」三节。

---

## 3. 常用命令

### 3.1 Java 构建 / 测试

```bash
# 编译所有模块
mvn -q -DskipTests clean install

# 跑全部单元测试
mvn test

# 仅测某个能力
mvn -pl mica-ai-core/mica-ai-tts -am test

# 跑集成测试（会读 model 目录，模型缺失时跳过）
mvn -pl mica-ai-core/mica-ai-voice -am test -Dtest="*IntegrationTest"

# 启动一个 Starter 跑冒烟（需要在 starter 模块里加 Demo）
mvn -pl mica-ai-starters/mica-ai-tts-spring-boot-starter -am spring-boot:run
```

JDK：**17+**（推荐 Temurin / Azul Zulu 17）。Surefire 已配 `-Djdk.net.URLClassPath.disableClassPathURLCheck=true` + `forkCount=1`，规避 Windows 跨盘符 fork 问题。

### 3.2 模型工具链（Python）

```bash
# 离线冒烟（不下载任何模型，仅验证脚本骨架）
make -C model-tools smoke

# 下载所有能力的原始模型（默认走 ModelScope 国内镜像）
make -C model-tools download

# 转换 / 导出 ONNX
make -C model-tools convert

# 单能力
make -C model-tools download-voice
make -C model-tools convert-intent

# 训练意图分类
make -C model-tools train-intent
```

模型下载源可通过 `--source modelscope|huggingface` 切换。环境变量 `MICA_MODELS_DIR` 可改变模型根目录。

---

## 4. 编码约定

### 4.1 Java 风格

- **JDK 17 特性**：可放心使用 `record` / `sealed` / `var` / text block。
- **构建器模式**：所有引擎配置走 `XxxConfig.builder()...build()`，**不要**用 Lombok `@Builder` 构造 public 实体；`@Builder` 仅用于内部 DTO。
- **资源管理**：引擎主类实现 `AutoCloseable`，**统一 try-with-resources**，禁止 finalize。
- **日志**：仅用 SLF4J，禁 `System.out.println`（测试 main 例外）。
- **空值语义**：使用 [`org.jspecify`](https://jspecify.dev/) 注解，**默认非空**；参数允许 null 时显式标 `@Nullable`。
- **异常**：业务异常继承 `MicaAiException`（在 `mica-ai-common`），禁止直接抛 `RuntimeException`。
- **Lombok**：可使用 `@Getter / @Setter / @RequiredArgsConstructor / @Slf4j`，**避免** `@Data`（破坏 builder 语义）。
- **缩进 / 命名**：4 空格缩进，类名 `UpperCamelCase`，包名全小写（`net.dreamlu.mica.ai.<capability>`）。
- **Maven 坐标**：`net.dreamlu:mica-ai-<capability>`，版本用 `${revision}` 占位。
- **注释**：**默认不加任何代码注释**（项目根 README 与各子 README 是事实来源），除非被显式要求。

### 4.2 Python 风格

- 仅在 `model-tools/` 下使用，**不得**反向依赖 Java 模块。
- 公共工具放 `common/`；每个能力目录独立 `requirements.txt`。
- 模型下载统一走 `common/downloader.py`（基于 `modelscope`），不在脚本里散写 `urllib` / `requests`。
- ONNX 导出后必须做 **PyTorch vs ONNX 推理一致性**校验（参考 `model-tools/intent/convert.py`）。
- INT8 量化参数：weight=`QUInt8`，精度损失阈值 < 1%。

### 4.3 提交 / 分支

- 提交信息：`<scope>: <verb> <object>`，例如 `tts: support zf_010 voice`、`ppocr: fix rec nms threshold`。
- 一个 PR 一个能力，**不要**把多个能力的修改混在同一个 commit。
- 任何修改**都不要**主动 commit（用户没要求时严禁 `git commit`）。

---

## 5. 架构与设计原则

1. **零 Spring 依赖**：核心模块不引任何 `spring-*`，确保能在非 Spring 环境直接用。
2. **零 Python 进程**：Java 端不能 spawn Python / subprocess，模型全部在 JVM 内 ONNX 推理。
3. **可插拔**：关键组件走接口注入，例如 `KokoroTtsConfig.Builder#g2p(G2P)`、`G2P#phonemize(String)`。
4. **ONNX 一致性优先**：所有能力默认 CPU bit-exact，需要 GPU 时再切 `onnxruntime_gpu`。
5. **国内友好**：模型工具链默认 ModelScope；Spring Boot Starter 全部走 `@ConfigurationProperties(prefix = "mica.ai.<cap>")`。
6. **不改 BOM 不引新依赖**：新增能力 / 新增三方库时，**先在根 `pom.xml` 评审**，避免子模块 `pom.xml` 散落版本号。

---

## 6. ⚠️ 硬性约束（Agent 必须遵守）

### 6.1 🟥 依赖模型必须可商用（Non-negotiable）

> **mica-ai 的目标是"让 Java 工程师在商业产品里直接落地 AI 能力"，因此任何被本项目收录 / 推荐的 AI 模型（"依赖模型"）的许可证必须允许商业使用，禁止使用纯研究 / 教学 / 非商业许可证（如 `CC BY-NC-*`、`Research Use Only`、`NonCommercial`、`NoDerivatives` 等）。**

具体落实：

- **新增 / 替换能力依赖的模型**时（含预训练权重、词表、配置、ONNX 产物），**必须**先在 PR 描述里贴出：
  1. 模型原始仓库 / 权重托管地址
  2. 官方 LICENSE 全文或链接
  3. 一句话结论：「可商用 / 不可商用」
- **仅接受以下协议（或同等宽松）**：
  - Apache License 2.0
  - MIT / BSD / ISC
  - MulanPSL-2.0
  - 其它 OSI-approved、商业友好的协议
  - 自定义协议需附 `LICENSE-COMMERCIAL-COMPATIBLE.md` 说明，并经维护者书面确认
- **现有能力的依赖模型 License 现状**（仅作参考，License 变更时及时更新）：

  | 能力 | 模型 | License | 商用 |
  |------|------|---------|------|
  | `mica-ai-tts` | Kokoro-82M（hexgrad） | Apache 2.0 | ✅ |
  | `mica-ai-voice` | SenseVoiceSmall（FunASR） | Apache 2.0 | ✅ |
  | `mica-ai-ppocr` | PP-OCRv6（PaddleOCR / PaddleX） | Apache 2.0 | ✅ |
  | `mica-ai-speaker` | ERes2NetV2（speechbrain / 3D-Speaker） | Apache 2.0 | ✅ |
  | `mica-ai-intent` | chinese-bert-wwm-ext（HFL） | Apache 2.0 | ✅ |
  | `mica-ai-face` | YuNet + SFace（OpenCV Zoo） | Apache 2.0 | ✅ |

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
- **不要**修改顶层 `<revision>`、`spring.boot.version`、`onnxruntime.version`、`opencv.version`，除非被显式要求。
- **不要**提交 `model/`、`output/`、`*.onnx`、`*.bin`、`target/` 等产物；`.gitignore` 已配置。
- **不要**在 Java 端引 `torch*` / `paddle*` / `python*` 依赖，破坏「零 Python 进程」原则。
- **不要**直接修改 `pom.xml` 中 `mica-auto` 插件配置（它负责生成 `@AutoConfiguration.imports`）。
- **不要**把模型 / 训练数据 commit 到仓库（已通过 `.gitignore` 屏蔽，但 Agent 也勿尝试绕过）。
- **不要**主动 commit / push / merge 任何东西；改动完等用户确认。
- **不要**引入与「商业可用」冲突的依赖（含 GPL 类 Java 库）；如有疑虑，宁可不加。

---

## 7. 常见改动场景的"标准操作"

| 场景 | 必读文件 | 标准动作 |
|------|---------|---------|
| 新增一种 G2P | `mica-ai-core/mica-ai-tts/README.md` §4 | 实现 `G2P` 接口 → 在 `g2p/` 包内新建类 → `ChineseG2P.pinyinToBopomofo` 复用 |
| 新增一个 ONNX 输入节点 | 对应能力的 `XxxConfig` + `XxxEngine` | 在 `Config` Builder 加字段 → 在 Engine 读出 → 更新该能力 README 的「I/O 格式」 |
| 替换底层模型 | `model-tools/<cap>/download.py` + `convert.py` | 先按 §6.1 自检 License → 改 `MODEL_*` 常量 → 重跑 smoke test → 更新 README 模型规格表 |
| 新增 Spring Boot 配置项 | Starter `*Properties.java` + `*AutoConfiguration.java` | 用 `mica-auto` 生成 import → 跑 `mvn install` → 在 `mica-ai-core/<cap>/README.md` 加示例 |
| 新增 Python 子能力 | `model-tools/<cap>/` | 复制 `intent/` 模板（最完整）→ 写 `download.py`/`convert.py` → 在顶层 `Makefile` 加 cap |
| 性能调优 | 对应能力 `XxxEngine` + `XxxConfig` | 优先调整 `intraOpNumThreads` / `interOpNumThreads` / `onnxProvider` |

---

## 8. 验证清单（改完跑一遍）

- [ ] `mvn -q -DskipTests clean install` 通过
- [ ] 受影响模块的 `mvn -pl <module> -am test` 通过
- [ ] 若改了 Python：`make -C model-tools smoke` 通过
- [ ] 若改了模型脚本：跑对应 `download.py` + `convert.py`，确认产物可被 Java 端加载
- [ ] 若改了 Starter：在 `application.yml` 加一段示例，且至少 1 个 `@Autowired` 使用点
- [ ] 若改了 README：用 `markdownlint` / IDE 自检（标题层级、代码块语言、链接）
- [ ] 若新增 / 替换模型：**§6.1 商用自检清单全部勾选**
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
