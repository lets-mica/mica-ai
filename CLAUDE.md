# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **必读**：[`AGENTS.md`](AGENTS.md) 描述了本项目对 AI 编码 Agent 的硬性约束（依赖模型商用协议、零 Python 进程、零 Spring 依赖、提交纪律、硬性约束清单、验证清单等）。在生成或修改任何代码前先通读。
> 模块细节（模型规格 / I/O 格式 / 关键组件）见各子模块 `README.md`：`mica-ai-core/mica-ai-*/README.md` 与 `mica-ai-starters/mica-ai-*-spring-boot-starter/README.md`。
> 用户面向的快速开始、依赖坐标、应用场景见根 [`README.md`](README.md)。

---

## 1. 项目一句话

**mica-ai** = Java 17 + ONNX Runtime 全家桶，把 6 大 AI 能力（TTS / ASR / OCR / 声纹 / 意图 / 人脸）封装成 **零 Python、零 PyTorch** 的 SDK + Spring Boot Starter。模型推理完全在 JVM 内完成，所有运行时依赖 Maven 拉取（`onnxruntime` 1.26.0、`opencv` 4.9.0-0）。Python 仅出现在 `model-tools/`，负责下载/转换/训练，不进 Java 运行时。

包前缀：`net.dreamlu.mica.ai.<capability>`；顶层 Maven `${revision}=2026.06.01-SNAPSHOT`。

## 2. 仓库地图

```
mica-ai/
├── pom.xml                    # 顶层 BOM（revision / spring-boot / onnxruntime / opencv）
├── mica-ai-common/            # ONNX Provider、统一异常、音频工具
├── mica-ai-core/              # 核心引擎（零 Spring，纯 Java 17）
│   ├── mica-ai-ppocr/         #   📷 PP-OCRv6 (det + rec)
│   ├── mica-ai-tts/           #   🎤 Kokoro-82M (可插拔 G2P)
│   ├── mica-ai-voice/         #   🎧 SenseVoice (Trie 树热词雷达)
│   ├── mica-ai-speaker/       #   👤 ERes2Net (256d Embedding)
│   ├── mica-ai-intent/        #   🧠 BERT 中文意图分类
│   └── mica-ai-face/          #   🎭 OpenCV Zoo (YuNet + SFace, 512d)
├── mica-ai-starters/          # Spring Boot Starter（`mica.ai.<cap>` 前缀 + 自动注入）
├── mica-ai-example/           # Spring Boot 集成示例（默认在 `develop` profile 下编译）
├── model-tools/               # Python 工具链：download / convert / train
└── docs/                      # 方案文档（WebSocket 实时识别、意图微调等）
```

每个 `mica-ai-core/mica-ai-*/` 子模块的 README 都有"模型规格 / 核心组件 / I/O 格式"三节 — 在该能力内做修改前**先读**对应 README。

## 3. 常用命令

### 3.1 Java 构建 / 测试

```bash
# 编译所有模块
mvn -q -DskipTests clean install

# 跑全部单元测试
mvn test

# 仅测某个能力（含依赖模块）
mvn -pl mica-ai-core/mica-ai-tts -am test

# 跑某个测试类
mvn -pl mica-ai-core/mica-ai-voice -am test -Dtest=ClassName

# 跑集成测试（缺失模型时通常会跳过）
mvn -pl mica-ai-core/mica-ai-voice -am test -Dtest="*IntegrationTest"

# 启动某个 Starter 的 Demo
mvn -pl mica-ai-starters/mica-ai-tts-spring-boot-starter -am spring-boot:run
```

JDK 17+（推荐 Temurin / Azul Zulu 17）。Surefire 已配 `-Djdk.net.URLClassPath.disableClassPathURLCheck=true` + `forkCount=0`，规避 Windows 跨盘符 fork 报错。

### 3.2 模型工具链（Python）

```bash
# 离线冒烟（不下载模型）
make -C model-tools smoke

# 下载所有能力的原始模型（默认 ModelScope 国内镜像）
make -C model-tools download

# 转换 / 导出 ONNX
make -C model-tools convert

# 单能力
make -C model-tools download-voice
make -C model-tools convert-intent

# 训练意图分类
make -C model-tools train-intent
```

下载源可用 `--source modelscope|huggingface` 切换；`MICA_MODELS_DIR` 环境变量改变模型根目录。

## 4. 架构与设计原则（核心要点）

1. **零 Spring 依赖**：核心模块不引任何 `spring-*`，非 Spring 环境直接可用。Spring Boot Starter 是**独立**的薄壳层（`mica-ai-starters/`）。
2. **零 Python 进程**：Java 端不能 spawn Python / subprocess；所有推理走 `onnxruntime`。GPU 切到 `onnxruntime_gpu` + `onnxProvider=cuda/dml`。
3. **Builder + try-with-resources**：所有引擎 `XxxConfig.builder()...build()`，主类实现 `AutoCloseable`。
4. **可插拔**：关键组件走接口注入，例如 `KokoroTtsConfig.Builder#g2p(G2P)`、`G2P#phonemize(String)`。
5. **公共契约**（在 `mica-ai-common`）：
   - `MicaAiException` — 统一业务异常
   - `OrtProviders` — ONNX Provider 管理
   - `AudioUtils` — 音频工具
6. **Spring Boot Starter 约定**：每个 Starter 提供 `XxxProperties`（`@ConfigurationProperties(prefix = "mica.ai.<cap>")`）+ `XxxAutoConfiguration`（`@AutoConfiguration`），自动装配由 `mica-auto` 插件生成 `imports` 文件，**不要**手改。
7. **依赖收口**：版本号统一在根 `pom.xml` 的 `dependencyManagement`；新增能力/三方库先在根 POM 评审，避免子模块散落版本号。**不要**改 `<revision>` / `spring.boot.version` / `onnxruntime.version` / `opencv.version`（除非被显式要求）。

## 5. 编码约定要点

- JDK 17：`record` / `sealed` / `var` / text block 可放心用。
- **不要**用 Lombok `@Builder` 构造 public 实体；`@Builder` 仅限内部 DTO。所有公开配置走手写 `XxxConfig.builder()...build()`。
- Lombok 可用：`@Getter / @Setter / @RequiredArgsConstructor / @Slf4j`；**避免** `@Data`（破坏 builder 语义）。
- 日志只用 SLF4J（测试 `main` 例外）。
- 空值语义：用 `org.jspecify`，**默认非空**，允许 null 时显式 `@Nullable`。
- 业务异常继承 `MicaAiException`，**禁止**直接 `throw new RuntimeException(...)`。
- **默认不加任何代码注释**（根 README 与子 README 是事实来源），除非被显式要求。
- 提交信息：`<scope>: <verb> <object>`（例如 `tts: support zf_010 voice`）；一个 PR 一个能力。
- **不要**主动 `git commit` / `push` / `merge`，等用户确认。

## 6. ⚠️ 硬性约束（速览，完整版见 `AGENTS.md` §6）

- **依赖模型必须可商用**：只接受 Apache-2.0 / MIT / BSD / ISC / MulanPSL-2.0 等商业友好协议；禁止 `CC BY-NC-*` / `Research Use Only` / `NonCommercial` / `NoDerivatives` / GPL / AGPL / LGPL。新增/替换模型必须在 PR 描述里贴 LICENSE 全文并明确「可商用 / 不可商用」结论。Agent 自检清单见 `AGENTS.md` §6.1。
- **不要**在 Java 端引 `torch*` / `paddle*` / `python*` 依赖，破坏「零 Python 进程」原则。
- **不要**修改 `mica-auto` 插件配置。
- **不要** commit `model/`、`output/`、`*.onnx`、`*.bin`、`target/`（`.gitignore` 已屏蔽）。
- **不要**创建无意义文件（无显式要求时不新建 `*.md`、空目录、占位脚本）。
- **不要**引入与「商业可用」冲突的 Java 库。

## 7. 常见改动场景的"标准操作"

| 场景 | 必读 | 标准动作 |
|------|------|---------|
| 新增 G2P | `mica-ai-tts/README.md` §4 | 实现 `G2P` 接口 → `g2p/` 包新建类 → 复用 `ChineseG2P.pinyinToBopomofo` |
| 新增 ONNX 输入节点 | 对应能力 `XxxConfig` + `XxxEngine` | `Config` Builder 加字段 → Engine 读出 → 更新该能力 README「I/O 格式」节 |
| 替换底层模型 | `model-tools/<cap>/download.py` + `convert.py` | 先按 §6 自检 License → 改 `MODEL_*` 常量 → 重跑 `make smoke` → 更新 README 模型规格表 |
| 新增 Spring Boot 配置项 | Starter `XxxProperties` + `XxxAutoConfiguration` | 用 `mica-auto` 生成 import → `mvn install` → `mica-ai-core/<cap>/README.md` 加示例 |
| 新增 Python 子能力 | `model-tools/<cap>/` | 复制 `intent/` 模板 → 写 `download.py`/`convert.py` → 顶层 `Makefile` 加 cap |
| 性能调优 | 对应能力 `XxxEngine` + `XxxConfig` | 优先调 `intraOpNumThreads` / `interOpNumThreads` / `onnxProvider` |

## 8. 验证清单（改完跑一遍）

- [ ] `mvn -q -DskipTests clean install` 通过
- [ ] 受影响模块 `mvn -pl <module> -am test` 通过
- [ ] 改了 Python：`make -C model-tools smoke` 通过
- [ ] 改了模型脚本：跑对应 `download.py` + `convert.py`，产物可被 Java 端加载
- [ ] 改了 Starter：`application.yml` 加示例 + 至少 1 个 `@Autowired` 使用点
- [ ] 改了 README：标题层级、代码块语言、链接自检
- [ ] 新增/替换模型：§6 商用自检清单全部勾选
- [ ] 未触发 §6 硬性约束任何一条

## 9. 给 Agent 的额外提示

- **回复语言**：与用户最新消息保持一致（默认中文）。代码 / 注释 / 标识符一律英文；中文文档与根 `README.md` 风格保持一致。
- **不要解释基础概念**（如"什么是 ONNX"），用户是资深 Java 工程师。
- **少而准**：能用一段代码说清就别写长篇说明；优先给「最小可运行示例 + 关键配置项」。
- **出错时优先复现**：先写一个失败测试 / 复现脚本，再改实现。
- **遇到不确定的模型 / 协议**（尤其是「依赖模型必须可商用」这条），**先停下问**，别凭印象判断。
