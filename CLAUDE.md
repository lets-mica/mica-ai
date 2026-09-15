# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **必读**：[`AGENTS.md`](AGENTS.md) 描述了本项目对 AI 编码 Agent 的硬性约束（依赖模型商用协议、零 Python 进程、零 Spring 依赖、提交纪律、硬性约束清单、验证清单等）。在生成或修改任何代码前先通读。
> 模块细节（模型规格 / I/O 格式 / 关键组件）见 `mica-ai-core/mica-ai-face/README.md` 与 `mica-ai-starters/mica-ai-face-spring-boot-starter/README.md`。
> 用户面向的快速开始、依赖坐标、应用场景见根 [`README.md`](README.md)。

---

## 1. 项目一句话

**mica-ai**（精简版）= Java 8 + ONNX Runtime + OpenCV，封装 **OpenCV Zoo 人脸识别**（YuNet 检测 + SFace 128d 向量 + MiniFASNetV2 活体）+ 头像/证件卡片提取。模型推理完全在 JVM 内完成，所有运行时依赖 Maven 拉取（`onnxruntime` 1.18.0 + `openpnp/opencv` 4.9.0）。音频 / OCR / 意图识别能力已抽离到独立 mica-* 项目，本仓库不再包含。

包前缀：`net.dreamlu.mica.ai.face`；顶层 Maven `${revision}=1.0.0`。

## 2. 仓库地图

```
mica-ai/
├── pom.xml                    # 顶层 BOM（revision / spring-boot / onnxruntime / opencv）
├── mica-ai-common/            # ONNX 通用基础设施（OnnxModelSession / OrtSessionFactory / MicaAiException）
├── mica-ai-core/              # 核心引擎（零 Spring，纯 Java 8）
│   └── mica-ai-face/          #   🎭 OpenCV Zoo（YuNet + SFace + MiniFASNetV2 + 头像 + 卡片）
├── mica-ai-starters/          # Spring Boot 2 Starter（`mica.ai.face` 前缀 + 自动注入）
│   └── mica-ai-face-spring-boot-starter/
├── mica-ai-example/           # Spring Boot 集成示例（默认在 `develop` profile 下编译）
└── model-tools/               # Python 工具链：download / convert / publish / package
    ├── common/                #   downloader / onnx_utils / progress
    ├── face/                  #   face 能力脚本
    └── scripts/               #   smoke_test / publish / package
```

`mica-ai-core/mica-ai-face/README.md` 有「模型规格 / 核心组件 / I/O 格式」三节 — 在该能力内做修改前**先读**。

## 3. 常用命令

### 3.1 Java 构建 / 测试

```bash
mvn -DskipTests install                        # 编译所有模块
mvn test                                       # 跑全部单元测试（face 12 + example 2）
mvn -pl mica-ai-core/mica-ai-face -am test     # 仅测 face（含依赖）
mvn -pl mica-ai-example -am spring-boot:run    # 跑 Demo
```

JDK 8+（推荐 Temurin / Azul Zulu 8/11/17）。Surefire 已配 `-Djdk.net.URLClassPath.disableClassPathURLCheck=true` + `forkCount=0`，规避 Windows 跨盘符 fork 报错。

### 3.2 模型工具链（Python）

```bash
make -C model-tools smoke       # 离线冒烟（不下载模型）
make -C model-tools download    # 下载 face 模型（默认 ModelScope）
make -C model-tools convert     # 把原始模型转换为 ONNX 产物
make -C model-tools publish     # 整理 out/ → models/，生成 manifest
make -C model-tools package     # 把 models/face 打包成 zip（用于 GitHub Release）
```

下载源可用 `--source modelscope|huggingface` 切换；`MICA_MODELS_DIR` 环境变量改变模型根目录。

## 4. 架构与设计原则（核心要点）

1. **零 Spring 依赖**：核心模块不引任何 `spring-*`，非 Spring 环境直接可用。Spring Boot Starter 是独立薄壳层。
2. **零 Python 进程**：Java 端不能 spawn Python / subprocess；所有推理走 `onnxruntime`。GPU 切到 `onnxruntime_gpu` + `device=gpu`。
3. **Builder + try-with-resources**：所有引擎配置走 `XxxConfig.builder()...build()`；`@PreDestroy` 关闭 `OrtSession`。
4. **公共契约**（在 `mica-ai-common`）：`MicaAiException`（统一异常）+ `OnnxModelSession`（单个 ONNX 会话持有者）+ `OrtSessionFactory`（跨能力 ONNX 会话工厂）。
5. **face 模块结构**：
   - `detection.FaceDetector` — YuNet 整图 + `detectTiled` 滑窗兜底
   - `recognition.FeatureExtractor` — SFace 128d + L2 归一化
   - `liveness.LivenessDetector` — MiniFASNetV2 + softmax
   - `verification.FaceVerifier` — 检测 → 对齐 → 特征 → 余弦相似度 一体化门面
   - `alignment.FaceAligner` — 5 点仿射到 112×112
   - `avatar.AvatarExtractor` — 头像提取（自动摆正 / 分块兜底 / 两阶段重采样）
   - `card.CardExtractor` — 证件卡片提取（掩膜 + 四边形拟合 + 透视矫正 + USM/CLAHE）
6. **Spring Boot Starter 约定**：`FaceProperties`（`@ConfigurationProperties(prefix = "mica.ai.face")`）+ `FaceAutoConfiguration`（`@Component`），自动装配由 `mica-auto` 插件生成 `spring.factories`，**不要**手改。
7. **依赖收口**：版本号统一在根 `pom.xml` 的 `dependencyManagement`；新增能力 / 三方库先在根 POM 评审。

## 5. 编码约定要点

- JDK 8 兼容：不使用 `var` / text block / records / sealed 等 JDK 9+ 语法。
- Lombok：`@Getter / @Setter / @RequiredArgsConstructor / @Slf4j / @Data / @Builder` 视场景选择；`@Builder(toBuilder=true)` 用于 immutable 参数对象。
- 日志只用 SLF4J（测试 `main` 例外）。
- 空值语义：用 `@Nullable` 显式标注可空参数；**默认非空**。
- 业务异常继承 `MicaAiException`，**禁止**直接 `throw new RuntimeException(...)`。
- **默认不加任何代码注释**（根 README 与子 README 是事实来源），除非被显式要求。
- 提交信息：`<scope>: <verb> <object>`（例如 `face: tighten det score threshold`）。
- **不要**主动 `git commit` / `push` / `merge`，等用户确认。

## 6. ⚠️ 硬性约束（速览，完整版见 `AGENTS.md` §6）

- **依赖模型必须可商用**：只接受 Apache-2.0 / MIT / BSD / ISC / MulanPSL-2.0 等商业友好协议；禁止 `CC BY-NC-*` / `Research Use Only` / `NonCommercial` / `NoDerivatives` / GPL / AGPL / LGPL。新增 / 替换模型必须在 PR 描述里贴 LICENSE 全文并明确「可商用 / 不可商用」结论。Agent 自检清单见 `AGENTS.md` §6.1。
- **不要**在 Java 端引 `torch*` / `paddle*` / `python*` 依赖，破坏「零 Python 进程」原则。
- **不要**修改 `mica-auto` 插件配置。
- **不要** commit `model/` / `output/` / `*.onnx` / `*.bin` / `target/`（`.gitignore` 已屏蔽）。
- **不要**创建无意义文件（无显式要求时不新建 `*.md`、空目录、占位脚本）。
- **不要**引入与「商业可用」冲突的 Java 库。
- **历史变更**：2026-06-01 起 mica-ai 由 Java 17 + Spring Boot 4.1 降级为 Java 8 + Spring Boot 2.7，对齐 mica-face 生态。降级需项目 owner 显式批准。

## 7. 常见改动场景的"标准操作"

| 场景 | 必读 | 标准动作 |
|------|------|---------|
| 新增 ONNX 输入节点 | `mica-ai-face/.../detection/FaceDetector.java` 或 `recognition/FeatureExtractor.java` 或 `liveness/LivenessDetector.java` | 改推理段常量 → 更新 mica-ai-face/README.md「I/O 格式」节 |
| 替换底层模型 | `model-tools/face/download.py` + `convert.py` | 先按 §6 自检 License → 改 `MODEL_*` 常量 → 重跑 `make smoke` → 更新 README 模型规格表 |
| 新增 Spring Boot 配置项 | Starter `FaceProperties.java` + `FaceAutoConfiguration.java` | 用 `mica-auto` 生成 import → `mvn install` → mica-ai-face/README.md 加示例 |
| 性能调优 | `mica-ai-face/onnx/OrtSessionFactory.java` + `mica-ai-face/onnx/OrtSessionOptions.java` | 优先调 `intraOpNumThreads` / `interOpNumThreads` / `device` |
| 新增头像提取参数 | `mica-ai-face/avatar/AvatarOptions.java` | Builder 加字段 → `validate()` 加范围校验 → Starter `FaceProperties.Avatar` 同步 → mica-ai-face/README.md「头像提取」节 |

## 8. 验证清单（改完跑一遍）

- [ ] `mvn -DskipTests install` 通过
- [ ] `mvn test` 通过（14 个用例：face 12 + example 2）
- [ ] 受影响模块 `mvn -pl <module> -am test` 通过
- [ ] 改了 Python：`make -C model-tools smoke` 通过
- [ ] 改了模型脚本：跑对应 `download.py` + `convert.py`，产物可被 Java 端加载
- [ ] 改了 Starter：`application.yml` 加示例 + 至少 1 个 `@Autowired` 使用点
- [ ] 改了 README：标题层级、代码块语言、链接自检
- [ ] 新增 / 替换模型：§6 商用自检清单全部勾选
- [ ] 未触发 §6 硬性约束任何一条

## 9. 给 Agent 的额外提示

- **回复语言**：与用户最新消息保持一致（默认中文）。代码 / 注释 / 标识符一律英文；中文文档与根 `README.md` 风格保持一致。
- **不要解释基础概念**（如"什么是 ONNX"），用户是资深 Java 工程师。
- **少而准**：能用一段代码说清就别写长篇说明；优先给「最小可运行示例 + 关键配置项」。
- **出错时优先复现**：先写一个失败测试 / 复现脚本，再改实现。
- **遇到不确定的模型 / 协议**（尤其是「依赖模型必须可商用」这条），**先停下问**，别凭印象判断。