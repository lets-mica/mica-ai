# mica-ai-example

> mica-ai Spring Boot 集成 Demo / Test：聚合 Face + Filetype + Plate + Layout + Matting + TextLine Starter 的可运行示例与集成测试。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run`（`develop` profile）启动后，可通过 REST 端点触发 face / filetype /
   plate / layout / matting / textline 能力。
2. **Starter 自动装配测试**：覆盖 `enabled` 开关、fail-fast、Bean 注入（基于轻量级
   `ApplicationContextRunner`，非 `@SpringBootTest`）。

---

## 1. 环境要求

| 组件          | 版本                                                      | 说明                             |
|-------------|---------------------------------------------------------|--------------------------------|
| JDK         | **8+**                                                  | 推荐 Temurin / Azul Zulu 8、11、17 |
| Maven       | 3.6+                                                    | 多模块构建                          |
| Spring Boot | 2.7.x                                                   | 由根 BOM 引入                      |
| ONNX 模型     | YuNet + SFace + Magika + HyperLPR3 + U²-Net + PP-LCNet | 见各能力 README；layout 模型需自行获取     |

---

## 2. 快速开始

### 2.1 默认启动

```bash
mvn -pl mica-ai-example -am -DskipTests clean install
mvn -pl mica-ai-example spring-boot:run
```

启动后访问 `http://localhost:8181`。

### 2.2 启用 face / filetype / plate

编辑 [src/main/resources/application.yml](src/main/resources/application.yml)，把模型路径改成你的实际路径（默认指向
仓库内 `model-tools/<cap>/models/`）：

```yaml
mica:
  ai:
    face:
      enabled: true
      model:                            # ⚠️ 模型路径统一挂在 model 下
        detection:
          path: model-tools/face/models/face_detection_yunet_2023mar.onnx
        recognition:
          path: model-tools/face/models/face_recognition_sface_2021dec.onnx
        liveness:
          path: model-tools/face/models/2.7_80x80_MiniFASNetV2.onnx
      liveness:
        enabled: true                   # 活体默认 true，关闭请显式设 false
    filetype:
      enabled: true
      model-path: model-tools/filetype/models/model.onnx
      config-path: model-tools/filetype/models/config.min.json
      content-types-path: model-tools/filetype/models/content_types_kb.min.json
    plate:
      enabled: true
      detection-model-path: model-tools/plate/models/y5fu_320x_sim.onnx
      recognition-model-path: model-tools/plate/models/rpv3_mdict_160_r3.onnx
      classification-model-path: model-tools/plate/models/litemodel_cls_96x_r1.onnx
    matting:
      enabled: true
      model-path: model-tools/matting/models/u2netp.onnx
    textline:
      enabled: true
      model-path: model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx
```

> matting（`u2netp` 4.36MB）/ textline（`PP-LCNet_x1_0` 6.46MB）模型均已入库，默认 `enabled: true` 即可用，
> 无需像 layout 那样兜底关闭。

> ⚠️ 当 `enabled=true` 但必填项缺失时，应用启动会 **fail-fast** 抛出 `MicaAiException`。
> 不希望加载该能力时，设 `enabled: false` 即可。

### 2.3 跑集成测试

```bash
mvn -pl mica-ai-example -am test
```

测试覆盖 `ExampleApplicationContextTest`（Starter 自动装配 / fail-fast）与 `SimpleMemoryRepositoryTest`（内存向量库）。

> 集成测试不依赖任何真实 ONNX 模型文件，可放心在 CI 中执行。

---

## 3. REST 端点

| 路径 | 方法 | 描述 |
|------|------|------|
| `/face/detect` | POST multipart | 上传图片，返回人脸框 + 关键点 + score |
| `/face/liveness` | POST multipart | 上传图片，逐脸做活体判断，返回 liveScore / isLive / attackType |
| `/face/extract` | POST multipart | 上传图片，返回 128d Embedding + L2 norm |
| `/face/register` | POST multipart | 上传图片，注册人脸到内存向量库，返回 personId |
| `/face/recognize` | POST multipart | 上传图片，1:N 检索返回最相似 personId |
| `/face/verify` | POST multipart | 上传两张图，1:1 比对 |
| `/face/identity-verify` | POST multipart | 人证核验：现场照 vs 已注册 personId |
| `/face/health` | GET | 健康检查：活体开关 / 阈值 / 各模型加载状态 |
| `/face/avatar` | POST multipart | 单图提取头像（正方形，自动摆正） |
| `/face/avatar/faces` | POST multipart | 多人图逐个提取头像 |
| `/face/avatar/annotate` | POST multipart | 在原图上标注头像裁剪窗口并返回图片 |
| `/face/card/extract` | POST multipart | 单张证件卡片提取 + 透视矫正 + 增强 |
| `/face/card/cards` | POST multipart | 多卡片提取 |
| `/filetype/detect` | POST multipart | 上传任意文件，返回 modelLabel / outputLabel / score / mimeType / group / description |
| `/filetype/detect-bytes` | POST octet-stream | 上传二进制，返回同上字段 |
| `/plate/recognize` | POST multipart | 上传车辆图片，返回车牌号 + 类型 + 置信度 |
| `/layout/detect` | POST multipart | 上传文档图片，返回版面区域（需本地有模型并置 `enabled=true`） |
| `/layout/detect-bytes` | POST octet-stream | 同上，octet-stream 入参 |
| `/matting/cutout` | POST multipart | 上传图片，返回透明底 PNG（BGRA） |
| `/matting/cutout-on-color` | POST multipart | 上传图片 + 底色，返回纯色底合成 PNG |
| `/matting/mask` | POST multipart | 上传图片，返回二值掩码 PNG（单通道 0/255） |
| `/textline/classify` | POST multipart | 上传**单行文本图**，返回 orientation / label / score / upsideDown / angle（JSON） |
| `/textline/upright` | POST multipart | 上传单行文本图，倒置则旋转 180° 输出 PNG，正常则原样回传 |

> layout 能力已提供 REST 端点（`LayoutController`），但因模型 125MB 不随仓库分发，本示例默认
> `mica.ai.layout.enabled=false` ⇒ 该端点**未装配**，访问会 404；本地放好模型后置 `true` 即可启用。

Swagger UI：`http://localhost:8181/swagger-ui.html`

---

## 4. 项目结构

```
mica-ai-example/
├── pom.xml                                          # 各能力 starter + spring-boot-starter-web
├── src/main/java/net/dreamlu/mica/ai/example/
│   ├── ExampleApplication.java                      # @SpringBootApplication 入口
│   ├── config/OpenApiConfig.java
│   ├── pipeline/FacePipeline.java                   # 检测 → 活体 → 对齐 → 特征 编排
│   ├── repository/VectorRepository.java             # 向量库抽象 + SimpleMemoryRepository 内存实现
│   └── controller/
│       ├── FaceController.java                      # /face/detect + /liveness + /extract
│       ├── FaceRecognitionController.java           # /face/register + /recognize + /verify + /identity-verify + /health
│       ├── AvatarController.java                    # /face/avatar（单图 / 多图 / 标注）
│       ├── CardController.java                      # /face/card/extract + /cards
│       ├── FiletypeController.java                  # /filetype/detect + /filetype/detect-bytes
│       ├── PlateController.java                     # /plate/recognize
│       ├── LayoutController.java                    # /layout/detect + /layout/detect-bytes
│       ├── MattingController.java                   # /matting/cutout + /cutout-on-color + /mask
│       └── TextLineController.java                  # /textline/classify + /textline/upright
├── src/main/resources/
│   ├── application.yml                              # 各能力示例配置
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    ├── ExampleApplicationContextTest.java           # Starter 自动装配测试（ApplicationContextRunner）
    └── repository/SimpleMemoryRepositoryTest.java   # 内存向量库单测
```

---

## 5. 已知约束

- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。