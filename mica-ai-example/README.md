# mica-ai-example

> mica-ai Spring Boot 集成 Demo / Test：聚合 Face + Filetype + Plate + Layout + Matting + HumanSeg + Detect + Track + DocAlign + Enhance + TextLine + Classify + Captcha Starter 的可运行示例与集成测试。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run`（`develop` profile）启动后，可通过 REST 端点触发 face / filetype /
   plate / layout / matting / humanseg / detect / track / docalign / enhance / textline / classify / captcha 能力。
2. **Starter 自动装配测试**：覆盖 `enabled` 开关、fail-fast、Bean 注入（基于轻量级
   `ApplicationContextRunner`，非 `@SpringBootTest`）。

---

## 1. 环境要求

| 组件          | 版本                                                      | 说明                             |
|-------------|---------------------------------------------------------|--------------------------------|
| JDK         | **8+**                                                  | 推荐 Temurin / Azul Zulu 8、11、17 |
| Maven       | 3.6+                                                    | 多模块构建                          |
| Spring Boot | 2.7.x                                                   | 由根 BOM 引入                      |
| ONNX 模型     | YuNet + SFace + Magika + HyperLPR3 + U²-Net + PP-HumanSeg + NanoDet + DocAligner + Real-ESRGAN + PP-LCNet + PULC + ddddocr | 见各能力 README；layout 模型需自行获取     |

---

## 2. 快速开始

### 2.1 默认启动

```bash
mvn -pl mica-ai-example -am -DskipTests clean install
mvn -pl mica-ai-example spring-boot:run -Dspring-boot.run.workingDirectory=..
```

> 模型路径相对**仓库根目录**（`model-tools/...`），而 `spring-boot:run` 默认以模块目录为工作目录，因此需要显式指定 `workingDirectory=..`。

启动后访问 `http://localhost:8181`，内置示例页 `src/main/resources/static/index.html`（人脸认证闭环 + 各能力上传演示）。

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
        quality:                        # eDifFIQA 人脸质量评估（CC-BY-4.0）
          path: model-tools/face/models/ediffiqa_tiny_jun2024.onnx
        expression:                     # 表情识别 7 类（Apache-2.0）
          path: model-tools/face/models/facial_expression_recognition_mobilefacenet_2022july.onnx
      detection:                        # 阈值配置在 detection 块下
        threshold: 0.9
        nms-threshold: 0.3
      liveness:
        enabled: true                   # 活体默认 true，关闭请显式设 false
        threshold: 0.85
        crop-scale: 2.7
      quality:
        enabled: true                   # 质量门控（默认 false）
        threshold: 0.5
      expression:
        enabled: true                   # 表情识别（默认 false）
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
    humanseg:
      enabled: true
      model-path: model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx
      input-width: 192
      input-height: 192
    detect:
      enabled: true
      family: NANODET_PLUS              # 切 YOLOX_S 时 input-size 必须同步改成 640
      model-path: model-tools/detect/models/object_detection_nanodet_2022nov.onnx
      input-size: 416
    track:
      enabled: true
      high-threshold: 0.6
      low-threshold: 0.1
      match-thresh: 0.8
      match-low-thresh: 0.5
      confirm-age: 3
      track-buffered-thresh: 30
    docalign:
      enabled: true
      model-path: model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx
      input-width: 256
      input-height: 256
      heatmap-threshold: 0.3
    enhance:
      enabled: true
      model-path: model-tools/enhance/models/realesr-general-x4v3.onnx
      scale: 4
      tile-size: 256
      tile-overlap: 16
    textline:
      enabled: true
      model-path: model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx
    classify:
      enabled: true
      model-path: model-tools/classify/models/text_image_orientation.onnx
      labels-path: model-tools/classify/models/text_image_orientation_labels.txt   # ⚠️ 必填，与模型成对
    captcha:
      enabled: true
      # ⚠️ 模型与字符集都是外置配置，两项必填且必须成对
      model-path: model-tools/captcha/models/ddddocr.onnx
      charset-path: model-tools/captcha/models/charset.txt
```

> matting（`u2netp` 4.36MB）/ humanseg（`PP-HumanSeg` 5.88MB）/ detect（`NanoDet-Plus` 3.62MB）/ docalign（4.55MB）/
> enhance（`Real-ESRGAN` 2.33MB）/ textline（`PP-LCNet_x1_0` 6.46MB）/ classify（4 个 PULC，共 4.14–6.79MB）/
> captcha（`ddddocr` 12.98MB）模型均已入库，默认 `enabled: true` 即可用，
> 无需像 layout 那样兜底关闭。

> ⚠️ 当 `enabled=true` 但必填项缺失时，应用启动会 **fail-fast** 抛出 `MicaAiException`。
> 不希望加载该能力时，设 `enabled: false` 即可。

### 2.3 PULC 多模型（分类族的真实用法）

`mica-ai-classify` 一个引擎绑定「一个模型 + 一份 label_list」，两者必须成对配置。要同时用多个 PULC 模型，
就再声明引擎 —— 示例把另外 3 个模型声明在**示例自有前缀** `mica.example.classify.models` 下，
按需（首次调用该模型时）懒加载，不占启动内存：

```yaml
mica:
  example:
    classify:
      models:
        clarity:
          model-path: model-tools/classify/models/clarity_assessment.onnx
          labels-path: model-tools/classify/models/clarity_assessment_labels.txt
        person:
          model-path: model-tools/classify/models/person_exists.onnx
          labels-path: model-tools/classify/models/person_exists_labels.txt
        table:
          model-path: model-tools/classify/models/table_attribute.onnx
          labels-path: model-tools/classify/models/table_attribute_labels.txt
          resize-short: 0        # 官方直接 resize 到 224，不做短边缩放 + 中心裁剪
          multi-label: true      # 6 个表格属性各自独立 sigmoid
          top-k: 6
```

调用时带 `model` 参数即可切换：`POST /classify/classify?model=clarity`；不传则用 starter 默认模型。

> 四个 PULC 模型的差异（标签、是否多标签、预处理）见
> [model-tools/classify/README.md](../model-tools/classify/README.md)。

### 2.4 跑集成测试

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
| `/face/quality` | POST multipart | 上传图片，逐脸做 eDifFIQA 质量评估，返回 score / threshold / passed |
| `/face/expression` | POST multipart | 上传图片，逐脸识别 7 类表情，返回 expression / desc / score / probabilities |
| `/face/register` | POST multipart | 上传图片，注册人脸到内存向量库，返回 personId |
| `/face/recognize` | POST multipart | 上传图片，1:N 检索返回最相似 personId |
| `/face/verify` | POST multipart | 上传两张图，1:1 比对 |
| `/face/identity-verify` | POST multipart | 人证核验：现场照 vs 已注册 personId |
| `/face/auth/session` | POST | 创建认证会话，返回 sessionId + 随机顺序的动作序列（120 秒过期、一次性） |
| `/face/auth/frame` | POST multipart | 逐帧上传（`sessionId` + `file`），返回质量 / 活体 / 姿态 / 动作进度 |
| `/face/auth/submit` | POST multipart | 动作完成后提交，用最佳帧特征与 `personId` 底库或 `reference` 参考照做 1:1 |
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
| `/humanseg/cutout` | POST multipart | 上传图片，返回人像透明底 PNG（BGRA） |
| `/humanseg/cutout-on-color` | POST multipart | 上传图片 + 底色，返回人像纯色底合成 PNG |
| `/humanseg/mask` | POST multipart | 上传图片，返回人像二值掩码 PNG |
| `/detect/detect` | POST multipart | 上传图片，返回 COCO 80 类检测框（label / classId / score / 坐标） |
| `/detect/annotate` | POST multipart | 上传图片，在原图上画出检测框与类别名并返回 PNG |
| `/track/update` | POST JSON | 提交一帧检测框数组，返回当前活跃轨迹（id / 框 / score / age / mature） |
| `/track/detect-and-track` | POST multipart | 上传一帧图片，自动检测 + 跟踪，返回带 track id 的框 |
| `/docalign/corners` | POST multipart | 上传文档图，返回左上 / 右上 / 右下 / 左下四个角点与置信度 |
| `/docalign/align` | POST multipart | 上传文档图，检测四角 + 透视拉平，返回 PNG |
| `/enhance/upscale` | POST multipart | 上传图片，4 倍超分后返回 PNG（大图自动 tile 推理） |
| `/textline/classify` | POST multipart | 上传**单行文本图**，返回 orientation / label / score / upsideDown / angle（JSON） |
| `/textline/upright` | POST multipart | 上传单行文本图，倒置则旋转 180° 输出 PNG，正常则原样回传 |
| `/classify/classify` | POST multipart | 上传图片做分类（`model` 参数可选：默认 / clarity / person / table），返回 topK + 全量概率 |
| `/captcha/recognize` | POST multipart | 上传验证码图片，返回 text（识别文本）/ confidence（平均置信度）/ charConfidences（逐字符置信度） |

> `/face/auth/*` 是**自建内控级**最小闭环（App 登录二次核验 / 考勤 / 门禁），不是银行开户级实名核身：
> YuNet 只有 5 点关键点，动作挑战用「左右转头 + 上下点头」的幅度，做不了眨眼与张口；MiniFASNetV2 也挡不住 3D 面具与深度伪造。
> 会话存在内存（单机 Demo），多实例部署请换 Redis 并补齐设备指纹、频次限制与审计。

> layout 能力已提供 REST 端点（`LayoutController`），但因模型 125MB 不随仓库分发，本示例默认
> `mica.ai.layout.enabled=false` ⇒ 该端点**未装配**，访问会 404；本地放好模型后置 `true` 即可启用。

> **ByteTrack 是有状态的**：同一个跟踪器实例要按顺序喂入同一路视频的连续帧，track id 才会稳定。
> 示例注入的是 starter 的单例 `ByteTracker`，多路并发时请改成「一路流一个实例」并自行加锁。

Swagger UI：`http://localhost:8181/swagger-ui.html`

---

## 4. 项目结构

```
mica-ai-example/
├── pom.xml                                          # 各能力 starter + spring-boot-starter-web
├── src/main/java/net/dreamlu/mica/ai/example/
│   ├── ExampleApplication.java                      # @SpringBootApplication 入口
│   ├── config/OpenApiConfig.java
│   ├── config/ClassifyModelProperties.java          # mica.example.classify.* 配置
│   ├── config/ClassifyModelRegistry.java            # PULC 多模型懒加载注册表
│   ├── pipeline/FacePipeline.java                   # 检测 → 活体 → 对齐 → 特征 编排
│   ├── repository/VectorRepository.java             # 向量库抽象 + SimpleMemoryRepository 内存实现
│   ├── auth/
│   │   ├── AuthAction.java                          # 动作挑战定义（转头 / 点头 + 所需幅度）
│   │   ├── FaceAuthSession.java                     # 会话状态（姿态极差 + 最佳帧特征）
│   │   └── FaceAuthStore.java                       # 内存会话存储（TTL + 一次性消费）
│   └── controller/
│       ├── FaceController.java                      # /face/detect + /liveness + /extract + /quality + /expression
│       ├── FaceRecognitionController.java           # /face/register + /recognize + /verify + /identity-verify + /health
│       ├── FaceAuthController.java                  # /face/auth/session + /frame + /submit
│       ├── AvatarController.java                    # /face/avatar（单图 / 多图 / 标注）
│       ├── CardController.java                      # /face/card/extract + /cards
│       ├── FiletypeController.java                  # /filetype/detect + /filetype/detect-bytes
│       ├── PlateController.java                     # /plate/recognize
│       ├── LayoutController.java                    # /layout/detect + /layout/detect-bytes
│       ├── MattingController.java                   # /matting/cutout + /cutout-on-color + /mask
│       ├── HumanSegController.java                  # /humanseg/cutout + /cutout-on-color + /mask
│       ├── DetectController.java                    # /detect/detect + /detect/annotate
│       ├── TrackController.java                     # /track/update + /track/detect-and-track
│       ├── DocAlignController.java                  # /docalign/corners + /docalign/align
│       ├── EnhanceController.java                   # /enhance/upscale
│       ├── TextLineController.java                  # /textline/classify + /textline/upright
│       ├── ClassifyController.java                  # /classify/classify
│       └── CaptchaController.java                   # /captcha/recognize
├── src/main/resources/
│   ├── application.yml                              # 各能力示例配置
│   ├── static/index.html                            # 示例页：人脸认证闭环 + 各能力上传演示
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    ├── ExampleApplicationContextTest.java           # Starter 自动装配测试（ApplicationContextRunner）
    └── repository/SimpleMemoryRepositoryTest.java   # 内存向量库单测
```

---

## 5. 已知约束

- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。
- **OpenCV 原生库**：由 face starter 的 `FaceAutoConfiguration` 在启动时 `OpenCV.loadLocally()` 加载；
  若把 `mica.ai.face.enabled` 设为 `false`，其它依赖 OpenCV 的能力（抠图 / 人像 / 检测 / 矫正 / 超分）会因原生库未加载而失败。
- **模型路径**：示例统一用相对仓库根目录的 `model-tools/...` 路径，因此启动时工作目录必须是仓库根。
