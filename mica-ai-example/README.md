# mica-ai-example

> mica-ai Spring Boot 集成 Demo / Test：聚合 Face + Filetype + Plate + Layout + Matting + HumanSeg + Detect + Track + DocAlign + Enhance + TextLine + Classify + Captcha Starter 的可运行示例与集成测试。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run`（`develop` profile）启动后，可通过 REST 端点触发 face / filetype /
   plate / layout / matting / humanseg / detect / track / docalign / enhance / textline / classify / captcha 能力。
2. **组合场景示例**：把多个能力编排成完整业务链路（智能文档流水线、证件照一键生成），见 [§3.1 组合场景](#31-组合场景)。
3. **Starter 自动装配测试**：覆盖 `enabled` 开关、fail-fast、Bean 注入（基于轻量级
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

启动后访问 `http://localhost:8181`，内置示例页 `src/main/resources/static/index.html`（人脸认证闭环 + 组合场景 + 各能力上传演示）。

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
        enabled: true                   # 活体/比对前的质量门控（属性默认 false），证件照的质量评估也依赖它
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

测试覆盖 `ExampleApplicationContextTest`（Starter 自动装配 / fail-fast）、`SimpleMemoryRepositoryTest`（内存向量库）、
`DocumentScenarioIntegrationTest`（智能文档流水线，含方向回转验证）与
`IdPhotoScenarioIntegrationTest`（证件照几何与端到端出片）。

> 集成测试不依赖任何真实 ONNX 模型文件，可放心在 CI 中执行。
> 场景测试会加载真实模型（模型缺失时整类跳过）。

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
| `/enhance/upscale-detail` | POST multipart | 同一次超分，但返回 JSON：送入模型的原图 + 超分图（base64）+ 尺寸 / 倍率 / 耗时 |
| `/textline/classify` | POST multipart | 上传**单行文本图**，返回 orientation / label / score / upsideDown / angle（JSON） |
| `/textline/upright` | POST multipart | 上传单行文本图，倒置则旋转 180° 输出 PNG，正常则原样回传 |
| `/classify/classify` | POST multipart | 上传图片做分类（`model` 参数可选：默认 / clarity / person / table），返回 topK + 全量概率 |
| `/captcha/recognize` | POST multipart | 上传验证码图片，返回 text（识别文本）/ confidence（平均置信度）/ charConfidences（逐字符置信度） |
| `/scenario/document/process` | POST multipart | **智能文档流水线**：拉正 → 整图方向 → 版面 → 文本块转正 → 超分，返回阶段报告 + 成片 base64 |
| `/scenario/document/preview` | POST multipart | 同上参数，直接返回处理后的 PNG 字节 |
| `/scenario/idphoto/generate` | POST multipart | **证件照一键生成**：检人脸 → 质量评估 → 抠像换底 → 按规格裁剪 → 超分 |
| `/scenario/idphoto/preview` | POST multipart | 同上参数，直接返回成片 PNG 字节（质量不通过返回 422） |
| `/scenario/idphoto/options` | GET | 列出可选的尺寸规格与底色 |

> `/face/auth/*` 是**自建内控级**最小闭环（App 登录二次核验 / 考勤 / 门禁），不是银行开户级实名核身：
> YuNet 只有 5 点关键点，动作挑战用「左右转头 + 上下点头」的幅度，做不了眨眼与张口；MiniFASNetV2 也挡不住 3D 面具与深度伪造。
> 会话存在内存（单机 Demo），多实例部署请换 Redis 并补齐设备指纹、频次限制与审计。

> layout 能力已提供 REST 端点（`LayoutController`），但因模型 125MB 不随仓库分发，本示例默认
> `mica.ai.layout.enabled=false` ⇒ 该端点**未装配**，访问会 404；本地放好模型后置 `true` 即可启用。

> **ByteTrack 是有状态的**：同一个跟踪器实例要按顺序喂入同一路视频的连续帧，track id 才会稳定。
> 示例注入的是 starter 的单例 `ByteTracker`，多路并发时请改成「一路流一个实例」并自行加锁。

Swagger UI：`http://localhost:8181/swagger-ui.html`

---

## 3.1 组合场景

> 单能力演示是「一个模型调一次」，组合场景才是**多个模型编排成一条业务链路**——
> 这才是 mica-ai 在真实产品里的用法。示例页的「组合场景」标签页可直接调用。

### 📄 智能文档流水线（`/scenario/document/*`）

把「手机拍的文档」变成「可读、结构正确、方向正确」的图，串联 **5 个模型能力**：

| 阶段 | 能力 | 作用 |
|------|------|------|
| 1. 拉正 | `mica-ai-docalign` | 检四角 + 透视拉平，把斜拍的文档拧正 |
| 2. 整图方向 | `mica-ai-classify`（PULC `text_image_orientation`） | 判 0° / 90° / 180° / 270° 并旋转到位 |
| 3. 版面分析 | `mica-ai-layout` | 出版面区域 + 阅读顺序 |
| 4. 文本块转正 | `mica-ai-textline` | 对文本块判 0° / 180°，倒置的原地转正 |
| 5. 画质增强 | `mica-ai-classify`（PULC `clarity_assessment`）+ `mica-ai-enhance` | 判清晰 / 模糊，模糊则超分 |

```bash
curl -X POST -F "file=@doc.jpg" http://localhost:8181/scenario/document/process
```

```json
{
  "width": 384, "height": 453,
  "stages": [
    {"name": "align",      "applied": false, "detail": "角点最低分 0.002 < 0.3，画面中可能没有成形文档，跳过拉正"},
    {"name": "orientation", "applied": true,  "detail": "预测 90°（0.856），逆时针旋转 90°"},
    {"name": "layout",     "applied": false, "detail": "layout 模型未启用（125MB 不随仓库分发），跳过版面分析"},
    {"name": "textLine",   "applied": false, "detail": "没有版面结果可供判定，跳过"},
    {"name": "clarity",    "applied": false, "detail": "判定 clear（0.917，clear 概率 0.917），无需超分"}
  ],
  "orientation": {"label": "90", "score": 0.856, "rotated": true, "rotation": "逆时针旋转 90°"},
  "blocks": [], "textBlocks": [],
  "imageBase64": "data:image/png;base64,..."
}
```

**三个设计取舍**：

1. **阶段顺序**：几何矫正与语义理解（版面 / 方向）都放在**原分辨率**上跑，最快；
   画质增强放最后，对整图一次性超分，不会让后续检测在 4 倍图上变慢。
2. **每个阶段都能单独关**（`align` / `orientation` / `enhance` / `layout` / `textLine` 参数），
   关掉的阶段既不推理也不出现在报告里。
3. **依赖缺失自动降级**，不抛异常：`layout` 模型 125MB 不随仓库分发，
   未启用时该阶段跳过并说明原因；`clarity` 模型未在 `mica.example.classify.models` 声明时同样跳过。

> ⚠️ **方向标签的约定（实测结论）**：PULC `text_image_orientation` 的标签是
> **图片当前相对正立的顺时针角度**，不是「需要旋转的角度」。
> 实测：顺时针转 90° 的图预测为 `90`；按 `360 - label`（等价于逆时针转 `label` 度）修正后
> 重新分类回到 `0`，置信度与正立原图逐位一致；按「顺时针转 label 度」修正则变成 `180`——正好转反。
> 该结论已固化为集成测试（`DocumentScenarioIntegrationTest#orientationStageShouldStraightenRotatedImage`）。

### 🪪 证件照一键生成（`/scenario/idphoto/*`）

把一张随手拍的人像照片，变成符合规格的证件照，串联 **4 个模型能力**：

| 阶段 | 能力 | 作用 |
|------|------|------|
| 1. 定位人脸 | `mica-ai-face`（YuNet） | 检人脸，多脸取面积最大的一张 |
| 2. 质量评估 | `mica-ai-face`（eDifFIQA） | 出人脸质量分与是否达标；是否因不达标而拒收由 `qualityGate` 决定（默认放行） |
| 3. 抠像换底 | `mica-ai-matting`（U²-Net）或 `mica-ai-humanseg` | 出人像掩码，合成到纯色底 |
| 4. 按规格裁剪 | 纯几何 | 头顶留白 7%、头高占比 75%、水平居中 |
| 5. 超分 | `mica-ai-enhance` | 输出 4 倍尺寸供印刷 |

```bash
curl -X POST -F "file=@me.jpg" -F "spec=ONE" -F "background=BLUE" \
     http://localhost:8181/scenario/idphoto/generate
```

```json
{
  "passed": true, "reason": "OK",
  "hint": "检测到 2 张人脸，已取面积最大的一张出片",
  "spec": "一寸", "background": "蓝底", "maskSource": "通用抠图",
  "qualityScore": 0.62, "qualityPassed": true, "faceScore": 0.88,
  "faceBox": [551, 143, 628, 255],
  "cropBox": [483, 67, 696, 365],
  "width": 1180, "height": 1652,
  "upscaled": true, "upscaleScale": 4,
  "imageBase64": "data:image/png;base64,..."
}
```

**质量门控（默认放行）**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `qualityGate` | `false` | `true` = 质量不达标直接拒收（`reason=LOW_QUALITY`，不出片）；`false` = 照常出片，只在 `hint` 里提示质量偏低 |
| `minQuality` | `0.5` | 质量门限，仅在 `qualityGate=true` 时起作用 |

质量**评估**始终执行、分数始终回填（`qualityScore` / `qualityPassed`），
门控只决定「要不要拦」，不决定「要不要评」——
所以示例页能直接展示质量分供用户判断，而不必把流程卡死。

**示例页的控件形态**：尺寸规格 / 底色 / 掩码来源 / 质量门控都是**下拉框**，
选项直接对应后端枚举（不用手打枚举名，选项里标了毫米数与 300 DPI 像素）；
**质量门限是滑动条**（0~1，步长 0.05，拖动时实时显示当前值）。
质量门控选「放行」时滑动条自动置灰——门限只对拒收生效。

> 前提：`mica.ai.face.quality.enabled=true` 时才装配质量评估器；
> 关掉它则整条链跳过质量评估，报告里 `qualityScore=-1`、`qualityPassed=null`。

**规格与底色**：

| 规格 | 毫米 | 300 DPI 像素 |
|------|------|------------|
| `SMALL_ONE` 小一寸 | 22×32 | 260×378 |
| `ONE` 一寸 | 25×35 | 295×413 |
| `LARGE_ONE` 大一寸 | 33×48 | 390×567 |
| `SMALL_TWO` 小二寸 | 35×45 | 413×531 |
| `TWO` 二寸 | 35×49 | 413×579 |
| `LARGE_TWO` 大两寸 | 35×53 | 413×626 |

底色：`WHITE` 白（255,255,255）/ `BLUE` 蓝（67,142,219）/ `RED` 红（255,0,0）。

**三个设计取舍**：

1. **越界用底色补齐，而不是拉伸**：裁剪框超出原图的部分由底色填满——
   这正是证件照的真实处理方式（背景本来就是纯色），也避免了把人脸拉变形。
2. **只处理交集区域再缩放贴进画布**，不对整张大图做浮点合成，省内存。
3. **质量评估与质量拦截分开**：评估恒做、分数恒回填，拦不拦由 `qualityGate` 决定且默认放行——
   拍糊了是用户该看见的信息（`qualityScore`），不该由流程替他拒收。

> `composeIdPhoto` / `computeCropRect` 是流水线里唯一与模型无关的纯几何步骤，
> 已单独暴露出来用**确定性合成图**做单测（头顶留白 29 行 = 7%、头高 310 行 = 75%、
> 背景像素精确等于底色），不依赖任何外部图片素材。

---

## 3.2 图像超分

`mica-ai-enhance`（Real-ESRGAN `realesr-general-x4v3`）在示例页里是**独立标签页**：
左侧是可拖动分割条的对比视图，右侧是上传区与参数区。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `outScale` | 模型原生倍率（4） | 输出倍率，取值 1 ~ 4 |
| `tileSize` | 256 | 分块边长，越小越省内存、越慢 |
| `maxInputSide` | 1200 | 输入长边上限，超过先等比缩小再超分 |

```bash
curl -X POST -F "file=@photo.jpg" -F "outScale=2" -F "tileSize=256" \
     http://localhost:8181/enhance/upscale-detail
```

```json
{
  "originalWidth": 384, "originalHeight": 453,
  "originalBase64": "data:image/png;base64,...",
  "outputWidth": 768, "outputHeight": 906,
  "outputBase64": "data:image/png;base64,...",
  "sourceWidth": 384, "sourceHeight": 453,
  "preDownscaled": false,
  "outScale": 2, "modelScale": 4, "tileSize": 256,
  "elapsedMs": 2829, "model": "realesr-general-x4v3"
}
```

### 对比视图交互

- **上传即预览**：选中文件后浏览器本地解码立刻显示原图，右半边盖一层「待生成」提示 ——
  不用等服务端返回就能先看原图。
- **默认适应窗口**：展示区本身随图片大小收缩（不铺黑底），图片默认按「适应窗口」缩放；
  要按原始像素查看就切到 `原图大小`。
- **缩放以原图为基准**：`原图大小` / `适应窗口`（默认）/ `200%` / `400%` / `800%`，
  百分比指的是**原图的像素倍数**。4× 模型下 `400%` 正好是超分图的 1:1 ——
  左侧原图被放大约 4 倍、右侧超分图 1:1，一拖分割条就是像素级对比。

### 三个设计取舍

1. **对比图严格同源**：`originalBase64` 是**实际送进模型的那张图**（触发输入长边上限时，
   是先缩小后的版本），`sourceWidth / sourceHeight` 另行保留上传时的真实尺寸。
   左右两侧同源同尺寸基准，拖分割条不会出现「拿大图比小图」的错觉。
2. **倍率小于原生值时先重建再降采样**：输出 2× 不是直接双三次插值到 2×，
   而是先跑完 4× 推理、再用 `INTER_AREA` 面积平均降采样——
   在已经重建出高频细节的图上做降采样，边缘比直接插值更实。
3. **每个档位一套引擎**：`tileSize` 在 `EnhanceConfig` 构造时就冻结，
   所以「处理档位」本质是每个档位一个 `EnhanceEngine`（各自一套 ONNX session）。
   `EnhanceEnginePool` 按档位缓存：首次用到才创建，后续复用，容器关闭时统一释放。

> ⚠️ **超分 ≠ 修复**：Real-ESRGAN 是超分模型，做的是「重建高频细节 + 把小图放大」。
> 撕裂、污损、缺失的部分它补不回来——严重老旧或破损照片的效果有限。
> 想换更高精度的权重，把 `mica.ai.enhance.model-path` 指向 `RealESRGAN_x4plus`（64MB，需自行下载）即可。

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
│   ├── config/ScenarioConfig.java                   # 组合场景流水线的 Bean 装配
│   ├── config/EnhanceEnginePool.java                 # 超分引擎池：按 tile 档位缓存 EnhanceEngine
│   ├── pipeline/FacePipeline.java                   # 检测 → 活体 → 对齐 → 特征 编排
│   ├── pipeline/FaceRecognitionResult.java
│   ├── pipeline/enhance/                            # ✨ 图像超分
│   │   ├── EnhanceUpscalePipeline.java              # 解码 → 预缩放 → 取引擎 → 推理 → 降采样 → 编码
│   │   ├── EnhanceUpscaleOptions.java               # 输出倍率 / 分块边长 / 输入长边上限
│   │   ├── EnhanceUpscaleResult.java                # 原图与输出的 PNG、尺寸、耗时
│   │   └── EnhanceUpscaleDetail.java                # REST 响应体（原图 + 超分图 base64）
│   ├── pipeline/document/                           # 📄 智能文档流水线
│   │   ├── DocumentPipeline.java                    # 5 阶段编排（docalign→classify→layout→textline→enhance）
│   │   ├── DocumentOptions.java                     # 阶段开关与门限
│   │   └── DocumentResult.java                      # 阶段报告 + 版面区域 + 文本块方向
│   ├── pipeline/idphoto/                            # 🪪 证件照一键生成
│   │   ├── IdPhotoPipeline.java                     # 检人脸 → 质量评估 → 抠像 → 按规格裁剪 → 超分
│   │   ├── IdPhotoOptions.java                      # 规格 / 底色 / 留白比例
│   │   ├── IdPhotoSpec.java                         # 6 种证件照尺寸规格
│   │   ├── IdPhotoBackground.java                   # 白 / 蓝 / 红底
│   │   ├── IdPhotoMaskSource.java                   # MATTING / HUMANSEG
│   │   └── IdPhotoResult.java
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
│       ├── EnhanceController.java                   # /enhance/upscale + /enhance/upscale-detail
│       ├── TextLineController.java                  # /textline/classify + /textline/upright
│       ├── ClassifyController.java                  # /classify/classify
│       ├── CaptchaController.java                   # /captcha/recognize
│       ├── DocumentScenarioController.java          # /scenario/document/* 智能文档流水线
│       └── IdPhotoScenarioController.java           # /scenario/idphoto/* 证件照一键生成
├── src/main/resources/
│   ├── application.yml                              # 各能力示例配置
│   ├── static/index.html                            # 示例页：人脸认证闭环 + 图像超分 + 组合场景 + 各能力上传演示
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    ├── ExampleApplicationContextTest.java           # Starter 自动装配测试（ApplicationContextRunner）
    ├── repository/SimpleMemoryRepositoryTest.java   # 内存向量库单测
    ├── enhance/
    │   ├── EnhanceUpscaleOptionsTest.java           # 参数校验单测（不依赖模型）
    │   └── EnhanceUpscaleIntegrationTest.java        # 真模型：倍率 / 预缩放 / 档位 / 引擎池
    └── scenario/
        ├── DocumentScenarioIntegrationTest.java     # 智能文档流水线（含方向回转验证）
        └── IdPhotoScenarioIntegrationTest.java      # 证件照几何 + 端到端出片
```

---

## 5. 已知约束

- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。
- **OpenCV 原生库**：由 face starter 的 `FaceAutoConfiguration` 在启动时 `OpenCV.loadLocally()` 加载；
  若把 `mica.ai.face.enabled` 设为 `false`，其它依赖 OpenCV 的能力（抠图 / 人像 / 检测 / 矫正 / 超分）会因原生库未加载而失败。
- **模型路径**：示例统一用相对仓库根目录的 `model-tools/...` 路径，因此启动时工作目录必须是仓库根。
- **组合场景的依赖缺失是降级而非报错**：某能力被 `enabled=false` 关掉时，
  对应阶段在报告里标记为「跳过」并说明原因，整条链路仍可跑完。
- **证件照的输入要求**：正面清晰人像。侧脸、逆光、远距离小脸的质量分偏低，
  默认仍会出片（`qualityGate=false`），但 `qualityScore` 会如实反映；
  需要严格把关时传 `qualityGate=true`，此时返回 `reason=LOW_QUALITY` 且不出片。
- **超分的输入长边上限**：`/enhance/upscale-detail` 默认 `maxInputSide=1200`，
  超过会先等比缩小再超分（响应里 `preDownscaled=true`，同时保留上传时的真实尺寸）。
  这是为了避免超大图把内存和响应体打爆——4× 输出意味着边长也要 ×4。
- **超分不是生成式修复**：Real-ESRGAN 重建高频细节、把小图放大，
  撕裂 / 污损 / 缺失的内容补不回来，严重老旧或破损照片的效果有限。
