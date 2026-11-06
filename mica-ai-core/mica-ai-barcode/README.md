# mica-ai-barcode

> [ZXing](https://github.com/zxing/zxing) `core` 驱动的二维码 / 一维条码解码 Java SDK：**零模型、零 ONNX、零 Spring、零 OpenCV**，纯 JVM 内存像素即可跑，Apache-2.0 可商用。
>
> 本能力**没有模型资产，也没有 `model-tools/barcode/` 目录**：它是纯算法模块，与 `mica-ai-track` 同类。

---

## 1. 为什么是这个方案

选型依据是 16 档 × N=12 的合成退化样本实测，结论已固化进本模块单测 `BarcodeSamples` / `BarcodeEngineTest`（同一套退化算子 + 命中判据）。三条实测结论直接决定了实现：

1. **只引 `com.google.zxing:core:3.5.3`**，不引 `javase`、不引 `opencv`、不引 ONNX Runtime。用 core 自带的 `RGBLuminanceSource` 从像素数组构造，零 AWT 类型泄漏、headless 安全；3.5.3 是 JDK 8 的最后一条线（3.6.0 起要求 Java 11+）。
2. **OpenCV `QRCodeDetector` / WeChatQRCode 没有兜底价值**：合成样本上纯 zxing + 多尺度 + 分块每一档都 ≥ OpenCV，且快 10–50 倍；OpenCV 在 1px 小码、模糊 σ=2、梯形透视、组合退化上全是 0%，`detectAndDecodeCurved` 连干净码都是 0%，还刷 `ECI is not supported properly`（中文 QR 有乱码风险）。WeChatQRCode 属 `opencv_contrib`（openpnp 官方 Java 包不含），依赖 4 个 Caffe 文件，其优势来自 zxing-cpp 的解码工程优化而非那两个 CNN 权重，`caffe2onnx` 收益/成本极差。
3. **固定一种二值化器是错的**：模糊 σ=2 时 `GlobalHistogramBinarizer` 还有 33.3%，`HybridBinarizer` 反而 0%。正确做法是多策略回退。

### 依赖 License 自检（对应 AGENTS.md §6.1）

| 项 | 内容 |
|---|---|
| 依赖库 | [`com.google.zxing:core:3.5.3`](https://github.com/zxing/zxing) |
| 官方 LICENSE | [Apache License 2.0](https://github.com/zxing/zxing/blob/master/LICENSE) |
| 依赖模型 / 权重 / 词表 | **无** |
| 结论 | **可商用**（允许商用、分发、修改；Apache-2.0 无 NC / 研究限制） |
| 需随分发保留 | Apache-2.0 许可声明；本模块不含任何第三方权重，故无 `MODEL_LICENSE` / NOTICE 二级文件 |

> 刻意**不引 `mica-ai-common`**：它依赖 `onnxruntime`，会把整个 ONNX 运行时拖进一个不需要推理的纯算法模块。因此参数校验与 `mica-ai-track` 一致，用 JDK 原生 `IllegalArgumentException`，而不是 `MicaAiException`。

---

## 2. 快速开始

```java
// 零配置：本能力没有 model-path 之类的必填项
BarcodeEngine engine = BarcodeEngine.defaults();

// ① 一图多码（跑完整链，返回去重后的全部结果）
List<BarcodeResult> codes = engine.decodeBytes(file.getBytes());
for (BarcodeResult code : codes) {
    System.out.println(code.getText() + " / " + code.getSymbology()
        + " / " + Arrays.toString(code.getBoundingBox()) + " / " + code.getStrategy());
}

// ② 只要一个码（命中即短路，单码场景快一个数量级）
Optional<BarcodeResult> first = engine.decodeFirstPath("/data/invoice.png");

// ③ 自定义策略
BarcodeConfig config = BarcodeConfig.builder()
    .formats(Arrays.asList(BarcodeSymbology.QR_CODE, BarcodeSymbology.CODE_128))
    .multiScale(true)
    .tileScan(true)
    .tileOverlapRatio(0.12f)
    .build();
BarcodeEngine custom = BarcodeEngine.create(config);
```

OpenCV 调用方（不引 AWT 的路径）：

```java
byte[] bgr = new byte[mat.rows() * mat.cols() * 3];
mat.get(0, 0, bgr);                                   // 需先保证是 CV_8UC3
BarcodeImage image = BarcodeImage.ofBgr(mat.cols(), mat.rows(), bgr);
List<BarcodeResult> codes = engine.decode(image);
```

Spring Boot 用法见 [`mica-ai-barcode-spring-boot-starter`](../../mica-ai-starters/mica-ai-barcode-spring-boot-starter/README.md)。

---

## 3. 解码链（策略按序，失败才进下一步）

| 顺序 | 策略 | `strategy` 值 | 解决的退化 |
|---|---|---|---|
| 1 | `HybridBinarizer + TRY_HARDER` 全图 | `DIRECT` | 干净码、旋转、低对比、噪声 |
| 2 | ×2 / ×3 **最近邻**放大重试 | `UPSCALE_x2` / `UPSCALE_x3` | **1px 小码 0→100%**、**模糊 σ=2 0→100%**、噪声 91.7→100% |
| 3 | 2×2 带 12% 重叠分块（每块再跑 1+2） | `TILE_r_c` / `TILE_r_c_UPSCALE_xN` | **一图 4 码 25→100%**（48/48），比 OpenCV `detectAndDecodeMulti` 的 75% 更高且快约 10 倍 |
| 4 | `GlobalHistogramBinarizer` 回退 | `GLOBAL_HISTOGRAM` | 模糊图上 Hybrid 全灭而 Global 尚有 33.3% 的情况 |

`decodeFirst(...)` 是**短路链**：命中即返回，单码图 ~0.1–0.5ms。`decode(...)` 是**全量扫描**：为找齐多个码会跑完分块，~1–11ms。两者共用同一套策略与去重规则（按「制式 + 文本」去重，保留最先命中的、坐标最准的结果），`maxResults` 达到上限即停。

`POSSIBLE_FORMATS` 显式限定为 `DEFAULT_FORMATS`（QR_CODE / CODE_128 / EAN_13 / EAN_8 / UPC_A / CODE_39 / ITF / CODABAR），**不全开**：Data Matrix / PDF417 / Aztec 未做退化样本验证，且全开会显著拖慢 `TRY_HARDER`。需要时在 `formats` 里显式加。

---

## 4. 公开 API

| 类 | 位置 | 说明 |
|----|------|------|
| `BarcodeEngine` | `core`（与 `mica-ai-track` 的 `ByteTracker` 同位置） | 门面；**不实现 `AutoCloseable`**（无 ONNX session / `Mat` / 文件句柄可释放），线程安全、可单例 |
| `BarcodeConfig` | `config` | Builder 配置 + `validate()`；默认值集中在 `BarcodeConfig.DEFAULTS` |
| `BarcodeSymbology` | `model` | 制式枚举（常量名与 ZXing `BarcodeFormat` 一致）+ `DEFAULT_FORMATS` |
| `BarcodeImage` | `model` | 解码输入（内存 ARGB 像素图） |
| `BarcodeResult` | `model` | 单个结果：文本 / 制式 / 字符集 / 近似包围盒 / 命中策略 |
| `BarcodeDecoder` | `decoding` | 解码链实现 |
| `BarcodeImageUtils` | `util` | 最近邻放大、分块裁剪、`LuminanceSource` 构造 |

| 方法 | 返回 | 说明 |
|------|------|------|
| `decode(BarcodeImage)` | `List<BarcodeResult>` | 全量扫描，一图多码 |
| `decodeFirst(BarcodeImage)` | `Optional<BarcodeResult>` | 短路链，只要一个码 |
| `decodeBytes(byte[])` / `decodeFirstBytes(byte[])` | 同上 | PNG / JPG 等编码字节（走 JDK ImageIO） |
| `decodePath(String)` / `decodeFirstPath(String)` | 同上 | 文件路径 |
| `formats()` | `List<BarcodeSymbology>` | 当前启用的制式 |
| `getConfig()` | `BarcodeConfig` | 当前配置 |

### 输入契约

| 工厂 | 入参 | 适用 |
|------|------|------|
| `BarcodeImage.ofArgb(w, h, int[])` | ARGB 像素 | `BufferedImage.getRGB()` / 自绘 |
| `BarcodeImage.ofGray(w, h, int[])` | 0–255 灰度 | 单通道传感器 / 已有灰度缓冲 |
| `BarcodeImage.ofGrayBytes(w, h, byte[])` | 灰度字节 | 上游返回 byte[] |
| `BarcodeImage.ofBgr(w, h, byte[])` | BGR 交错字节（长度 `w*h*3`） | OpenCV `Mat.get(0,0,byte[])` |
| `BarcodeImage.fromEncodedBytes(byte[])` | PNG / JPG 等 | 便捷入口 |

> `fromEncodedBytes` 的 `java.awt` / `javax.imageio` 类型被隔离在包内可见的 `BarcodeImageIO` 中，
> `BarcodeEngine` 与 `BarcodeImage` 的公开方法签名不含任何 AWT 类型，精简 JRE 只要不调用该入口就不会触发类加载。
> `ImageIO.read` 只做内存解码、不接触显示设备，`java.awt.headless=true` 下可用。

### 结果与坐标

`boundingBox` 是 `[x1, y1, x2, y2]` 的**原图坐标**，已经还原放大倍数与分块偏移。但 ZXing 的 `ResultPoint` 通常只覆盖定位图形（QR 的三个角、一维码的起止点），因此它是**近似包围盒**，不保证完整框住整个码；需要精确框请自行做角点检测。`strategy` 记录是哪一档命中的，便于线上排查解码率问题。

---

## 5. 配置项（`BarcodeConfig`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `formats` | `DEFAULT_FORMATS`（8 种） | 启用的制式，至少 1 种 |
| `charset` | `UTF-8` | 解码字符集；中文 QR 依赖此项 |
| `tryHarder` | `true` | ZXing `TRY_HARDER`，提高旋转 / 部分遮挡召回，代价是耗时 |
| `multiScale` | `true` | 失败后按 `upscaleFactors` 放大重试 |
| `upscaleFactors` | `[2, 3]` | 每个放大倍数（最近邻），范围 `[2,8]` |
| `maxSideLength` | `4096` | 放大后长边上限；超过该值的整档跳过，避免大图 ×3 OOM |
| `tileScan` | `true` | 分块扫描（一图多码的关键） |
| `tileRows` / `tileCols` | `2` / `2` | 分块行列数，范围 `[1,4]`；1×1 时不重复扫全图 |
| `tileOverlapRatio` | `0.12` | 相邻块重叠比例，避免码被切在分界线上，范围 `[0, 0.5)` |
| `globalHistogramFallback` | `true` | 末档用 `GlobalHistogramBinarizer` 再试一次 |
| `maxResults` | `16` | 结果数上限，达到即停止后续策略 |

非法配置在构造期抛 `IllegalArgumentException`（零依赖模块，不用 `MicaAiException`）。

---

## 6. 明确不做的事

- **不做 OpenCV 定位、不做 WeChatQRCode**：实测无兜底价值，详见 §1 与探针报告。
- **不做生成 / 编码**：`MultiFormatWriter` 在 ZXing core 里本来就有，本模块只负责读。
- **不做 90° 一维码纠正**：QR 对任意角度旋转稳健，一维码必须水平；文档图请先过 `mica-ai-docalign`（透视拉正）与 `mica-ai-classify`（整图方向）。
- **不做生成式修复**：模糊、断码不做超分或 inpainting 猜测内容，返回「未解出」比返回一个错码安全。

---

## 7. 已知局限（落地前务必读）

数据来自合成退化样本、N=12/档、非实拍，**不是生产承诺**：

| 场景 | 表现 | 建议 |
|------|------|------|
| 梯形透视 | 全策略 25%，OpenCV 0% | 接在 `docalign` 拉正之后；`mica-ai-example` 的 `DocumentPipeline` 已这样挂接 |
| 组合退化（2px + 旋转 20° + 模糊 + 噪声） | 全策略 0%，无解 | 业务层提示重拍，不要猜 |
| **一维码缩小 / 模糊** | 所有策略 0% | **必须原分辨率、禁止缩放**，建议原图裁剪后再送解码 |
| 摩尔纹 / 屏幕翻拍 / 强光照不均 / JPEG 强压缩 | 未测 | 上线前用真实票据、面单、资产标签复跑探针 |
| 中文 QR | zxing 正常 | 不要拿 OpenCV 结果兜底（ECI 警告） |

复现命令见探针报告 §1.1；`BarcodeEngineTest` 已把上述关键结论固化成回归测试。

---

## 8. 线程安全与资源

- `BarcodeConfig` 构造后按不可变使用；`getFormats()` / `getUpscaleFactors()` 返回只读视图。
- `MultiFormatReader` 有状态，解码器每次调用新建一个；冻结的 hints Map 只读共享。
- `BarcodeEngine` **线程安全**，可作为 Spring 单例；**不实现 `AutoCloseable`**，Spring Bean 不需要 `destroyMethod`。
- `BarcodeImage` 的 `getArgb()` 返回内部数组以避免解码链反复拷贝大图，**调用方不得修改其内容**。

---

## 9. 测试

```bash
mvn -pl mica-ai-core/mica-ai-barcode -am test
```

19 个用例，覆盖：干净 QR 直解、1px 小码必须靠多尺度、模糊 σ=2、一图 4 码必须靠分块、Code128 / EAN-13 原分辨率、中文 UTF-8 QR、`POSSIBLE_FORMATS` 限定、包围盒坐标映射、最近邻放大与裁剪像素、PNG 字节入口、配置默认值与 fail-fast。全部只依赖 ZXing core，不加载任何模型或原生库。
