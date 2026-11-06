# mica-ai-barcode-spring-boot-starter

> [ZXing](https://github.com/zxing/zxing) 二维码 / 一维条码解码 Spring Boot Starter，基于 [mica-ai-barcode](../../mica-ai-core/mica-ai-barcode/README.md) 核心模块。**Apache-2.0 可商用**。
>
> **零模型、零 ONNX、零 OpenCV**：不需要下载权重、不需要配模型路径，什么都不配也能直接注入可用引擎。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-barcode-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 传递依赖只有 `com.google.zxing:core:3.5.3`（约 0.6MB）与 `slf4j-api`。

---

## 2. 配置项（`mica.ai.barcode` 前缀）

**全部可选**，以下是默认值全量展开：

```yaml
mica:
  ai:
    barcode:
      enabled: true
      formats: [ QR_CODE, CODE_128, EAN_13, EAN_8, UPC_A, CODE_39, ITF, CODABAR ]
      charset: UTF-8
      try-harder: true
      multi-scale: true
      upscale-factors: [ 2, 3 ]
      max-side-length: 4096
      tile-scan: true
      tile-rows: 2
      tile-cols: 2
      tile-overlap-ratio: 0.12
      global-histogram-fallback: true
      max-results: 16
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | `false` 时不装配 `BarcodeEngine` Bean |
| `formats` | 8 种（见上） | 启用的制式；可选 `QR_CODE` / `DATA_MATRIX` / `PDF_417` / `AZTEC` / `CODE_128` / `CODE_39` / `CODE_93` / `EAN_13` / `EAN_8` / `UPC_A` / `UPC_E` / `ITF` / `CODABAR` |
| `charset` | `UTF-8` | 解码字符集；中文 QR 依赖此项 |
| `try-harder` | `true` | ZXing `TRY_HARDER`，提高旋转 / 遮挡召回，代价是耗时 |
| `multi-scale` | `true` | 失败后按 `upscale-factors` 最近邻放大重试（1px 小码、模糊图的关键） |
| `upscale-factors` | `[2, 3]` | 每个放大倍数，范围 `[2,8]` |
| `max-side-length` | `4096` | 放大后长边上限，超过的整档跳过，避免大图 ×3 OOM |
| `tile-scan` | `true` | 分块扫描（一图多码的关键；关掉后多码只能解出 1 个） |
| `tile-rows` / `tile-cols` | `2` / `2` | 分块行列数，范围 `[1,4]` |
| `tile-overlap-ratio` | `0.12` | 相邻块重叠比例，避免码被切在分界线上，范围 `[0, 0.5)` |
| `global-histogram-fallback` | `true` | 末档换 `GlobalHistogramBinarizer` 再试一次 |
| `max-results` | `16` | 结果数上限 |

> 默认值只在 core 的 `BarcodeConfig.DEFAULTS` 定义一次，Properties 从它读取，不会两边漂移。
> 非法配置在启动装配 Bean 时抛 `IllegalArgumentException`（本模块刻意零依赖，不引 `mica-ai-common` 的 `MicaAiException`）。

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class BarcodeController {

    private final BarcodeEngine barcodeEngine;

    @PostMapping("/barcode/decode")
    public List<Map<String, Object>> decode(@RequestParam("file") MultipartFile file) throws IOException {
        return barcodeEngine.decodeBytes(file.getBytes()).stream()
            .map(code -> {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("text", code.getText());
                item.put("symbology", code.getSymbology().name());
                item.put("boundingBox", code.getBoundingBox());
                item.put("strategy", code.getStrategy());
                return item;
            })
            .collect(Collectors.toList());
    }
}
```

可运行版本见 [`mica-ai-example`](../../mica-ai-example/README.md) 的 `/barcode/decode` 与 `/barcode/decode-first`。

### 公开 API（`BarcodeEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `decode(BarcodeImage)` | `List<BarcodeResult>` | 全量扫描，一图多码 |
| `decodeFirst(BarcodeImage)` | `Optional<BarcodeResult>` | 短路链，只要一个码（单码图快一个数量级） |
| `decodeBytes(byte[])` / `decodeFirstBytes(byte[])` | 同上 | PNG / JPG 等编码字节 |
| `decodePath(String)` / `decodeFirstPath(String)` | 同上 | 文件路径 |
| `formats()` | `List<BarcodeSymbology>` | 当前启用的制式 |
| `getConfig()` | `BarcodeConfig` | 当前配置 |

---

## 4. 解码链与边界

策略顺序：`Hybrid + TRY_HARDER` → `×2/×3 最近邻放大` → `2×2 带重叠分块` → `GlobalHistogram` 回退。
实测依据（16 档 × N=12 的退化样本）见 `mica-ai-core/mica-ai-barcode` 的单测 `BarcodeSamples` / `BarcodeEngineTest`：1px 小码 0→100%、模糊 σ=2 的 0→100%、一图 4 码 25→100%，且不依赖任何 CNN 定位模型。

落地时请注意：

- **一维码必须原分辨率**：缩小或模糊后所有策略都是 0%，不要先缩放再送解码。
- **透视 / 组合退化是共同短板**：文档图建议先过 `mica-ai-docalign` 拉正；`mica-ai-example` 的 `DocumentPipeline` 已把条码阶段挂在拉正与整图转正之后、超分之前（原分辨率）。
- **中文 QR 以 zxing 为准**，不要拿 OpenCV 的 `QRCodeDetector` 结果兜底（ECI 警告、乱码风险）。
- 探针是合成样本，非实拍；上线前建议用真实票据 / 面单 / 资产标签复跑一遍。

---

## 5. 线程安全与生命周期

`BarcodeEngine` **线程安全**，可作为单例共享；解码内部每次新建有状态的 `MultiFormatReader`。
它**不持有任何需要释放的资源**（无 ONNX session、无 `Mat`、无文件句柄），因此自动装配的 Bean **不设 `destroyMethod`**，也无需 try-with-resources。

---

## 6. 自定义 Bean

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 Bean：

```java
@Bean
public BarcodeEngine myBarcodeEngine(BarcodeProperties props) {
    return BarcodeEngine.create(props.toConfig().toBuilder()
        .tileRows(3)
        .tileCols(3)
        .build());
}
```

---

## 7. 测试覆盖

`BarcodePropertiesTest` 覆盖：默认值与 core `DEFAULTS` 逐项一致、kebab-case / 枚举列表 / 整数列表绑定、
`toConfig()` 把每个属性都装进 `BarcodeConfig`（漏装配一项即失败）、非法配置 fail-fast。
核心解码链的 19 个回归测试在 [`mica-ai-barcode`](../../mica-ai-core/mica-ai-barcode/README.md) 模块。
