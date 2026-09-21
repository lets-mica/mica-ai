# mica-ai-matting-spring-boot-starter

> [U²-Net / u2netp](https://github.com/xuebinqin/U-2-Net) 通用抠图 Spring Boot Starter，基于 [mica-ai-matting](../../mica-ai-core/mica-ai-matting/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `MattingEngine` Bean，对外提供 `alpha` / `matte*` / `cutout*` / `matteBinary*` 四组 API：透明底 PNG、纯色底合成、二值掩码。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-matting-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `u2netp.onnx` 仅 **4.36 MB**，**已随仓库分发**，无需额外下载。

---

## 2. 配置项（`mica.ai.matting` 前缀）

```yaml
mica:
  ai:
    matting:
      enabled: true
      model-path: model-tools/matting/models/u2netp.onnx   # 支持 classpath:
      input-size: 320                                      # 必须等于模型输入边长
      output-select: AUTO                                  # AUTO / FIRST / D0
      binary-threshold: 0.5
      min-max-normalize: true
      interpolation: LINEAR                                # LINEAR / NEAREST / CUBIC
      background-color: [255, 255, 255]                    # RGB
      mean: [0.485, 0.456, 0.406]                          # ImageNet，RGB 顺序
      std:  [0.229, 0.224, 0.225]
      onnx:
        device: cpu                                        # cpu / gpu
        intra-op-num-threads: 0                            # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `MattingEngine` Bean |
| `model-version` | `u2netp` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/matting/u2netp.onnx` | 模型路径，支持 `classpath:`；**指向本地文件即可切换到 168MB 级外置模型** |
| `input-size` | `320` | 模型输入边长；**必须等于模型输入边长**，不一致启动即抛 `MicaAiException` |
| `output-select` | `AUTO` | d0 定位策略：`AUTO`（名称提示→首个）/ `FIRST`（强制首个）/ `D0`（强制精确 `d0`，否则启动失败） |
| `mean` / `std` | `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | 归一化参数（**RGB** 顺序，ImageNet 统计量）。通道顺序敏感，非专家不要改 |
| `interpolation` | `LINEAR` | 掩码缩放插值：`LINEAR` / `NEAREST` / `CUBIC` |
| `binary-threshold` | `0.5` | 二值掩码阈值（0~1），仅二值输出接口使用 |
| `min-max-normalize` | `true` | 是否把 d0 min-max 拉伸到 `[0,1]`（rembg 参考行为）；低对比度输入差别巨大 |
| `background-color` | `[255,255,255]` | 纯色底输出的默认底色（RGB） |
| `onnx.device` | `cpu` | `cpu` / `gpu`（GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> 与 layout 不同，本能力模型**已入库**，默认配置即可启动，无需 `enabled: false` 兜底。

### 切换到其它 U²-Net 族模型

`u2net`（168MB，通用显著性完整版）与 `u2net_human_seg`（168MB，人体分割）与内置
`u2netp` 的 ONNX 契约**完全一致**，因此**只改 `model-path` 就行**，无需改代码：

```yaml
mica:
  ai:
    matting:
      # 指向你自行下载的模型文件（两者均 168MB，超出 50MB 入库上限，不随仓库分发）
      model-path: /data/models/u2net_human_seg.onnx
```

模型获取地址与选型建议（含「为什么不要靠文件名判断权重语义」的实测数据）见
[`model-tools/matting/README.md`](../../model-tools/matting/README.md)。

---

## 3. 使用示例

### 3.1 REST 端点：上传图片 → 返回透明底 PNG

```java
@RestController
@RequiredArgsConstructor
public class MattingController {

    private final MattingEngine mattingEngine;

    /** 去背，直接返回透明底 PNG */
    @PostMapping(value = "/matting/cutout", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] cutout(@RequestParam("file") MultipartFile file) throws IOException {
        return mattingEngine.cutoutBytes(file.getBytes());
    }

    /** 去背 + 换底色（如证件照蓝底） */
    @PostMapping(value = "/matting/cutout-blue", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] cutoutBlue(@RequestParam("file") MultipartFile file,
                             @RequestParam(defaultValue = "0,128,255") String rgb)
        throws IOException {
        String[] parts = rgb.split(",");
        int[] bg = {Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2])};
        return mattingEngine.cutoutOnColorBytes(file.getBytes(), bg);
    }

    /** 只要二值掩码 */
    @PostMapping(value = "/matting/mask", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] mask(@RequestParam("file") MultipartFile file) throws IOException {
        return mattingEngine.matteBinaryBytes(file.getBytes());
    }
}
```

### 3.2 业务用法：把 alpha 叠加到自己的画布

```java
@Service
@RequiredArgsConstructor
public class ProductImageService {

    private final MattingEngine mattingEngine;

    /** 生成白底商品主图 */
    public byte[] toWhiteBackground(byte[] src) {
        return mattingEngine.cutoutOnColorBytes(src, new int[]{255, 255, 255});
    }

    /** 自行合成：拿到 alpha 后用自己的逻辑处理（羽化 / 描边 / 多图层） */
    public void custom(byte[] src) throws IOException {
        try (MattingResult r = mattingEngine.matteBytes(src)) {
            Mat alpha = r.getAlpha();     // CV_32FC1，[0,1]，尺寸 = r.getWidth() x r.getHeight()
            // 交由你自己的 OpenCV 合成逻辑处理
        }
    }
}
```

### 3.3 公开 API（`MattingEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `alpha(Mat)` / `alphaBytes(byte[])` / `alphaPath(String)` | `Mat` | 原尺寸 `CV_32FC1` alpha（`[0,1]`），调用方负责 release |
| `matte(Mat)` / `matteBytes(byte[])` | `MattingResult` | 带尺寸的 alpha 包装，`AutoCloseable` |
| `cutoutBytes(byte[])` / `cutoutPath(String)` | `byte[]` | **透明底 PNG**（BGRA 4 通道） |
| `cutoutOnColorBytes(byte[], int[])` | `byte[]` | 合成到指定纯色底（RGB）后的 PNG |
| `matteBinaryBytes(byte[])` | `byte[]` | 二值掩码 PNG（单通道 `0/255`） |
| `modelInputSize()` / `outputCount()` | `int` | 模型自检：输入边长 / 输出节点数（u2netp 应为 320 / 7） |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `MattingEngine` Bean：

```java
@Configuration
public class MyMattingConfig {

    @Bean(destroyMethod = "close")
    public MattingEngine myEngine(MattingProperties props) {
        MattingConfig cfg = MattingConfig.builder()
            .modelPath(props.getModelPath())
            .inputSize(props.getInputSize())
            .binaryThreshold(props.getBinaryThreshold())
            .build();
        return MattingEngine.create(cfg);
    }
}
```

只要返回 `MattingEngine` 即可被 Starter 识别为自定义实现，**业务代码零改动**。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 完整示例

参见 [`mica-ai-example`](../../mica-ai-example/README.md)：已聚合 face / filetype / plate / layout 各 Starter。
本 Starter 的 Bean 装配与配置绑定由 `MattingPropertiesTest` 覆盖，核心链路由
[`MattingIntegrationTest`](../../mica-ai-core/mica-ai-matting/src/test/java/net/dreamlu/mica/ai/matting/MattingIntegrationTest.java) 用真模型覆盖。
