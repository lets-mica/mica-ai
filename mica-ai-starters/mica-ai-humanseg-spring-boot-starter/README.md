# mica-ai-humanseg-spring-boot-starter

> [PP-HumanSeg V2-Lite](https://github.com/PaddlePaddle/PaddleSeg) 人像分割 Spring Boot Starter，基于 [mica-ai-humanseg](../../mica-ai-core/mica-ai-humanseg/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `HumanSegEngine` Bean，对外提供 `alpha*` / `matte*` / `cutout*` / `maskBinaryBytes` 四组 API：透明底 PNG、纯色底合成、二值掩码。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-humanseg-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `human_segmentation_pphumanseg_2023mar.onnx` 仅 **5.88 MB**，**已随仓库分发**，无需额外下载。

---

## 2. 配置项（`mica.ai.humanseg` 前缀）

```yaml
mica:
  ai:
    humanseg:
      enabled: true
      model-path: model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx
      input-width: 192                                     # 必须等于模型输入宽
      input-height: 192                                    # 必须等于模型输入高
      binary-threshold: 0.5
      background-color: [255, 255, 255]                    # RGB
      onnx:
        device: CPU                                        # CPU / GPU（枚举）
        intra-op-num-threads: 0                            # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `HumanSegEngine` Bean |
| `model-version` | `human_segmentation_pphumanseg_2023mar` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | **必填** | 模型路径，支持 `classpath:`；**模型不随 jar 分发**，未配置时启动即失败并给出提示；指向本地文件即可切换到竖版 2.3MB 模型 |
| `input-width` / `input-height` | `192` / `192` | 必须等于模型输入宽高，不一致启动即抛 `MicaAiException`（竖版模型为 `256` / `144`） |
| `mean` / `std` | `[0.5,0.5,0.5]` | 归一化参数（**RGB** 顺序，PaddleSeg 默认），非专家不要改 |
| `binary-threshold` | `0.5` | 二值掩码阈值（0~1），仅二值输出接口使用 |
| `background-color` | `[255,255,255]` | 纯色底输出的默认底色（RGB） |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举，GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> 本能力模型**已入库**，默认配置即可启动，无需 `enabled: false` 兜底。

### 切换到竖版 Portrait-PP-HumanSegV2-Lite

同族契约一致（单输出 + 2 通道 Softmax + 通道 1 = 人像），**只改配置**：

```yaml
mica:
  ai:
    humanseg:
      model-path: /data/models/pphumansegv2_lite_portrait.onnx   # 自行 paddle2onnx 转换
      input-width: 256
      input-height: 144
```

模型获取与转换说明见 [`model-tools/humanseg/README.md`](../../model-tools/humanseg/README.md)。

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class HumanSegController {

    private final HumanSegEngine humanSegEngine;

    /** 人像抠像，直接返回透明底 PNG */
    @PostMapping(value = "/humanseg/cutout", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] cutout(@RequestParam("file") MultipartFile file) throws IOException {
        return humanSegEngine.cutoutBytes(file.getBytes());
    }

    /** 抠像 + 换底色（如证件照蓝底） */
    @PostMapping(value = "/humanseg/cutout-blue", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] cutoutBlue(@RequestParam("file") MultipartFile file) throws IOException {
        return humanSegEngine.cutoutOnColorBytes(file.getBytes(), new int[]{0, 128, 255});
    }

    /** 只要二值掩码 */
    @PostMapping(value = "/humanseg/mask", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] mask(@RequestParam("file") MultipartFile file) throws IOException {
        return humanSegEngine.maskBinaryBytes(file.getBytes());
    }
}
```

### 公开 API（`HumanSegEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `alpha(Mat)` / `alphaBytes(byte[])` / `alphaPath(String)` | `Mat` | 原尺寸 `CV_32FC1` alpha（`[0,1]`），调用方负责 release |
| `matte(Mat)` / `matteBytes(byte[])` | `HumanSegResult` | 带尺寸的 alpha 包装，`AutoCloseable` |
| `cutoutBytes(byte[])` / `cutoutPath(String)` | `byte[]` | **透明底 PNG**（BGRA 4 通道） |
| `cutoutOnColorBytes(byte[], int[])` | `byte[]` | 合成到指定纯色底（RGB）后的 PNG，`null` 时用配置的默认底色 |
| `maskBinaryBytes(byte[])` | `byte[]` | 二值掩码 PNG（单通道 `0/255`） |
| `modelInputWidth()` / `modelInputHeight()` / `outputCount()` | `int` | 模型自检：输入宽高 / 输出节点数（应为 192 / 192 / 1） |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `HumanSegEngine` Bean：

```java
@Configuration
public class MyHumanSegConfig {

    @Bean(destroyMethod = "close")
    public HumanSegEngine myEngine(HumanSegProperties props) {
        return HumanSegEngine.create(HumanSegAutoConfiguration.toConfig(props));
    }
}
```

只要返回 `HumanSegEngine` 即可被 Starter 识别为自定义实现，**业务代码零改动**。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 测试覆盖

- `HumanSegPropertiesTest`：默认值、字段全量映射、`onnx` 实例共享
- [`HumanSegIntegrationTest`](../../mica-ai-core/mica-ai-humanseg/src/test/java/net/dreamlu/mica/ai/humanseg/HumanSegIntegrationTest.java)：真模型推理、透明底 / 二值输出、尺寸不符快速失败
