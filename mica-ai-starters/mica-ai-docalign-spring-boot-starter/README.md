# mica-ai-docalign-spring-boot-starter

> [DocAligner](https://github.com/DocsaidLab/DocAligner) 文档角点检测 / 透视矫正 Spring Boot Starter，基于 [mica-ai-docalign](../../mica-ai-core/mica-ai-docalign/README.md) 核心模块。**Apache-2.0 可商用**。
>
> 零配置即可注入 `DocAlignEngine` Bean，对外提供 `corners*` / `warp*` / `alignBytes` / `alignPath` 四组 API：四角检测、拉伸至指定尺寸、一步拉平成 PNG。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-docalign-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `lcnet100_h_e_bifpn_256_fp32.onnx` 仅 **4.55 MB**，**已随仓库分发**，无需额外下载。

---

## 2. 配置项（`mica.ai.docalign` 前缀）

```yaml
mica:
  ai:
    docalign:
      enabled: true
      model-path: model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx   # 支持 classpath:
      input-width: 256                                     # 必须等于模型输入宽
      input-height: 256                                    # 必须等于模型输入高
      heatmap-threshold: 0.3
      onnx:
        device: CPU                                        # CPU / GPU（枚举）
        intra-op-num-threads: 0                            # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `DocAlignEngine` Bean |
| `model-version` | `lcnet100_h_e_bifpn_256_fp32` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/docalign/lcnet100_h_e_bifpn_256_fp32.onnx` | 模型路径，支持 `classpath:`；**指向本地文件即可切换到 fastvit 等更高精度模型** |
| `input-width` / `input-height` | `256` / `256` | 必须等于模型输入宽高，不一致启动即抛 `MicaAiException` |
| `heatmap-threshold` | `0.3` | 热力图二值化阈值（`[0,1]`），越低越容易把弱响应也算作角点 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举，GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> 本能力模型**已入库**，默认配置即可启动，无需 `enabled: false` 兜底。

### 切换到更高精度模型

同族契约一致（单输入 `img` + 单输出 4 通道热力图 + 输入 256×256），**只改配置**：

```yaml
mica:
  ai:
    docalign:
      # 自行从 DocsaidLab/DocAligner releases 下载（83MB，超出 50MB 入库上限）
      model-path: /data/models/fastvit_sa24_h_e_bifpn_256_fp32.onnx
```

模型清单与获取方式见 [`model-tools/docalign/README.md`](../../model-tools/docalign/README.md)。

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class DocAlignController {

    private final DocAlignEngine docAlignEngine;

    /** 一步拉平：上传斜拍文档 → 返回拉平后的 PNG */
    @PostMapping(value = "/docalign/flatten", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] flatten(@RequestParam("file") MultipartFile file) throws IOException {
        return docAlignEngine.alignBytes(file.getBytes());
    }

    /** 只返回四个角点，前端自己画选框 */
    @PostMapping("/docalign/corners")
    public DocAlignResult corners(@RequestParam("file") MultipartFile file) throws IOException {
        return docAlignEngine.cornersBytes(file.getBytes());
    }
}
```

### 公开 API（`DocAlignEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `corners(Mat)` / `cornersBytes(byte[])` / `cornersPath(String)` | `DocAlignResult` | 4 个角点（原图坐标系，顺序左上 / 右上 / 右下 / 左下）；空图返回 `null` |
| `warp(Mat, DocAlignResult)` | `Mat` | 按角点边长自动推算尺寸拉平，调用方负责 release |
| `warp(Mat, DocAlignResult, int, int)` | `Mat` | 拉平到指定尺寸（如 A4 2480×3508） |
| `alignBytes(byte[])` / `alignPath(String)` | `byte[]` | 检测 + 拉平一步完成，返回 PNG |
| `modelInputWidth()` / `modelInputHeight()` / `outputCount()` | `int` | 模型自检（应为 256 / 256 / 1） |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `DocAlignEngine` Bean：

```java
@Configuration
public class MyDocAlignConfig {

    @Bean(destroyMethod = "close")
    public DocAlignEngine myEngine(DocAlignProperties props) {
        return DocAlignEngine.create(DocAlignAutoConfiguration.toConfig(props));
    }
}
```

只要返回 `DocAlignEngine` 即可被 Starter 识别为自定义实现，**业务代码零改动**。
⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 测试覆盖

- `DocAlignPropertiesTest`：默认值（模型标识 / 输入宽高 / 阈值）、字段全量映射、`onnx` 实例共享
- [`DocAlignIntegrationTest`](../../mica-ai-core/mica-ai-docalign/src/test/java/net/dreamlu/mica/ai/docalign/DocAlignIntegrationTest.java)：真模型推理、4 通道角点顺序、拉平结果尺寸与画面内容、尺寸不符快速失败、空图返回 `null`
