# mica-ai-enhance-spring-boot-starter

> [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-general-x4v3` 图像超分 Spring Boot Starter，基于 [mica-ai-enhance](../../mica-ai-core/mica-ai-enhance/README.md) 核心模块。**BSD-3-Clause 可商用**。
>
> 零配置即可注入 `EnhanceEngine` Bean，对外提供 `upscale*` API：4 倍放大（大图自动 tile 切分，不爆内存）。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-enhance-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 模型 `realesr-general-x4v3.onnx` 仅 **2.33 MB**，**已随仓库分发**，无需额外下载。

---

## 2. 配置项（`mica.ai.enhance` 前缀）

```yaml
mica:
  ai:
    enhance:
      enabled: true
      model-path: model-tools/enhance/models/realesr-general-x4v3.onnx   # 支持 classpath:
      scale: 4                                           # 必须等于模型实际放大倍数
      tile-size: 256                                     # 原图像素；越小越省内存
      tile-overlap: 16                                   # 消除拼接接缝
      onnx:
        device: CPU                                      # CPU / GPU（枚举）
        intra-op-num-threads: 0                          # 0 = ORT 默认
        inter-op-num-threads: 0
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `EnhanceEngine` Bean |
| `model-version` | `realesr-general-x4v3` | 模型标识（文件名前缀，兼作 classpath 资源名） |
| `model-path` | `classpath:mica-ai/models/enhance/realesr-general-x4v3.onnx` | 模型路径，支持 `classpath:`；指向本地文件即可切换到外置模型 |
| `scale` | `4` | 放大倍数；与模型实际输出/输入比值不一致时**推理期**抛异常 |
| `tile-size` | `256` | 单次推理 tile 边长（原图像素）；图像小于该值时整图推理 |
| `tile-overlap` | `16` | tile 重叠边宽，必须 `< tile-size`；`0` 表示不重叠 |
| `onnx.device` | `CPU` | `CPU` / `GPU`（枚举，GPU 需 classpath 换 `onnxruntime_gpu`） |
| `onnx.intra-op-num-threads` | `0` | ORT 内部线程 |
| `onnx.inter-op-num-threads` | `0` | ORT 交互线程 |

> 本能力模型**已入库**，默认配置即可启动，无需 `enabled: false` 兜底。

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class EnhanceController {

    private final EnhanceEngine enhanceEngine;

    /** 上传图片 → 返回 4 倍放大的 PNG */
    @PostMapping(value = "/enhance/upscale", produces = MediaType.IMAGE_PNG_VALUE)
    public byte[] upscale(@RequestParam("file") MultipartFile file) throws IOException {
        return enhanceEngine.upscaleBytes(file.getBytes());
    }
}
```

### 公开 API（`EnhanceEngine`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `upscale(Mat)` | `Mat` | 4 倍放大后的 BGR Mat，调用方负责 release；空图返回 `null` |
| `upscaleBytes(byte[])` | `byte[]` | 解码 + 放大 + PNG 编码 |
| `upscalePath(String)` | `byte[]` | 读文件 + 放大 + PNG 编码 |
| `outputCount()` | `int` | 模型自检：输出节点数（应为 1） |

---

## 4. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `EnhanceEngine` Bean：

```java
@Configuration
public class MyEnhanceConfig {

    @Bean(destroyMethod = "close")
    public EnhanceEngine myEngine(EnhanceProperties props) {
        return EnhanceEngine.create(EnhanceAutoConfiguration.toConfig(props));
    }
}
```

⚠️ 自定义 Bean 记得保留 `destroyMethod = "close"`，否则容器关闭时底层 `OrtSession` 不会释放。

---

## 5. 测试覆盖

- `EnhancePropertiesTest`：默认值与内置模型一致、字段全量映射到核心配置、`onnx` 实例共享
- [`EnhanceIntegrationTest`](../../mica-ai-core/mica-ai-enhance/src/test/java/net/dreamlu/mica/ai/enhance/EnhanceIntegrationTest.java)：真模型 4 倍输出、与双三次放大对比 MAE < 6%、tile 与整图两条路径、空图返回 `null`
