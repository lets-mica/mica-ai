# mica-ai-common

> mica-ai 公共模块，提供统一接口、工具类和异常定义，被所有核心模块依赖。

零业务逻辑，纯基础设施层。所有模块通过它共享一致的基础能力。

---

## 1. 环境要求

| 组件 | 版本     | 说明                                    |
|------|--------|---------------------------------------|
| JDK | 8+     | 推荐 Azul Zulu 8 / Temurin 8 / Oracle 8 |
| Maven | 3.6+   | 编译 / 打包                               |
| ONNX Runtime | 1.18.0 | 通过 Maven 自动拉取                         |

---

## 2. 核心组件

| 组件 | 类 | 说明 |
|------|-----|------|
| **统一异常** | `MicaAiException` / `ErrorCode` | 所有模块异常均继承 `MicaAiException`，便于统一处理 |
| **ONNX 会话持有者** | `OnnxModelSession` | 单模型 `OrtSession` 持有者，支持 `classpath:` 前缀资源解析 |
| **ONNX 会话工厂** | `OrtSessionFactory` | `OrtSessionOptions` → 原生 `OrtSession.SessionOptions` 的映射 |
| **ONNX 配置** | `OrtSessionOptions` / `OrtDevice` / `OrtExecutionMode` / `OrtGraphOptimizationLevel` | 跨能力共享的会话基础配置（线程数 / 优化级别 / 设备 等） |
| **ONNX Provider** | `OrtProviders` | ONNX Runtime 执行提供者（Execution Provider）自动选择与注册 |

### ONNX Provider 管理

`OrtProviders` 提供两个静态方法，由 [OrtSessionFactory](src/main/java/net/dreamlu/mica/ai/common/onnx/OrtSessionFactory.java) 在 `device = GPU` 时调用：

```java
import ai.onnxruntime.OrtSession;
import net.dreamlu.mica.ai.common.onnx.OrtProviders;

// 解析当前运行时可用的 provider 名称；preferCpu=true 强制 CPU
// false 时按 CoreML (macOS) > CUDA > CPU 自动选择
String[] providers = OrtProviders.resolve(false);

// 把解析到的首个 provider 注册到 SessionOptions
// 仅 CUDA / CoreML 会真正注册；注册失败仅 warn，自动回退 CPU
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
OrtProviders.apply(providers, opts, /* deviceId = */ 0);
```

通常业务代码不需要直接调用 `OrtProviders`，只需要在 `OrtSessionOptions.builder().device(OrtDevice.GPU)` 启用 GPU，工厂会自动按上述流程解析并注册。

---

## 3. 使用

作为本项目的内部模块直接依赖即可，坐标由根 POM 统一管理：

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-common</artifactId>
</dependency>
```
