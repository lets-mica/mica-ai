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
| **统一异常** | `MicaAiException` | 所有模块异常均继承此类，便于统一处理 |
| **ONNX Provider** | `OrtProviders` | ONNX Runtime 执行提供者管理（CPU/CUDA/DML 等） |

### ONNX Provider 管理

```java
import net.dreamlu.mica.ai.common.onnx.OrtProviders;

// 获取 CPU provider
OrtProvider cpu = OrtProviders.cpu();

// 获取 CUDA provider（需 onnxruntime_gpu）
OrtProvider cuda = OrtProviders.cuda(0);  // deviceId = 0

// 创建 session options
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
OrtProviders.cpu().applyTo(opts);
```

---

## 3. 使用

作为本项目的内部模块直接依赖即可，坐标由根 POM 统一管理：

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-common</artifactId>
</dependency>
```
