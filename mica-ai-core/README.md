# mica-ai-core

> mica-ai 核心能力聚合模块，零 Spring 依赖，纯 Java 实现。

本模块为聚合 POM（packaging=pom），包含 1 个子模块：

---

## 子模块一览

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| [mica-ai-face](mica-ai-face/README.md) | OpenCV Zoo 人脸识别（检测 + 512d 向量） | ONNX Runtime |

子模块依赖 [mica-ai-common](../mica-ai-common/README.md)，共享 `MicaAiException` 统一异常、`OrtProviders` Provider 管理。

---

## 设计原则

- **零 Spring 依赖** — 核心模块不引入任何 Spring 类，可在任何 Java 17+ 项目中独立使用
- **Builder 模式** — 所有引擎均通过 Builder 风格的 Config 类构造，链式调用清晰
- **AutoCloseable** — 所有引擎实现 `AutoCloseable`，支持 try-with-resources 自动释放 ONNX 资源
