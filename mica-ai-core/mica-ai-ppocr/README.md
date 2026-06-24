# mica-ai-ppocr

> PP-OCRv6 纯 ONNXRuntime 推理的 **Java 17** 实现，移植自 [`AIwork4me/ppocrv6_onnx`](https://github.com/AIwork4me/ppocrv6_onnx) 的 `ppocrv6_onnx.py` 单文件实现。

零 PaddlePaddle 依赖。完整复现检测 + 识别的预处理 / 后处理逻辑（DB 后处理、CTC 解码、 pyclipper 等价的多边形 unclip）。

本项目优先 CPU 以保证 bit-exact；如需 CUDA，把 `onnxruntime` 替换为 `onnxruntime_gpu` 并设置 `preferAccelerator(true)`。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.26.0 | 通过 Maven 自动拉取 |
| OpenCV | 4.10.0-0 | 通过 Maven 自动拉取（含 Windows/Linux/macOS 原生库） |
| JTS | 1.20.0 | 多边形偏移（pyclipper 等价物） |

## 2. 模型准备

把 `PP-OCRv6` 的两个 ONNX 模型与字符字典放到 `models/` 下：

```
models/
├── PP-OCRv6_tiny_det_onnx/
│   └── inference.onnx
├── PP-OCRv6_tiny_rec_0515_onnx/
│   └── inference.onnx
└── rec_char_dict.txt
```

`rec_char_dict.txt` 已在原 Python 项目中提供，可直接复用
（路径：`ppocrv6_onnx/models/rec_char_dict.txt`，共 7180 字符）。

ONNX 模型从 PaddleX 官方下载：

```bash
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_det_onnx.tar
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/PP-OCRv6_tiny_rec_0515_onnx.tar
tar xf PP-OCRv6_tiny_det_onnx.tar
tar xf PP-OCRv6_tiny_rec_0515_onnx.tar
```

## 3. 快速使用

```java
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Config;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Result;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import nu.pattern.OpenCV;

public class Demo {
  public static void main(String[] args) {
    OpenCV.loadShared();
    Mat img = Imgcodecs.imread("test.png");
    
    PPOcrV6Config config = PPOcrV6Config.builder()
        .detModelPath("models/PP-OCRv6_tiny_det_onnx/inference.onnx")
        .recModelPath("models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx")
        .recCharDictPath("models/rec_char_dict.txt")
        .build();
    
    try (PPOcrV6Engine ocr = new PPOcrV6Engine(config)) {
      List<PPOcrV6Result> results = ocr.run(img);
      for (PPOcrV6Result r : results) {
        System.out.printf("%s  (%.3f)%n", r.text(), r.score());
      }
    }
  }
}
```

## 4. 已知差异

- pyclipper 使用整数内部表示（带 scale 因子），JTS Buffer 使用 double。
  对典型 100~4000 像素范围的文本框，扩展后像素差异在 1px 以内。
- ONNX Runtime Java 端无 CoreML provider 选项，CoreML 加速不可用。


## 5. 模型规格

PP-OCRv6 提供三种规格，替换下载 URL 即可：

| 规格 | 检测模型 | 识别模型 |
|------|----------|----------|
| **tiny**（小） | `PP-OCRv6_tiny_det_onnx.tar` | `PP-OCRv6_tiny_rec_0515_onnx.tar` |
| **small**（中） | `PP-OCRv6_small_det_onnx.tar` | `PP-OCRv6_small_rec_0515_onnx.tar` |
| **medium**（大） | `PP-OCRv6_medium_det_onnx.tar` | `PP-OCRv6_medium_rec_0515_onnx.tar` |

基础 URL：`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp/`
