# ppocr 模型工具

> 对应 mica-ai-ppocr（PP-OCRv6 ONNX 推理）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 文字检测（det）+ 文字识别（rec） |
| 原始仓库 | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| 官方分发 | PaddleX 提供的 ONNX 推理包（`inference.onnx`） |
| 推荐规格 | server（默认）、mobile、mobile_tiny |

> mica-ai-ppocr 的 Java 侧不需要 PaddlePaddle，只要 ONNX + OpenCV + JTS。
> 因此 `download.py` 拿到的就是 **已转换好的 ONNX**，`convert.py` 只做目录整理。

## 快速使用

```bash
# 安装依赖
pip install -r ../requirements.txt
pip install -r requirements.txt

# 下载 det + rec
python download.py
# 转换：把解压后的子目录扁平化为 mica-ai-ppocr 期望的目录结构
python convert.py
```

最终产物：

```
model-tools/ppocr/model/
├── det/
│   └── inference.onnx
├── rec/
│   └── inference.onnx
└── rec_char_dict.txt
```

## Java 侧配置

`application.yml`：

```yaml
mica:
  ai:
    ppocr:
      det-model-path: <abs>/model/det/inference.onnx
      rec-model-path: <abs>/model/rec/inference.onnx
      rec-char-dict-path: <abs>/model/rec_char_dict.txt
```
