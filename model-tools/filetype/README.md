# mica-ai-filetype 模型工具链

Google Magika standard_v3_3（Apache-2.0，可商用）的下载与导出脚本。

## 快速开始

```bash
# 1. 下载 ONNX 模型 + config + 知识库
python filetype/download.py

# 2. 校验 + 拷贝到 model/out/
python filetype/convert.py
```

可指定 `--source direct`（默认）/ `huggingface` / `modelscope`。
可设 `MICA_MODELS_DIR` 自定义模型根目录。

## 产物

```
model-tools/filetype/model/out/
├── model.onnx                  ≈3.1 MB，ONNX 推理图
├── config.min.json             模型超参 + 阈值
└── content_types_kb.min.json   ≈45 KB，标签 → mime/group/desc 映射
```

## License

Google Magika: Apache License 2.0（magika 源码 + 模型 + 知识库均 Apache-2.0）。