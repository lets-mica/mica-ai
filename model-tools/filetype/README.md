# filetype 模型目录

Google Magika `standard_v3_3`（Apache-2.0，可商用）文件类型识别模型，**已直接入库**，无需下载脚本。

## 模型清单（models/）

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.onnx` | ≈2.4 MB | Magika ONNX 模型 |
| `config.min.json` | ≈0.2 MB | 模型配置（压缩版） |
| `content_types_kb.min.json` | ≈0.5 MB | 214 类内容类型知识库（压缩版） |

- 来源：<https://github.com/google/magika>（`models/standard_v3_3`）
- License：Apache License 2.0，**可商用** ✅

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    filetype:
      model-path: model-tools/filetype/models/model.onnx
      config-path: model-tools/filetype/models/config.min.json
      kb-path: model-tools/filetype/models/content_types_kb.min.json
```

模型规格 / I/O 格式见 [`mica-ai-core/mica-ai-filetype/README.md`](../../mica-ai-core/mica-ai-filetype/README.md)。

## 替换 / 升级

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 覆盖 `models/` 下的三件套
3. `make -C model-tools smoke` 确认 ONNX 结构合法
4. 更新 `mica-ai-core/mica-ai-filetype/README.md` 的模型规格表
