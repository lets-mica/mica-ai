# filetype 模型目录

Google Magika `standard_v3_3`（Apache-2.0，可商用）文件类型识别模型，**已直接入库**，无需下载脚本。

## 模型清单（`models/`）

| 文件 | 说明 |
|------|------|
| `model.onnx` | Magika ONNX 模型（输入 `int32[1,2048]` → 输出 `float32[1,214]`） |
| `config.min.json` | 模型配置（`beg_size` / `end_size` / `block_size` / `padding_token` / `thresholds` / `overwrite_map`） |
| `content_types_kb.min.json` | 214 类内容类型知识库（`mime_type` / `group` / `description` / `extensions` / `is_text`） |

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
      content-types-path: model-tools/filetype/models/content_types_kb.min.json
```

模型规格 / I/O 格式见 [`mica-ai-core/mica-ai-filetype/README.md`](../../mica-ai-core/mica-ai-filetype/README.md)。

## 替换 / 升级

1. 按 `AGENTS.md` §6.1 完成 License 商用自检
2. 覆盖 `models/` 下的三件套
3. 跑 filetype 模块集成测试确认可加载且推理正常：`mvn -pl mica-ai-core/mica-ai-filetype -am test`
4. 更新 `mica-ai-core/mica-ai-filetype/README.md` 的模型规格表