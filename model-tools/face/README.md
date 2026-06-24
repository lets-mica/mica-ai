# face 模型工具

> 对应 mica-ai-face（InsightFace 人脸检测 + 识别）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 人脸检测 + 识别（检测 + 1:N 检索由调用方实现） |
| 模型集 | buffalo_l（精度优先） |
| 检测 ONNX | `det_10g.onnx`（RetinaFace，640×640 BGR 输入） |
| 识别 ONNX | `w600k_r50.onnx`（ArcFace w600k_r50，112×112 BGR 输入，512d 输出） |
| Embedding 维度 | 512，已 L2 归一化 |
| 来源 | <https://github.com/deepinsight/insightface/releases/tag/v0.7> |

> buffalo_l 包内还包含 `2d106det.onnx`（106 关键点）和 `genderage.onnx`（性别/年龄），本模块**不依赖**这两个，按需可自行扩展。

## 快速使用

### 安装依赖

```bash
pip install -r ../requirements.txt
# 私有依赖
pip install -r requirements.txt
```

### 端到端

```bash
# 1. 下载 buffalo_l.zip（~340 MB，含全部 4 个 ONNX）
python download.py

# 2. 把 mica-ai-face 真正用到的两个 ONNX 拷贝到 model/out/
python convert.py

# 可选：用符号链接代替复制（节省磁盘）
python convert.py --link
```

最终产物：

```
model-tools/face/model/out/det_10g.onnx     # RetinaFace
model-tools/face/model/out/w600k_r50.onnx   # ArcFace
```

## Java 侧配置

```yaml
mica:
  ai:
    face:
      det-model-path: <abs>/model/out/det_10g.onnx
      rec-model-path: <abs>/model/out/w600k_r50.onnx
      det-score-threshold: 0.5
      det-nms-threshold: 0.4
```

## 模型替换

需要换别的模型集（如 buffalo_s 轻量、buffalo_sc 嵌入式），只需要：

1. 把 `download.py` 里的 `BUFFALO_L_ZIP_URL` 替换为对应 release 的 zip
2. 检查 `convert.py` 里的 `REQUIRED_FILES` 是否还需要调整（det_500m.onnx / w600k_r50.onnx 等）
3. 不需要改 Java 端代码

## 检索不在本模块

本模块只负责把图片转成 512d 向量。1:N 检索请用向量数据库（Milvus / pgvector / Qdrant）实现。详见 [`mica-ai-face/README.md`](../../mica-ai-core/mica-ai-face/README.md)。
