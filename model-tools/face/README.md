# face 模型工具

> 对应 mica-ai-face（OpenCV Zoo 人脸检测 + 识别，Apache-2.0 可商用）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 人脸检测 + 识别（1:N 检索由调用方实现） |
| 检测 ONNX | `face_detection_yunet_2023mar.onnx`（YuNet，320×320 RGB 输入，~340 KB） |
| 识别 ONNX | `face_recognition_sface_2021dec.onnx`（SFace，112×112 RGB 输入，512d 输出，~89 MB） |
| Embedding 维度 | 512，已 L2 归一化 |
| 来源 | <https://github.com/opencv/opencv_zoo> |
| License | Apache-2.0（含代码与模型权重，**可放心商用**） |

## 快速使用

### 安装依赖

```bash
pip install -r ../requirements.txt
# 私有依赖
pip install -r requirements.txt
```

### 端到端

```bash
# 1. 下载 OpenCV Zoo YuNet + SFace（裸 .onnx，总计 ~90 MB）
python download.py

# 2. 把 mica-ai-face 真正用到的两个 ONNX 拷贝到 model/out/
python convert.py

# 可选：用符号链接代替复制（节省磁盘）
python convert.py --link
```

最终产物：

```
model-tools/face/model/out/face_detection_yunet_2023mar.onnx     # YuNet
model-tools/face/model/out/face_recognition_sface_2021dec.onnx   # SFace
```

## Java 侧配置

```yaml
mica:
  ai:
    face:
      det-model-path: <abs>/model/out/face_detection_yunet_2023mar.onnx
      rec-model-path: <abs>/model/out/face_recognition_sface_2021dec.onnx
      det-score-threshold: 0.6
      det-nms-threshold: 0.3
```

## 模型替换

mica-ai-face 把 `FaceDetector` 和 `FaceRecognizer` 抽象成了接口（`net.dreamlu.mica.ai.face.engine`），
未来要换别的模型集（如 MobileFaceNet、ArcFace R50），只需要：

1. 实现 `FaceDetector` / `FaceRecognizer` 接口（ONNX 推理 + 预处理）
2. 在 `FaceEngine.createDefault*` 里加 `case`（或在调用方用 `.detector(...).recognizer(...)` 注入）
3. 不需要改 mica-ai-face 框架代码

如果要换整个模型集的源头（比如想用其他开源模型），改 `download.py` 里的 URL 即可。

## 检索不在本模块

本模块只负责把图片转成 512d 向量。1:N 检索请用向量数据库（Milvus / pgvector / Qdrant）实现。详见 [`mica-ai-face/README.md`](../../mica-ai-core/mica-ai-face/README.md)。