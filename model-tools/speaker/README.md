# speaker 模型工具

> 对应 mica-ai-speaker（ERes2Net 声纹识别）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 声纹验证 / 说话人识别 |
| ModelScope | [`iic/speech_eres2netv2_sv_zh-cn_16k-common`](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common) |
| Embedding 维度 | 192 |
| 采样率 | 16 kHz |
| 特征 | 80 维 log-Mel FBank |

> 上游 ModelScope 仓库的 ONNX 文件可能不齐全，必要时从 PyTorch 权重自行导出。

## 快速使用

### 安装依赖

```bash
# 公共依赖
pip install -r ../requirements.txt
# 私有依赖（PyTorch + onnx + onnxsim + onnxruntime）
pip install -r requirements.txt

# threed-speaker 路径（推荐）需要：
pip install 3D-Speaker
```

### 端到端

```bash
# 1. 下载
python download.py --variant eres2netv2

# 2. 导出 ONNX
# 路径 A（推荐）：用 3D-Speaker 加载并剥离 frontend
python convert.py --method threed-speaker --variant eres2netv2

# 路径 B：用 funasr 加载
python convert.py --method funasr --variant eres2netv2

# 路径 C（占位）：打印操作指南
python convert.py --method manual
```

`convert.py` 详细文档见 [convert.py](convert.py) 顶部 docstring。

最终产物：

```
model-tools/speaker/model/out/eres2net.onnx
```

## Java 侧配置

```yaml
mica:
  ai:
    speaker:
      model-path: <abs>/model/out/eres2net.onnx
```
