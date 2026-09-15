# Model Tools

> mica-ai 的 **Python 端模型工具链**：下载 + 转换。
> 与 Java 业务代码同级、互不依赖、互不污染。

当前只覆盖 `face`（OpenCV Zoo YuNet + SFace，Apache-2.0 可商用）。其他 mica-* 项目的工具链独立维护。

| 脚本 | 用途 |
|------|------|
| `download.py` | 从 OpenCV Zoo GitHub 下载 YuNet + SFace（裸 .onnx，无需转换） |
| `convert.py`  | 校验 ONNX，并把 mica-ai-face 真正用到的两个文件拷贝 / 链接到 `model/out/` |

---

## 📁 目录结构

```
model-tools/
├── README.md                   # 本文件
├── Makefile                    # 一键命令：make download / convert / publish / package
├── requirements.txt            # 公共依赖（modelscope, tqdm, rich, pyyaml）
├── .gitignore                  # 忽略 model/、output/、__pycache__/
│
├── common/                     # 跨能力复用的工具
│   ├── downloader.py           # 统一下载器（ModelScope / HuggingFace / DIRECT）
│   ├── onnx_utils.py           # ONNX 检查 / 简化 / 量化
│   ├── paths.py                # mica_root / cap_models_dir / 版本号
│   └── progress.py             # 带颜色的日志 / 进度条
│
├── face/                       # 对应 mica-ai-face
│   ├── README.md
│   ├── download.py             # OpenCV Zoo YuNet + SFace
│   ├── convert.py              # 校验 + 拷贝到 model/out/
│   └── requirements.txt        # face 私有依赖（仅 requests）
│
└── scripts/                    # 跨能力工具
    ├── smoke_test.py           # 离线冒烟（不下载模型）
    ├── publish.py              # out/ → models/，生成 manifest.json
    └── package.py              # models/face → mica-ai-models-face-*.zip
```

---

## 🚀 快速开始

### 1. 安装 Python 依赖

```bash
# 建议 Python 3.10+
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 公共依赖
pip install -r model-tools/requirements.txt

# face 私有依赖
pip install -r model-tools/face/requirements.txt
```

### 2. 下载 + 转换

```bash
# 顶层 Makefile 一键命令
make -C model-tools download     # 仅 face
make -C model-tools convert      # 仅 face

# 或者直接进入子目录
cd model-tools/face && python download.py
cd model-tools/face && python convert.py       # 默认复制；可加 --link 用符号链接
```

### 3. 整理 + 打包（用于 GitHub Release）

```bash
python model-tools/scripts/publish.py        # out/ → models/face/ + manifest.json
python model-tools/scripts/publish.py --verify   # 校验 manifest
python model-tools/scripts/package.py            # 打 zip：model-tools/release/mica-ai-models-face-*.zip
```

---

## 🔧 设计原则

- **零侵入**：所有脚本放在 `model-tools/` 下，不修改 Java 模块的 `pom.xml`。
- **能力维度切分**：每个能力一个子目录，与 Java 的 `mica-ai-core/mica-ai-xxx/` 一一对应。
- **可重入**：`download.py` 重复执行会跳过已下载文件。
- **版本对齐**：与根 `pom.xml` 的 `<revision>` 同号，写在 `common/paths.py`。

---

## ❓ 与 Java 端如何对接

转换完成后 face 的产物：

```
model-tools/face/model/out/
├── face_detection_yunet_2023mar.onnx     # YuNet（320x320 RGB）
└── face_recognition_sface_2021dec.onnx   # SFace（112x112 RGB, 512d）
```

发布时经 `publish.py` 整理到 `model-tools/models/face/`，再由 `package.py` 打 zip 上传到 GitHub Release。Java 端把目录路径配到 Spring Boot 的 `application.yml` 即可使用：

```yaml
mica:
  ai:
    face:
      det-model-path: <abs>/model/out/face_detection_yunet_2023mar.onnx
      rec-model-path: <abs>/model/out/face_recognition_sface_2021dec.onnx
```
