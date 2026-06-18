"""mica-ai 模型工具链的共享工具包。

各能力子目录通过 ``from common import ...`` 复用以下能力：

- :func:`common.downloader.download_model` —— 统一的模型下载入口
- :func:`common.paths.cap_models_dir` —— 各能力的默认模型根目录
- :func:`common.onnx_utils.*` —— ONNX 验证 / 简化 / 量化
- :func:`common.progress.step` —— 带颜色和 emoji 的步骤打印
"""

from common.downloader import download_model, DownloadSource, DownloadSpec
from common.paths import cap_models_dir, ensure_dir, mica_root
from common.progress import step, info, ok, warn, fail
from common.onnx_utils import (
    check_onnx,
    quantize_dynamic,
    simplify_onnx,
    verify_torch_vs_onnx,
)

__all__ = [
    "download_model",
    "DownloadSource",
    "DownloadSpec",
    "cap_models_dir",
    "ensure_dir",
    "mica_root",
    "step",
    "info",
    "ok",
    "warn",
    "fail",
    "check_onnx",
    "simplify_onnx",
    "quantize_dynamic",
    "verify_torch_vs_onnx",
]
