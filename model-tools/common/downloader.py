"""统一的模型下载器。

设计要点：
- 默认走 **ModelScope**（国内速度快，覆盖 PaddleOCR / SenseVoice / Kokoro 等）
- 可通过参数切换到 **Hugging Face**
- 内置重试（网络抖动时自动重试 3 次）
- 解析模型目录下的特定文件（如 ``vocab.txt``），返回绝对路径
- 与 mica-ai 版本号联动，写入 manifest 文件

用法：

.. code-block:: python

    from common import download_model, DownloadSource

    model_dir = download_model(
        cap="intent",
        modelscope_id="AI-ModelScope/chinese-bert-wwm-ext",
    )
    print(model_dir)  # .../model-tools/intent/model/chinese-bert-wwm-ext
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from common.paths import MICA_VERSION, cap_models_dir, ensure_dir
from common.progress import fail, info, ok, step, warn


class DownloadSource(str, Enum):
    """模型下载来源。"""

    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    DIRECT = "direct"  # 适合没有 ModelScope 镜像的官方资源（如 PaddlePaddle tar）


@dataclass
class DownloadSpec:
    """单个模型下载规格。

    - 默认走 ModelScope（``modelscope_id``）
    - 若 ``source=DIRECT``，则下载 ``url``（可附带 ``url_sha256`` 校验）
    """

    target_subdir: str
    """下载到 ``<cap>/model/<target_subdir>``。"""

    modelscope_id: str | None = None
    """ModelScope 模型 ID（国内镜像仓库的 id）。"""

    huggingface_id: str | None = None
    """HuggingFace 模型 ID，仅在 ``source=HUGGINGFACE`` 时使用。"""

    url: str | None = None
    """直接下载 URL。"""

    url_sha256: str | None = None
    """直链文件 SHA256 校验值，可选。"""

    url_archive: str | None = None
    """直链下载的压缩格式：``tar`` / ``zip`` / ``none``。默认自动识别。"""

    required_files: tuple[str, ...] = ()
    """下载后必须存在的文件列表（用于完整性校验）。"""


def _huggingface_snapshot(repo_id: str, local_dir: Path) -> Path:
    """从 HuggingFace 下载模型（通过 ``huggingface_hub``）。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        fail("请先安装 huggingface_hub: pip install huggingface_hub")
        raise e

    info(f"开始从 HuggingFace 下载 {repo_id} ...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        max_workers=4,
    )
    return Path(path)


def _direct_download(
    url: str,
    target_dir: Path,
    *,
    sha256: str | None = None,
    archive: str | None = None,
) -> Path:
    """直接下载 URL 到 ``target_dir``，可选 sha256 校验和解压。"""
    import hashlib
    import tarfile
    import zipfile

    import requests

    target_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("?")[0].rsplit("/", 1)[-1]
    archive_path = target_dir / filename

    info(f"下载直链: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(archive_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
        if total and downloaded != total:
            warn(f"下载大小不匹配: 期望 {total} 字节，实际 {downloaded} 字节")

    if sha256:
        actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual.lower() != sha256.lower():
            archive_path.unlink(missing_ok=True)
            raise ValueError(f"SHA256 校验失败: 期望 {sha256}，实际 {actual}")
        info("SHA256 校验通过 ✓")

    # 解压
    fmt = (archive or _guess_archive(archive_path.name)).lower()
    if fmt == "tar":
        info(f"解压 tar: {archive_path.name}")
        with tarfile.open(archive_path) as tf:
            tf.extractall(target_dir)
        archive_path.unlink()
    elif fmt == "zip":
        info(f"解压 zip: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
        archive_path.unlink()
    elif fmt == "none":
        info(f"无需解压，保留原始文件: {archive_path.name}")
    else:
        warn(f"未知的归档格式 '{fmt}'，跳过解压")

    return target_dir


def _guess_archive(filename: str) -> str:
    """根据文件名后缀猜测压缩格式。"""
    name = filename.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    return "none"


def _modelscope_snapshot(modelscope_id: str, local_dir: Path) -> Path:
    """从 ModelScope 下载模型。"""
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        fail("请先安装 modelscope: pip install modelscope")
        raise e

    info(f"开始从 ModelScope 下载 {modelscope_id} ...")
    path = snapshot_download(
        model_id=modelscope_id,
        cache_dir=str(local_dir.parent),  # modelscope 会自己建子目录
    )
    # modelscope 会把模型放到 cache_dir/<modelscope_id>，移动到 local_dir
    if path != str(local_dir):
        if local_dir.exists():
            warn(f"目标目录已存在，先清理: {local_dir}")
            shutil.rmtree(local_dir)
        shutil.move(path, local_dir)
    return local_dir


def download_model(
    cap: str,
    spec: DownloadSpec | Iterable[DownloadSpec],
    *,
    source: DownloadSource | str = DownloadSource.MODELSCOPE,
    retries: int = 3,
    skip_if_exists: bool = True,
) -> list[Path]:
    """下载一个或多个模型到 ``<cap>/model/<target_subdir>``。

    Parameters
    ----------
    cap : str
        能力名（ppocr / tts / voice / speaker / intent），用于确定默认根目录
    spec : DownloadSpec or Iterable[DownloadSpec]
        一个或多个下载规格
    source : DownloadSource
        下载来源，默认 ModelScope
    retries : int
        网络异常时的重试次数
    skip_if_exists : bool
        若目标目录已存在则跳过（用于断点续传）

    Returns
    -------
    list[Path]
        每个 spec 解析后的实际模型目录（按输入顺序）
    """
    if isinstance(source, str):
        source = DownloadSource(source)
    if isinstance(spec, DownloadSpec):
        spec = [spec]
    spec_list = list(spec)

    cap_root = ensure_dir(cap_models_dir(cap))
    step(f"[{cap}] 目标根目录: {cap_root}  (source={source.value})")

    results: list[Path] = []
    for s in spec_list:
        target_dir = cap_root / s.target_subdir
        if skip_if_exists and target_dir.exists() and any(target_dir.iterdir()):
            info(f"已存在，跳过下载: {target_dir}")
            results.append(target_dir)
            _verify(s, target_dir)
            _write_manifest(target_dir, cap, s, source)
            continue

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if source is DownloadSource.MODELSCOPE:
                    if not s.modelscope_id:
                        raise ValueError(f"spec {s.target_subdir} 未指定 modelscope_id")
                    _modelscope_snapshot(s.modelscope_id, target_dir)
                elif source is DownloadSource.HUGGINGFACE:
                    if not s.huggingface_id:
                        raise ValueError(f"spec {s.target_subdir} 未指定 huggingface_id")
                    _huggingface_snapshot(s.huggingface_id, target_dir)
                else:  # DIRECT
                    if not s.url:
                        raise ValueError(f"spec {s.target_subdir} 未指定 url")
                    _direct_download(
                        s.url, target_dir, sha256=s.url_sha256, archive=s.url_archive
                    )
                last_err = None
                break
            except Exception as e:  # 网络/校验失败等
                last_err = e
                warn(f"第 {attempt}/{retries} 次下载失败: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
        if last_err is not None:
            fail(f"下载 {s.modelscope_id} 失败: {last_err}")
            raise last_err

        ok(f"下载完成: {target_dir}")
        _verify(s, target_dir)
        _write_manifest(target_dir, cap, s, source)
        results.append(target_dir)

    return results


def _verify(spec: DownloadSpec, target_dir: Path) -> None:
    """检查 ``required_files`` 是否齐全。"""
    missing = [f for f in spec.required_files if not (target_dir / f).exists()]
    if missing:
        fail(f"{target_dir} 缺少必要文件: {missing}")
        raise FileNotFoundError(missing)


def _write_manifest(target_dir: Path, cap: str, spec: DownloadSpec, source: DownloadSource) -> None:
    """在模型目录写一个 ``.mica-manifest.json``，方便追溯。"""
    import json

    manifest = {
        "mica_version": MICA_VERSION,
        "capability": cap,
        "target_subdir": spec.target_subdir,
        "modelscope_id": spec.modelscope_id,
        "huggingface_id": spec.huggingface_id,
        "url": spec.url,
        "source": source.value,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (target_dir / ".mica-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
