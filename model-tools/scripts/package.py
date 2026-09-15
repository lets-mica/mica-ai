"""把 ``models/face/`` 打包成 zip，用于 GitHub Release 上传。

文件名包含 mica-ai 版本号：
    mica-ai-models-face-2026.06.01.zip

zip 内顶层目录前缀 `mica-ai/`，方便解压后直接配 application.yml。

用法：
    python scripts/package.py                # 默认打包 face
    python scripts/package.py --out /tmp     # 输出到别处
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.paths import MICA_VERSION
from common.progress import fail, ok, step, warn

DEFAULT_IN = ROOT / "models"


def collect_files(in_root: Path) -> list[Path]:
    """返回要打包的文件列表。"""
    if not (in_root / "manifest.json").exists():
        fail(f"未找到 {in_root / 'manifest.json'}，请先跑 publish.py")
        return []

    manifest = json.loads((in_root / "manifest.json").read_text(encoding="utf-8"))
    files: list[Path] = []
    for e in manifest["entries"]:
        cap = e["cap"]
        if cap != "face":
            continue
        src = in_root / cap / e["relpath"]
        if not src.exists():
            warn(f"  缺失: {src}")
            continue
        files.append(src)
    return files


def make_zip(zip_path: Path, files: list[Path], *, prefix: str) -> tuple[Path, int]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in files:
            parts = f.parts
            try:
                idx = parts.index("models")
                inside = "/".join(parts[idx + 1:])
            except ValueError:
                inside = f.name
            zf.write(f, arcname=f"{prefix}/{inside}")
            total += f.stat().st_size
    return zip_path, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把 models/face 打包成 zip")
    p.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, help="zip 输出目录（默认 model-tools/release/）")
    p.add_argument("--prefix", default="mica-ai",
                   help="zip 内顶层目录前缀（默认 mica-ai）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_root: Path = args.in_dir

    files = collect_files(in_root)
    if not files:
        fail("没有可打包的文件")
        return 1
    out_dir = args.out or (in_root.parent / "release")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"mica-ai-models-face-{MICA_VERSION}.zip"

    step(f"打包根目录: {in_root}")
    step(f"zip 输出目录: {out_dir}")
    zip_path, total_size = make_zip(zip_path, files, prefix=args.prefix)
    ok(f"  + {zip_path.name}  ({len(files)} files, {total_size // 1024 // 1024} MB)")
    ok(f"全部完成，1 个 zip，{total_size // 1024 // 1024} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
