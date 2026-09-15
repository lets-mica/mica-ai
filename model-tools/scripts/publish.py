"""把 ``model-tools/face/model/out/`` 的最终 ONNX 产物整理到
``model-tools/models/face/`` 目录，方便打包成 GitHub Release。

设计原则
========

1. **零侵入**：不动 ``face/model/`` 下的任何原始文件，只读 ``out/`` 然后拷贝。
2. **可重入**：目标已存在时按 SHA256 校验决定是否覆盖。
3. **可审计**：同时生成 ``manifest.json`` + ``manifest.csv``，每行含
   ``cap/file/size/sha256/source`` 字段。
4. **离线可校验**：``--verify`` 模式只读 manifest 重新计算 SHA256 并比对。

用法
====

::

    # 把 face 的 out/ 整理到 model-tools/models/
    python scripts/publish.py

    # 验证已有 manifest（不重新拷贝）
    python scripts/publish.py --verify

    # 自定义输出根目录（默认 model-tools/models）
    python scripts/publish.py --out /tmp/mica-models
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.paths import MICA_VERSION, mica_root
from common.progress import fail, info, ok, step, warn

DEFAULT_OUT = ROOT / "models"
MANIFEST = "manifest.json"
MANIFEST_CSV = "manifest.csv"

# ---------------------------------------------------------------------------
# face 搬运规则
# ---------------------------------------------------------------------------

CAP_RULES: list = [
    {
        "cap": "face",
        "out_dir": ROOT / "face" / "model" / "out",
        "scope": None,
        "files": [
            "face_detection_yunet_2023mar.onnx",
            "face_recognition_sface_2021dec.onnx",
        ],
        "source_url": "https://github.com/opencv/opencv_zoo",
        "license": "Apache-2.0",
    },
    {
        "cap": "filetype",
        "out_dir": ROOT / "filetype" / "model" / "out",
        "scope": None,
        "files": [
            "model.onnx",
            "config.min.json",
            "content_types_kb.min.json",
        ],
        "source_url": "https://github.com/google/magika",
        "license": "Apache-2.0",
    },
]


@dataclass
class Entry:
    cap: str
    relpath: str
    abs_src: str
    size: int
    sha256: str
    license: str
    source_url: str

    @property
    def id(self) -> str:
        return f"{self.cap}/{self.relpath}"


def sha256_file(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_files(rule: dict) -> list:
    out_dir: Path = rule["out_dir"]
    paths: list = []
    for rel in rule.get("files", ()):
        p = out_dir / rel
        if p.exists():
            paths.append(p)
        else:
            warn(f"  缺失: {p}")
    return paths


def copy_one(src: Path, dst: Path) -> tuple:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size, sha256_file(dst)


def gather_rules(caps):
    rules: list = []
    for r in CAP_RULES:
        rules.append(r)
    if not caps:
        return rules
    wanted = {c.strip() for c in caps if c.strip()}
    return [r for r in rules if r["cap"] in wanted]


def publish(rules: list, out_root: Path) -> list:
    entries: list = []
    for rule in rules:
        cap = rule["cap"]
        out_dir: Path = rule["out_dir"]
        if not out_dir.exists():
            warn(f"[{cap}] {out_dir} 不存在，跳过（先跑 convert.py）")
            continue

        step(f"[{cap}] 整理 {out_dir} -> {out_root / cap}")
        sources = collect_files(rule)
        if not sources:
            warn(f"  没有可搬运的文件")
            continue

        for src in sources:
            rel = src.relative_to(out_dir).as_posix()
            tgt = out_root / cap / rel
            size, sha = copy_one(src, tgt)
            info(f"  OK {cap}/{rel}  ({size // 1024} KB)")
            entries.append(Entry(
                cap=cap,
                relpath=rel,
                abs_src=str(src.resolve()),
                size=size,
                sha256=sha,
                license=rule.get("license", "Apache-2.0"),
                source_url=rule.get("source_url", ""),
            ))

    return entries


def write_manifest(entries: list, out_root: Path) -> None:
    payload = {
        "mica_version": MICA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "out_root": str(out_root.resolve()),
        "total_files": len(entries),
        "total_size": sum(e.size for e in entries),
        "entries": [asdict(e) for e in entries],
    }
    json_path = out_root / MANIFEST
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ok(f"  + {MANIFEST}  ({len(entries)} 项, {payload['total_size'] // 1024 // 1024} MB)")

    csv_path = out_root / MANIFEST_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "cap", "relpath", "size", "sha256", "license", "source_url"])
        for e in entries:
            w.writerow([e.id, e.cap, e.relpath, e.size, e.sha256, e.license, e.source_url])
    ok(f"  + {MANIFEST_CSV}")


def verify(out_root: Path) -> int:
    json_path = out_root / MANIFEST
    if not json_path.exists():
        fail(f"未找到 {json_path}")
        return 1
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entries = payload["entries"]
    step(f"校验 {len(entries)} 个文件 SHA256")
    bad = 0
    for e in entries:
        p = out_root / e["cap"] / e["relpath"]
        if not p.exists():
            warn(f"  缺失: {p}")
            bad += 1
            continue
        actual = sha256_file(p)
        if actual != e["sha256"]:
            fail(f"  不一致: {p}  expect={e['sha256'][:12]}  actual={actual[:12]}")
            bad += 1
        else:
            info(f"  OK {e['cap']}/{e['relpath']}")
    if bad:
        fail(f"{bad} 项校验失败")
        return 1
    ok("全部一致")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把 face 的 out/ 整理到 model-tools/models/")
    p.add_argument("--cap", help="只搬运指定 cap（默认 face）")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出根目录")
    p.add_argument("--verify", action="store_true", help="只校验已有 manifest")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root: Path = args.out
    if args.verify:
        return verify(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    step(f"输出根目录: {out_root}")
    rules = gather_rules(None if not args.cap else args.cap.split(","))
    info(f"共 {len(rules)} 条 cap 规则：")
    for r in rules:
        info(f"  - {r['cap']}")
    entries = publish(rules, out_root)
    if not entries:
        fail("没有可搬运的文件（请先跑 face/convert.py）")
        return 1
    write_manifest(entries, out_root)
    ok("全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
