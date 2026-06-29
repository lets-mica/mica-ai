"""把 ``model-tools/<cap>/model/out/`` 的最终 ONNX 产物整理到
``model-tools/models/<cap>/`` 目录，方便打包成 GitHub Release。

设计原则
========

1. **零侵入**：不动 ``<cap>/model/`` 下的任何原始文件，只读 ``out/`` 然后拷贝。
2. **可重入**：目标已存在时按 SHA256 校验决定是否覆盖。
3. **可分卷**：每个能力单独成一个子目录；ppocr 按 spec 拆子目录。
4. **可审计**：同时生成 ``manifest.json`` + ``manifest.csv``，每行含
   ``cap/scope/file/size/sha256/source`` 字段。
5. **离线可校验**：``--verify`` 模式只读 manifest 重新计算 SHA256 并比对。

用法
====

::

    # 把所有已完成 convert 的 cap 整理到 model-tools/models/
    python scripts/publish.py

    # 只整理指定 cap
    python scripts/publish.py --cap face,tts

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
from typing import Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.paths import MICA_VERSION, mica_root
from common.progress import fail, info, ok, step, warn

DEFAULT_OUT = ROOT / "models"
MANIFEST = "manifest.json"
MANIFEST_CSV = "manifest.csv"

# ---------------------------------------------------------------------------
# 每个 cap 的搬运规则
# ---------------------------------------------------------------------------

PPOCR_SPECS = ("tiny", "small", "medium")

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
        "cap": "tts",
        "out_dir": ROOT / "tts" / "model" / "out",
        "scope": None,
        "files_top": ["model_dynamic.onnx", "config.json"],
        "files_globs": ["voices/*.bin"],
        "source_url": "https://www.modelscope.cn/models/KeanuX/Kokoro-82M-v1.1-dynamic-static-ONNX",
        "license": "Apache-2.0",
    },
    {
        "cap": "voice",
        "out_dir": ROOT / "voice" / "model" / "out",
        "scope": None,
        # 文件名遵循 HaujetZhao/SenseVoice-ONNX 仓库命名（与 mica-ai-voice Java 测试一致）
        # 外部 data 文件与 .onnx 同目录，ONNX Runtime 自动加载
        "files": [
            "SenseVoice-Encoder.fp32.onnx",
            "SenseVoice-Encoder.fp32.onnx.data",
            "SenseVoice-CTC.fp32.onnx",
            "SenseVoice-CTC.fp32.onnx.data",
            "Tokenizer.bpe.model",
        ],
        "source_url": "https://www.modelscope.cn/models/iic/SenseVoiceSmall",
        "license": "Apache-2.0",
    },
    {
        "cap": "speaker",
        "out_dir": ROOT / "speaker" / "model" / "out",
        "scope": None,
        "files": ["eres2net.onnx"],
        "source_url": "https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "license": "Apache-2.0",
    },
]


def build_ppocr_rules() -> list:
    # ppocr 的 convert.py 把 det/rec 输出到固定路径，多次跑会互相覆盖。
    # 我们在 scripts/ 流程里把每次跑的结果备份到 out-by-spec/<spec>/，
    # publish 从这里读 —— 这样 tiny/small/medium 三套互不污染。
    by_spec = ROOT / "ppocr" / "model" / "out-by-spec"
    rules: list = []
    for spec in PPOCR_SPECS:
        out_dir = by_spec / spec
        if not out_dir.exists():
            continue
        files = ["det/inference.onnx", "rec/inference.onnx"]
        dict_file = out_dir / f"rec_char_dict_{spec}.txt"
        if dict_file.exists():
            files.append(f"rec_char_dict_{spec}.txt")
        rules.append({
            "cap": "ppocr",
            "spec": spec,
            "scope": spec,
            "out_dir": out_dir,
            "files": files,
            "source_url": (
                "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
                f"official_inference_model/paddle3.0.0/tmp/PP-OCRv6_{spec}_"
            ),
            "license": "Apache-2.0",
        })
    return rules


@dataclass
class Entry:
    cap: str
    scope: str
    relpath: str
    abs_src: str
    size: int
    sha256: str
    license: str
    source_url: str

    @property
    def id(self) -> str:
        s = self.scope or "-"
        return f"{self.cap}/{s}/{self.relpath}"


def sha256_file(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_globs(rule: dict) -> list:
    out_dir: Path = rule["out_dir"]
    paths: list = []
    for key in ("files", "files_top"):
        for rel in rule.get(key, ()):
            p = out_dir / rel
            if p.exists():
                paths.append(p)
            else:
                warn(f"  缺失: {p}")
    for pattern in rule.get("files_globs", ()):
        matched = sorted(out_dir.glob(pattern))
        if not matched:
            warn(f"  无匹配: {out_dir}/{pattern}")
        paths.extend(matched)
    return paths


def copy_one(src: Path, dst: Path) -> tuple:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size, sha256_file(dst)


def gather_rules(caps):
    rules: list = []
    rules.extend(build_ppocr_rules())
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
        scope = rule.get("scope") or "-"
        out_dir: Path = rule["out_dir"]
        if not out_dir.exists():
            warn(f"[{cap}/{scope}] {out_dir} 不存在，跳过（先跑 convert.py）")
            continue

        step(f"[{cap}/{scope}] 整理 {out_dir} -> {out_root / cap / (rule.get('scope') or '')}")
        sources = collect_globs(rule)
        if not sources:
            warn(f"  没有可搬运的文件")
            continue

        for src in sources:
            rel = src.relative_to(out_dir).as_posix()
            tgt = out_root / cap / (rule.get("scope") or "") / rel
            size, sha = copy_one(src, tgt)
            info(f"  OK {cap}/{rule.get('scope') or '-'}/{rel}  ({size // 1024} KB)")
            entries.append(Entry(
                cap=cap,
                scope=rule.get("scope") or "",
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
        w.writerow(["id", "cap", "scope", "relpath", "size", "sha256", "license", "source_url"])
        for e in entries:
            w.writerow([e.id, e.cap, e.scope, e.relpath, e.size, e.sha256, e.license, e.source_url])
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
        p = out_root / e["cap"] / (e["scope"] or "") / e["relpath"]
        if not p.exists():
            warn(f"  缺失: {p}")
            bad += 1
            continue
        actual = sha256_file(p)
        if actual != e["sha256"]:
            fail(f"  不一致: {p}  expect={e['sha256'][:12]}  actual={actual[:12]}")
            bad += 1
        else:
            eid = f"{e['cap']}/{e['scope'] or '-'}/{e['relpath']}"
            info(f"  OK {eid}")
    if bad:
        fail(f"{bad} 项校验失败")
        return 1
    ok("全部一致")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把各 cap 的 out/ 整理到 model-tools/models/")
    p.add_argument("--cap", help="只搬运指定 cap（逗号分隔），默认全部")
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
        info(f"  - {r['cap']}/{r.get('scope') or '-'}")
    entries = publish(rules, out_root)
    if not entries:
        fail("没有可搬运的文件（请先跑各 cap 的 convert.py）")
        return 1
    write_manifest(entries, out_root)
    ok("全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
