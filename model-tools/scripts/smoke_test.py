"""model-tools 工具链的离线冒烟测试。

不下载任何模型、不连外网，只验证：
  1. Python 版本满足要求（>=3.10）
  2. common/ 模块可以正常 import
  3. face 能力子目录文件齐全
  4. 所有 .py 文件语法合法（用 py_compile 重核一次）

用法：
    python scripts/smoke_test.py
    python -m scripts.smoke_test          # 等价

返回码：
    0 = 全部通过
    1 = 至少一项失败
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

# 把自己加进 sys.path，便于 ``python scripts/smoke_test.py`` 也能 import common
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import cap_models_dir, mica_root
from common.progress import fail, ok, step, warn

MIN_PY = (3, 10)

CAPS = ("face",)

EXPECTED_FILES: dict[str, tuple[str, ...]] = {
    "face": ("README.md", "download.py", "convert.py", "requirements.txt"),
}

ALL_PY_FILES = [
    "common/__init__.py", "common/paths.py", "common/progress.py",
    "common/downloader.py", "common/onnx_utils.py",
    "face/download.py", "face/convert.py",
    "scripts/smoke_test.py",
]

# 真正能 import 的脚本（不带 argparse 副作用）
IMPORTABLE_SCRIPTS = [
    "face.download", "face.convert",
]


def check_python_version() -> bool:
    step(f"检查 Python 版本（要求 >= {MIN_PY[0]}.{MIN_PY[1]}）")
    if sys.version_info < MIN_PY:
        fail(f"当前 Python {sys.version.split()[0]}，过低")
        return False
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_common_imports() -> bool:
    step("导入 common 模块")
    try:
        mod = importlib.import_module("common")
        for name in ("download_model", "DownloadSource", "DownloadSpec",
                     "cap_models_dir", "mica_root", "step", "ok", "fail", "warn",
                     "check_onnx", "quantize_dynamic"):
            if not hasattr(mod, name):
                fail(f"common.{name} 缺失")
                return False
        ok(f"common 模块导出齐全（{len(mod.__all__)} 个公开符号）")
        return True
    except Exception as e:
        fail(f"导入 common 失败: {e}")
        return False


def check_capabilities() -> bool:
    step("检查 face 能力子目录")
    all_ok = True
    for cap in CAPS:
        cap_dir = ROOT / cap
        if not cap_dir.is_dir():
            fail(f"[{cap}] 目录不存在: {cap_dir}")
            all_ok = False
            continue
        missing = [f for f in EXPECTED_FILES[cap] if not (cap_dir / f).is_file()]
        if missing:
            fail(f"[{cap}] 缺少文件: {missing}")
            all_ok = False
            continue
        ok(f"[{cap}] {len(EXPECTED_FILES[cap])} 个文件齐全")
    return all_ok


def check_py_syntax() -> bool:
    step(f"重核 {len(ALL_PY_FILES)} 个 .py 文件语法")
    all_ok = True
    for rel in ALL_PY_FILES:
        path = ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            fail(f"[{rel}] 语法错误: {e}")
            all_ok = False
    if all_ok:
        ok("所有 .py 语法通过")
    return all_ok


def check_imports() -> bool:
    """真正 import 每个 capability 脚本（不执行 main），确保顶层 import 链不破。"""
    all_ok = True
    for mod in IMPORTABLE_SCRIPTS:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if any(p in str(e) for p in ("modelscope", "onnxruntime")):
                warn(f"[{mod}] 跳过（缺可选依赖: {e.name}）")
                continue
            fail(f"[{mod}] 不可 import: {e}")
            all_ok = False
        except Exception as e:
            fail(f"[{mod}] import 失败: {e}")
            all_ok = False
    if all_ok:
        ok(f"所有 {len(IMPORTABLE_SCRIPTS)} 个轻量脚本 import 成功")
    return all_ok


def check_root_layout() -> bool:
    step("检查根目录结构")
    expected = ("README.md", "Makefile", "requirements.txt", ".gitignore", "common")
    missing = [f for f in expected if not (ROOT / f).exists()]
    if missing:
        fail(f"根目录缺少: {missing}")
        return False
    ok("根目录结构完整")
    return True


def main() -> int:
    print("=" * 60)
    print(" mica-ai model-tools 冒烟测试")
    print("=" * 60)
    print(f" repo root: {mica_root()}")
    print(f" face models dir: {cap_models_dir('face')}")
    print()

    results = {
        "Python 版本": check_python_version(),
        "common 导入":  check_common_imports(),
        "根目录":       check_root_layout(),
        "能力子目录":    check_capabilities(),
        "Python 语法":  check_py_syntax(),
        "脚本 import":  check_imports(),
    }

    print()
    print("=" * 60)
    print(" 汇总")
    print("=" * 60)
    for name, ok_flag in results.items():
        print(f"  {'✅' if ok_flag else '❌'}  {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  通过 {passed}/{total}")

    if all(results.values()):
        ok("冒烟测试全部通过 🎉")
        return 0
    fail("冒烟测试未通过，请根据上方提示修复")
    return 1


if __name__ == "__main__":
    sys.exit(main())
