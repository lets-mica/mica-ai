"""model-tools 工具链的离线冒烟测试。

模型已直接入库（各能力子目录 models/），本脚本不连外网，只验证：
  1. Python 版本满足要求（>=3.10）
  2. 根目录结构完整（README / Makefile / .gitignore / requirements.txt）
  3. 各能力子目录的 README 与 models/ 模型文件齐全
  4. ONNX 模型结构合法（需本机安装 onnx，缺失则跳过该项）

用法：
    python scripts/smoke_test.py

返回码：
    0 = 全部通过
    1 = 至少一项失败
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MIN_PY = (3, 10)

# 各能力子目录：README + 必须存在的模型文件
EXPECTED_MODELS: dict[str, tuple[str, ...]] = {
    "face": (
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
    ),
    "filetype": (
        "model.onnx",
        "config.min.json",
        "content_types_kb.min.json",
    ),
    "plate": (
        "y5fu_320x_sim.onnx",
        "y5fu_640x_sim.onnx",
        "rpv3_mdict_160_r3.onnx",
        "litemodel_cls_96x_r1.onnx",
    ),
}


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def step(msg: str) -> None:
    print(f"\n🔹 {msg}")


def check_python_version() -> bool:
    step(f"检查 Python 版本（要求 >= {MIN_PY[0]}.{MIN_PY[1]}）")
    if sys.version_info < MIN_PY:
        fail(f"当前 Python {sys.version.split()[0]}，过低")
        return False
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_root_layout() -> bool:
    step("检查根目录结构")
    expected = ("README.md", "Makefile", "requirements.txt", ".gitignore", "scripts/smoke_test.py")
    missing = [f for f in expected if not (ROOT / f).exists()]
    if missing:
        fail(f"根目录缺少: {missing}")
        return False
    ok("根目录结构完整")
    return True


def check_capabilities() -> bool:
    step("检查各能力子目录（README + models/）")
    all_ok = True
    for cap, models in EXPECTED_MODELS.items():
        cap_dir = ROOT / cap
        if not cap_dir.is_dir():
            fail(f"[{cap}] 目录不存在: {cap_dir}")
            all_ok = False
            continue
        if not (cap_dir / "README.md").is_file():
            fail(f"[{cap}] 缺少 README.md")
            all_ok = False
            continue
        missing = [m for m in models if not (cap_dir / "models" / m).is_file()]
        if missing:
            fail(f"[{cap}] models/ 缺少文件: {missing}")
            all_ok = False
            continue
        ok(f"[{cap}] README + {len(models)} 个模型文件齐全")
    return all_ok


def check_onnx_structure() -> bool:
    step("校验 ONNX 结构（可选，缺 onnx 包则跳过）")
    try:
        import onnx
    except ImportError:
        warn("未安装 onnx，跳过结构校验（pip install onnx）")
        return True

    all_ok = True
    for cap, models in EXPECTED_MODELS.items():
        for name in models:
            path = ROOT / cap / "models" / name
            if path.suffix != ".onnx":
                continue
            try:
                model = onnx.load(str(path))
                onnx.checker.check_model(model)
                ok(f"[{cap}] {name} 结构合法")
            except Exception as e:
                fail(f"[{cap}] {name} 校验失败: {e}")
                all_ok = False
    return all_ok


def main() -> int:
    print("=" * 60)
    print(" mica-ai model-tools 冒烟测试")
    print("=" * 60)

    results = {
        "Python 版本": check_python_version(),
        "根目录": check_root_layout(),
        "能力子目录": check_capabilities(),
        "ONNX 结构": check_onnx_structure(),
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
        print("\n✅ 冒烟测试全部通过 🎉")
        return 0
    print("\n❌ 冒烟测试未通过，请根据上方提示修复")
    return 1


if __name__ == "__main__":
    sys.exit(main())
