"""把 ERes2Net 系列声纹模型导出为 mica-ai-speaker 期望的 ONNX。

mica-ai-speaker Java 侧期望的形态（见
``mica-ai-speaker/src/main/java/.../engine/SpeakerVerifier.java``）：

==============================================================
eres2net.onnx
==============================================================
  Inputs : （任意名, 一般是 feats / feature / input） float32  [1, T, 80]
  Outputs: （任意名, 一般是 embedding / emb / output） float32  [1, 192]
  说明：Java 端用 ``session.getInputInfo().keySet().iterator().next()``
        取输入名，不依赖具体名字。

==============================================================
FBank 特征
==============================================================
mica-ai-speaker 的 FBankExtractor 自己负责 FBank 提取，所以本 ONNX
**不包含** frontend（FBank + CMVN）。只导出 backbone。

==============================================================
实现路径
==============================================================
提供 3 选 1：

  --method threed-speaker  【推荐】使用 [3D-Speaker](https://github.com/modelscope/3D-Speaker)
                            项目的 ERes2Net / ERes2NetV2 加载器，剥离 frontend 后导出。
                            需要 `pip install 3D-Speaker`。

  --method funasr          【备选】使用 funasr.AutoModel 加载，剥离 frontend 后导出。
                            需要 `pip install funasr`。

  --method manual          【占位】打印操作指南。

用法：
    pip install 3D-Speaker
    python convert.py --method threed-speaker
    python convert.py --method threed-speaker --variant eres2net
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, fail, info, ok, step, warn
from common.onnx_utils import check_onnx, simplify_onnx


# ============================================================================
# 工具
# ============================================================================

def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _try_import_3dspeaker():
    try:
        import speakerlab  # type: ignore  # 3D-Speaker 提供的顶层包
        return speakerlab
    except ImportError:
        return None


def _try_import_funasr():
    try:
        from funasr import AutoModel  # type: ignore
        return AutoModel
    except ImportError:
        return None


# ============================================================================
# 子模型抽取：从前端-后端结构中取出"只吃 FBank"的 backbone
# ============================================================================

def _extract_backbone(model: Any, *, framework: str) -> Any:
    """从 3D-Speaker / funasr 加载的模型中抽出 backbone（FBank 后的部分）。

    返回的 backbone.forward(feats) 应输出 [B, 192] embedding。
    """
    # 3D-Speaker / speakerlab 模型通常: model.frontend / model.backbone
    if hasattr(model, "backbone") and hasattr(model, "frontend"):
        info(f"      [{framework}] 检测到 frontend/backbone 结构，剥离 frontend")
        return model.backbone

    # 某些 funasr 版本：直接就是一个 backbone
    if hasattr(model, "forward"):
        return model

    raise RuntimeError(
        f"无法从 {framework} 模型中抽取 backbone。"
        f"模型属性: {[a for a in dir(model) if not a.startswith('_')]}"
    )


# ============================================================================
# 路径一：3D-Speaker
# ============================================================================

def _convert_threed_speaker(model_dir: Path, out_dir: Path, *, opset: int, variant: str) -> None:
    speakerlab = _try_import_3dspeaker()
    torch = _try_import_torch()
    if speakerlab is None:
        fail("未安装 3D-Speaker，请先：pip install 3D-Speaker")
        return
    if torch is None:
        fail("未安装 torch，请先：pip install torch")
        return

    # 3D-Speaker 提供了 ERes2Net / ERes2NetV2 的预定义类
    # 不同版本路径略不同，做兜底
    ERes2NetCls = None
    for path in (
        ("speakerlab.models.eres2net", "ERes2Net"),
        ("speakerlab.models.eres2net.main", "ERes2Net"),
    ):
        try:
            mod = __import__(path[0], fromlist=[path[1]])
            ERes2NetCls = getattr(mod, path[1])
            break
        except (ImportError, AttributeError):
            continue
    if ERes2NetCls is None:
        fail(
            "未在 3D-Speaker 中找到 ERes2Net 类。"
            "请确认 3D-Speaker >= 0.2 版本，或改用 --method funasr。"
        )
        return

    step(f"用 3D-Speaker 加载 {variant} ...")
    # 3D-Speaker 通常通过 .from_pretrained(local_path) 加载
    try:
        model = ERes2NetCls.from_pretrained(str(model_dir.resolve()))
    except Exception as e:
        fail(f"3D-Speaker 加载失败: {e}\n请确认 model_dir 是 ERes2Net 完整权重目录。")
        return

    model.eval()
    backbone = _extract_backbone(model, framework="3D-Speaker")

    T = 300  # ~3 秒音频对应的 FBank 帧数
    dummy = torch.randn(1, T, 80)

    out_path = out_dir / "eres2net.onnx"
    step(f"导出 backbone -> {out_path}")

    with torch.no_grad():
        torch.onnx.export(
            backbone,
            (dummy,),
            str(out_path),
            input_names=["feats"],
            output_names=["embedding"],
            dynamic_axes={
                "feats":     {0: "B", 1: "T"},
                "embedding": {0: "B"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    step("校验 ONNX 结构")
    check_onnx(out_path)

    if os.environ.get("MICA_SKIP_SIMPLIFY") != "1":
        simplify_onnx(out_path)

    ok("3D-Speaker 路径完成")


# ============================================================================
# 路径二：funasr
# ============================================================================

def _convert_funasr(model_dir: Path, out_dir: Path, *, opset: int) -> None:
    AutoModel = _try_import_funasr()
    torch = _try_import_torch()
    if AutoModel is None:
        fail("未安装 funasr，请先：pip install funasr")
        return
    if torch is None:
        fail("未安装 torch，请先：pip install torch")
        return

    step("用 funasr 加载 ERes2Net ...")
    model = AutoModel(model=str(model_dir.resolve()))
    inner = getattr(model, "model", model)
    backbone = _extract_backbone(inner, framework="funasr")

    T = 300
    dummy = torch.randn(1, T, 80)

    out_path = out_dir / "eres2net.onnx"
    step(f"导出 backbone -> {out_path}")

    with torch.no_grad():
        torch.onnx.export(
            backbone,
            (dummy,),
            str(out_path),
            input_names=["feats"],
            output_names=["embedding"],
            dynamic_axes={
                "feats":     {0: "B", 1: "T"},
                "embedding": {0: "B"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    step("校验 ONNX 结构")
    check_onnx(out_path)

    if os.environ.get("MICA_SKIP_SIMPLIFY") != "1":
        simplify_onnx(out_path)

    ok("funasr 路径完成")


# ============================================================================
# 路径三：manual
# ============================================================================

def _convert_manual(model_dir: Path, out_dir: Path, *, opset: int) -> None:
    fail(
        "manual 模式仅打印操作指南。\n"
        "请二选一：\n"
        "  1) pip install 3D-Speaker 后执行：\n"
        "       python convert.py --method threed-speaker --variant eres2netv2\n"
        "  2) pip install funasr 后执行：\n"
        "       python convert.py --method funasr\n"
        "\n"
        "两条路径都会：\n"
        "  - 加载 PyTorch 权重\n"
        "  - 剥离 frontend（FBank 提取）\n"
        "  - 仅导出 backbone 接受 FBank[1,T,80] -> embedding[1,192]\n"
        "  - 产物在 model/out/eres2net.onnx"
    )


# ============================================================================
# 入口
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 ERes2Net ONNX")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument(
        "--variant",
        default="eres2netv2",
        choices=("eres2netv2", "eres2netv2_en", "eres2net", "eres2net_base_200k"),
        help="与 download.py --variant 保持一致",
    )
    parser.add_argument(
        "--method",
        choices=("threed-speaker", "funasr", "manual"),
        default="threed-speaker",
        help="导出路径：threed-speaker / funasr / manual",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = cap_models_dir("speaker")
    model_dir = root / f"{args.variant}_raw"
    if not model_dir.exists():
        fail(f"未找到 {model_dir}，请先执行 download.py --variant {args.variant}")
        sys.exit(1)

    out_dir = ensure_dir(root / "out")
    step(f"开始导出 ERes2Net ONNX，输出: {out_dir}，方法: {args.method}，variant: {args.variant}")

    if args.method == "threed-speaker":
        _convert_threed_speaker(model_dir, out_dir, opset=args.opset, variant=args.variant)
    elif args.method == "funasr":
        _convert_funasr(model_dir, out_dir, opset=args.opset)
    else:
        _convert_manual(model_dir, out_dir, opset=args.opset)

    ok(f"全部完成。产物: {out_dir / 'eres2net.onnx'}")


if __name__ == "__main__":
    main()
