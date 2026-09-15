"""把 download.py 下载的 Magika 原始三件套导出为 mica-ai-filetype 期望的最终文件。

- 校验 ONNX 结构（输入 int64 [1,2048]、输出 float32 [1,N]）
- 校验 config.min.json 的 key 完整
- 把 model.onnx / config.min.json / content_types_kb.min.json
  拷贝 / 链接到 model/out/ 下，方便 Java 端直接读取。

最终产物：

    model-tools/filetype/model/out/model.onnx                    ≈3.1 MB
    model-tools/filetype/model/out/config.min.json
    model-tools/filetype/model/out/content_types_kb.min.json     ≈45 KB

Java 侧配置：

    mica.ai.filetype.model-path:        <abs>/model/out/model.onnx
    mica.ai.filetype.model-config-path: <abs>/model/out/config.min.json
    mica.ai.filetype.content-types-path: <abs>/model/out/content_types_kb.min.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, info, ok, step, warn
from common.onnx_utils import check_onnx


RAW_SUBDIR = "magika_v3"

REQUIRED_FILES = {
    "model.onnx": "Magika 主模型（int64 [1,2048] 输入 → float32 [1,N] 概率分布）",
    "config.min.json": "Magika 模型配置（beg/end/mid/padding/thresholds/overwrite_map）",
    "content_types_kb.min.json": "Magika 内容类型知识库（label → mime / group / description）",
}

REQUIRED_CONFIG_KEYS = (
    "beg_size", "end_size", "mid_size", "padding_token",
    "block_size", "min_file_size_for_dl",
    "medium_confidence_threshold", "thresholds", "overwrite_map",
    "target_labels_space", "use_inputs_at_offsets",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 mica-ai-filetype ONNX 产物")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验 raw 模型有效性，不复制",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="用符号链接代替复制（节省磁盘）",
    )
    args = parser.parse_args()

    root = cap_models_dir("filetype")
    raw_dir = root / RAW_SUBDIR
    if not raw_dir.exists():
        warn(f"未找到 {raw_dir}，请先执行：python download.py")
        sys.exit(1)

    step("校验 raw 模型文件...")
    for fname in REQUIRED_FILES:
        src = raw_dir / fname
        if not src.exists():
            warn(f"raw 目录缺少 {src}")
            sys.exit(1)

    onnx_meta = check_onnx(raw_dir / "model.onnx")
    in0 = onnx_meta["inputs"][0]
    out0 = onnx_meta["outputs"][0]
    # ONNX 把 batch 维度声明为 0（dynamic），运行时实际为 1
    if in0["shape"][-1] != 2048:
        warn(f"model.onnx 输入特征维度异常: {in0['shape']}（期望 [..., 2048]）")
    if len(out0["shape"]) != 2 or out0["shape"][0] != 0:
        warn(f"model.onnx 输出 shape 异常: {out0['shape']}（期望 [0, N]）")
    info(f"ONNX opset: {onnx_meta['opset']}")

    # 检查输入 dtype（Java 端用 IntBuffer 喂 int32 张量）
    try:
        import onnx
        model_proto = onnx.load(str(raw_dir / "model.onnx"))
        in_dtype = model_proto.graph.input[0].type.tensor_type.elem_type
        in_dtype_name = onnx.TensorProto.DataType.Name(in_dtype)
        if in_dtype_name not in ("INT32", "INT64"):
            warn(f"model.onnx 输入 dtype 异常: {in_dtype_name}（期望 INT32/INT64）")
        info(f"input dtype: {in_dtype_name}")
    except Exception as e:
        warn(f"读取 model.onnx dtype 失败: {e}")

    cfg = json.loads((raw_dir / "config.min.json").read_text(encoding="utf-8"))
    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing_keys:
        warn(f"config.min.json 缺少字段: {missing_keys}")
        sys.exit(1)
    info(f"config.min.json: {len(cfg['target_labels_space'])} labels, "
         f"{len(cfg.get('thresholds', {}))} thresholds")

    kb = json.loads((raw_dir / "content_types_kb.min.json").read_text(encoding="utf-8"))
    info(f"content_types_kb.min.json: {len(kb)} content types")

    if args.check_only:
        for fname, desc in REQUIRED_FILES.items():
            ok(f"raw/{fname} 存在 ({desc})")
        ok("check-only 完成")
        return

    out_dir = ensure_dir(root / "out")
    step(f"导出到 {out_dir}")

    for fname in REQUIRED_FILES:
        src = raw_dir / fname
        dst = out_dir / fname
        if args.link:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            info(f"  链接 {dst} -> {src}")
        else:
            shutil.copy2(src, dst)
            info(f"  复制 {dst}")

    ok(f"导出完成。产物目录: {out_dir}")
    info("Java 侧 yml 配置：")
    info("  mica.ai.filetype.model-path:         <abs>/model/out/model.onnx")
    info("  mica.ai.filetype.model-config-path:  <abs>/model/out/config.min.json")
    info("  mica.ai.filetype.content-types-path: <abs>/model/out/content_types_kb.min.json")


if __name__ == "__main__":
    main()