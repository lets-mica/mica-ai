"""把微调后的 BERT 意图分类模型导出为 mica-ai-intent 可用的 ONNX。

与 docs/意图识别模型微调与ONNX导出.md 同步演进。

用法：
    python convert.py \\
      --model_dir model/intent-model \\
      --output_dir model/out \\
      --max_length 128

    # 可选：导出后做 INT8 动态量化
    python convert.py --model_dir model/intent-model --output_dir model/out --quantize int8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import check_onnx
from common.onnx_utils import quantize_dynamic
from common.progress import fail, info, ok, step, warn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 BERT 意图分类 ONNX 模型")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="微调后的模型目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="ONNX 模型输出目录")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度（需与微调时一致）")
    parser.add_argument("--opset_version", type=int, default=14,
                        help="ONNX opset 版本")
    parser.add_argument("--quantize", choices=("none", "int8"), default="none",
                        help="导出后是否做 INT8 动态量化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from transformers import BertForSequenceClassification

    # 1. 加载微调模型
    step(f"[1/4] 加载微调模型: {args.model_dir}")
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    num_labels = model.config.num_labels
    info(f"      num_labels={num_labels}, max_length={args.max_length}")

    # 2. 构造 dummy 输入 + 验证 PyTorch 前向
    dummy_input_ids = torch.zeros(1, args.max_length, dtype=torch.long)
    dummy_attention_mask = torch.zeros(1, args.max_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, args.max_length, dtype=torch.long)

    step("[2/4] 验证 PyTorch 前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
        )
    assert outputs.logits.shape == (1, num_labels), (
        f"logits shape 应为 (1, {num_labels})，实际为 {outputs.logits.shape}"
    )
    ok(f"      logits shape: {outputs.logits.shape} ✓")

    # 3. 导出 ONNX
    onnx_path = os.path.join(args.output_dir, "bert_intent.onnx")
    step(f"[3/4] 导出 ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=args.opset_version,
            do_constant_folding=True,
        )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    info(f"      ONNX 文件大小: {file_size_mb:.1f} MB")

    # 复制 vocab.txt
    vocab_src = os.path.join(args.model_dir, "vocab.txt")
    if os.path.exists(vocab_src):
        vocab_dst = os.path.join(args.output_dir, "vocab.txt")
        shutil.copy2(vocab_src, vocab_dst)
        info(f"      复制 vocab.txt -> {vocab_dst}")
    else:
        warn(f"      未找到 vocab.txt，请手动从预训练模型目录拷贝到 {args.output_dir}")

    # 复制 labels.json
    labels_src = os.path.join(args.model_dir, "labels.json")
    if os.path.exists(labels_src):
        shutil.copy2(labels_src, os.path.join(args.output_dir, "labels.json"))
        info(f"      复制 labels.json -> {args.output_dir}")

    # 4. 验证 ONNX 模型
    step("[4/4] 验证 ONNX 模型...")
    check_onnx(onnx_path)

    # PyTorch vs ONNX 推理一致性
    sess = ort.InferenceSession(onnx_path)
    with torch.no_grad():
        pt_logits = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
        ).logits.numpy()
    ort_logits = sess.run(None, {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
        "token_type_ids": dummy_token_type_ids.numpy(),
    })[0]
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    info(f"      PyTorch vs ONNX 最大差异: {max_diff:.6f}")
    if max_diff > 1e-4:
        fail(f"      ONNX 推理与 PyTorch 差异过大: {max_diff}")
        raise SystemExit(1)
    ok("      推理一致性验证通过 ✓")

    # 可选量化
    if args.quantize == "int8":
        int8_path = quantize_dynamic(onnx_path)
        info(f"      INT8 量化文件: {int8_path}")

    ok("ONNX 导出完成!")
    info(f"   模型文件: {onnx_path}")
    info(f"   词表文件: {os.path.join(args.output_dir, 'vocab.txt')}")
    info(f"   标签文件: {os.path.join(args.output_dir, 'labels.json')}")


if __name__ == "__main__":
    main()
