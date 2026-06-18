"""ONNX 工具：检查 / 简化 / 量化 / 推理一致性验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from common.progress import fail, info, ok, step, warn


def check_onnx(onnx_path: Path | str) -> dict:
    """用 ``onnx.checker`` 验证 ONNX 结构。返回输入输出元信息。"""
    import onnx

    onnx_path = Path(onnx_path)
    step(f"检查 ONNX 结构: {onnx_path}")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    info("ONNX 结构合法 ✓")

    graph = model.graph
    inputs = [
        {"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]}
        for i in graph.input
    ]
    outputs = [
        {"name": o.name, "shape": [d.dim_value for d in o.type.tensor_type.shape.dim]}
        for o in graph.output
    ]
    info(f"inputs : {inputs}")
    info(f"outputs: {outputs}")
    return {"inputs": inputs, "outputs": outputs, "opset": model.opset_import[0].version}


def simplify_onnx(onnx_path: Path | str, output_path: Path | str | None = None) -> Path:
    """用 ``onnxsim`` 做图简化，减小文件体积和推理时延。"""
    try:
        import onnxsim
    except ImportError as e:
        fail("请先安装 onnxsim: pip install onnxsim")
        raise e

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path
    step(f"简化 ONNX 图: {onnx_path} -> {output_path}")
    model_simp, ok_flag = onnxsim.simplify(str(onnx_path))
    if not ok_flag:
        warn("onnxsim 未能进一步简化（已是最简形态）")
        return onnx_path
    import onnx
    onnx.save(model_simp, str(output_path))
    ok(f"简化完成: {output_path}")
    return output_path


def quantize_dynamic(
    onnx_path: Path | str,
    output_path: Path | str | None = None,
    weight_type: str = "u8",
) -> Path:
    """动态量化（INT8），适合 BERT 等内存带宽受限的模型。

    体积通常可压缩 3-4 倍，精度损失 < 1%。
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        fail("请先安装 onnxruntime: pip install onnxruntime")
        raise e

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_name(
        onnx_path.stem + "_int8" + onnx_path.suffix
    )
    step(f"动态量化: {onnx_path} -> {output_path}")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8 if weight_type == "u8" else QuantType.QInt8,
    )
    ok(f"量化完成: {output_path}")
    return output_path


def verify_torch_vs_onnx(
    torch_forward: Callable[[], np.ndarray],
    onnx_path: Path | str,
    onnx_inputs: dict[str, np.ndarray],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> float:
    """对比 PyTorch 与 ONNX Runtime 的推理结果。返回最大绝对误差。"""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    step(f"对比 PyTorch vs ONNX Runtime: {onnx_path}")

    torch_out = torch_forward()
    info(f"PyTorch output shape: {torch_out.shape}")

    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, onnx_inputs)[0]
    info(f"ONNX output shape   : {onnx_out.shape}")

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff > atol and not np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
        fail(f"推理结果不一致: max_diff={max_diff}")
        raise AssertionError(f"max_diff={max_diff} 超过阈值 atol={atol}")
    ok(f"推理一致性通过 (max_diff={max_diff:.6f})")
    return max_diff
