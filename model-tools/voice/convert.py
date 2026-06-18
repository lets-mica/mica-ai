"""把 SenseVoiceSmall 导出为 mica-ai-voice 期望的双 ONNX 模型。

mica-ai-voice Java 侧期望的形态（见
``mica-ai-voice/src/main/java/.../engine/SenseVoiceEncoder.java`` 与
``SenseVoiceDecoder.java``）：

==============================================================
sensevoice_encoder.onnx
==============================================================
  Inputs : speech_feat   float32  [1, T, 560]   (LFR 特征, 7 帧 * 80 维 FBank)
           mask          float32  [1, T]        (有效帧 mask)
           prompt_ids    int64    [1, 4]        (lid_idx, 1, 2, itn_idx)
  Output : (任意名)        float32  [1, T+4, 512] (encoder 隐藏态, 前 4 帧 = prompt)
  Metadata: lid_dict       = '{"auto":0,"zh":1,"en":2,"yue":3,"ja":4,"ko":5,"nospeech":6}'
            textnorm_dict  = '{"withitn":14,"woitn":15}'

==============================================================
sensevoice_ctc.onnx
==============================================================
  Inputs : enc_out        float32  [1, T+4, 512]
  Outputs: (任意名)        float32  [1, T+4, K] (top-K log_probs)
           (任意名)        int64    [1, T+4, K] (top-K 索引)
  K 由训练决定（一般是 4 或 5）

==============================================================
其它资源
==============================================================
  - tokens.txt    SentencePiece 词表
  - config.json   推理配置（含 reverse_weight 等）
  - am.mvn        FBank 均值/方差（CMVN）

==============================================================
实现路径
==============================================================
提供 3 选 1：

  --method upstream    【推荐】克隆 [SenseVoice-ONNX](https://github.com/lovemefan/SenseVoice-ONNX)
                       仓库并执行其 01-Export-Encoder.py / 02-Export-CTC.py
                       / 03-Prepare-Assets.py，最稳。

  --method funasr      【自动】直接用 funasr.AutoModel 加载 SenseVoiceSmall，
                       按上述接口手工切 encoder + CTC head 并导出 ONNX。
                       不需要克隆外部仓库，但 funasr 版本敏感。
                       （提示：funasr 不同版本里 encoder / prompt_embed 命名不同，
                        跑不通时请用 --method upstream。）

  --method manual      【占位】打印操作指南（无 GPU/funasr 时回退）。

用法：
    python convert.py --method upstream
    python convert.py --method funasr --top-k 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import cap_models_dir, ensure_dir, fail, info, ok, step, warn
from common.onnx_utils import check_onnx, simplify_onnx


LID_DICT = {
    "auto": 0, "zh": 1, "en": 2, "yue": 3,
    "ja": 4, "ko": 5, "nospeech": 6,
}
TEXTNORM_DICT = {"withitn": 14, "woitn": 15}

UPSTREAM_REPO = "https://github.com/lovemefan/SenseVoice-ONNX.git"


# ============================================================================
# 公共：资源整理
# ============================================================================

def _copy_assets(model_dir: Path, out_dir: Path) -> None:
    candidates = {
        "tokens.txt":  ["tokens.txt", "chn_jpn_yue_eng_ko_spectok.bpe.model"],
        "config.json": ["config.json", "configuration.json"],
        "am.mvn":      ["am.mvn"],
    }
    for target, names in candidates.items():
        for n in names:
            src = model_dir / n
            if src.exists():
                shutil.copy2(src, out_dir / target)
                info(f"      复制 {n} -> {target}")
                break
        else:
            warn(f"      未找到 {target}（尝试过: {names}）")


# ============================================================================
# 路径一：upstream
# ============================================================================

def _convert_upstream(model_dir: Path, out_dir: Path, *, opset: int) -> None:
    import tempfile

    step("克隆 SenseVoice-ONNX 仓库...")
    work_root = Path(tempfile.mkdtemp(prefix="sensevoice-onnx-"))
    upstream_dir = work_root / "SenseVoice-ONNX"
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", UPSTREAM_REPO, str(upstream_dir)],
            check=True, capture_output=True, text=True,
        )
        env = os.environ.copy()
        env.setdefault("MODEL_DIR", str(model_dir.resolve()))

        for script, label in [
            ("01-Export-Encoder.py", "encoder"),
            ("02-Export-CTC.py",     "ctc"),
            ("03-Prepare-Assets.py", "assets"),
        ]:
            sp = upstream_dir / script
            if not sp.exists():
                fail(f"上游脚本不存在: {sp}（SenseVoice-ONNX 仓库结构可能变了）")
                return
            step(f"执行 {script} ({label})")
            result = subprocess.run(
                [sys.executable, str(sp)],
                cwd=upstream_dir,
                env=env,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                fail(f"{script} 失败:\n{result.stdout}\n{result.stderr}")
                return
            tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "OK"
            info(tail)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    for name in ("sensevoice_encoder.onnx", "sensevoice_ctc.onnx"):
        for c in (upstream_dir / name, upstream_dir / "models" / name):
            if c.exists():
                shutil.copy2(c, out_dir / name)
                break
        else:
            warn(f"未找到上游产物 {name}，请检查 SenseVoice-ONNX 仓库的输出路径")

    _copy_assets(model_dir, out_dir)
    ok("upstream 路径完成")


# ============================================================================
# 路径二：funasr 直接导出
# ============================================================================

def _try_import_funasr():
    try:
        from funasr import AutoModel  # type: ignore
        return AutoModel
    except ImportError:
        return None


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _resolve_sense_voice_submodules(inner: Any) -> dict[str, Any]:
    """从 funasr AutoModel.model 中按启发式找出 embed / encoder / ctc / prompt。

    funasr 不同版本属性名差异较大，这里做兜底查找并抛出清晰错误。
    """
    result: dict[str, Any] = {}

    result["embed"] = (
        getattr(inner, "embed", None)
        or getattr(inner, "embedding", None)
        or getattr(inner, "frontend_embed", None)
    )

    result["encoder"] = (
        getattr(inner, "encoder", None)
        or getattr(inner, "speech_encoder", None)
    )

    ctc = getattr(inner, "ctc", None) or getattr(inner, "ctc_lo", None)
    result["ctc"] = ctc

    result["prompt_embed"] = (
        getattr(inner, "prompt_embed", None)
        or getattr(inner, "embed_prompt", None)
        or getattr(inner, "prompt_embedding", None)
    )

    missing = [k for k in ("embed", "encoder", "ctc", "prompt_embed") if result[k] is None]
    if missing:
        raise RuntimeError(
            f"funasr SenseVoice 模型缺少关键子模块: {missing}。"
            f"funasr 版本可能不兼容，请改用 --method upstream。"
        )
    return result


def _build_encoder_wrapper(parts: dict[str, Any]) -> Any:
    """构造一个 nn.Module，把 LFR → embed → prompt prepend → encoder 串起来。"""
    import torch
    import torch.nn as nn

    class EncoderWrapper(nn.Module):
        def __init__(self, embed, encoder, prompt_embed):
            super().__init__()
            self.embed = embed
            self.encoder = encoder
            self.prompt_embed = prompt_embed

        def forward(self, speech_feat, mask, prompt_ids):
            # 1) LFR + Conv1d embed → (B, T, hidden)
            xs, _ = self.embed(speech_feat)
            # 2) prompt embedding → (B, 4, hidden)
            #    prompt_embed 接受 int64 token ids
            prompt = self.prompt_embed(prompt_ids)
            # 3) concat → (B, T+4, hidden)
            xs = torch.cat([prompt, xs], dim=1)
            # 4) SAN-M 编码
            enc_out, _ = self.encoder(xs, mask)
            return enc_out

    return EncoderWrapper(parts["embed"], parts["encoder"], parts["prompt_embed"])


def _build_ctc_wrapper(ctc_proj: Any, top_k: int) -> Any:
    import torch
    import torch.nn as nn

    class CTCWrapper(nn.Module):
        def __init__(self, ctc_proj, top_k):
            super().__init__()
            self.ctc_proj = ctc_proj
            self.top_k = top_k

        def forward(self, enc_out):
            logits = self.ctc_proj(enc_out)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_lp, topk_idx = torch.topk(log_probs, self.top_k, dim=-1)
            return topk_lp, topk_idx

    return CTCWrapper(ctc_proj, top_k)


def _convert_funasr(model_dir: Path, out_dir: Path, *, opset: int, top_k: int) -> None:
    AutoModel = _try_import_funasr()
    if AutoModel is None:
        fail("funasr 未安装，请先：pip install -r requirements.txt")
        return
    torch = _try_import_torch()
    if torch is None:
        fail("torch 未安装，请先：pip install torch")
        return

    step("用 funasr 加载 SenseVoiceSmall ...")
    model = AutoModel(model=str(model_dir.resolve()))
    inner = getattr(model, "model", model)

    parts = _resolve_sense_voice_submodules(inner)
    info(
        f"      embed={type(parts['embed']).__name__}, "
        f"encoder={type(parts['encoder']).__name__}, "
        f"ctc={type(parts['ctc']).__name__}, "
        f"prompt_embed={type(parts['prompt_embed']).__name__}"
    )

    # 1) encoder
    enc_path = out_dir / "sensevoice_encoder.onnx"
    step(f"导出 encoder -> {enc_path}")
    enc_wrapper = _build_encoder_wrapper(parts).eval()

    T = 30
    speech_feat = torch.randn(1, T, 560)
    mask = torch.ones(1, T)
    prompt_ids = torch.tensor([[0, 1, 2, 14]], dtype=torch.long)

    import onnx
    with torch.no_grad():
        torch.onnx.export(
            enc_wrapper,
            (speech_feat, mask, prompt_ids),
            str(enc_path),
            input_names=["speech_feat", "mask", "prompt_ids"],
            output_names=["enc_out"],
            dynamic_axes={
                "speech_feat": {0: "B", 1: "T"},
                "mask":        {0: "B", 1: "T"},
                "prompt_ids":  {0: "B"},
                "enc_out":     {0: "B", 1: "T"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    m = onnx.load(str(enc_path))
    from onnx import StringStringEntryProto
    m.metadata_props.append(StringStringEntryProto(
        key="lid_dict", value=json.dumps(LID_DICT, ensure_ascii=False)
    ))
    m.metadata_props.append(StringStringEntryProto(
        key="textnorm_dict", value=json.dumps(TEXTNORM_DICT, ensure_ascii=False)
    ))
    onnx.save(m, str(enc_path))
    info("      元数据 lid_dict / textnorm_dict 已写入")

    # 2) CTC head
    ctc_path = out_dir / "sensevoice_ctc.onnx"
    step(f"导出 CTC head (top_k={top_k}) -> {ctc_path}")

    ctc = parts["ctc"]
    ctc_proj = getattr(ctc, "ctc_lo", ctc)
    ctc_wrapper = _build_ctc_wrapper(ctc_proj, top_k).eval()

    enc_out_dummy = torch.randn(1, T + 4, 512)
    with torch.no_grad():
        torch.onnx.export(
            ctc_wrapper,
            (enc_out_dummy,),
            str(ctc_path),
            input_names=["enc_out"],
            output_names=["topk_log_probs", "topk_indices"],
            dynamic_axes={
                "enc_out":          {0: "B", 1: "T"},
                "topk_log_probs":   {0: "B", 1: "T"},
                "topk_indices":     {0: "B", 1: "T"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    _copy_assets(model_dir, out_dir)

    step("校验 ONNX 结构")
    check_onnx(enc_path)
    check_onnx(ctc_path)

    if os.environ.get("MICA_SKIP_SIMPLIFY") != "1":
        simplify_onnx(enc_path)
        simplify_onnx(ctc_path)

    ok("funasr 路径完成")


# ============================================================================
# 路径三：manual
# ============================================================================

def _convert_manual(model_dir: Path, out_dir: Path, *, opset: int) -> None:
    fail(
        "manual 模式仅打印操作指南。\n"
        "请二选一：\n"
        "  1) 克隆 [SenseVoice-ONNX](https://github.com/lovemefan/SenseVoice-ONNX) "
        "并执行 01-Export-Encoder.py / 02-Export-CTC.py / 03-Prepare-Assets.py\n"
        "  2) pip install funasr onnx onnxsim 后执行：\n"
        "       python convert.py --method funasr --top-k 4"
    )


# ============================================================================
# 入口
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 SenseVoice ONNX")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=4, help="CTC decoder top-K 深度")
    parser.add_argument(
        "--method",
        choices=("upstream", "funasr", "manual"),
        default="upstream",
        help="导出路径：upstream=克隆 SenseVoice-ONNX；funasr=直接用 funasr；manual=只打印指南",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = cap_models_dir("voice")
    model_dir = root / "SenseVoiceSmall"
    if not model_dir.exists():
        fail(f"未找到 {model_dir}，请先执行 download.py")
        sys.exit(1)

    out_dir = ensure_dir(root / "out")
    step(f"开始导出 SenseVoice ONNX，输出: {out_dir}，方法: {args.method}")

    if args.method == "upstream":
        _convert_upstream(model_dir, out_dir, opset=args.opset)
    elif args.method == "funasr":
        _convert_funasr(model_dir, out_dir, opset=args.opset, top_k=args.top_k)
    else:
        _convert_manual(model_dir, out_dir, opset=args.opset)

    ok(f"全部完成。产物: {out_dir}")


if __name__ == "__main__":
    main()
