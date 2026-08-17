"""
FairScan 桌面移植：验证桌面端 (JVM/Bytedeco TFLite) 与 Python 参考实现 (ai-edge-litert)
对同一输入张量推理出的分割 mask 是否一致。

背景：桌面端 Bytedeco 构建使用 TFLite 内置参考内核（无 XNNPACK 默认委托），
与 Python 的 BUILTIN_WITHOUT_DEFAULT_DELEGATES 完全一致（逐位相同）。

流程：
    1. 先运行桌面 CLI 导出输入张量与输出 mask：
       fairscan-desktop scan <image> --dump-input input.f32 --dump-mask mask.f32
    2. 再用本脚本对比：
       python verify_model.py input.f32 mask.f32

判定标准：
    - 主参考（参考内核）：期望与桌面逐位一致（max abs diff < 1e-6，二值一致 100%）
    - 次参考（XNNPACK，近似 Android LiteRT 行为）：二值一致性 > 0.999
"""
import sys
from pathlib import Path

import numpy as np

from ai_edge_litert.interpreter import Interpreter, OpResolverType

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TFLITE = SCRIPT_DIR.parent / "models" / "fairscan-segmentation-model.tflite"


def run_tflite(x: np.ndarray, resolver_type: OpResolverType) -> np.ndarray:
    interp = Interpreter(
        model_path=str(MODEL_TFLITE),
        experimental_op_resolver_type=resolver_type,
    )
    interp.allocate_tensors()
    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    interp.set_tensor(in_details["index"], x)
    interp.invoke()
    return interp.get_tensor(out_details["index"])[0, :, :, 0].ravel()  # [256,256]


def compare(tag: str, a: np.ndarray, b: np.ndarray) -> bool:
    diff = np.abs(a - b)
    agree = (a >= 0.5) == (b >= 0.5)
    mismatch = int((~agree).sum())
    print(f"[{tag}] max diff={diff.max():.8f}  mean diff={diff.mean():.8f}  "
          f"binary agreement={agree.mean():.4%}  mismatch={mismatch}")
    return diff.max() < 1e-6 and mismatch == 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    input_path = Path(sys.argv[1])
    mask_path = Path(sys.argv[2])
    if not input_path.exists() or not mask_path.exists():
        print(f"文件不存在: {input_path} / {mask_path}")
        return 2

    # 桌面端导出的预处理输入张量 [1,256,256,3] NHWC float32 与输出 mask [256,256] float32
    x = np.fromfile(input_path, dtype=np.float32).reshape(1, 256, 256, 3)
    mask_desktop = np.fromfile(mask_path, dtype=np.float32).reshape(256, 256).ravel()

    print(f"input   : shape={x.shape} mean={x.mean():.6f} min={x.min():.4f} max={x.max():.4f}")
    print(f"model   : {MODEL_TFLITE}")

    # 主参考：内置参考内核（与桌面端 Bytedeco 构建一致），期望逐位相同
    mask_ref_ref = run_tflite(x, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    primary_ok = compare("主参考(参考内核,期望逐位一致)", mask_desktop, mask_ref_ref)

    # 次参考：XNNPACK（近似 Android LiteRT 默认行为），允许浮点内核差异，仅要求分类一致
    mask_ref_xnn = run_tflite(x, OpResolverType.BUILTIN)
    agree = ((mask_desktop >= 0.5) == (mask_ref_xnn >= 0.5)).mean()
    mismatch = int(((mask_desktop >= 0.5) != (mask_ref_xnn >= 0.5)).sum())
    print(f"[次参考(XNNPACK≈Android)] max diff={np.abs(mask_desktop - mask_ref_xnn).max():.4f}  "
          f"binary agreement={agree:.4%}  mismatch={mismatch}")
    secondary_ok = agree > 0.999

    ok = primary_ok and secondary_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
