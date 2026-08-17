"""对比两模型分割掩膜：阈值 0.5 后的面积占比 + 归一化包围盒。

用法:
  python tools/compare_masks.py <u2netp.f32> <tflite.f32> [宽1] [宽2]
  默认宽: u2netp=320, tflite=256（按需传入）
"""
import sys

import numpy as np


def load(path, w):
    data = np.fromfile(path, dtype=np.float32)
    n = int(np.sqrt(data.size))
    return data.reshape(n, n), n


def analyze(name, prob):
    binm = (prob >= 0.5)
    frac = binm.mean() * 100
    ys, xs = np.nonzero(binm)
    if len(xs) == 0:
        print(f"{name}: empty mask")
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    n = prob.shape[0]
    norm = lambda p: (round(p[0] / n, 3), round(p[1] / n, 3))
    print(f"{name}: white={frac:.1f}% quad={[norm(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]}")
    return binm


if __name__ == "__main__":
    a_path, b_path = sys.argv[1], sys.argv[2]
    wa = int(sys.argv[3]) if len(sys.argv) > 3 else 320
    wb = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    a, na = load(a_path, wa)
    b, nb = load(b_path, wb)
    ba = analyze(f"u2netp ({na})", a)
    bb = analyze(f"fairscan ({nb})", b)
    if ba is not None and bb is not None:
        from PIL import Image
        bb_rs = np.array(Image.fromarray((bb * 255).astype(np.uint8)).resize((na, na), Image.NEAREST)) > 0
        extra = ba & ~bb_rs
        print(f"u2netp 相比 fairscan 额外标记前景: {extra.mean() * 100:.1f}%")
