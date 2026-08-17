"""测试 uvdoc 对输入分辨率敏感性：不同缩放尺寸下展平质量对比。

用法: python tools/test_uvdoc_size.py
"""
import cv2
import numpy as np
import onnxruntime as ort

IMG = "../app/src/debug/assets/uncropped/img01.jpg"
SESS = ort.InferenceSession("models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = SESS.get_inputs()[0].name


def run_uvdoc(bgr):
    h, w = bgr.shape[:2]
    inp = bgr.astype(np.float32) / 255.0
    chw = np.transpose(inp, (2, 0, 1))[None]
    outs = SESS.run(None, {in_name: chw})[0]
    out = np.transpose(outs[0], (1, 2, 0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8), (w, h)


def curl_warp(bgr, amp=40.0):
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(bgr, xx + amp * np.sin(np.pi * yy / h), yy, cv2.INTER_LINEAR)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


orig = cv2.imread(IMG)
warped = curl_warp(orig, amp=40.0)
gt = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

h, w = warped.shape[:2]
print(f"warped vs gt: SSIM={ssim(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), gt):.4f}")

for max_side in (640, 512, 384, 256):
    scale = max_side / max(h, w)
    small = cv2.resize(warped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out, out_size = run_uvdoc(small)
    # 输出与输入同尺寸，放大回原尺寸对比
    out_r = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    out_g = cv2.cvtColor(out_r, cv2.COLOR_BGR2GRAY)
    s = ssim(out_g, gt)
    # 文本行弯曲残差
    dark = out_g < 128
    drifts = []
    for y in range(0, out_g.shape[0], 8):
        row = dark[y]
        if row.sum() < 20:
            continue
        drifts.append((y, np.nonzero(row)[0].mean()))
    if len(drifts) > 2:
        d = np.array(drifts)
        fit = np.polyfit(d[:, 0], d[:, 1], 1)
        resid = np.abs(d[:, 1] - np.polyval(fit, d[:, 0])).mean()
    else:
        resid = float("nan")
    print(f"max_side={max_side}: 输入 {small.shape[1]}x{small.shape[0]} -> 输出 {out_size} "
          f"SSIM={s:.4f} 行弯曲残差={resid:.1f}px")
    cv2.imwrite(f"cmpre/uvdoc-out-{max_side}.png", out_r)
