"""验证 UVDoc 效果差的可能原因：输入是否为增强（美白/灰度）后的图。

方法：用 img01 合成"卷曲文档"（水平圆柱卷），对每个像素做已知形变，ground truth
就是原图。分别以 (a) 原始彩图 (b) 灰度+白点拉伸（模拟 enhanceCapturedImage）作为输入
喂给 uvdoc.onnx，用 SSIM/PSNR 对比展平输出与原图的相似度。

用法: python tools/test_uvdoc_input.py
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
    chw = np.transpose(inp, (2, 0, 1))[None]  # 1,3,H,W
    outs = SESS.run(None, {in_name: chw})[0]
    out = np.transpose(outs[0], (1, 2, 0))  # H,W,C
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def curl_warp(bgr, amp=40.0):
    """水平卷曲：x 坐标按正弦沿 y 位移，模拟页面弯折。"""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + amp * np.sin(np.pi * yy / h)
    map_y = yy
    return cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)


def sim_enhance(bgr):
    """模拟 extractDocument 的增强：灰度 + 白点拉伸（去掉亮度梯度）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    stretch = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return np.stack([stretch] * 3, axis=-1).astype(np.uint8)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


orig = cv2.imread(IMG)
warped = curl_warp(orig, amp=40.0)
cv2.imwrite("cmpre/uvdoc-warped.png", warped)

gt = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
out_raw = run_uvdoc(warped)
out_enh = run_uvdoc(sim_enhance(warped))
cv2.imwrite("cmpre/uvdoc-out-raw.png", out_raw)
cv2.imwrite("cmpre/uvdoc-out-enhanced.png", out_enh)

print(f"warped vs gt        SSIM={ssim(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), gt):.4f}")
print(f"unwarp(raw input)   SSIM={ssim(cv2.cvtColor(out_raw, cv2.COLOR_BGR2GRAY), gt):.4f}")
print(f"unwarp(enh input)   SSIM={ssim(cv2.cvtColor(out_enh, cv2.COLOR_BGR2GRAY), gt):.4f}")
