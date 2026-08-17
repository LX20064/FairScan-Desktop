"""判别 uvdoc 预处理是否正确的两个实验：
A. 平面文档（无卷曲）过模型：模型应基本保持原图（输出≈输入）。
B. 卷曲文档 + 模拟真实光照渐变（uvdoc 训练分布内的阴影线索）。
用法: python tools/test_uvdoc_ab.py
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
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


def curl_warp(bgr, amp=40.0, shading=False):
    """水平卷曲；shading=True 时叠加与弯曲一致的光照（亮顶暗边模拟纸张反光）。"""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(bgr, xx + amp * np.sin(np.pi * yy / h), yy, cv2.INTER_LINEAR)
    if shading:
        # 纸张卷曲朝左/右时，两侧暗中间亮（圆柱反光）
        gain = 0.6 + 0.4 * np.cos(2 * np.pi * yy / h)
        warped = np.clip(warped.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    return warped


orig = cv2.imread(IMG)
gt_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# A. 平面文档
out_flat = run_uvdoc(orig)
s = ssim(cv2.cvtColor(out_flat, cv2.COLOR_BGR2GRAY), gt_g)
cv2.imwrite("cmpre/uvdoc-A-flat.png", out_flat)
print(f"A. 平面文档: SSIM(输出, 原图)={s:.4f}  (接近1=模型基本保持原图)")

# B. 卷曲 + 光照
for amp in (30, 60):
    warped = curl_warp(orig, amp=amp, shading=True)
    out = run_uvdoc(warped)
    out_r = cv2.resize(out, (orig.shape[1], orig.shape[0]))
    s = ssim(cv2.cvtColor(out_r, cv2.COLOR_BGR2GRAY), gt_g)
    # 行弯曲残差
    dark = cv2.cvtColor(out_r, cv2.COLOR_BGR2GRAY) < 128
    drifts = []
    for y in range(0, out_r.shape[0], 8):
        row = dark[y]
        if row.sum() < 20:
            continue
        drifts.append((y, np.nonzero(row)[0].mean()))
    d = np.array(drifts)
    fit = np.polyfit(d[:, 0], d[:, 1], 1)
    resid = np.abs(d[:, 1] - np.polyval(fit, d[:, 0])).mean()
    print(f"B. 卷曲 amp={amp}+光照: SSIM={s:.4f} 行弯曲残差={resid:.1f}px")
    cv2.imwrite(f"cmpre/uvdoc-B-amp{amp}.png", out_r)
    cv2.imwrite(f"cmpre/uvdoc-B-amp{amp}-input.png", warped)
