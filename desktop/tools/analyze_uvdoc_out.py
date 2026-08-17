"""分析 uvdoc 输出：尺寸、内容相关性、文本行直线度。"""
import cv2
import numpy as np

warped = cv2.imread("cmpre/uvdoc-warped.png", cv2.IMREAD_GRAYSCALE)
out_raw = cv2.imread("cmpre/uvdoc-out-raw.png", cv2.IMREAD_GRAYSCALE)
out_enh = cv2.imread("cmpre/uvdoc-out-enhanced.png", cv2.IMREAD_GRAYSCALE)
orig = cv2.imread("../app/src/debug/assets/uncropped/img01.jpg", cv2.IMREAD_GRAYSCALE)

print("warped:", warped.shape, "out_raw:", out_raw.shape, "out_enh:", out_enh.shape, "orig:", orig.shape)

for name, img in [("warped", warped), ("out_raw", out_raw), ("out_enh", out_enh)]:
    print(f"{name}: mean={img.mean():.1f} std={img.std():.1f} min={img.min()} max={img.max()}")

# 输出与原图/卷曲图的相关性（重采样到原图尺寸）
def resize(img, h, w):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

h, w = orig.shape
out_raw_r = resize(out_raw, h, w)
out_enh_r = resize(out_enh, h, w)
warped_r = resize(warped, h, w)
for name, img in [("out_raw", out_raw_r), ("out_enh", out_enh_r), ("warped", warped_r)]:
    c_orig = np.corrcoef(img.ravel(), orig.ravel())[0, 1]
    c_warp = np.corrcoef(img.ravel(), warped_r.ravel())[0, 1]
    print(f"{name}: corr(orig)={c_orig:.3f} corr(warped)={c_warp:.3f}")

# 文本行直线度：对每行计算暗像素质心 x 均值，看是否沿 y 呈正弦（卷曲）还是平直
def line_centroid_drift(img):
    dark = img < 128
    drifts = []
    for y in range(0, img.shape[0], 8):
        row = dark[y]
        if row.sum() < 20:
            continue
        xs = np.nonzero(row)[0]
        drifts.append((y, xs.mean()))
    return np.array(drifts)

for name, img in [("warped", warped), ("out_raw", out_raw_r), ("out_enh", out_enh_r)]:
    d = line_centroid_drift(img)
    if len(d) > 2:
        # 拟合一次项，看残差（残差大 = 弯曲）
        fit = np.polyfit(d[:, 0], d[:, 1], 1)
        resid = np.abs(d[:, 1] - np.polyval(fit, d[:, 0]))
        print(f"{name}: 行质心斜率={fit[0]:.3f} 弯曲残差 mean={resid.mean():.1f}px max={resid.max():.1f}px")
