import numpy as np
import cv2
import onnxruntime as ort
import sys

img = cv2.imread("app/src/debug/assets/uncropped/img01.jpg")
official = cv2.imread("desktop/cmpre/uvdoc-official.png")  # (2800,2100,3)

# 我们的 onnx: 直灌 2800x2100
SESS = ort.InferenceSession("desktop/models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = SESS.get_inputs()[0].name
chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
raw = SESS.run(None, {in_name: chw})[0][0]  # (3,H,W)
ours = np.clip(np.transpose(raw, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
print("ours shape:", ours.shape, "official:", official.shape)

print("ours mean:", round(ours.mean(), 1), " official mean:", round(official.mean(), 1))
print("ours std:", round(ours.std(), 1), " official std:", round(official.std(), 1))
for c in range(3):
    a = ours[:, :, c].astype(np.float32)
    b = official[:, :, c].astype(np.float32)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print("ch%d 相关: %.4f  均值差: %.1f" % (c, corr, (a - b).mean()))

cv2.imwrite("desktop/cmpre/uvdoc-ours-vs-official.png", np.hstack([cv2.resize(ours, (1050, 1400)), cv2.resize(official, (1050, 1400))]))
