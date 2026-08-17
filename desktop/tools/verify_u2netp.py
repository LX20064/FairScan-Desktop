"""验证 u2netp.onnx：输入归一化方式与输出范围（是否内置 sigmoid）。

用法: python tools/verify_u2netp.py [图片路径]
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort

MODEL = "models/u2netp.onnx"
IMG = sys.argv[1] if len(sys.argv) > 1 else "../app/src/debug/assets/uncropped/img01.jpg"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

img = cv2.imread(IMG)
if img is None:
    sys.exit("cannot read image: " + IMG)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

variants = {
    "imagenet mean/std": None,
    "plain /255": None,
}
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
variants["imagenet mean/std"] = np.transpose((rgb - mean) / std, (2, 0, 1))[None]
variants["plain /255"] = np.transpose(rgb, (2, 0, 1))[None]

for name, x in variants.items():
    outs = sess.run(None, {in_name: x})
    print(f"\n== {name} ==")
    for i, o in enumerate(outs):
        print(f"  out{i} shape={o.shape} min={o.min():.4f} max={o.max():.4f} mean={o.mean():.4f}")
    # 以第一个输出在 0.5 阈值下文档占比做粗判
    d1 = outs[0]
    for th in (0.0, 0.5):
        frac = float((d1 > th).mean())
        print(f"  out0 > {th}: {frac * 100:.1f}%")
