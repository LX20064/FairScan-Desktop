"""对比 UVDoc 管线改造前后效果（img01 平铺文档）：
A = 基线增强页（无展平，uvdoc-new-base 输出）
B = 新流程：raw extracted -> UVDoc -> 增强（uvdoc-new 输出）
C = 旧流程模拟：A(增强页) -> uvdoc.onnx（PaddleX 预处理/后处理）
指标：A vs B、A vs C 的结构相似度 + 网格相对恒等偏差（间接看行弯曲）。
用法: python tools/_cmp_uvdoc_flow.py
"""
import glob
import os
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = "../../cmpre/uvdoc-new-base/图片"
NEW_DIR = "../../cmpre/uvdoc-new/图片"


def load(p):
    files = sorted(glob.glob(p))
    assert files, f"no file: {p}"
    return cv2.imdecode(np.fromfile(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)


def run_uvdoc(bgr):
    h, w = bgr.shape[:2]
    inp = bgr.astype(np.float32) / 255.0
    chw = np.transpose(inp, (2, 0, 1))[None]
    sess = ort.InferenceSession("models/uvdoc.onnx", providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: chw})[0]
    out = np.transpose(out[0], (1, 2, 0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


a = load(os.path.join(BASE_DIR, "*.jpg"))
b = load(os.path.join(NEW_DIR, "*.jpg"))
if a.shape != b.shape:
    b = cv2.resize(b, (a.shape[1], a.shape[0]))

c = run_uvdoc(a)
if c.shape != a.shape:
    c = cv2.resize(c, (a.shape[1], a.shape[0]))

def stats(name, img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  {name}: mean={g.mean():.1f} std={g.std():.1f} bgmode={np.argmax(np.bincount(g.ravel()))}")

print("=== img01 平铺文档 UVDoc 流程对比 ===")
stats("A 基线增强页(无展平)", a)
stats("B 新流程 raw->UVDoc->增强", b)
stats("C 旧流程模拟 增强页->UVDoc", c)
print(f"SSIM(A,B 新流程) = {ssim(a, b):.3f}")
print(f"SSIM(A,C 旧流程) = {ssim(a, c):.3f}")
print(f"SSIM(B,C)        = {ssim(b, c):.3f}")
print(f"均方根差(B,C)    = {np.sqrt(((b.astype(float)-c.astype(float))**2).mean()):.2f}")
cv2.imwrite("../../cmpre/uvdoc-flow-A.png", a)
cv2.imwrite("../../cmpre/uvdoc-flow-B-new.png", b)
cv2.imwrite("../../cmpre/uvdoc-flow-C-old.png", c)
print("已导出 uvdoc-flow-A/B/C.png")
