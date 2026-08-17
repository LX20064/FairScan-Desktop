"""生成硬场景测试图（深色底 / 低光 / 阴影），用于对比分割模型。

输出: cmpre/hard-darkbg.jpg, hard-lowlight.jpg, hard-shadow.jpg
"""
import numpy as np
from PIL import Image

IMG = "../app/src/debug/assets/uncropped/img01.jpg"
doc = Image.open(IMG).convert("RGB")
W, H = doc.size
print("doc size:", doc.size)


def to_np(img):
    return np.asarray(img, dtype=np.uint8).copy()


# 1) 深色底：把文档缩到 60% 贴到黑色桌面上（文档占中间区域）
canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas[...] = 20  # 近黑桌面
d = to_np(doc.resize((int(W * 0.6), int(H * 0.6))))
x0, y0 = int(W * 0.2), int(H * 0.2)
x1, y1 = x0 + d.shape[1], y0 + d.shape[0]
canvas[y0:y1, x0:x1] = d
Image.fromarray(canvas).save("cmpre/hard-darkbg.jpg")

# 2) 低光：整体压暗 + 轻度噪声
low = (to_np(doc) * 0.35).astype(np.uint8)
low = np.clip(low.astype(np.int16) + np.random.default_rng(0).normal(0, 3, low.shape).astype(np.int16), 0, 255).astype(np.uint8)
Image.fromarray(low).save("cmpre/hard-lowlight.jpg")

# 3) 阴影：对角方向渐变压暗（右上亮、左下暗）
shadow = to_np(doc).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
t = (xx / W * 0.5 + yy / H * 0.7)  # 0..1.2
gain = 0.35 + 0.65 * np.clip(1.0 - t, 0, 1)
shadow = np.clip(shadow * gain[..., None], 0, 255).astype(np.uint8)
Image.fromarray(shadow).save("cmpre/hard-shadow.jpg")

print("saved 3 hard cases")
