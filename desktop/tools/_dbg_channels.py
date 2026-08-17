import numpy as np
import cv2
import onnxruntime as ort
from _official_uvdoc import run_uvdoc

S = ort.InferenceSession("desktop/models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = S.get_inputs()[0].name

print("=== 纯色测试 (BGR) ===")
for name, col in [("红(0,0,255)", (0, 0, 255)), ("绿(0,255,0)", (0, 255, 0)),
                  ("蓝(255,0,0)", (255, 0, 0)), ("白", (255, 255, 255)), ("黑", (0, 0, 0))]:
    img = np.full((200, 200, 3), col, np.uint8)
    chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
    ours = np.clip(np.transpose(S.run(None, {in_name: chw})[0][0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
    off = run_uvdoc(img)
    print("%s: 输入%s  我们BGR=%s  官方BGR=%s" % (name, col, tuple(ours[100, 100]), tuple(off[100, 100])))

print()
print("=== 水平渐变 (BGR 顺序构造) ===")
grad = np.zeros((200, 300, 3), np.uint8)
grad[:, :, 0] = np.linspace(0, 255, 300).astype(np.uint8)  # B 渐变
grad[:, :, 1] = 128
grad[:, :, 2] = 255 - grad[:, :, 0].astype(np.uint8)  # R 反向渐变
chw = np.transpose(grad.astype(np.float32) / 255.0, (2, 0, 1))[None]
ours = np.clip(np.transpose(S.run(None, {in_name: chw})[0][0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
off = run_uvdoc(grad)
for x in (10, 100, 290):
    print("x=%d 输入=%s 我们=%s 官方=%s" % (x, tuple(grad[100, x]), tuple(ours[100, x]), tuple(off[100, x])))
