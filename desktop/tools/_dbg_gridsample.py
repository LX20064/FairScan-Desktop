import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# 构造最小模型: X(1,3,712,488) + Grid(1,712,488,2) -> GridSample -> out
H, W = 712, 488
yy, xx = np.mgrid[0:H, 0:W]
# 恒等网格: gx: -1..1 沿列(宽), gy: -1..1 沿行(高), align_corners=True
gx = (xx / (W - 1)) * 2 - 1
gy = (yy / (H - 1)) * 2 - 1
grid = np.stack([gx, gy], axis=-1).astype(np.float32)[None]  # (1,H,W,2)

# 棋盘格输入
chk = (((xx // 16) + (yy // 16)) % 2 * 255).astype(np.uint8)
img = np.stack([chk] * 3, axis=-1).astype(np.float32) / 255.0  # (H,W,3)

x_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, H, W])
g_t = helper.make_tensor_value_info("g", TensorProto.FLOAT, [1, H, W, 2])
y_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, H, W])

node = helper.make_node("GridSample", ["x", "g"], ["y"], mode="bilinear",
                        padding_mode="zeros", align_corners=1)
graph = helper.make_graph([node], "gs", [x_t, g_t], [y_t])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
onnx.save(model, "desktop/models/_gs_test.onnx")

s = ort.InferenceSession("desktop/models/_gs_test.onnx", providers=["CPUExecutionProvider"])
out = s.run(None, {"x": np.transpose(img, (2, 0, 1))[None], "g": grid})[0][0]  # (C,H,W)
o = (np.clip(np.transpose(out, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)[:, :, 0]
print("恒等网格: 匹配比例 =", round(float((o == chk).mean()), 4), "(应≈1.0)")

# 再测: 用我们的真实 flow(45x31) 上采样到 H,W 后的网格
import cv2
flow = np.load("desktop/cmpre/_flow45x31.npy") if False else None
