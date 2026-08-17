import cv2
import numpy as np
import onnxruntime as ort

SESS = ort.InferenceSession("desktop/models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = SESS.get_inputs()[0].name
yy, xx = np.mgrid[0:712, 0:488]
chk = (((xx // 16) + (yy // 16)) % 2 * 255).astype(np.uint8)
img = np.stack([chk] * 3, axis=-1).astype(np.float32) / 255.0

import onnx
m = onnx.load(r"desktop\models\uvdoc.onnx")
g = m.graph
for n in g.node:
    if "p2o.pd_op.add.41.0" in n.output:
        v = onnx.helper.ValueInfoProto(); v.name = "p2o.pd_op.add.41.0"; g.output.append(v)
onnx.save(m, r"desktop\models\uvdoc_tmp.onnx")
S2 = ort.InferenceSession(r"desktop\models\uvdoc_tmp.onnx", providers=["CPUExecutionProvider"])
outs = dict(zip([i.name for i in S2.get_outputs()], S2.run(None, {S2.get_inputs()[0].name: np.transpose(img, (2, 0, 1))[None]})))
flow = outs["p2o.pd_op.add.41.0"][0]
out_full = np.transpose(outs["fetch_name_0"][0], (1, 2, 0))
o = (np.clip(out_full, 0, 1) * 255).astype(np.uint8)[:, :, 0]

# flow 中心行/列的完整值
print("flow ch0 行15 全部31列:")
print([round(float(v), 3) for v in flow[0, 15]])
print("flow ch1 列15 全部45行:")
print([round(float(v), 3) for v in flow[1, :, 15]])

# 抽查输出像素与输入棋盘的关系
H, W = 712, 488
def where(gx, gy):
    u = (gx + 1) / 2 * (W - 1)
    v = (gy + 1) / 2 * (H - 1)
    return u, v

for (oy, ox) in [(0, 0), (100, 100), (356, 244), (600, 400), (711, 487)]:
    gy = flow[1, oy // 16, ox // 16]
    gx = flow[0, oy // 16, ox // 16]
    u, v = where(gx, gy)
    print("out(%d,%d)=%3d  grid=(%.3f,%.3f) -> 采样 in(%.1f,%.1f)=%3d"
          % (oy, ox, o[oy, ox], gx, gy, u, v, chk[int(v) % 712, int(u) % 488]))
cv2.imwrite("desktop/cmpre/uvdoc-chk-out.png", o)
cv2.imwrite("desktop/cmpre/uvdoc-chk-in.png", chk)
print("saved")
