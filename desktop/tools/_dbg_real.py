import cv2
import numpy as np
import onnxruntime as ort

SESS = ort.InferenceSession("desktop/models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = SESS.get_inputs()[0].name
orig = cv2.imread("app/src/debug/assets/uncropped/img01.jpg")
inp = cv2.resize(orig, (488, 712), interpolation=cv2.INTER_LINEAR)

import onnx
m = onnx.load(r"desktop\models\uvdoc.onnx")
g = m.graph
for n in g.node:
    if "p2o.pd_op.add.41.0" in n.output:
        v = onnx.helper.ValueInfoProto(); v.name = "p2o.pd_op.add.41.0"; g.output.append(v)
onnx.save(m, r"desktop\models\uvdoc_tmp.onnx")
S2 = ort.InferenceSession(r"desktop\models\uvdoc_tmp.onnx", providers=["CPUExecutionProvider"])
outs = dict(zip([i.name for i in S2.get_outputs()],
                S2.run(None, {S2.get_inputs()[0].name: np.transpose(inp.astype(np.float32) / 255.0, (2, 0, 1))[None]})))
flow = outs["p2o.pd_op.add.41.0"][0]
out_full = np.clip(np.transpose(outs["fetch_name_0"][0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

H, W = 712, 488
gi = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
go = cv2.cvtColor(out_full, cv2.COLOR_BGR2GRAY)
print("抽样: out(y,x) vs 对应输入采样点(应为近似值)")
print("out mean=%d, in mean=%d" % (go.mean(), gi.mean()))
for (oy, ox) in [(0, 0), (200, 120), (356, 244), (500, 380), (711, 487)]:
    gy = flow[1, oy // 16, ox // 16]
    gx = flow[0, oy // 16, ox // 16]
    u = (gx + 1) / 2 * (W - 1)
    v = (gy + 1) / 2 * (H - 1)
    print("out(%d,%d)=%3d  in(%.0f,%.0f)=%3d   (流=(%.3f,%.3f))"
          % (oy, ox, go[oy, ox], u, v, gi[int(v), int(u)], gx, gy))

# 网格位移量（相对恒等）
mx = np.abs(flow[0] - np.linspace(-1, 1, 31)[None, :]).max()
my = np.abs(flow[1] - np.linspace(-1, 1, 45)[:, None]).max()
print("相对恒等网格最大偏差: dx=%.3f dy=%.3f (全图跨度2, 偏差应<0.1)" % (mx, my))
cv2.imwrite("desktop/cmpre/uvdoc-real-out.png", out_full)
cv2.imwrite("desktop/cmpre/uvdoc-real-in.png", inp)
