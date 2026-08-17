import cv2
import numpy as np
import onnxruntime as ort

# 1) 从模型 dump flow
SESS = ort.InferenceSession("desktop/models/uvdoc.onnx", providers=["CPUExecutionProvider"])
in_name = SESS.get_inputs()[0].name

yy, xx = np.mgrid[0:712, 0:488]
chk = (((xx // 16) + (yy // 16)) % 2 * 255).astype(np.uint8)
img = np.stack([chk] * 3, axis=-1).astype(np.float32) / 255.0

# 中间输出 flow 的名字
import onnx
m = onnx.load(r"desktop\models\uvdoc.onnx")
g = m.graph
flow_name = None
for n in g.node:
    if "p2o.pd_op.add.41.0" in n.output:
        flow_name = "p2o.pd_op.add.41.0"
        v = onnx.helper.ValueInfoProto()
        v.name = flow_name
        g.output.append(v)
onnx.save(m, r"desktop\models\uvdoc_tmp.onnx")
S2 = ort.InferenceSession(r"desktop\models\uvdoc_tmp.onnx", providers=["CPUExecutionProvider"])
outs = dict(zip([i.name for i in S2.get_outputs()], S2.run(None, {S2.get_inputs()[0].name: np.transpose(img, (2, 0, 1))[None]})))
flow = outs[flow_name][0]  # (2,45,31)
out_full = outs["fetch_name_0"][0]  # (3,712,488)
print("flow:", flow.shape, "out:", out_full.shape)

# 2) 手动: flow -> bilinear 上采样到 712x488 (align_corners) -> transpose -> grid_sample
H, W = 712, 488
fh, fw = flow.shape[1], flow.shape[2]
flow_up = np.zeros((2, H, W), np.float32)
for c in range(2):
    src = flow[c]  # (fh,fw)
    flow_up[c] = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR)
grid = np.transpose(flow_up, (1, 2, 0))  # (H,W,2)

# 用同一个 GridSample 迷你模型做手动采样
import onnx
from onnx import helper, TensorProto
x_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, H, W])
g_t = helper.make_tensor_value_info("g", TensorProto.FLOAT, [1, H, W, 2])
y_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, H, W])
node = helper.make_node("GridSample", ["x", "g"], ["y"], mode="bilinear",
                        padding_mode="zeros", align_corners=1)
graph = helper.make_graph([node], "gs2", [x_t, g_t], [y_t])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
onnx.save(model, "desktop/models/_gs2.onnx")
S3 = ort.InferenceSession("desktop/models/_gs2.onnx", providers=["CPUExecutionProvider"])
man = S3.run(None, {"x": np.transpose(img, (2, 0, 1))[None], "g": grid[None]})[0][0]

print("模型输出 vs 手动重现 相关 =",
      round(float(np.corrcoef(out_full.ravel(), man.ravel())[0, 1]), 4))
print("模型输出 ch0 与输入棋盘 相关 =",
      round(float(np.corrcoef(out_full[0].ravel(), chk.ravel())[0, 1]), 4))
print("手动重现 ch0 与输入棋盘 相关 =",
      round(float(np.corrcoef(man[0].ravel(), chk.ravel())[0, 1]), 4))
