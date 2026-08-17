"""查看 uvdoc.onnx 输入/输出规格。"""
import onnxruntime as ort

sess = ort.InferenceSession("models/uvdoc.onnx", providers=["CPUExecutionProvider"])
for n, i in enumerate(sess.get_inputs()):
    print(f"in[{n}] name={i.name} shape={i.shape} type={i.type}")
for n, o in enumerate(sess.get_outputs()):
    print(f"out[{n}] name={o.name} shape={o.shape} type={o.type}")
