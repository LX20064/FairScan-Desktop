"""把 uvdoc.onnx 的 IR version 从 10 降到 8（兼容 onnxruntime <= 1.14 的 Win7 支持线）。

IR version 只是容器版本号，opset(16) 与算子语义不变，推理结果不受影响。
仅对官方预编译最低支持 IR=8 的旧版 onnxruntime 有意义；新版 ORT 无影响。
用法: python tools/patch_ir_version.py
"""
import os
import shutil

import onnx

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "uvdoc.onnx")

BACKUP_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "_orig_ir10")
os.makedirs(BACKUP_DIR, exist_ok=True)
shutil.copy2(MODEL, os.path.join(BACKUP_DIR, "uvdoc.onnx"))

m = onnx.load(MODEL, load_external_data=False)
print("before: ir_version =", m.ir_version, "opset =", [(o.domain, o.version) for o in m.opset_import])
if m.ir_version > 8:
    m.ir_version = 8
    # IR10 新增字段，ORT 1.14 不认识，一并清除
    if len(m.training_info) > 0:
        del m.training_info[:]
    onnx.save(m, MODEL)
    m2 = onnx.load(MODEL, load_external_data=False)
    print("after : ir_version =", m2.ir_version, "opset =", [(o.domain, o.version) for o in m2.opset_import])
    print("backup ->", os.path.join(BACKUP_DIR, "uvdoc.onnx"))
else:
    print("无需修改")
