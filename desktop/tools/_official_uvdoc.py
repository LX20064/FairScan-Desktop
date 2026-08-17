"""用官方 PaddlePaddle 加载 UVDoc_infer (PIR) 模型并推理，严格按 PaddleX 前后处理。"""
import sys
import numpy as np
import cv2
from paddle.inference import Config, create_predictor

MODEL_DIR = r"desktop\cmpre\uvdoc-infer\UVDoc_infer"


def build_predictor():
    config = Config(MODEL_DIR + r"\inference.json", MODEL_DIR + r"\inference.pdiparams")
    config.disable_gpu()
    config.switch_ir_optim(True)
    predictor = create_predictor(config)
    return predictor


def run_uvdoc(bgr):
    """PaddleX 流程: BGR -> /255 -> CHW -> batch -> model -> CHW -> *255 -> [::-1]"""
    predictor = build_predictor() if not hasattr(run_uvdoc, "pred") else run_uvdoc.pred
    run_uvdoc.pred = predictor
    h, w = bgr.shape[:2]
    inp = bgr.astype(np.float32) / 255.0
    chw = np.transpose(inp, (2, 0, 1))[None].astype(np.float32)
    in_names = predictor.get_input_names()
    in_handle = predictor.get_input_handle(in_names[0])
    in_handle.copy_from_cpu(chw)
    predictor.run()
    out_names = predictor.get_output_names()
    out = predictor.get_output_handle(out_names[0]).copy_to_cpu()  # (1,3,H,W)
    im = np.transpose(out[0], (1, 2, 0)) * 255.0
    im = im[:, :, ::-1]
    return np.clip(im, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    img = sys.argv[1]
    bgr = cv2.imread(img)
    print("input:", img, bgr.shape)
    out = run_uvdoc(bgr)
    print("output:", out.shape, "mean:", round(out.mean(), 1), "range:", out.min(), out.max())
    cv2.imwrite("desktop/cmpre/uvdoc-official.png", out)
    print("saved desktop/cmpre/uvdoc-official.png")
