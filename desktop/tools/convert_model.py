"""
FairScan 桌面移植：将 tflite 分割模型转换为 ONNX。

用法：
    python convert_model.py [input.tflite] [output.onnx]

说明：
    tflite2onnx 0.4.1 不支持 SUM op（模型解码器使用 reduce_sum）。
    这里在运行时给 tflite2onnx 打补丁：把 SUM 注册为 ONNX 的 ReduceSum。
"""
import logging
import sys
from pathlib import Path

import numpy as np
import tflite  # tflite2onnx 的依赖
import tflite2onnx
from tflite2onnx.op.reduce import Reduce
from tflite2onnx.op.common import Operator, OpFactory

logger = logging.getLogger('tflite2onnx')

# SUM -> ReduceSum（ONNX opset <= 12 时 axes 为属性；tflite2onnx 输出 opset 11）
if tflite.BuiltinOperator.SUM not in Reduce.TypeMapping:
    Reduce.TypeMapping[tflite.BuiltinOperator.SUM] = "ReduceSum"
    # OpFactory.registry 在模块 import 时已固化，需同步注册
    OpFactory.registry[tflite.BuiltinOperator.SUM] = Reduce


class GatherNd(Operator):
    """GATHER_ND -> ONNX GatherND（opset 11 起支持）。"""
    TypeMapping = {
        tflite.BuiltinOperator.GATHER_ND: 'GatherND',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        return 'GatherND'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)

        self.parseInput(0)
        self.parseInput(1)  # indices
        self.parseOutput(0)

        # 可选的 batch_dims（opset 11 的 GatherND 不支持该属性，仅在非 0 时添加）
        if op.BuiltinOptions() is not None:
            option = tflite.GatherNdOptions()
            option.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            batch_dims = option.BatchDims()
            if batch_dims != 0:
                self.attrs['batch_dims'] = batch_dims

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass


OpFactory.register(GatherNd)


class SharedPad(Operator):
    """修复 tflite2onnx 的 PAD 共享 pads 张量 bug：
    多个 PAD op 共用同一 pads initializer 时，第一个 transform 会把
    pt.shape 展平成 1D，后续 op 按 [n, 2] 取列会 IndexError。
    这里改为只依赖数据长度（len % 2 == 0）。"""
    TypeMapping = {
        tflite.BuiltinOperator.PAD: 'Pad',
        tflite.BuiltinOperator.MIRROR_PAD: 'Pad',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.attrs['mode'] = 'constant'
        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Pad'
        opcode = self.model.OperatorCodes(self.tflite.OpcodeIndex()).BuiltinCode()
        return self.TypeMapping[opcode]

    def parse(self):
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if opcode is tflite.BuiltinOperator.MIRROR_PAD:
            self.attrs['mode'] = 'reflect'
        else:
            self.attrs['mode'] = 'constant'
        assert(op.InputsLength() == 2)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        pt = self.parseInput(1)
        assert(pt.isInitializer)
        pt.asDtype('int64')
        self.parseOutput(0)
        self.setParsed()

    def propagatableTensors(self):
        return [self.inputs[0], self.outputs[0]]

    def transform(self):
        layout = self.inputs[0].layout
        pt = self.inputs[1]
        pads = np.reshape(pt.data, (-1, 2))
        if layout is None:
            pads = np.transpose(pads)
        else:
            pads_begin = pads[:, 0]
            pads_end = pads[:, 1]
            pads_begin = layout.transform(pads_begin)
            pads_end = layout.transform(pads_end)
            pads = np.array([pads_begin, pads_end])
        pt.data = pads.flatten()
        pt.shape = [pt.data.shape[0]]


# 覆盖原有 Padding（PAD/MIRROR_PAD 已在 registry 中）
OpFactory.registry[tflite.BuiltinOperator.PAD] = SharedPad
OpFactory.registry[tflite.BuiltinOperator.MIRROR_PAD] = SharedPad


def _patch_reshape_transform():
    """修复 forFakeBroadcasting 的 Reshape：当输入输出都无 layout（例如
    AdaptiveAvgPool2d 全局池化链路：MEAN/SUM -> RESHAPE）时，reshape 完全透明，
    无需做任何布局变换；原实现会抛 ValueError。其余情况委托原实现。"""
    from tflite2onnx.op.reshape import Reshape

    original_transform = Reshape.transform

    def transform(self):
        if self.forFakeBroadcasting and self.outputs[0].layout is None:
            logger.debug("Skipping layout transform for transparent reshape %s", self.shorty)
            return
        original_transform(self)

    Reshape.transform = transform


_patch_reshape_transform()


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/fairscan-segmentation-model.tflite")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".onnx")

    print(f"Converting {input_path} -> {output_path}")
    tflite2onnx.convert(str(input_path), str(output_path))
    print("Conversion done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
