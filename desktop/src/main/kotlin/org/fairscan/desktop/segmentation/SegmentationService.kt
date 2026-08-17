/*
 * Copyright 2025-2026 The FairScan authors
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package org.fairscan.desktop.segmentation

import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.tensorflowlite.TfLiteInterpreter
import org.bytedeco.tensorflowlite.TfLiteInterpreterOptions
import org.bytedeco.tensorflowlite.TfLiteModel
import org.bytedeco.tensorflowlite.TfLiteTensor
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterAllocateTensors
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterCreate
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterDelete
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterGetInputTensor
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterGetOutputTensor
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterInvoke
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterOptionsCreate
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterOptionsDelete
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteInterpreterOptionsSetNumThreads
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteModelCreateFromFile
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteModelDelete
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorByteSize
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorCopyFromBuffer
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorCopyToBuffer
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorDim
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorNumDims
import org.bytedeco.tensorflowlite.global.tensorflowlite.TfLiteTensorType
import org.bytedeco.tensorflowlite.global.tensorflowlite.kTfLiteFloat32
import org.bytedeco.tensorflowlite.global.tensorflowlite.kTfLiteOk
import org.opencv.core.Mat

/**
 * 使用 Bytedeco JavaCPP 封装的 TFLite C API 直接加载原始 .tflite 模型做推理。
 *
 * 与 Android 端（LiteRT Interpreter）使用同一模型文件与同一运行内核，
 * 输入/输出布局 NHWC float32：输入 [1,256,256,3]，输出 [1,256,256,1]。
 */
class SegmentationService(
    private val modelPath: String,
    private val numThreads: Int = 2,
) : MaskSegmenter {

    private val model: TfLiteModel
    private val interpreter: TfLiteInterpreter
    private val inputTensor: TfLiteTensor
    private val outputTensor: TfLiteTensor

    /** 输入张量形状，如 [1, 256, 256, 3]。 */
    val inputShape: IntArray

    /** 输出张量形状，如 [1, 256, 256, 1]。 */
    val outputShape: IntArray

    /** 输出元素总数 = 256*256。 */
    val outputElementCount: Int

    override val imageSize: Int = 256

    init {
        model = TfLiteModelCreateFromFile(modelPath)
            ?: throw IllegalStateException("Failed to load TFLite model: $modelPath")

        val options = TfLiteInterpreterOptionsCreate()
        try {
            TfLiteInterpreterOptionsSetNumThreads(options, numThreads)
            interpreter = TfLiteInterpreterCreate(model, options)
                ?: throw IllegalStateException("Failed to create TFLite interpreter")
        } finally {
            TfLiteInterpreterOptionsDelete(options)
        }

        // C API 中 TfLiteInterpreterCreate 不会自动分配张量，
        // 必须先调用 AllocateTensors，否则输入张量 data 为 NULL，CopyFromBuffer 会崩溃。
        val allocStatus = TfLiteInterpreterAllocateTensors(interpreter)
        check(allocStatus == kTfLiteOk) { "TfLiteInterpreterAllocateTensors failed: $allocStatus" }

        inputTensor = checkNotNull(TfLiteInterpreterGetInputTensor(interpreter, 0))
        outputTensor = checkNotNull(TfLiteInterpreterGetOutputTensor(interpreter, 0))

        inputShape = shapeOf(inputTensor)
        outputShape = shapeOf(outputTensor)
        outputElementCount = outputShape.fold(1) { acc, d -> acc * d }

        val inputType = TfLiteTensorType(inputTensor)
        if (inputType != kTfLiteFloat32) {
            throw IllegalStateException(
                "Unsupported input tensor type: $inputType (expected kTfLiteFloat32=$kTfLiteFloat32)"
            )
        }
        checkInputShape(inputShape)
    }

    private fun checkInputShape(shape: IntArray) {
        val expected = intArrayOf(1, 256, 256, 3)
        if (!shape.contentEquals(expected)) {
            throw IllegalStateException(
                "Unexpected model input shape ${shape.contentToString()}, expected ${expected.contentToString()}"
            )
        }
    }

    private fun shapeOf(tensor: TfLiteTensor): IntArray {
        val dims = TfLiteTensorNumDims(tensor)
        return IntArray(dims) { TfLiteTensorDim(tensor, it) }
    }

    /**
     * 预处理与 Android 端（ImageSegmentation.kt + TensorImage）完全一致：
     * BGR→RGB、BILINEAR 缩放 256x256、NHWC、(x-127.5)/127.5 归一化。
     */
    override fun preprocess(bgr: Mat): FloatArray = ImagePreprocessor.preprocess(bgr)

    /**
     * 执行一次推理。
     *
     * @param input NHWC 布局的 float32 输入，长度须等于输入元素总数（256*256*3）。
     * @return 输出 mask 的概率图，长度 [width*height]（256*256），取值通常为 [0,1]。
     */
    override fun run(input: FloatArray): FloatArray {
        val expected = inputShape.fold(1) { acc, d -> acc * d }
        require(input.size == expected) {
            "Input size ${input.size} != expected $expected"
        }

        val inputBytes = TfLiteTensorByteSize(inputTensor)
        val inPtr = FloatPointer(input.size.toLong())
        try {
            inPtr.put(input, 0, input.size)
            val copyStatus = TfLiteTensorCopyFromBuffer(inputTensor, inPtr, inputBytes)
            check(copyStatus == kTfLiteOk) { "TfLiteTensorCopyFromBuffer failed: $copyStatus" }

            val invokeStatus = TfLiteInterpreterInvoke(interpreter)
            check(invokeStatus == kTfLiteOk) { "TfLiteInterpreterInvoke failed: $invokeStatus" }

            val outputBytes = TfLiteTensorByteSize(outputTensor)
            val result = FloatArray(outputElementCount)
            val outPtr = FloatPointer(outputElementCount.toLong())
            try {
                val copyStatus = TfLiteTensorCopyToBuffer(outputTensor, outPtr, outputBytes)
                check(copyStatus == kTfLiteOk) { "TfLiteTensorCopyToBuffer failed: $copyStatus" }
                outPtr.get(result)
            } finally {
                outPtr.deallocate()
            }
            return result
        } finally {
            inPtr.deallocate()
        }
    }

    override fun close() {
        TfLiteInterpreterDelete(interpreter)
        TfLiteModelDelete(model)
    }
}
