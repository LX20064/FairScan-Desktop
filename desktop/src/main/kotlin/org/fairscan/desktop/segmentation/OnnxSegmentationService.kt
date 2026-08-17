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

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * ONNX 通用分割服务（U²-Net u2netp 等 saliency/前景分割模型）。
 *
 * 与 rembg 的 u2netp 预处理/输出约定保持一致：
 *  - 输入 NCHW float32 [1,3,320,320]，BILINEAR 缩放，ImageNet mean/std 归一化；
 *  - 输出多个 [1,1,320,320] 张量（U²-Net 多尺度 d1..d7，均已内置 sigmoid），
 *    取第一个（d1 融合输出）作为最终概率图，长度 imageSize*imageSize。
 */
class OnnxSegmentationService(
    private val modelPath: String,
    private val numThreads: Int = 2,
) : MaskSegmenter {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    override val imageSize: Int

    /** 输出元素总数 = imageSize*imageSize。 */
    val outputElementCount: Int

    init {
        val opts = OrtSession.SessionOptions()
        try {
            opts.setIntraOpNumThreads(numThreads)
            session = env.createSession(modelPath, opts)
        } finally {
            opts.close()
        }

        val inEntry = session.inputInfo.entries.firstOrNull()
            ?: throw IllegalStateException("ONNX 模型没有输入张量: $modelPath")
        inputName = inEntry.key
        val inInfo = inEntry.value.info as? TensorInfo
            ?: throw IllegalStateException("ONNX 输入不是张量: ${inEntry.value.info.javaClass.simpleName}")
        val inShape = inInfo.shape
        if (inShape.size != 4) {
            throw IllegalStateException("分割模型输入应为 NCHW 4 维，实际: ${inShape.contentToString()}")
        }
        if (inShape[1].toInt() != 3) {
            throw IllegalStateException("分割模型输入通道数应为 3，实际: ${inShape[1]}")
        }
        val h = inShape[2].toInt().coerceAtLeast(1)
        val w = inShape[3].toInt().coerceAtLeast(1)
        if (h != w) {
            throw IllegalStateException("分割模型输入应为正方形，实际: ${inShape.contentToString()}")
        }
        imageSize = h

        val outInfo = session.outputInfo.entries.first().value.info as? TensorInfo
            ?: throw IllegalStateException("ONNX 输出不是张量")
        outputElementCount = outInfo.shape.fold(1L) { acc, d -> acc * d }.toInt()
        if (outputElementCount != imageSize * imageSize) {
            throw IllegalStateException(
                "分割模型输出元素数 $outputElementCount 与 ${imageSize}x$imageSize 不符"
            )
        }
    }

    /** BGR → RGB、BILINEAR 缩放、ImageNet 归一化，输出 NCHW float32（与 rembg 一致）。 */
    override fun preprocess(bgr: Mat): FloatArray {
        val resized = Mat()
        val floatMat = Mat()
        try {
            Imgproc.cvtColor(bgr, resized, Imgproc.COLOR_BGR2RGB)
            Imgproc.resize(
                resized,
                floatMat,
                Size(imageSize.toDouble(), imageSize.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR,
            )
            floatMat.convertTo(floatMat, CvType.CV_32F)

            val n = imageSize * imageSize
            val data = FloatArray(n * 3)
            val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            val std = floatArrayOf(0.229f, 0.224f, 0.225f)
            val rgbVals = FloatArray(3)
            var idx = 0
            for (y in 0 until imageSize) {
                for (x in 0 until imageSize) {
                    floatMat.get(y, x, rgbVals) // rgbVals = [B, G, R]（仍为 BGR 内存序）
                    val r = (rgbVals[2] / 255f - mean[0]) / std[0]
                    val g = (rgbVals[1] / 255f - mean[1]) / std[1]
                    val b = (rgbVals[0] / 255f - mean[2]) / std[2]
                    data[idx] = r
                    data[idx + n] = g
                    data[idx + 2 * n] = b
                    idx++
                }
            }
            return data
        } finally {
            resized.release()
            floatMat.release()
        }
    }

    override fun run(input: FloatArray): FloatArray {
        require(input.size == outputElementCount * 3) {
            "Input size ${input.size} != expected ${outputElementCount * 3}"
        }
        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1L, 3L, imageSize.toLong(), imageSize.toLong()),
        )
        try {
            val result = session.run(mapOf(inputName to tensor))
            try {
                val outTensor = result[0] as OnnxTensor
                val buf = outTensor.floatBuffer
                val prob = FloatArray(outputElementCount)
                buf.get(prob)
                return prob
            } finally {
                result.close()
            }
        } finally {
            tensor.close()
        }
    }

    override fun close() {
        session.close()
    }
}
