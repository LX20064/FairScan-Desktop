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
package org.fairscan.desktop.orientation

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * 文档方向分类（PP-LCNet_x1_0_doc_ori 转 ONNX）。
 *
 * 输入为任意 BGR 图，内部缩放至模型输入尺寸并归一化；
 * 输出四分类角度（0°/90°/180°/270°，表示图像相对正方向顺时针偏转的角度）。
 * [detectRotation] 返回使文档转正所需的顺时针旋转角度（0/90/180/270）。
 */
class OrientationService(
    private val modelPath: String,
    private val numThreads: Int = 2,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val inputH: Int
    private val inputW: Int

    init {
        val opts = OrtSession.SessionOptions()
        try {
            opts.setIntraOpNumThreads(numThreads)
            session = env.createSession(modelPath, opts)
        } finally {
            opts.close()
        }

        val entry = session.inputInfo.entries.firstOrNull()
            ?: throw IllegalStateException("ONNX 模型没有输入张量: $modelPath")
        inputName = entry.key
        val tensorInfo = entry.value.info as? TensorInfo
            ?: throw IllegalStateException("ONNX 输入不是张量: ${entry.value.info.javaClass.simpleName}")
        val shape = tensorInfo.shape
        if (shape.size != 4) {
            throw IllegalStateException("方向分类模型输入应为 NCHW 4 维，实际: ${shape.contentToString()}")
        }
        inputH = shape[2].toInt().coerceAtLeast(1)
        inputW = shape[3].toInt().coerceAtLeast(1)
        if (shape[1].toInt() != 3) {
            throw IllegalStateException("方向分类模型输入通道数应为 3，实际: ${shape[1]}")
        }
    }

    /**
     * 检测文档方向，返回需要顺时针旋转的角度（0/90/180/270）。
     */
    fun detectRotation(bgr: Mat): Int {
        val input = preprocess(bgr)
        try {
            val result = session.run(mapOf(inputName to input))
            try {
                val probs = result[0].value as Array<FloatArray>
                val row = probs[0]
                val label = row.indices.maxByOrNull { row[it] } ?: 0
                // 模型输出 0/1/2/3 表示内容相对正方向顺时针偏转 0/90/180/270；
                // 转正需要顺时针旋转 (360 - label*90) % 360。
                return ((360 - label * 90) % 360 + 360) % 360
            } finally {
                result.close()
            }
        } finally {
            input.close()
        }
    }

    /** BGR → RGB、缩放至输入尺寸、ImageNet 归一化，输出 NCHW float32。 */
    private fun preprocess(bgr: Mat): OnnxTensor {
        val resized = Mat()
        Imgproc.resize(bgr, resized, Size(inputW.toDouble(), inputH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val floatMat = Mat()
        resized.convertTo(floatMat, CvType.CV_32F)

        val data = FloatArray(inputH * inputW * 3)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        var idx = 0
        val bgrVals = FloatArray(3)
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                floatMat.get(y, x, bgrVals) // bgrVals = [B, G, R]
                val r = (bgrVals[2] / 255f - mean[0]) / std[0]
                val g = (bgrVals[1] / 255f - mean[1]) / std[1]
                val b = (bgrVals[0] / 255f - mean[2]) / std[2]
                data[idx] = r; data[idx + inputH * inputW] = g; data[idx + 2 * inputH * inputW] = b
                idx++
            }
        }
        resized.release()
        floatMat.release()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1L, 3L, inputH.toLong(), inputW.toLong()))
    }

    override fun close() {
        session.close()
    }
}
