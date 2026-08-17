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
package org.fairscan.desktop.unwarp

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.FloatBuffer

/**
 * 文档展平（UVDoc 转 ONNX，PaddleOCR 生态，Apache 2.0）。
 *
 * 输入为任意尺寸 BGR 图（无需缩放，模型输入输出均动态），输出同尺寸的展开图。
 * 预处理/后处理与 PaddleX image_unwarping 参考实现一致：
 * - 输入: BGR -> /255 -> CHW float32（PaddleX Normalize scale=1/255）
 * - 输出: CHW float32（值域 [0,1]）-> HWC -> *255 -> 保持 BGR 通道序
 *
 * 注意：本模型逐通道保持（纯色测试无通道置换），故不做 PaddleX 的通道反转。
 */
class UVDocService(
    private val modelPath: String,
    private val numThreads: Int = 2,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val opts = OrtSession.SessionOptions()
        try {
            opts.setIntraOpNumThreads(numThreads)
            session = env.createSession(modelPath, opts)
        } finally {
            opts.close()
        }

        val entry = session.inputInfo.entries.firstOrNull()
            ?: throw IllegalStateException("UVDoc 模型没有输入张量: $modelPath")
        inputName = entry.key
    }

    /**
     * 展开文档：输入 BGR Mat，返回同尺寸 BGR 展开图（调用方负责 release）。
     */
    fun unwarp(bgr: Mat): Mat {
        val h = bgr.rows()
        val w = bgr.cols()
        val input = preprocess(bgr, w, h)
        return try {
            val result = session.run(mapOf(inputName to input))
            try {
                val tensor = result[0] as? OnnxTensor
                    ?: throw IllegalStateException("UVDoc 输出不是 OnnxTensor")
                // 4D 输出 [1,3,H,W] 的 value 是四层嵌套数组，统一用 FloatBuffer 读展平数据
                val flat = FloatArray(3 * h * w)
                tensor.getFloatBuffer().get(flat)
                postprocess(flat, w, h)
            } finally {
                result.close()
            }
        } finally {
            input.close()
        }
    }

    /** BGR -> /255 -> CHW float32，NCHW 张量。 */
    private fun preprocess(bgr: Mat, w: Int, h: Int): OnnxTensor {
        val floatMat = Mat()
        bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

        val data = FloatArray(3 * h * w)
        val bgrVals = FloatArray(3)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                floatMat.get(y, x, bgrVals) // [B, G, R]
                data[idx] = bgrVals[0]
                data[idx + h * w] = bgrVals[1]
                data[idx + 2 * h * w] = bgrVals[2]
                idx++
            }
        }
        floatMat.release()
        return OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(data),
            longArrayOf(1L, 3L, h.toLong(), w.toLong()),
        )
    }

    /** CHW float -> HWC -> *255 -> CV_8UC3 BGR。 */
    private fun postprocess(flat: FloatArray, w: Int, h: Int): Mat {
        val channels = arrayOf(Mat(h, w, CvType.CV_32F), Mat(h, w, CvType.CV_32F), Mat(h, w, CvType.CV_32F))
        val row = FloatArray(w)
        for (c in 0 until 3) {
            val plane = channels[c]
            val base = c * h * w
            for (y in 0 until h) {
                System.arraycopy(flat, base + y * w, row, 0, w)
                plane.put(y, 0, row)
            }
        }
        val merged = Mat()
        Core.merge(channels.toList(), merged)
        val out = Mat()
        merged.convertTo(out, CvType.CV_8UC3, 255.0) // 饱和截断到 [0,255]
        channels.forEach { it.release() }
        merged.release()
        return out
    }

    override fun close() {
        session.close()
    }
}
