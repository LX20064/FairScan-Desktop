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

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 复刻 Android 端（ImageSegmentation.kt + TensorImage）的输入预处理：
 *  1. BILINEAR 缩放到模型输入尺寸 256x256（Android 用 ResizeOp BILINEAR）；
 *  2. (x - 127.5) / 127.5 归一化（Android 用 NormalizeOp(127.5f, 127.5f)）；
 *  3. 通道序为 RGB（Android TensorImage 加载 Bitmap 即 RGB），布局 NHWC float32。
 */
object ImagePreprocessor {

    const val MODEL_SIZE = 256

    /**
     * @param bgr OpenCV BGR 图像（与 Android 端 decodeJpeg 输出一致）。
     * @return 长度 [256*256*3] 的 float32 数组，NHWC、RGB、归一化到约 [-1, 1]。
     */
    fun preprocess(bgr: Mat): FloatArray {
        val rgb = Mat()
        val resized = Mat()
        try {
            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
            Imgproc.resize(
                rgb,
                resized,
                Size(MODEL_SIZE.toDouble(), MODEL_SIZE.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR,
            )

            val pixelCount = MODEL_SIZE * MODEL_SIZE * 3
            val data = ByteArray(pixelCount)
            resized.get(0, 0, data)

            val result = FloatArray(pixelCount)
            for (i in 0 until pixelCount) {
                val v = data[i].toInt() and 0xFF
                result[i] = (v - 127.5f) / 127.5f
            }
            return result
        } finally {
            rgb.release()
            resized.release()
        }
    }
}
