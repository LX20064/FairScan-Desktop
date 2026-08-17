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

import org.fairscan.imageprocessing.ImageSize
import org.fairscan.imageprocessing.Mask
import org.opencv.core.CvType
import org.opencv.core.Mat

/**
 * 分割结果，实现了 imageprocessing 的 [Mask] 接口。
 * 二值化行为与 Android 端 Segmentation.toMat() 完全一致（阈值 0.5）。
 */
class SegmentationMask(
    private val probmap: FloatArray,
    override val width: Int,
    override val height: Int,
) : Mask {

    fun get(x: Int, y: Int): Float = probmap[y * width + x]

    fun toFloatArray(): FloatArray = probmap

    override fun toMat(): Mat {
        val mask = Mat(height, width, CvType.CV_8UC1)
        val data = ByteArray(width * height)
        for (i in probmap.indices) {
            data[i] = if (probmap[i] >= 0.5f) 255.toByte() else 0.toByte()
        }
        mask.put(0, 0, data)
        return mask
    }

    fun maskSize(): ImageSize = ImageSize(width, height)
}
