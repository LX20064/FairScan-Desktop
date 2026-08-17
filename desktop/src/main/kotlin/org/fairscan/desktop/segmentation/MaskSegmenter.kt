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

/**
 * 分割服务统一接口，TFLite（fairscan-segmentation-model）与 ONNX（u2netp 等）双实现。
 *
 * 预处理与推理布局由各实现自行决定（TFLite 为 NHWC 256、[-1,1] 归一化；
 * ONNX 为 NCHW 320、ImageNet 归一化），流水线只依赖 [imageSize] 与两个方法。
 */
interface MaskSegmenter : AutoCloseable {

    /** 模型输入/掩膜边长（方形输入），如 256（TFLite）或 320（ONNX）。 */
    val imageSize: Int

    /** BGR 原图 → 模型输入张量（float32 一维展开，布局由实现决定）。 */
    fun preprocess(bgr: Mat): FloatArray

    /** 推理，返回 [imageSize]*[imageSize] 的概率图，取值 [0,1]。 */
    fun run(input: FloatArray): FloatArray
}

/**
 * 按模型文件扩展名选择分割实现：
 *  - `.onnx`  → [OnnxSegmentationService]（u2netp 等通用分割模型）
 *  - 其他     → [SegmentationService]（fairscan TFLite 模型，与 Android 端一致）
 */
fun createMaskSegmenter(modelPath: String, threads: Int = 2): MaskSegmenter {
    return if (modelPath.endsWith(".onnx", ignoreCase = true)) {
        OnnxSegmentationService(modelPath, threads)
    } else {
        SegmentationService(modelPath, threads)
    }
}
