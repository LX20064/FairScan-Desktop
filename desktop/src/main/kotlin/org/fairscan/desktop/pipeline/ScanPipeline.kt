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
package org.fairscan.desktop.pipeline

import org.fairscan.desktop.segmentation.MaskSegmenter
import org.fairscan.desktop.segmentation.SegmentationMask
import org.fairscan.desktop.unwarp.UVDocService
import org.fairscan.imageprocessing.ColorMode
import org.fairscan.imageprocessing.EstimatedDimensions
import org.fairscan.imageprocessing.ImageSize
import org.fairscan.imageprocessing.Mode
import org.fairscan.imageprocessing.Quad
import org.fairscan.imageprocessing.autoColorMode
import org.fairscan.imageprocessing.detectDocumentQuad
import org.fairscan.imageprocessing.detectDocumentQuads
import org.fairscan.imageprocessing.enhanceCapturedImage
import org.fairscan.imageprocessing.estimateRealDimensions
import org.fairscan.imageprocessing.extractDocument
import org.fairscan.imageprocessing.resizeForMaxPixels
import org.fairscan.imageprocessing.rotate
import org.fairscan.imageprocessing.scaledTo
import org.opencv.core.Mat

/**
 * 输出质量参数，与 Android 端 ExportQuality 保持一致。
 */
enum class ExportQuality(val jpegQuality: Int, val maxPixels: Long) {
    LOW(jpegQuality = 60, maxPixels = 1_000_000),
    BALANCED(jpegQuality = 75, maxPixels = 2_000_000),
    HIGH(jpegQuality = 80, maxPixels = 4_000_000),
}

data class ScanResult(
    val mask: SegmentationMask,
    /** 检测到的四边形（在 256x256 mask 坐标系下）。 */
    val quadInMask: Quad?,
    /** 归一化到 [0,1] 的四边形，用于复用/调试。 */
    val normalizedQuad: Quad?,
    /** 实际使用的色彩模式。 */
    val colorMode: ColorMode,
    /** 自动判定的色彩模式。 */
    val autoColorMode: ColorMode,
    /** 裁剪+增强+旋转后的页面图（BGR）。调用方负责 release。 */
    val pageMat: Mat,
    /** 输出的预估物理尺寸（用于 PDF 页面大小）。 */
    val estimatedDimensions: EstimatedDimensions?,
    val inferenceTimeMs: Long,
    val totalTimeMs: Long,
)

/** 多文档模式下的一张文档页。调用方负责释放 [pageMat]。 */
data class ScanPage(
    val quadInMask: Quad,
    val normalizedQuad: Quad,
    val pageMat: Mat,
    val estimatedDimensions: EstimatedDimensions?,
    /** 实际使用的色彩模式。 */
    val colorMode: ColorMode,
    /** 自动判定的色彩模式。 */
    val autoColorMode: ColorMode,
)

/** 多文档扫描结果：整图 mask + 各文档页。 */
data class ScanMultiResult(
    val mask: SegmentationMask,
    val pages: List<ScanPage>,
    val inferenceTimeMs: Long,
    val totalTimeMs: Long,
)

/**
 * 桌面端扫描流水线：分割推理 -> 文档四边形检测 -> 透视校正/增强 -> 旋转。
 * 流程与 Android 端 ImageProcessor.extractDocumentFromBitmap 一一对应。
 */
class ScanPipeline(
    private val segmentation: MaskSegmenter,
) {
    fun scan(
        source: Mat,
        rotationDegrees: Int = 0,
        mode: Mode = Mode.CAPTURE,
        colorModeOverride: ColorMode? = null,
        exportQuality: ExportQuality = ExportQuality.BALANCED,
        onPreprocessedInput: ((FloatArray) -> Unit)? = null,
        /**
         * 手动指定文档四边形（源图像素坐标：TL,TR,BR,BL）。
         * 非 null 时跳过自动检测，用于 UI 四角拖拽后的重新处理。
         */
        quadOverride: Quad? = null,
        /** 去阴影（形态学背景估计 + 除法归一化）。 */
        removeShadow: Boolean = false,
        /** 文档展平（UVDoc）。非 null 时对裁剪后的页面图做展开。 */
        unwarpService: UVDocService? = null,
    ): ScanResult {
        val startMs = System.currentTimeMillis()

        // 1. 分割推理（预处理与 Android TensorImage 完全一致）
        val input = segmentation.preprocess(source)
        onPreprocessedInput?.invoke(input)
        val probmap = segmentation.run(input)
        val mask = SegmentationMask(probmap, segmentation.imageSize, segmentation.imageSize)
        val inferenceTimeMs = System.currentTimeMillis() - startMs

        // 2. 文档四边形检测（mask 坐标系；手动覆盖时从源图坐标换算回来）
        val sourceSize = ImageSize(source.width(), source.height())
        val quadInMask = if (quadOverride != null) {
            quadOverride.scaledTo(source.width(), source.height(), mask.width, mask.height)
        } else {
            detectDocumentQuad(mask, sourceSize, mode)
        }

        // 3. 透视校正 + 增强 + 旋转（复刻 Android processedImage/extractDocumentFromBitmap）
        var colorMode = ColorMode.COLOR
        var autoColorMode = colorMode
        var normalizedQuad: Quad? = null
        var estimatedDimensions: EstimatedDimensions? = null
        val page: Mat = if (quadInMask == null) {
            // 未检测到文档：整图缩放 + 旋转
            val resized = resizeForMaxPixels(source, exportQuality.maxPixels.toDouble())
            try {
                rotate(resized, rotationDegrees)
            } finally {
                resized.release()
            }
        } else {
            val quad = quadInMask.scaledTo(mask.width, mask.height, source.width(), source.height())
            normalizedQuad = quad.scaledTo(source.width(), source.height(), 1, 1)
            autoColorMode = autoColorMode(source, mask, quad)
            colorMode = colorModeOverride ?: autoColorMode
            estimatedDimensions = estimateRealDimensions(quad, source.width(), source.height(), null)
                .snapToStandardFormat()
            val extracted = extractDocument(source, quad, rotationDegrees, colorMode, exportQuality.maxPixels, null, removeShadow, enhance = unwarpService == null)
            if (unwarpService != null) {
                // 先展平后增强：UVDoc 依赖光照纹理线索估计网格，吃未增强图，展开后再增强/去阴影
                val unwrapped = unwarpService.unwarp(extracted)
                extracted.release()
                val enhanced = enhanceCapturedImage(unwrapped, colorMode, removeShadow)
                unwrapped.release()
                enhanced
            } else {
                extracted
            }
        }

        return ScanResult(
            mask = mask,
            quadInMask = quadInMask,
            normalizedQuad = normalizedQuad,
            colorMode = colorMode,
            autoColorMode = autoColorMode,
            pageMat = page,
            estimatedDimensions = estimatedDimensions,
            inferenceTimeMs = inferenceTimeMs,
            totalTimeMs = System.currentTimeMillis() - startMs,
        )
    }

    /**
     * 多文档扫描：分割推理 -> 对所有可靠连通域独立检测四边形 -> 逐个透视校正/增强/旋转。
     * 未检测到任何文档时返回空 pages（调用方可回退到 [scan] 的整图缩放逻辑）。
     */
    fun scanMulti(
        source: Mat,
        rotationDegrees: Int = 0,
        mode: Mode = Mode.CAPTURE,
        colorModeOverride: ColorMode? = null,
        exportQuality: ExportQuality = ExportQuality.BALANCED,
        maxDocs: Int = 4,
        /** 去阴影（形态学背景估计 + 除法归一化）。 */
        removeShadow: Boolean = false,
        /** 文档展平（UVDoc）。非 null 时对每个裁剪后的页面图做展开。 */
        unwarpService: UVDocService? = null,
    ): ScanMultiResult {
        val startMs = System.currentTimeMillis()

        val input = segmentation.preprocess(source)
        val probmap = segmentation.run(input)
        val mask = SegmentationMask(probmap, segmentation.imageSize, segmentation.imageSize)
        val inferenceTimeMs = System.currentTimeMillis() - startMs

        val sourceSize = ImageSize(source.width(), source.height())
        val quads = detectDocumentQuads(mask, sourceSize, mode, maxDocs)
        val pages = quads.map { q ->
            val quad = q.scaledTo(mask.width, mask.height, source.width(), source.height())
            val normalizedQuad = quad.scaledTo(source.width(), source.height(), 1, 1)
            val autoColor = autoColorMode(source, mask, quad)
            val cm = colorModeOverride ?: autoColor
            val dims = estimateRealDimensions(quad, source.width(), source.height(), null)
                .snapToStandardFormat()
            val extracted = extractDocument(source, quad, rotationDegrees, cm, exportQuality.maxPixels, null, removeShadow, enhance = unwarpService == null)
            val pageMat = if (unwarpService != null) {
                // 先展平后增强：UVDoc 依赖光照纹理线索估计网格，吃未增强图，展开后再增强/去阴影
                val unwrapped = unwarpService.unwarp(extracted)
                extracted.release()
                val enhanced = enhanceCapturedImage(unwrapped, cm, removeShadow)
                unwrapped.release()
                enhanced
            } else {
                extracted
            }
            ScanPage(q, normalizedQuad, pageMat, dims, cm, autoColor)
        }

        return ScanMultiResult(
            mask = mask,
            pages = pages,
            inferenceTimeMs = inferenceTimeMs,
            totalTimeMs = System.currentTimeMillis() - startMs,
        )
    }
}
