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
package org.fairscan.imageprocessing

import org.fairscan.imageprocessing.quad.findQuadFromContourOrientation
import org.fairscan.imageprocessing.quad.minAreaRect
import org.fairscan.imageprocessing.quad.scoreQuadAgainstProbmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

interface Mask {
    val width: Int
    val height: Int
    fun toMat(): Mat
}

enum class Mode {
    CAPTURE, IMPORT, LIVE_ANALYSIS
}

fun detectDocumentQuad(mask: Mask, originalSize: ImageSize, mode: Mode): Quad? {
    val quads = detectDocumentQuads(mask, originalSize, mode, maxDocs = 1)
    if (quads.isNotEmpty()) return quads.first()

    // Fallback（仅单文档/拍照模式）：bounding rectangle
    if (mode == Mode.CAPTURE) {
        val mat = mask.toMat()
        val biggest = biggestContour(mat)
        mat.release()
        if (biggest != null) {
            val polygon = biggest.toList().map { Point(it.x, it.y) }
            val vertices = minAreaRect(polygon, mask.width, mask.height)
            val maskSize = ImageSize(mask.width, mask.height)
            return if (vertices?.size == 4 && vertices.all { isInsideImage(it, maskSize) })
                createQuad(vertices)
            else null
        }
    }
    return null
}

/**
 * 多文档检测：返回 mask 坐标系下、按面积降序的多个四边形。
 * 对每个可靠连通域独立做边缘方向拟合；不同阈值下的重复候选会按评分去重。
 *
 * @param maxDocs 最多返回的文档数。
 */
fun detectDocumentQuads(
    mask: Mask,
    originalSize: ImageSize,
    mode: Mode,
    maxDocs: Int = 4,
): List<Quad> {
    val mat = mask.toMat()
    // Best thresholds on test dataset: {0.95=146, 0.85=39, 0.75=35, 0.90=8, 0.70=1, 0.35=1}
    val thresholds =
        if (mode != Mode.CAPTURE) listOf(0.9) else listOf(0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    val probFloat = Mat()
    // toMat() 返回 0/255 二值图，归一化到 [0,1] 作为概率供评分使用
    mat.convertTo(probFloat, CvType.CV_32F, 1.0 / 255.0)

    val probmapU8 = Mat()
    mat.convertTo(probmapU8, CvType.CV_8U, 255.0)
    val probmapSmooth = Mat()
    Imgproc.GaussianBlur(probmapU8, probmapSmooth, Size(3.0, 3.0), 0.0)
    probmapU8.release()

    // 候选 quad（mask 坐标，opencv Point）+ 评分
    val candidates = mutableListOf<Pair<List<org.opencv.core.Point>, Double>>()
    val minArea = mat.total() * 0.005 // 过滤过小区域（< 0.5% 图幅）

    for (thr in thresholds) {
        val bin = Mat()
        Imgproc.threshold(probmapSmooth, bin, thr * 255.0, 255.0, Imgproc.THRESH_BINARY)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        hierarchy.release()
        for (contour in contours) {
            val area = abs(Imgproc.contourArea(contour))
            if (area < minArea) {
                contour.release()
                continue
            }
            val quad = fitQuadFromContour(contour, mat.size(), originalSize)
            if (quad != null) {
                val score = scoreQuadAgainstProbmap(quad, probFloat, minQuadAreaRatio = 0.02)
                if (score > 0.0) candidates += quad to score
            }
            contour.release()
        }
        bin.release()
    }
    probmapSmooth.release()
    probFloat.release()
    mat.release()

    val maskSize = ImageSize(mask.width, mask.height)
    val selected = mutableListOf<Quad>()
    for ((corners, _) in candidates.sortedByDescending { it.second }) {
        if (corners.size != 4 || !corners.all { isInsideImage(Point(it.x, it.y), maskSize) }) continue
        val quad = createQuad(corners.map { Point(it.x, it.y) })
        if (selected.any { quadsOverlap(quad, it) }) continue
        selected += quad
        if (selected.size >= maxDocs) break
    }
    return selected
}

/** 两个四边形中心距离小于任一对角线的一半时视为同一文档（跨阈值重复检测）。 */
private fun quadsOverlap(a: Quad, b: Quad): Boolean {
    val ptsA = listOf(a.topLeft, a.topRight, a.bottomRight, a.bottomLeft)
    val ptsB = listOf(b.topLeft, b.topRight, b.bottomRight, b.bottomLeft)
    val cxA = ptsA.map { it.x }.average()
    val cyA = ptsA.map { it.y }.average()
    val cxB = ptsB.map { it.x }.average()
    val cyB = ptsB.map { it.y }.average()
    val dist = hypot(cxB - cxA, cyB - cyA)
    val diagA = norm(a.topLeft, a.bottomRight)
    val diagB = norm(b.topLeft, b.bottomRight)
    return dist < minOf(diagA, diagB) / 2.0
}

fun findQuadFromOrientationWithAdaptiveThreshold(
    maskMat: Mat, originalSize: ImageSize, thresholds: List<Double>
): List<org.opencv.core.Point>? {
    val probmapU8 = Mat()
    val probmap = maskMat
    probmap.convertTo(probmapU8, CvType.CV_8U, 255.0)
    val probmapSmooth = Mat()
    Imgproc.GaussianBlur(probmapU8, probmapSmooth, Size(3.0, 3.0), 0.0)

    var bestQuad: List<org.opencv.core.Point>? = null
    var bestScore = 0.0
    for (thr in thresholds) {
        val bin = Mat()
        Imgproc.threshold(probmapSmooth, bin, thr * 255.0, 255.0, Imgproc.THRESH_BINARY)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel)
        val quad = findQuadFromOrientation(bin, originalSize)
        if (quad != null) {
            val probFloat = Mat()
            probmap.convertTo(probFloat, CvType.CV_32F)
            val score = scoreQuadAgainstProbmap(quad, probFloat, minQuadAreaRatio = 0.02)
            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }
        bin.release()
    }

    probmapSmooth.release()
    probmapU8.release()
    return bestQuad
}

fun isInsideImage(p: Point, imageSize: ImageSize): Boolean {
    return p.x >= 0 && p.x <= imageSize.width
       && p.y >= 0 && p.y <= imageSize.height
}

fun findQuadFromOrientation(maskMat: Mat, originalSize: ImageSize): List<org.opencv.core.Point>? {
    val contour = biggestContour(maskMat) ?: return null
    return fitQuadFromContour(contour, maskMat.size(), originalSize)
}

/**
 * 对单个轮廓（mask 坐标）做边缘方向拟合。
 * mask 与源图宽高比可能不同，先缩放到源图比例拟合（角度依赖宽高比），再换算回 mask 坐标。
 */
private fun fitQuadFromContour(
    contour: MatOfPoint,
    maskSize: Size,
    originalSize: ImageSize,
): List<org.opencv.core.Point>? {
    val scaleX = originalSize.width / maskSize.width
    val scaleY = originalSize.height / maskSize.height
    return findQuadFromContourOrientation(
        contour.toList().map { org.opencv.core.Point(it.x * scaleX, it.y * scaleY) }
    )?.map { org.opencv.core.Point(it.x / scaleX, it.y / scaleY) }
}

fun biggestContour(mat: Mat): MatOfPoint? {
    val refinedMask = refineMask(mat)

    val blurred = Mat()
    Imgproc.GaussianBlur(refinedMask, blurred, Size(5.0, 5.0), 0.0)

    val edges = Mat()
    Imgproc.Canny(blurred, edges, 75.0, 200.0)

    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE)

    var biggest: MatOfPoint? = null
    var maxArea = 0.0

    for (contour in contours) {
        val area = abs(Imgproc.contourArea(contour))
        if (area > maxArea) {
            maxArea = area
            biggest = contour
        }
    }
    return biggest
}

/**
 * Applies morphological operations to improve a document mask.
 */
fun refineMask(original: Mat): Mat {
    // Step 0: Ensure the mask is binary (just in case)
    val binaryMask = Mat()
    Imgproc.threshold(original, binaryMask, 128.0, 255.0, Imgproc.THRESH_BINARY)

    // Step 1: Closing (fills small holes)
    val kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
    val closed = Mat()
    Imgproc.morphologyEx(binaryMask, closed, Imgproc.MORPH_CLOSE, kernelClose)

    // Step 2: Gentle opening (removes isolated noise)
    val kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
    val opened = Mat()
    Imgproc.morphologyEx(closed, opened, Imgproc.MORPH_OPEN, kernelOpen)

    return opened
}

fun extractDocument(
    inputMat: Mat,
    quad: Quad,
    rotationDegrees: Int,
    colorMode: ColorMode,
    maxPixels: Long,
    opticalMeasures: OpticalMeasures? = null,
    removeShadow: Boolean = false,
    /**
     * 是否在透视校正后立即增强。桌面端 UVDoc 展平时需传 false：
     * 展平依赖光照纹理线索，增强会破坏线索，应先展平再增强。
     */
    enhance: Boolean = true,
): Mat {
    val estimatedDimensions = estimateRealDimensions(
        quad,
        inputMat.cols(),
        inputMat.rows(),
        opticalMeasures,
    ).snapToStandardFormat()
    val (targetWidth, targetHeight) = estimatedDimensions.toPixelDimensions(quad)
    val srcPoints = MatOfPoint2f(
        quad.topLeft.toCv(),
        quad.topRight.toCv(),
        quad.bottomRight.toCv(),
        quad.bottomLeft.toCv(),
    )
    val dstPoints = MatOfPoint2f(
        org.opencv.core.Point(0.0, 0.0),
        org.opencv.core.Point(targetWidth, 0.0),
        org.opencv.core.Point(targetWidth, targetHeight),
        org.opencv.core.Point(0.0, targetHeight)
    )
    val transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)

    val warped = Mat()
    val outputSize = Size(targetWidth, targetHeight)
    Imgproc.warpPerspective(inputMat, warped, transform, outputSize)

    val resized = resizeForMaxPixels(warped, maxPixels.toDouble())
    val page = if (enhance) {
        val enhanced = enhanceCapturedImage(resized, colorMode, removeShadow)
        resized.release()
        enhanced
    } else {
        resized
    }
    val rotated = rotate(page, rotationDegrees)

    warped.release()
    page.release()

    return rotated
}

fun EstimatedDimensions.toPixelDimensions(quad: Quad): Pair<Double, Double> {
    val w = (norm(quad.topLeft, quad.topRight) + norm(quad.bottomLeft, quad.bottomRight)) / 2
    val h = (norm(quad.topLeft, quad.bottomLeft) + norm(quad.topRight, quad.bottomRight)) / 2
    val projectedArea = w * h

    val ratio = aspectRatio
    val targetWidth = sqrt(projectedArea / ratio)
    val targetHeight = targetWidth * ratio
    return Pair(targetWidth, targetHeight)
}

fun rotate(input: Mat, degrees: Int): Mat {
    val output = Mat()
    when ((degrees % 360 + 360) % 360) {
        0 -> input.copyTo(output)
        90 -> Core.rotate(input, output, Core.ROTATE_90_CLOCKWISE)
        180 -> Core.rotate(input, output, Core.ROTATE_180)
        270 -> Core.rotate(input, output, Core.ROTATE_90_COUNTERCLOCKWISE)
        else -> throw IllegalArgumentException("Only 0, 90, 180, 270 degrees are supported")
    }
    return output
}

fun Point.toCv(): org.opencv.core.Point {
    return org.opencv.core.Point(x, y)
}

fun Size.toImageSize(): ImageSize {
    return ImageSize(width, height)
}
