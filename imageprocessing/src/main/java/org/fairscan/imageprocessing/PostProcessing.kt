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

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

enum class ColorMode {
    COLOR,
    GRAYSCALE,
}

fun enhanceCapturedImage(img: Mat, colorMode: ColorMode, removeShadow: Boolean = false): Mat {
    val pre = if (removeShadow) removeShadows(img) else img
    val out = when (colorMode) {
        ColorMode.COLOR -> multiScaleRetinexOnL(pre)
        ColorMode.GRAYSCALE -> enhanceGrayscaleImage(pre)
    }
    if (removeShadow) pre.release()
    return out
}

/**
 * 形态学背景估计 + 除法归一化去阴影。
 * 对每个通道独立操作：膨胀估计局部背景 -> 高斯模糊平滑 -> 原图/背景*255。
 * 阴影区域背景值低，除法归一化将其抬回白底，同时保留内容相对对比。
 * 适用于低光/强投影拍摄的文档；对彩色与灰度图均有效。
 */
fun removeShadows(img: Mat): Mat {
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(25.0, 25.0))
    val background = Mat()
    Imgproc.morphologyEx(img, background, Imgproc.MORPH_DILATE, kernel)
    Imgproc.GaussianBlur(background, background, Size(51.0, 51.0), 0.0)

    val bgFloat = Mat()
    background.convertTo(bgFloat, CvType.CV_32F)
    Core.max(bgFloat, Scalar(1.0), bgFloat) // 防止除零

    val imgFloat = Mat()
    img.convertTo(imgFloat, CvType.CV_32F)

    val norm = Mat()
    Core.divide(imgFloat, bgFloat, norm, 255.0)
    Core.min(norm, Scalar(255.0), norm)

    val result = Mat()
    norm.convertTo(result, CvType.CV_8U)

    background.release()
    bgFloat.release()
    imgFloat.release()
    norm.release()
    return result
}

fun multiScaleRetinexOnL(bgr: Mat): Mat {

    // --- 1. BGR -> Lab ---
    val lab = Mat()
    Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)

    val labChannels = ArrayList<Mat>(3)
    Core.split(lab, labChannels)

    val l = labChannels[0] // CV_8U [0..255]

    // --- 2. Prepare L (float) ---
    val lFloat = Mat()
    l.convertTo(lFloat, CvType.CV_32F)
    Core.add(lFloat, Scalar(1.0), lFloat)

    val scaleFactor = 2.0
    val smallSize = Size(
        lFloat.cols() / scaleFactor,
        lFloat.rows() / scaleFactor
    )

    val lSmall = Mat()
    Imgproc.resize(lFloat, lSmall, smallSize, 0.0, 0.0, Imgproc.INTER_AREA)

    // --- 3. log(L) once ---
    val logLSmall = Mat()
    Core.log(lSmall, logLSmall)

    val maxDimSmall = max(smallSize.width, smallSize.height)
    val kernelSizes = listOf(
        maxDimSmall / 80.0,
        maxDimSmall / 10.0,
        maxDimSmall / 2.0,
    )

    val weight = 1.0 / kernelSizes.size
    val retinexSmall = Mat.zeros(lSmall.size(), CvType.CV_32F)

    val blurLog = Mat()
    val diff = Mat()

    for (ks in kernelSizes) {
        val k = ks.toInt().coerceAtLeast(3) or 1

        Imgproc.boxFilter(
            logLSmall,
            blurLog,
            -1,
            Size(k.toDouble(), k.toDouble())
        )

        Core.subtract(logLSmall, blurLog, diff)
        Core.addWeighted(retinexSmall, 1.0, diff, weight, 0.0, retinexSmall)
    }

    // --- 4. Normalize Retinex (relative [0..1]) ---
    val minMax = Core.minMaxLoc(retinexSmall)
    val retinexNormSmall = Mat()
    Core.subtract(retinexSmall, Scalar(minMax.minVal), retinexNormSmall)

    val range = minMax.maxVal - minMax.minVal
    if (range > 1e-6) {
        Core.multiply(retinexNormSmall, Scalar(1.0 / range), retinexNormSmall)
    }

    // --- Upscale Retinex back to full resolution ---
    val retinexNorm = Mat()
    Imgproc.resize(
        retinexNormSmall,
        retinexNorm,
        lFloat.size(),
        0.0,
        0.0,
        Imgproc.INTER_CUBIC
    )

    // --- 5. Re-center around original luminance ---
    val lOriginalFloat = Mat()
    l.convertTo(lOriginalFloat, CvType.CV_32F)

    val meanL = Core.mean(lOriginalFloat).`val`[0]
    val amplitude = 60.0

    val correctedL = Mat()
    Core.multiply(retinexNorm, Scalar(amplitude), correctedL)
    Core.add(correctedL, Scalar(meanL - amplitude / 2.0), correctedL)

    // --- 6. Blend with original L ---
    val alpha = 0.6
    Core.addWeighted(
        lOriginalFloat, 1.0 - alpha,
        correctedL, alpha,
        0.0,
        correctedL
    )

    // --- 7. Restore contrast ---
    val pLowOrig = percentileL(lOriginalFloat, 0.001)
    val pLow = percentileL(correctedL, 0.001)
    val pHigh = percentileL(correctedL, 0.995)

    val targetLow = min(pLow, pLowOrig)
    val targetHigh = 245.0
    val scale = (targetHigh - targetLow) / (pHigh - pLow + 1e-6)

    Core.subtract(correctedL, Scalar(pLow), correctedL)
    Core.multiply(correctedL, Scalar(scale), correctedL)
    Core.add(correctedL, Scalar(targetLow), correctedL)

    // --- 8. Clamp and write back ---
    Core.min(correctedL, Scalar(255.0), correctedL)
    Core.max(correctedL, Scalar(0.0), correctedL)

    correctedL.convertTo(labChannels[0], CvType.CV_8U)

    // --- 9. Lab -> BGR ---
    Core.merge(labChannels, lab)
    val result = Mat()
    Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR)

    // --- Cleanup ---
    lab.release()
    lFloat.release()
    lSmall.release()
    logLSmall.release()
    blurLog.release()
    diff.release()
    retinexSmall.release()
    retinexNormSmall.release()
    retinexNorm.release()
    lOriginalFloat.release()
    correctedL.release()
    labChannels.forEach { it.release() }

    return result
}

fun percentileL(l: Mat, p: Double): Double {
    val hist = Mat()
    Imgproc.calcHist(
        listOf(l),
        MatOfInt(0),
        Mat(),
        hist,
        MatOfInt(256),
        MatOfFloat(0f, 256f)
    )

    val total = l.total()
    var sum = 0.0
    for (i in 0 until 256) {
        sum += hist.get(i, 0)[0]
        if (sum / total >= p) {
            hist.release()
            return i.toDouble()
        }
    }
    hist.release()
    return 255.0
}

fun enhanceGrayscaleImage(img: Mat): Mat {

    // -- 1. Convert to grayscale --------
    val gray = Mat()
    when (img.channels()) {
        4    -> Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGRA2GRAY)
        3    -> Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        else -> img.copyTo(gray)
    }

    // -- 2. Multi-scale Retinex ---------
    val maxDim = max(gray.cols(), gray.rows()).toDouble()

    val imgFloat = Mat()
    gray.convertTo(imgFloat, CvType.CV_32F)
    Core.add(imgFloat, Scalar(1.0), imgFloat)

    val logImg = Mat()
    Core.log(imgFloat, logImg)

    val kernelSizes = listOf(maxDim / 6, maxDim / 50)
    val weight = 1.0 / kernelSizes.size
    val retinex = Mat.zeros(gray.size(), CvType.CV_32F)
    val blur = Mat()
    val logBlur = Mat()
    val diff = Mat()

    for (kernelSize in kernelSizes) {
        Imgproc.boxFilter(imgFloat, blur, -1, Size(kernelSize, kernelSize))
        Core.add(blur, Scalar(1.0), blur)
        Core.log(blur, logBlur)
        Core.subtract(logImg, logBlur, diff)
        val diffGray = Mat()
        if (diff.channels() > 1) {
            Imgproc.cvtColor(diff, diffGray, Imgproc.COLOR_BGRA2GRAY)
        } else {
            diff.copyTo(diffGray)
        }
        Core.addWeighted(retinex, 1.0, diffGray, weight, 0.0, retinex)
        diffGray.release()
    }

    // -- 3. exp() + p1/p99 normalization ---------
    // exp() compensates for the compression of bright tones caused by
    // the Retinex log-space computation, making annotations and light
    // gray areas more visible.
    val retinexExp = Mat()
    Core.exp(retinex, retinexExp)

    val flat = Mat()
    retinexExp.reshape(1, 1).copyTo(flat)
    val sorted = Mat()
    Core.sort(flat, sorted, Core.SORT_ASCENDING)
    val n = sorted.cols()
    val pLow  = sorted.get(0, (n * 0.004).toInt())[0]
    val pHigh = sorted.get(0, (n * 0.99).toInt())[0]
    flat.release(); sorted.release()

    val normalized = Mat()
    Core.subtract(retinexExp, Scalar(pLow), normalized)
    val scale = if (pHigh > pLow) 255.0 / (pHigh - pLow) else 1.0
    Core.multiply(normalized, Scalar(scale), normalized)
    Core.min(normalized, Scalar(255.0), normalized)
    Core.max(normalized, Scalar(0.0), normalized)
    retinexExp.release()

    val result8u = Mat()
    normalized.convertTo(result8u, CvType.CV_8U)
    normalized.release()

    // -- 4. Stretch toward white (conditional) --------
    // Find the histogram mode in [180..255] as an estimate of the background level.
    // Normal-light documents have a bright background (modeVal typically 240+),
    // so mapping that level to 255 is safe. Dark / shadowed documents have a gray
    // background (modeVal clearly below 255): stretching by 255/modeVal there
    // pushes the background together with the text towards full white, destroying
    // contrast. In that case keep the Retinex-normalized result from step 3.
    val hist = Mat()
    Imgproc.calcHist(listOf(result8u), MatOfInt(0), Mat(), hist,
        MatOfInt(256), MatOfFloat(0f, 256f))

    var modeVal = 220; var modeCount = 0.0
    for (i in 180 until 256) {
        val c = hist.get(i, 0)[0]
        if (c > modeCount) { modeCount = c; modeVal = i }
    }
    hist.release()

    val stretched8u = Mat()

    when {
        // Retinex 过曝（文档含大面积暗区把背景推到饱和）：回退到原始灰度归一化
        modeVal >= 254 -> {
            val grayF = Mat()
            gray.convertTo(grayF, CvType.CV_32F)
            val grayFlat = Mat()
            grayF.reshape(1, 1).copyTo(grayFlat)
            val graySorted = Mat()
            Core.sort(grayFlat, graySorted, Core.SORT_ASCENDING)
            val gN = graySorted.cols()
            val gLow  = graySorted.get(0, (gN * 0.01).toInt())[0]
            val gHigh = graySorted.get(0, (gN * 0.99).toInt())[0]
            grayFlat.release(); graySorted.release()
            Core.subtract(grayF, Scalar(gLow), grayF)
            Core.multiply(grayF, Scalar(255.0 / (gHigh - gLow + 1e-6)), grayF)
            Core.min(grayF, Scalar(255.0), grayF)
            Core.max(grayF, Scalar(0.0), grayF)
            grayF.convertTo(stretched8u, CvType.CV_8U)
            grayF.release()
        }
        // 其余情况（含低光/阴影文档）：把背景众数映射到 255 提亮。
        // 与上游原代码行为一致（对 modeVal < 240 的低光文档同样无条件拉伸）。
        else -> {
            val stretchedF = Mat()
            result8u.convertTo(stretchedF, CvType.CV_32F)
            Core.multiply(stretchedF, Scalar(255.0 / modeVal), stretchedF)
            Core.min(stretchedF, Scalar(255.0), stretchedF)
            stretchedF.convertTo(stretched8u, CvType.CV_8U)
            stretchedF.release()
        }
    }

    // -- 5. Bilateral denoising ---------
    // Smooths background texture and fine grain amplified by exp() and stretch,
    // while preserving sharp edges (text, lines, annotations).
    val denoised = Mat()
    Imgproc.bilateralFilter(stretched8u, denoised, 9, 20.0, 10.0)

    val finalBgr = Mat()
    Imgproc.cvtColor(denoised, finalBgr, Imgproc.COLOR_GRAY2BGR)

    // -- Cleanup -----------
    gray.release(); imgFloat.release(); logImg.release()
    blur.release(); logBlur.release(); diff.release()
    retinex.release(); result8u.release()
    stretched8u.release(); denoised.release()

    return finalBgr
}
