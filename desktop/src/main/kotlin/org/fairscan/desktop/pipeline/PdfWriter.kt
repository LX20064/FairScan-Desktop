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

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.pdmodel.PDPageContentStream.AppendMode
import org.apache.pdfbox.pdmodel.common.PDRectangle
import org.apache.pdfbox.pdmodel.graphics.image.JPEGFactory
import org.fairscan.imageprocessing.EstimatedDimensions
import org.fairscan.imageprocessing.OcrTextBox
import org.fairscan.imageprocessing.PaperFormats
import java.io.OutputStream
import java.util.Calendar

data class PdfPageInput(
    val jpegBytes: ByteArray,
    val estimatedDimensions: EstimatedDimensions?,
    /** 该页 OCR 词条（图像像素坐标），非空时追加不可见文本层。 */
    val ocrTextBoxes: List<OcrTextBox> = emptyList(),
)

/**
 * 标准 Apache PDFBox 生成 PDF。
 * 页面尺寸逻辑与 Android 端 AndroidPdfWriter 保持一致：
 * 物理尺寸可用时直接换算，否则按 A4 高度约束整图比例。
 * OCR 词条以不可见文本层写入（可选、可搜索）。
 */
object PdfWriter {
    private const val POINTS_PER_MM = 72f / 25.4f

    fun writePdf(pages: List<PdfPageInput>, outputStream: OutputStream) {
        val doc = PDDocument()
        doc.use {
            doc.documentInformation.creator = "FairScan Desktop"
            doc.documentInformation.creationDate = Calendar.getInstance()

            val ocrLayer = OcrPdfTextLayer(doc)

            for (page in pages) {
                val image = JPEGFactory.createFromByteArray(doc, page.jpegBytes)
                val widthPx = image.width.toFloat()
                val heightPx = image.height.toFloat()

                val (widthMm, heightMm) = pageMm(page.estimatedDimensions, widthPx, heightPx)
                val pageWidthPoints = (widthMm * POINTS_PER_MM).toFloat()
                val pageHeightPoints = (heightMm * POINTS_PER_MM).toFloat()

                val pdPage = PDPage(PDRectangle(pageWidthPoints, pageHeightPoints))
                doc.addPage(pdPage)

                val contentStream = PDPageContentStream(doc, pdPage, AppendMode.OVERWRITE, false)
                try {
                    contentStream.drawImage(
                        image,
                        0f,
                        0f,
                        pageWidthPoints,
                        pageHeightPoints,
                    )
                } finally {
                    contentStream.close()
                }

                if (page.ocrTextBoxes.isNotEmpty()) {
                    ocrLayer.addPage(
                        page = pdPage,
                        ocrTextBoxes = page.ocrTextBoxes,
                        imageWidth = image.width,
                        imageHeight = image.height,
                        pageWidth = pageWidthPoints,
                        pageHeight = pageHeightPoints,
                    )
                }
            }
            doc.save(outputStream)
        }
    }

    private fun pageMm(
        estimatedDimensions: EstimatedDimensions?,
        widthPx: Float,
        heightPx: Float,
    ): Pair<Double, Double> {
        val (widthMm, heightMm) = when (estimatedDimensions) {
            is EstimatedDimensions.Physical ->
                estimatedDimensions.widthMm to estimatedDimensions.heightMm
            else -> {
                // 无物理尺寸：按 A4 高度约束整图比例（与 Android 一致）
                val maxDimMm = PaperFormats.A4.heightMm
                val scalePxToMm = maxDimMm / maxOf(widthPx, heightPx)
                (widthPx * scalePxToMm).toDouble() to (heightPx * scalePxToMm).toDouble()
            }
        }
        return constrainToMaxFormat(widthMm, heightMm)
    }

    /** 复刻 Android AndroidPdfWriter.constrainToMaxFormat。 */
    private fun constrainToMaxFormat(widthMm: Double, heightMm: Double): Pair<Double, Double> {
        val maxDim = 297.0   // A4 height
        val minDim = 215.9   // Letter width
        val scale = minOf(
            maxDim / maxOf(widthMm, heightMm),
            minDim / minOf(widthMm, heightMm),
            1.0,
        )
        return widthMm * scale to heightMm * scale
    }
}
