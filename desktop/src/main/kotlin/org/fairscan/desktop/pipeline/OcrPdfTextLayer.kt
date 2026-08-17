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

import org.apache.pdfbox.cos.COSArray
import org.apache.pdfbox.cos.COSDictionary
import org.apache.pdfbox.cos.COSName
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDResources
import org.apache.pdfbox.pdmodel.common.PDRectangle
import org.apache.pdfbox.pdmodel.common.PDStream
import org.apache.pdfbox.pdmodel.font.PDFontDescriptor
import org.fairscan.imageprocessing.OcrTextBox
import java.util.Locale

/**
 * 向 PDF 页面追加不可见 OCR 文本层。
 *
 * 移植自 Android AndroidPdfWriter.OcrDocument（源于 tesseract-ocr/tesseract 的
 * src/api/pdfrenderer.cpp）：嵌入 GlyphLessFont 字体 + Identity-H 编码 + ToUnicode CMap，
 * 用 3 Tr（不可见）渲染模式写出每个单词，使 PDF 文本可选/可搜索。
 */
class OcrPdfTextLayer(private val document: PDDocument) {

    private val fontBytes: ByteArray by lazy {
        checkNotNull(javaClass.getResourceAsStream("/fonts/TesseractGlyphLessFont.ttf")) {
            "缺少字体资源 /fonts/TesseractGlyphLessFont.ttf"
        }.use { it.readBytes() }
    }

    private val cidToGidMap: ByteArray by lazy {
        ByteArray(65536 * 2) { i -> if (i % 2 == 0) 0x00.toByte() else 0x01.toByte() }
    }

    private var fontDict: COSDictionary? = null

    private fun ensureResourcesCreated() {
        if (fontDict != null) return

        // -- Object 1 : embedded TTF --
        val fontFileStream = PDStream(document)
        fontFileStream.createOutputStream().use { it.write(fontBytes) }
        fontFileStream.cosObject.setInt(COSName.LENGTH1, fontBytes.size)

        // -- Object 2 : CIDToGIDMap stream --
        val cidToGidStream = PDStream(document)
        cidToGidStream.createOutputStream(COSName.FLATE_DECODE).use { it.write(cidToGidMap) }

        // -- Object 3 : FontDescriptor --
        val fontDescriptor = PDFontDescriptor(COSDictionary())
        fontDescriptor.fontName = "GlyphLessFont"
        fontDescriptor.flags = 5
        fontDescriptor.fontBoundingBox = PDRectangle(0f, 0f, 500f, 750f)
        fontDescriptor.italicAngle = 0f
        fontDescriptor.ascent = 750f
        fontDescriptor.descent = 0f
        fontDescriptor.capHeight = 750f
        fontDescriptor.stemV = 80f
        fontDescriptor.cosObject.setItem(COSName.FONT_FILE2, fontFileStream)

        // -- Object 4 : CIDFont descendant --
        val cidFont = COSDictionary()
        cidFont.setName(COSName.TYPE, "Font")
        cidFont.setName(COSName.SUBTYPE, "CIDFontType2")
        cidFont.setName(COSName.BASE_FONT, "GlyphLessFont")
        val cidSystemInfo = COSDictionary()
        cidSystemInfo.setString(COSName.getPDFName("Registry"), "Adobe")
        cidSystemInfo.setString(COSName.getPDFName("Ordering"), "Identity")
        cidSystemInfo.setInt(COSName.getPDFName("Supplement"), 0)
        cidFont.setItem(COSName.getPDFName("CIDSystemInfo"), cidSystemInfo)
        cidFont.setItem(COSName.FONT_DESC, fontDescriptor)
        cidFont.setInt(COSName.getPDFName("DW"), 500)
        cidFont.setItem(COSName.getPDFName("CIDToGIDMap"), cidToGidStream)

        // -- Object 5 : ToUnicode CMap --
        val toUnicodeStream = PDStream(document)
        toUnicodeStream.createOutputStream().use {
            it.write(buildToUnicodeCMap().toByteArray(Charsets.US_ASCII))
        }

        // -- Object 6 : Font Type0 --
        val fontDict = COSDictionary()
        fontDict.setName(COSName.TYPE, "Font")
        fontDict.setName(COSName.SUBTYPE, "Type0")
        fontDict.setName(COSName.BASE_FONT, "GlyphLessFont")
        fontDict.setName(COSName.ENCODING, "Identity-H")
        val descendants = COSArray()
        descendants.add(cidFont)
        fontDict.setItem(COSName.DESCENDANT_FONTS, descendants)
        fontDict.setItem(COSName.TO_UNICODE, toUnicodeStream)
        this.fontDict = fontDict
    }

    private fun buildToUnicodeCMap(): String = buildString {
        append("/CIDInit /ProcSet findresource begin\n")
        append("12 dict begin\n")
        append("begincmap\n")
        append("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n")
        append("/CMapName /Adobe-Identify-UCS def\n")
        append("/CMapType 2 def\n")
        append("1 begincodespacerange\n")
        append("<0000> <FFFF>\n")
        append("endcodespacerange\n")
        append("1 beginbfrange\n")
        append("<0000> <FFFF> <0000>\n")
        append("endbfrange\n")
        append("endcmap\n")
        append("CMapName currentdict /CMap defineresource pop\n")
        append("end\nend\n")
    }

    /**
     * @param page 目标页（图像已绘入）。
     * @param ocrTextBoxes 该页 OCR 词条（图像像素坐标系，y 向下）。
     * @param imageWidth 图像像素宽（即 OCR 时图像宽度）。
     * @param imageHeight 图像像素高。
     * @param pageWidth 页面宽度（PDF 点）。
     * @param pageHeight 页面高度（PDF 点）。
     */
    fun addPage(
        page: PDPage,
        ocrTextBoxes: List<OcrTextBox>,
        imageWidth: Int,
        imageHeight: Int,
        pageWidth: Float,
        pageHeight: Float,
    ) {
        if (ocrTextBoxes.isEmpty()) return

        ensureResourcesCreated()
        val resources = page.resources ?: PDResources().also { page.resources = it }
        val fontResources = resources.cosObject
            .getDictionaryObject(COSName.FONT) as? COSDictionary
            ?: COSDictionary().also { resources.cosObject.setItem(COSName.FONT, it) }
        fontResources.setItem(COSName.getPDFName("F1"), fontDict)

        val textStream = buildTextStream(ocrTextBoxes, imageWidth, imageHeight, pageWidth, pageHeight)
        val pdTextStream = PDStream(document)
        pdTextStream.createOutputStream(COSName.FLATE_DECODE).use {
            it.write(textStream.toByteArray(Charsets.US_ASCII))
        }

        val contentsArray = COSArray()
        val iter = page.contentStreams
        while (iter.hasNext()) contentsArray.add(iter.next().cosObject)
        contentsArray.add(pdTextStream.cosObject)
        page.cosObject.setItem(COSName.CONTENTS, contentsArray)
    }

    private fun buildTextStream(
        ocrTextBoxes: List<OcrTextBox>,
        imageWidth: Int,
        imageHeight: Int,
        pageWidth: Float,
        pageHeight: Float,
    ): String {
        val scaleX = pageWidth / imageWidth
        val scaleY = pageHeight / imageHeight
        val sb = StringBuilder()

        for (textBox in ocrTextBoxes) {
            val x = textBox.box.left * scaleX
            val wordWidth = textBox.box.width * scaleX
            val fontSize = textBox.lineHeight * scaleY * 0.8f
            // nominal width: fontSize / kCharWidth (Tesseract convention)
            val nominalWidth = textBox.text.length * fontSize / 2f
            val hScale = if (nominalWidth > 0f) (wordWidth / nominalWidth) * 100f else 100f
            val baselineY = pageHeight - (textBox.lineBottom * scaleY) + fontSize * 0.2f

            val utf16hex = buildString {
                textBox.text.codePoints().forEach { cp -> append(codepointToUtf16beHex(cp)) }
            }

            sb.append("BT\n")
            sb.append(String.format(Locale.US, "/F1 %.3f Tf\n", fontSize))
            sb.append(String.format(Locale.US, "%.3f Tz\n", hScale))
            sb.append("3 Tr\n")
            sb.append(String.format(Locale.US, "%.3f %.3f Td\n", x, baselineY))
            sb.append("<${utf16hex}0020> Tj\n")
            sb.append("ET\n")
        }
        return sb.toString()
    }

    private fun codepointToUtf16beHex(cp: Int): String {
        return if (cp < 0x10000) {
            "%04X".format(cp)
        } else {
            val a = cp - 0x10000
            val high = (a shr 10) + 0xD800
            val low = (a and 0x3FF) + 0xDC00
            "%04X%04X".format(high, low)
        }
    }
}
