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
package org.fairscan.desktop.ocr

import net.sourceforge.tess4j.ITessAPI
import net.sourceforge.tess4j.TessAPI
import org.fairscan.imageprocessing.ImageRect
import org.fairscan.imageprocessing.OcrTextBox
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.IntBuffer

/**
 * 桌面端 OCR 服务，行为与 Android 端 OcrService（TessBaseAPI ResultIterator）一致：
 * 遍历 RIL_WORD，取单词文本 + 单词 bbox + 行 bbox（lineHeight/lineBottom），置信度 > 50 才保留。
 *
 * 使用 Tess4J 暴露的低层 Tesseract C API（JNA）。
 * 输入图像先转灰度（Tesseract 内部自行二值化）。
 */
class OcrService(
    /** 含 *.traineddata 的目录（Tesseract 约定 datapath 的 tessdata 子目录）。 */
    tessdataDir: File,
    /** 语言串，如 "eng"、"chi_sim"、"eng+chi_sim"。 */
    private val language: String,
) : AutoCloseable {

    private val api: TessAPI = TessAPI.INSTANCE
    private val handle: ITessAPI.TessBaseAPI = api.TessBaseAPICreate()

    init {
        val rc = api.TessBaseAPIInit3(handle, tessdataDir.absolutePath, language)
        check(rc == 0) {
            "Tesseract init failed (rc=$rc): datapath=${tessdataDir.absolutePath}, lang=$language"
        }
    }

    /**
     * @param page 已裁剪/增强的页面图（BGR）。
     * @return OCR 词条列表（按 Tesseract 行序）。
     */
    fun runOcr(page: Mat): List<OcrTextBox> {
        val gray = Mat()
        try {
            Imgproc.cvtColor(page, gray, Imgproc.COLOR_BGR2GRAY)
            val width = gray.width()
            val height = gray.height()
            val pixels = ByteArray(width * height)
            gray.get(0, 0, pixels)

            // JNA 需要 direct buffer；Tesseract SetImage 不拷贝，需保持存活到迭代结束
            val buffer = ByteBuffer.allocateDirect(pixels.size)
            buffer.put(pixels)
            buffer.rewind()

            api.TessBaseAPISetImage(handle, buffer, width, height, 1, width)

            // 触发识别（结果文本此处不需要，仅驱动迭代器）
            val fullText = api.TessBaseAPIGetUTF8Text(handle)
            if (fullText != null) api.TessDeleteText(fullText)

            val resultIterator = api.TessBaseAPIGetIterator(handle) ?: return emptyList()
            val pageIterator = api.TessResultIteratorGetPageIterator(resultIterator)
            api.TessPageIteratorBegin(pageIterator)

            val wordLevel = ITessAPI.TessPageIteratorLevel.RIL_WORD
            val lineLevel = ITessAPI.TessPageIteratorLevel.RIL_TEXTLINE

            // 各输出参数须使用独立 buffer（JNA 从各自 position 写入）
            val wLeft = IntBuffer.allocate(1)
            val wTop = IntBuffer.allocate(1)
            val wRight = IntBuffer.allocate(1)
            val wBottom = IntBuffer.allocate(1)
            val lLeft = IntBuffer.allocate(1)
            val lTop = IntBuffer.allocate(1)
            val lRight = IntBuffer.allocate(1)
            val lBottom = IntBuffer.allocate(1)
            val result = mutableListOf<OcrTextBox>()

            do {
                val wordPtr = api.TessResultIteratorGetUTF8Text(resultIterator, wordLevel)
                if (wordPtr == null) continue // 空词条

                val confidence = api.TessResultIteratorConfidence(resultIterator, wordLevel)
                if (confidence <= 50f) {
                    api.TessDeleteText(wordPtr)
                    continue
                }

                val wordText = wordPtr.getString(0, "UTF-8")
                api.TessDeleteText(wordPtr)

                api.TessPageIteratorBoundingBox(
                    pageIterator, wordLevel,
                    wLeft, wTop, wRight, wBottom,
                )
                val rect = ImageRect(wLeft[0], wTop[0], wRight[0], wBottom[0])

                api.TessPageIteratorBoundingBox(
                    pageIterator, lineLevel,
                    lLeft, lTop, lRight, lBottom,
                )
                val lineHeight = lBottom[0] - lTop[0]
                val lineBottom = lBottom[0]

                result.add(OcrTextBox(wordText, rect, lineHeight, lineBottom))
            } while (api.TessPageIteratorNext(pageIterator, wordLevel) == 1)

            api.TessResultIteratorDelete(resultIterator)
            return result
        } finally {
            gray.release()
        }
    }

    override fun close() {
        api.TessBaseAPIDelete(handle)
    }
}
