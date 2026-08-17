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
package org.fairscan.desktop

import java.io.FileDescriptor
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import org.fairscan.desktop.ocr.OcrService
import org.fairscan.desktop.orientation.OrientationService
import org.fairscan.desktop.pipeline.ExportQuality
import org.fairscan.desktop.pipeline.PdfPageInput
import org.fairscan.desktop.pipeline.PdfWriter
import org.fairscan.desktop.pipeline.ScanPipeline
import org.fairscan.desktop.segmentation.SegmentationMask
import org.fairscan.desktop.segmentation.createMaskSegmenter
import org.fairscan.desktop.unwarp.UVDocService
import org.fairscan.imageprocessing.ColorMode
import org.fairscan.imageprocessing.EstimatedDimensions
import org.fairscan.imageprocessing.ImageRect
import org.fairscan.imageprocessing.ImageSize
import org.fairscan.imageprocessing.Mode
import org.fairscan.imageprocessing.OcrTextBox
import org.fairscan.imageprocessing.Quad
import org.fairscan.imageprocessing.decodeJpeg
import org.fairscan.imageprocessing.detectDocumentQuad
import org.fairscan.imageprocessing.encodeJpeg
import org.fairscan.imageprocessing.scaledTo
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.system.exitProcess

const val VERSION = "0.1.0"

private val USAGE = """
    FairScan Desktop CLI v$VERSION
    用法:
      fairscan-desktop scan <input>... [选项]      批量扫描
      fairscan-desktop detect <input> [选项]       仅文档检测（返回四边形，供摄像头实时定位）
      fairscan-desktop pdf <page.jpg> [选项]       由页面图 + 词条 JSON 生成 PDF（OCR 文本编辑回写）

    输入支持（可同时指定多个，支持通配符与目录）:
      scan a.jpg b.jpg             多个文件
      scan "photos/*.jpg"          通配符（* ?）
      scan photos/                 目录下全部图片

    扫描选项:
      --model <path>          分割模型：.tflite（原版）或 .onnx（u2netp 等）
                               (默认: models/fairscan-segmentation-model.tflite)
      --out <dir>             输出目录 (默认: output)
      --rotation <0|90|180|270>  额外旋转角度 (默认: 0)
      --mode <CAPTURE|IMPORT>  检测模式 (默认: CAPTURE)
      --color <AUTO|COLOR|GRAYSCALE>  色彩模式 (默认: AUTO)
      --quality <LOW|BALANCED|HIGH>   输出质量 (默认: BALANCED)
      --no-image              不导出裁剪后的图片（仅导出 PDF）
      --multi-doc             多文档检测：一张图内多个文档分别扫描（默认关闭）
      --max-docs <n>          多文档模式最多检测的文档数 (默认: 4)
      --remove-shadow         去阴影（形态学背景估计 + 除法归一化）
      --auto-rotate           自动方向感知（PP-LCNet 方向分类，0/90/180/270）
      --orientation-model <path>  方向分类 ONNX 模型路径
      --dewarp                文档展平（UVDoc，处理弯曲/卷曲文档）
      --unwarp-model <path>   UVDoc 展平模型路径 (默认: models/uvdoc.onnx)
      --quad <8 floats>       手动指定文档四边形（源图像素坐标, 逗号分隔:
                              TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy）
      --pdf                   同时生成 PDF（默认开启 OCR 文本层）
      --no-ocr                禁用 OCR 文本层
      --lang <langs>          OCR 语言，+ 连接 (默认: eng+chi_sim)
      --tessdata <dir>        OCR 训练数据目录 (默认: tessdata)
      --threads <n>           TFLite 推理线程数 (默认: 2)
      --json                  结构化输出（每行一个 JSON 对象，供 UI/脚本消费）
      --dump-mask <tpl>       导出原始 float32 mask（验证用，尺寸随模型）
      --dump-input <tpl>      导出预处理后的输入张量（验证用，尺寸随模型）

    PDF 选项:
      --ocr-words <file>      词条 JSON（{"width":..,"height":..,"words":[{text,left,top,right,bottom,lineHeight,lineBottom}]}）
      --out <file>            输出 PDF 路径（pdf 子命令下为文件而非目录）
      --tessdata <dir>        （可选，pdf 子命令仅回写词条时无需 tesseract）

    detect 选项（子集）:
      --mode <CAPTURE|IMPORT>  检测模式 (默认: CAPTURE)
      --json                  结构化输出（每行一个 JSON 对象）
""".trimIndent()

private class Args {
    val input = mutableListOf<String>()
    var model: String = "models/fairscan-segmentation-model.tflite"
    var outDir: String = "output"
    var rotation: Int = 0
    var mode: Mode = Mode.CAPTURE
    var color: ColorMode? = null
    var quality: ExportQuality = ExportQuality.BALANCED
    var noImage: Boolean = false
    var multiDoc: Boolean = false
    var maxDocs: Int = 4
    var removeShadow: Boolean = false
    var autoRotate: Boolean = false
    var orientationModel: String? = null
    var dewarp: Boolean = false
    var unwarpModel: String = "models/uvdoc.onnx"
    var quad: DoubleArray? = null
    var pdf: Boolean = false
    var ocrEnabled: Boolean = true
    var lang: String = "eng+chi_sim"
    var tessdata: String = "tessdata"
    var json: Boolean = false
    var ocrWords: String? = null
    var dumpMask: String? = null
    var dumpInput: String? = null
    var threads: Int = 2
}

private fun parseArgs(raw: Array<String>): Args {
    val args = Args()
    var i = 0
    while (i < raw.size) {
        when (val a = raw[i]) {
            "scan", "detect", "pdf" -> {} // subcommand
            "--model" -> { args.model = raw[++i] }
            "--out" -> { args.outDir = raw[++i] }
            "--rotation" -> { args.rotation = raw[++i].toInt() }
            "--mode" -> { args.mode = Mode.valueOf(raw[++i].uppercase()) }
            "--color" -> {
                args.color = when (raw[++i].uppercase()) {
                    "AUTO" -> null
                    "COLOR" -> ColorMode.COLOR
                    "GRAYSCALE" -> ColorMode.GRAYSCALE
                    else -> throw IllegalArgumentException("invalid color mode")
                }
            }
            "--quality" -> { args.quality = ExportQuality.valueOf(raw[++i].uppercase()) }
            "--no-image" -> { args.noImage = true }
            "--multi-doc" -> { args.multiDoc = true }
            "--max-docs" -> { args.maxDocs = raw[++i].toInt() }
            "--remove-shadow" -> { args.removeShadow = true }
            "--auto-rotate" -> { args.autoRotate = true }
            "--orientation-model" -> { args.orientationModel = raw[++i] }
            "--dewarp" -> { args.dewarp = true }
            "--unwarp-model" -> { args.unwarpModel = raw[++i] }
            "--quad" -> {
                args.quad = raw[++i].split(",").map { it.trim().toDouble() }.toDoubleArray()
                require(args.quad!!.size == 8) { "--quad 需要 8 个坐标值 (TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy)" }
            }
            "--pdf" -> { args.pdf = true }
            "--no-ocr" -> { args.ocrEnabled = false }
            "--lang" -> { args.lang = raw[++i] }
            "--tessdata" -> { args.tessdata = raw[++i] }
            "--json" -> { args.json = true }
            "--ocr-words" -> { args.ocrWords = raw[++i] }
            "--dump-mask" -> { args.dumpMask = raw[++i] }
            "--dump-input" -> { args.dumpInput = raw[++i] }
            "--threads" -> { args.threads = raw[++i].toInt() }
            "-h", "--help" -> { println(USAGE); exitProcess(0) }
            else -> {
                if (a.startsWith("-")) throw IllegalArgumentException("unknown option: $a")
                args.input.add(a)
            }
        }
        i++
    }
    return args
}

private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

/**
 * 展开输入：支持文件、目录、通配符（* ?），结果按字典序排序。
 */
private fun expandInputs(raw: List<String>): List<File> {
    val result = mutableListOf<File>()
    for (r in raw) {
        val f = File(r)
        when {
            f.isDirectory -> {
                val files = f.listFiles()
                    ?.filter { it.isFile && it.extension.lowercase() in IMAGE_EXTENSIONS }
                    ?.sortedBy { it.name.lowercase() }
                    ?: emptyList()
                if (files.isEmpty()) System.err.println("警告: 目录中没有图片: $f")
                result += files
            }
            f.name.any { it == '*' || it == '?' || it == '[' || it == ']' } -> {
                val parent = f.parentFile ?: File(".")
                val pattern = f.name
                val files = parent.listFiles()
                    ?.filter {
                        it.isFile && globMatch(pattern, it.name) &&
                            it.extension.lowercase() in IMAGE_EXTENSIONS
                    }
                    ?.sortedBy { it.name.lowercase() }
                    ?: emptyList()
                if (files.isEmpty()) System.err.println("警告: 没有匹配文件: $f")
                result += files
            }
            else -> result += f
        }
    }
    return result
}

/** 简易 glob 匹配，支持 *（任意串）与 ?（单字符）。 */
private fun globMatch(pattern: String, name: String): Boolean {
    var p = 0
    var n = 0
    var starP = -1
    var starN = 0
    while (n < name.length) {
        if (p < pattern.length && (pattern[p] == '?' || pattern[p] == name[n])) {
            p++
            n++
        } else if (p < pattern.length && pattern[p] == '*') {
            starP = p++
            starN = n
        } else if (starP >= 0) {
            p = starP + 1
            n = ++starN
        } else {
            return false
        }
    }
    while (p < pattern.length && pattern[p] == '*') p++
    return p == pattern.length
}

/**
 * 解析导出路径模板：
 * 含 {name} 时替换为输入文件名；否则自动加文件名前缀（batch 下避免覆盖）。
 */
private fun resolveDumpPath(template: String, base: String, outDir: File): File {
    if ("{name}" in template) {
        val expanded = File(template.replace("{name}", base))
        // 相对路径默认落到输出目录，避免散落在进程工作目录
        return if (expanded.parent == null) File(outDir, expanded.name) else expanded
    }
    val tpl = File(template)
    val name = tpl.name
    val newName = if (name.startsWith("$base-") || name.startsWith("$base.")) name else "$base-$name"
    return if (tpl.parent == null) File(outDir, newName) else File(tpl.parent, newName)
}

/** NDJSON 输出（--json 时 stdout 只写 JSON 行）。 */
private fun emitJson(obj: Any) {
    val json = GsonHolder.gson.toJson(obj)
    println(json)
    System.out.flush()
}

/** 避免引入额外依赖，用最小 JSON 序列化器。 */
private object GsonHolder {
    val gson: com.google.gson.Gson = com.google.gson.Gson()
}

fun main(rawArgs: Array<String>) {
    // openpnp:opencv 桌面版需显式加载原生库（Windows 下从 jar 提取 DLL）
    nu.pattern.OpenCV.loadLocally()

    // JDK 18+ 的 System.out/err 会跟随 Windows 控制台编码(GBK)输出，导致 --json 中文路径乱码；
    // 显式重定向为 UTF-8 字节流，保证 CLI 输出编码与 JVM 版本/平台无关（UI 侧始终按 UTF-8 解码）。
    val utf8Out = PrintStream(FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8)
    System.setOut(utf8Out)
    System.setErr(PrintStream(FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8))

    if (rawArgs.isEmpty() || rawArgs[0] !in setOf("scan", "detect", "pdf", "-h", "--help")) {
        println(USAGE)
        exitProcess(if (rawArgs.isEmpty()) 1 else 0)
    }
    val subcommand = rawArgs[0]

    val args = try {
        parseArgs(rawArgs)
    } catch (e: Exception) {
        System.err.println("参数错误: ${e.message}")
        println(USAGE)
        exitProcess(1)
    }

    when (subcommand) {
        "pdf" -> runPdfSubcommand(args)
        "detect" -> runDetectSubcommand(args)
        else -> runScanSubcommand(args)
    }
}

// ---------------------------------------------------------------------------
// detect 子命令：仅分割 + 文档四边形检测（不增强、不导出）
// 供摄像头实时预览自动定位文档使用；--json 输出归一化 quad 供 UI 叠加。
// ---------------------------------------------------------------------------

private fun runDetectSubcommand(args: Args) {
    if (args.input.isEmpty()) {
        System.err.println("缺少输入文件")
        println(USAGE)
        exitProcess(1)
    }

    val inputs = expandInputs(args.input)
    if (inputs.isEmpty()) {
        System.err.println("没有可处理的输入文件")
        exitProcess(1)
    }

    val modelFile = File(args.model)
    if (!modelFile.exists()) {
        System.err.println("模型文件不存在: $modelFile（可用 --model 指定）")
        exitProcess(1)
    }

    createMaskSegmenter(modelFile.absolutePath, args.threads).use { segmentation ->
        for (input in inputs) {
            val startMs = System.currentTimeMillis()
            val source = decodeJpeg(Files.readAllBytes(input.toPath()))
            try {
                    val preprocessed = segmentation.preprocess(source)
                    val probmap = segmentation.run(preprocessed)
                    val mask = SegmentationMask(
                        probmap,
                        segmentation.imageSize,
                        segmentation.imageSize,
                    )
                    val sourceSize = ImageSize(source.width(), source.height())
                    val quadInMask = detectDocumentQuad(mask, sourceSize, args.mode)
                    val normalizedQuad = quadInMask?.scaledTo(mask.width, mask.height, 1, 1)
                    val elapsed = System.currentTimeMillis() - startMs

                    if (args.json) {
                        emitJson(
                            mapOf(
                                "t" to "detect",
                                "file" to input.name,
                                "width" to source.width(),
                                "height" to source.height(),
                                "detected" to (normalizedQuad != null),
                                "quad" to normalizedQuad?.let { q ->
                                    listOf(
                                        q.topLeft.x, q.topLeft.y, q.topRight.x, q.topRight.y,
                                        q.bottomRight.x, q.bottomRight.y, q.bottomLeft.x, q.bottomLeft.y,
                                    )
                                },
                                "inferenceMs" to elapsed,
                            ),
                        )
                    } else {
                        if (normalizedQuad != null) {
                            println(
                                "${input.name}: 检测到文档（归一化） " +
                                    "TL(${"%.3f".format(normalizedQuad.topLeft.x)},${"%.3f".format(normalizedQuad.topLeft.y)}) " +
                                    "TR(${"%.3f".format(normalizedQuad.topRight.x)},${"%.3f".format(normalizedQuad.topRight.y)}) " +
                                    "BR(${"%.3f".format(normalizedQuad.bottomRight.x)},${"%.3f".format(normalizedQuad.bottomRight.y)}) " +
                                    "BL(${"%.3f".format(normalizedQuad.bottomLeft.x)},${"%.3f".format(normalizedQuad.bottomLeft.y)}) " +
                                    "${elapsed}ms"
                            )
                        } else {
                            println("${input.name}: 未检测到文档 (${elapsed}ms)")
                        }
                    }
                } finally {
                    source.release()
                }
            }
        }
}

// ---------------------------------------------------------------------------
// scan 子命令
// ---------------------------------------------------------------------------

private fun runScanSubcommand(args: Args) {
    if (args.input.isEmpty()) {
        System.err.println("缺少输入文件")
        println(USAGE)
        exitProcess(1)
    }

    val inputs = expandInputs(args.input)
    if (inputs.isEmpty()) {
        System.err.println("没有可处理的输入文件")
        exitProcess(1)
    }
    for (input in inputs) {
        if (!input.exists()) {
            System.err.println("输入文件不存在: $input")
            exitProcess(1)
        }
    }

    val modelFile = File(args.model)
    if (!modelFile.exists()) {
        System.err.println("模型文件不存在: $modelFile（可用 --model 指定）")
        exitProcess(1)
    }

    val outDir = File(args.outDir).apply { mkdirs() }

    if (!args.json) {
        println("== FairScan Desktop ==")
        println("模型    : ${modelFile.absolutePath}")
        println("输入    : ${inputs.size} 个文件")
        println("模式    : ${args.mode}  旋转=${args.rotation}°  质量=${args.quality}")
        if (args.quad != null) println("四边形  : 手动指定（跳过自动检测）")
        if (args.pdf) {
            println("PDF     : 开启  OCR=${if (args.ocrEnabled) args.lang else "禁用"}")
        }
        println()
    } else {
        emitJson(mapOf("t" to "config", "model" to modelFile.absolutePath, "total" to inputs.size))
    }

    // 方向感知服务（仅 --auto-rotate 时初始化；加载失败仅告警）
    val orientationService: OrientationService? = if (args.autoRotate) {
        val model = File(args.orientationModel ?: "models/doc_ori.onnx")
        if (!model.exists()) {
            System.err.println("警告: 方向分类模型不存在: $model，--auto-rotate 无效")
            null
        } else {
            try {
                OrientationService(model.absolutePath).also {
                    if (!args.json) println("方向感知 : ${model.absolutePath}")
                }
            } catch (e: Exception) {
                System.err.println("警告: 方向分类模型加载失败: ${e.message}")
                null
            }
        }
    } else null

    // 文档展平服务（仅 --dewarp 时初始化；加载失败仅告警）
    val unwarpService: UVDocService? = if (args.dewarp) {
        val model = File(args.unwarpModel)
        if (!model.exists()) {
            System.err.println("警告: UVDoc 展平模型不存在: $model，--dewarp 无效")
            null
        } else {
            try {
                UVDocService(model.absolutePath).also {
                    if (!args.json) println("文档展平 : ${model.absolutePath}")
                }
            } catch (e: Exception) {
                System.err.println("警告: UVDoc 展平模型加载失败: ${e.message}")
                null
            }
        }
    } else null

    try {
        createMaskSegmenter(modelFile.absolutePath, args.threads).use { segmentation ->
        val pipeline = ScanPipeline(segmentation)

        // OCR 服务（仅当生成 PDF 且未禁用 OCR 时初始化；失败仅告警，不影响主流程）
        val ocrService: OcrService? = if (args.pdf && args.ocrEnabled) {
            val tessdataDir = File(args.tessdata)
            if (!tessdataDir.exists()) {
                System.err.println("警告: tessdata 目录不存在: $tessdataDir，跳过 OCR")
                null
            } else {
                try {
                    OcrService(tessdataDir, args.lang).also {
                        if (!args.json) println("OCR     : ${args.lang} (${tessdataDir.absolutePath})")
                    }
                } catch (e: Exception) {
                    System.err.println("警告: OCR 初始化失败，跳过 OCR - ${e.message}")
                    null
                }
            }
        } else null

        try {
            var failed = 0
            var detected = 0
            var totalMs = 0L

            for ((index, inputFile) in inputs.withIndex()) {
                if (!args.json) {
                    println("[${index + 1}/${inputs.size}] ${inputFile.name}")
                } else {
                    emitJson(
                        mapOf(
                            "t" to "start", "index" to index + 1, "total" to inputs.size,
                            "file" to inputFile.name, "path" to inputFile.absolutePath,
                        ),
                    )
                }
                try {
                    val source = decodeJpeg(Files.readAllBytes(inputFile.toPath()))
                    try {
                        var preprocessedInput: FloatArray? = null
                    val quadOverride = args.quad?.let {
                        Quad(
                            topLeft = org.fairscan.imageprocessing.Point(it[0], it[1]),
                            topRight = org.fairscan.imageprocessing.Point(it[2], it[3]),
                            bottomRight = org.fairscan.imageprocessing.Point(it[4], it[5]),
                            bottomLeft = org.fairscan.imageprocessing.Point(it[6], it[7]),
                        )
                    }

                    // 自动方向感知：仅当未手动指定旋转角度时生效
                    val effectiveRotation = if (args.autoRotate && args.rotation == 0) {
                        val auto = orientationService?.detectRotation(source) ?: 0
                        if (auto != 0 && !args.json) println("方向感知 : 自动旋转 ${auto}°")
                        auto
                    } else args.rotation

                    // ---- 页面收集：多文档 或 单文档 ----
                    val pageOuts = mutableListOf<PageOutput>()
                    var segMask: SegmentationMask? = null
                    var detectedThisFile = 0
                    var scanTimeMs = 0L

                    if (args.multiDoc && quadOverride == null) {
                        val mr = pipeline.scanMulti(
                            source = source,
                            rotationDegrees = effectiveRotation,
                            mode = args.mode,
                            colorModeOverride = args.color,
                            exportQuality = args.quality,
                            maxDocs = args.maxDocs,
                            removeShadow = args.removeShadow,
                            unwarpService = unwarpService,
                        )
                        segMask = mr.mask
                        scanTimeMs = mr.totalTimeMs
                        if (mr.pages.isNotEmpty()) {
                            detectedThisFile = mr.pages.size
                            for (p in mr.pages) {
                                pageOuts += PageOutput(p.pageMat, p.quadInMask, p.normalizedQuad, p.estimatedDimensions, p.colorMode, p.autoColorMode)
                            }
                            if (!args.json) println("多文档   : 检测到 ${mr.pages.size} 个文档")
                        } else {
                            // 未检测到任何文档：回退单文档整图缩放
                            val fallback = pipeline.scan(
                                source = source,
                                rotationDegrees = effectiveRotation,
                                mode = args.mode,
                                colorModeOverride = args.color,
                                exportQuality = args.quality,
                                onPreprocessedInput = { input ->
                                    if (args.dumpInput != null) preprocessedInput = input
                                },
                                removeShadow = args.removeShadow,
                                unwarpService = unwarpService,
                            )
                            segMask = fallback.mask
                            pageOuts += PageOutput(fallback.pageMat, fallback.quadInMask, fallback.normalizedQuad, fallback.estimatedDimensions, fallback.colorMode, fallback.autoColorMode)
                            if (fallback.quadInMask != null) detectedThisFile = 1
                        }
                    } else {
                        val result = pipeline.scan(
                            source = source,
                            rotationDegrees = effectiveRotation,
                            mode = args.mode,
                            colorModeOverride = args.color,
                            exportQuality = args.quality,
                            onPreprocessedInput = { input ->
                                if (args.dumpInput != null) preprocessedInput = input
                            },
                            quadOverride = quadOverride,
                            removeShadow = args.removeShadow,
                            unwarpService = unwarpService,
                        )
                        segMask = result.mask
                        scanTimeMs = result.totalTimeMs
                        pageOuts += PageOutput(result.pageMat, result.quadInMask, result.normalizedQuad, result.estimatedDimensions, result.colorMode, result.autoColorMode)
                        if (result.quadInMask != null) detectedThisFile = 1
                    }

                    // OCR 需在 pageMat 释放前执行（pageMat 由 saveOutputs 释放）
                    for (page in pageOuts) {
                        page.ocrTextBoxes = if (ocrService != null && page.pageMat.total() > 0) {
                            val t0 = System.currentTimeMillis()
                            try {
                                val boxes = ocrService.runOcr(page.pageMat)
                                if (!args.json) {
                                    println("OCR     : ${boxes.size} 个词条, ${System.currentTimeMillis() - t0} ms")
                                } else {
                                    emitJson(
                                        mapOf(
                                            "t" to "ocr", "file" to inputFile.name,
                                            "width" to page.pageMat.width(),
                                            "height" to page.pageMat.height(),
                                            "words" to boxes.map { box ->
                                                mapOf(
                                                    "text" to box.text,
                                                    "left" to box.box.left,
                                                    "top" to box.box.top,
                                                    "right" to box.box.right,
                                                    "bottom" to box.box.bottom,
                                                    "lineHeight" to box.lineHeight,
                                                    "lineBottom" to box.lineBottom,
                                                )
                                            },
                                        ),
                                    )
                                }
                                boxes
                            } catch (e: Exception) {
                                System.err.println("警告: OCR 失败 - ${e.message}")
                                emptyList()
                            }
                        } else emptyList()
                    }

                    val outputs = saveOutputs(checkNotNull(segMask), pageOuts, args, outDir, preprocessedInput)
                    detected += detectedThisFile
                    totalMs += scanTimeMs

                    val first = pageOuts.firstOrNull()
                    if (args.json) {
                        emitJson(
                            mapOf(
                                "t" to "result",
                                "file" to inputFile.name,
                                "detected" to (detectedThisFile > 0),
                                "docCount" to detectedThisFile,
                                "quad" to first?.normalizedQuad?.let { q ->
                                    listOf(
                                        q.topLeft.x, q.topLeft.y, q.topRight.x, q.topRight.y,
                                        q.bottomRight.x, q.bottomRight.y, q.bottomLeft.x, q.bottomLeft.y,
                                    )
                                },
                                "colorMode" to (first?.colorMode?.name ?: "COLOR"),
                                "autoColorMode" to (first?.autoColorMode?.name ?: "COLOR"),
                                "inferenceMs" to 0,
                                "totalMs" to scanTimeMs,
                                "ocrWords" to pageOuts.sumOf { it.ocrTextBoxes.size },
                                "page" to outputs.pageJpeg?.absolutePath,
                                "pdf" to outputs.pdf?.absolutePath,
                            ),
                        )
                    } else {
                        printResult(inputFile, pageOuts, args)
                    }
                    } finally {
                        source.release()
                    }
                } catch (e: Exception) {
                    failed++
                    if (args.json) {
                        emitJson(mapOf("t" to "error", "file" to inputFile.name, "message" to (e.message ?: e.toString())))
                    } else {
                        System.err.println("处理失败: $inputFile - ${e.message}")
                    }
                }
                if (!args.json) println()
            }

            if (args.json) {
                emitJson(
                    mapOf(
                        "t" to "summary",
                        "total" to inputs.size,
                        "success" to (inputs.size - failed),
                        "failed" to failed,
                        "detected" to detected,
                        "totalMs" to totalMs,
                    ),
                )
            } else {
                println(
                    "== 批量完成: 成功=${inputs.size - failed} 失败=$failed " +
                        "检测到文档=$detected 平均耗时=${if (inputs.isEmpty()) 0 else totalMs / inputs.size}ms " +
                        "输出目录=${outDir.absolutePath}"
                )
            }
            if (failed > 0) exitProcess(1)
        } finally {
            ocrService?.close()
        }
        } // SegmentationService.use
    } finally {
        orientationService?.close()
        unwarpService?.close()
    }
}

private class ScanOutputs(
    val pageJpeg: File?,
    val pdf: File?,
)

// 输出命名：日期时间 + 批内序号，保证唯一
private var outputSeq = 0
private val tsFormat = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")

private fun uniqueBaseName(): String {
    outputSeq++
    return "扫描结果-${LocalDateTime.now().format(tsFormat)}-$outputSeq"
}

/** 一张输出页（单文档/多文档统一表示）。pageMat 由 saveOutputs 释放。 */
private class PageOutput(
    val pageMat: Mat,
    val quadInMask: Quad?,
    val normalizedQuad: Quad?,
    val estimatedDimensions: EstimatedDimensions?,
    val colorMode: ColorMode,
    val autoColorMode: ColorMode,
    var ocrTextBoxes: List<OcrTextBox> = emptyList(),
)

private fun printResult(inputFile: File, pages: List<PageOutput>, args: Args) {
    if (pages.size > 1) {
        println("检测到文档: ${pages.size} 个（多文档模式）")
    }
    pages.forEachIndexed { i, page ->
        if (pages.size > 1) println("-- 文档 ${i + 1} --")
        val quad = page.quadInMask
        if (quad != null) {
            println(
                "检测到文档四边形 (mask 坐标): " +
                    "TL(${"%.2f".format(quad.topLeft.x)},${"%.2f".format(quad.topLeft.y)}) " +
                    "TR(${"%.2f".format(quad.topRight.x)},${"%.2f".format(quad.topRight.y)}) " +
                    "BR(${"%.2f".format(quad.bottomRight.x)},${"%.2f".format(quad.bottomRight.y)}) " +
                    "BL(${"%.2f".format(quad.bottomLeft.x)},${"%.2f".format(quad.bottomLeft.y)})"
            )
            println("色彩模式 : ${page.autoColorMode}（自动）")
            page.estimatedDimensions?.let {
                println("预估尺寸 : ${it}")
            }
        } else {
            println("未检测到文档，使用整图缩放输出")
        }
    }
}

private fun saveOutputs(
    mask: SegmentationMask,
    pages: List<PageOutput>,
    args: Args,
    outDir: File,
    preprocessedInput: FloatArray?,
): ScanOutputs {
    // 结果分目录：图片/ 与 PDF/
    val imageDir = File(outDir, "图片").apply { mkdirs() }
    val pdfDir = File(outDir, "PDF").apply { mkdirs() }
    val base = uniqueBaseName()

    // 每页裁剪 JPEG（PDF 回写需要字节数据，图片导出关闭时仅不落盘）
    val jpegs = pages.map { encodeJpeg(it.pageMat, args.quality.jpegQuality) }
    var firstPageFile: File? = null
    if (!args.noImage) {
        for ((i, jpeg) in jpegs.withIndex()) {
            val pageFile = if (pages.size == 1) {
                File(imageDir, "$base.jpg")
            } else {
                File(imageDir, "$base-${i + 1}.jpg")
            }
            pageFile.writeBytes(jpeg)
            if (!args.json) println("已输出   : $pageFile")
            if (i == 0) firstPageFile = pageFile
        }
    }

    // 原始 float32 mask 导出（验证用，批量时按模板落盘）
    args.dumpMask?.let { template ->
        val file = resolveDumpPath(template, base, outDir)
        dumpRawMask(mask, file)
        if (!args.json) println("已导出   : mask float32 -> $file")
    }

    // 预处理输入张量导出（验证用，批量时按模板落盘）
    args.dumpInput?.let { template ->
        preprocessedInput?.let { input ->
            val file = resolveDumpPath(template, base, outDir)
            val buffer = ByteBuffer.allocate(input.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.asFloatBuffer().put(input)
            file.writeBytes(buffer.array())
            if (!args.json) println("已导出   : input float32 -> $file")
        }
    }

    // PDF（多页合并，附带 OCR 隐形文本层）
    var pdfFile: File? = null
    if (args.pdf && jpegs.isNotEmpty()) {
        pdfFile = File(pdfDir, "$base.pdf")
        FileOutputStream(pdfFile).use { fos ->
            PdfWriter.writePdf(
                pages.mapIndexed { i, page -> PdfPageInput(jpegs[i], page.estimatedDimensions, page.ocrTextBoxes) },
                fos,
            )
        }
        val ocrCount = pages.sumOf { it.ocrTextBoxes.size }
        if (!args.json) {
            println("已输出   : $pdfFile${if (ocrCount > 0) " (OCR $ocrCount 词条)" else ""}")
        }
    }

    pages.forEach { it.pageMat.release() }
    return ScanOutputs(firstPageFile, pdfFile)
}

private fun dumpRawMask(mask: SegmentationMask, file: File) {
    val prob = mask.toFloatArray()
    val buffer = ByteBuffer.allocate(prob.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(prob)
    file.writeBytes(buffer.array())
}

// ---------------------------------------------------------------------------
// pdf 子命令：页面图 + 词条 JSON -> PDF（OCR 文本编辑回写）
// ---------------------------------------------------------------------------

private data class WordsFile(
    val width: Int,
    val height: Int,
    val words: List<OcrTextBox>,
)

private fun parseWordsFile(content: String): WordsFile {
    val tree = GsonHolder.gson.fromJson(content, com.google.gson.JsonObject::class.java)
    val width = tree.get("width").asInt
    val height = tree.get("height").asInt
    val words = tree.getAsJsonArray("words").map { el ->
        val obj = el.asJsonObject
        OcrTextBox(
            text = obj.get("text").asString,
            box = ImageRect(
                left = obj.get("left").asInt,
                top = obj.get("top").asInt,
                right = obj.get("right").asInt,
                bottom = obj.get("bottom").asInt,
            ),
            lineHeight = obj.get("lineHeight").asInt,
            lineBottom = obj.get("lineBottom").asInt,
        )
    }
    return WordsFile(width, height, words)
}

private fun runPdfSubcommand(args: Args) {
    if (args.input.size != 1) {
        System.err.println("pdf 子命令需要一个页面图: fairscan-desktop pdf <page.jpg> [选项]")
        exitProcess(1)
    }
    val pageFile = File(args.input[0])
    if (!pageFile.exists()) {
        System.err.println("页面图不存在: $pageFile")
        exitProcess(1)
    }
    val wordsFile = File(args.ocrWords ?: "")
    if (args.ocrWords == null || !wordsFile.exists()) {
        System.err.println("缺少词条 JSON（--ocr-words <file>）或文件不存在")
        exitProcess(1)
    }

    val words = try {
        parseWordsFile(Files.readString(wordsFile.toPath()))
    } catch (e: Exception) {
        System.err.println("词条 JSON 解析失败: ${e.message}")
        exitProcess(1)
    }

    // 页面图按原尺寸读入（页面图与 OCR 尺寸一致，词条坐标直接映射）。
    // 用字节流解码而非 Imgcodecs.imread：OpenCV 的 imread 在 Windows 下无法读取中文路径。
    val pageMat: Mat
    try {
        pageMat = decodeJpeg(Files.readAllBytes(pageFile.toPath()))
    } catch (e: Exception) {
        System.err.println("无法读取页面图: $pageFile (${e.message})")
        exitProcess(1)
    }
    if (pageMat.empty()) {
        System.err.println("无法读取页面图: $pageFile")
        exitProcess(1)
    }
    val pageJpeg: ByteArray
    try {
        pageJpeg = encodeJpeg(pageMat, 95)
    } finally {
        pageMat.release()
    }

    val outFile = File(args.outDir) // pdf 子命令中 --out 为输出文件路径
    outFile.parentFile?.mkdirs()
    FileOutputStream(outFile).use { fos ->
        PdfWriter.writePdf(listOf(PdfPageInput(pageJpeg, null, words.words)), fos)
    }

    if (args.json) {
        emitJson(
            mapOf(
                "t" to "pdf", "page" to pageFile.absolutePath,
                "out" to outFile.absolutePath, "words" to words.words.size,
            ),
        )
    } else {
        println("已输出   : $outFile (${words.words.size} 词条)")
    }
}
