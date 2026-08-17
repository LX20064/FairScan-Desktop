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

plugins {
    kotlin("jvm") version "2.2.0"
    application
}

group = "org.fairscan"
version = "0.1.0"

repositories {
    mavenCentral()
}

// 复用 Android 工程 :imageprocessing 模块的算法源码（纯 JVM，无 Android 依赖），
// 保证桌面端与移动端使用同一份图像处理实现。
sourceSets.main {
    java.srcDir("../imageprocessing/src/main/java")
    kotlin.srcDir("../imageprocessing/src/main/java")
}

dependencies {
    // 图像处理（与 imageprocessing 模块同款，JavaCPP 分发，Windows x86_64 自带原生库）
    implementation("org.openpnp:opencv:4.9.0-0")

    // TFLite 推理：Bytedeco JavaCPP 封装，直接加载原始 .tflite（与 Android 端同模型同权重）
    implementation("org.bytedeco:tensorflow-lite:2.18.0-1.5.11")
    runtimeOnly("org.bytedeco:tensorflow-lite:2.18.0-1.5.11:windows-x86_64")

    // PDF 生成：标准 Apache PDFBox（Android 端为 tom-roush 分支，桌面用原版，API 兼容）
    implementation("org.apache.pdfbox:pdfbox:2.0.31")

    // OCR：Tess4J（JNA 封装 Tesseract，主 jar 自带 win32-x86_64/linux 原生库）
    implementation("net.sourceforge.tess4j:tess4j:5.15.0")

    // JSON：--json 结构化输出与 pdf 子命令的词条回写解析
    implementation("com.google.code.gson:gson:2.11.0")

    // 方向感知（Step 2）：PP-LCNet 方向分类转 ONNX，ONNX Runtime Java 推理
    // 1.14.0：官方预编译最后支持 Win7 SP1 的版本线（v1.15 起预编译弃 Win7）；
    // 我们的模型均 <= opset 16（uvdoc GridSample-16），1.14 完全支持
    implementation("com.microsoft.onnxruntime:onnxruntime:1.14.0")

    testImplementation(kotlin("test"))
    testImplementation("org.assertj:assertj-core:3.27.7")
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
    }
}

application {
    mainClass.set("org.fairscan.desktop.MainKt")
}

tasks.test {
    useJUnitPlatform()
}
