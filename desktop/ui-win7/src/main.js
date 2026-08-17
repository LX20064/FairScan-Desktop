/*
 * FairScan Desktop (Win7 兼容版) 主进程
 *
 * 职责：
 *  - 创建亚克力背景窗口（Electron 22 无 backgroundMaterial，全平台走 SWCA）
 *  - 通过 IPC 暴露：文件对话框 / spawn JVM 处理内核 / OCR 词条回写 PDF
 *  - 解析 CLI 的 NDJSON stdout（--json）逐行转发给渲染进程
 *
 * JVM 内核调用方式（避免 .bat 控制台编码问题）：
 *   java -Dfile.encoding=UTF-8 -cp "<install>/lib/*" org.fairscan.desktop.MainKt ...
 */
const { app, BrowserWindow, ipcMain, dialog, shell, nativeTheme } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');
const readline = require('readline');

// Windows 亚克力分层（Win7 兼容版：Electron 22 不支持 backgroundMaterial）：
//   - build >= 17763（Win10 1809+ / Win11）：调用 Win32
//     SetWindowCompositionAttribute(ACCENT_ENABLE_ACRYLICBLURBEHIND) 实现亚克力
//   - 更老版本（含 Win7 SP1）：CSS 毛玻璃回退（透明窗口 + backdrop-filter 模拟）
const WIN_BUILD = process.platform === 'win32'
  ? parseInt((os.release().split('.')[2]) || '0', 10)
  : 0;
const IS_WIN11 = WIN_BUILD >= 22000;
const IS_WIN10_ACRYLIC = WIN_BUILD >= 17763 && !IS_WIN11;
const USE_ACRYLIC = IS_WIN11 || IS_WIN10_ACRYLIC;
// 低于 1809（含 Win7 SP1）：无系统亚克力 API，回退 CSS 毛玻璃（透明窗口 + backdrop-filter 模拟质感）
const IS_WIN10_LEGACY = process.platform === 'win32' && WIN_BUILD > 0 && WIN_BUILD < 17763;

// 应用图标（src 目录随打包进入 app.asar，开发模式即 ui/src）
const APP_ICON = path.join(__dirname, 'icon.ico');

// Windows 11 圆角：通过 DWM API 显式启用
let enableRoundedCorners = null;
// Windows 10 1809+ 亚克力：SetWindowCompositionAttribute(ACCENT_ENABLE_ACRYLICBLURBEHIND)
let enableWin10Acrylic = null;
if (process.platform === 'win32') {
  try {
    const koffi = require('koffi');
    const HANDLE = koffi.pointer('HANDLE', koffi.opaque());
    const HWND = koffi.alias('HWND', HANDLE);
    const dwmapi = koffi.load('dwmapi.dll');
    const DwmSetWindowAttribute = dwmapi.func(
      '__stdcall',
      'DwmSetWindowAttribute',
      'long',
      [HWND, 'uint32', 'uint32 *', 'uint32']
    );
    const DWMWA_WINDOW_CORNER_PREFERENCE = 33;
    const DWMWCP_ROUND = 2;
    enableRoundedCorners = (hwndBuf) => {
      try {
        // getNativeWindowHandle() 返回的 Buffer 解码为 HWND 地址
        const addr = hwndBuf.readBigUInt64LE(0);
        console.log('native window handle address:', addr.toString(16));
        const hwnd = koffi.decode(hwndBuf, HANDLE);
        const val = Buffer.alloc(4);
        val.writeUInt32LE(DWMWCP_ROUND, 0);
        const hr = DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, val, 4);
        console.log('DwmSetWindowAttribute corner preference result:', hr);
      } catch (e) {
        console.warn('DwmSetWindowAttribute failed:', e.message);
      }
    };
    const user32 = koffi.load('user32.dll');
    const SetWindowCompositionAttribute = user32.func(
      '__stdcall',
      'SetWindowCompositionAttribute',
      'bool',
      [HWND, 'void *']
    );
    // WINDOWCOMPOSITIONATTRIBDATA（x64：Attrib@0 + 对齐 + pvData@8 + cbData@16，共 24 字节）
    // 内部 ACCENT_POLICY（16 字节）：AccentState=4(ACRYLICBLURBEHIND)、AccentFlags、GradientColor、AnimationId
    const WCA_ACCENT_POLICY = 19;
    const ACCENT_ENABLE_ACRYLICBLURBEHIND = 4;
    enableWin10Acrylic = (hwndBuf) => {
      try {
        const hwnd = koffi.decode(hwndBuf, HANDLE);
        const accent = Buffer.alloc(16);
        accent.writeUInt32LE(ACCENT_ENABLE_ACRYLICBLURBEHIND, 0);
        accent.writeUInt32LE(0, 4);  // AccentFlags
        accent.writeUInt32LE(0, 8);  // GradientColor（0 = 不额外染色，透出 CSS 半透明白）
        accent.writeUInt32LE(0, 12); // AnimationId
        const data = Buffer.alloc(24);
        data.writeUInt32LE(WCA_ACCENT_POLICY, 0);
        data.writeBigUInt64LE(BigInt(koffi.address(accent)), 8); // pvData
        data.writeBigUInt64LE(16n, 16);                           // cbData
        const ok = SetWindowCompositionAttribute(hwnd, data);
        console.log('SetWindowCompositionAttribute(acrylic) result:', ok);
        return !!ok;
      } catch (e) {
        console.warn('Win10 acrylic failed:', e.message);
        return false;
      }
    };
  } catch (e) {
    console.warn('koffi load failed:', e.message);
  }
}

const DESKTOP_DIR = path.join(__dirname, '..', '..'); // ui/src -> desktop
const DEFAULT_INSTALL = path.join(DESKTOP_DIR, 'build', 'install', 'fairscan-desktop');
const MAIN_CLASS = 'org.fairscan.desktop.MainKt';

// 打包后（electron-builder）extraResources 位于 process.resourcesPath：
//   resources/kernel     JVM 内核（lib/*、bin/*）
//   resources/models     分割/方向/UVDoc 模型
//   resources/tessdata   Tesseract 语言包
//   resources/jre        内置精简 JRE
const PACKAGED = !!app.isPackaged;

function resourceBase() {
  return PACKAGED ? process.resourcesPath : DESKTOP_DIR;
}

function kernelDir() {
  return PACKAGED ? path.join(process.resourcesPath, 'kernel') : DEFAULT_INSTALL;
}

// 把相对路径（模型/tessdata 等）解析为安装包/开发目录内的绝对路径
function resolveAsset(p) {
  if (!p) return p;
  if (path.isAbsolute(p)) return p;
  return path.join(resourceBase(), p);
}

let mainWindow = null;
let child = null;

// ---------------------------------------------------------------------------
// JVM 内核定位
// ---------------------------------------------------------------------------

function findJava() {
  // 1) 优先使用内置 JRE（发布包）
  const bundled = path.join(process.resourcesPath, 'jre', 'bin', 'java.exe');
  if (PACKAGED && fs.existsSync(bundled)) return bundled;
  // 2) 开发模式：JAVA_HOME
  const envJava = process.env.JAVA_HOME;
  if (envJava) {
    const p = path.join(envJava, 'bin', 'java.exe');
    if (fs.existsSync(p)) return p;
  }
  return 'java'; // 回退到 PATH
}

function cliInfo(cliDirOverride) {
  const cliDir = cliDirOverride || kernelDir();
  const lib = path.join(cliDir, 'lib');
  return {
    cliDir,
    lib,
    java: findJava(),
    mainClass: MAIN_CLASS,
    exists: fs.existsSync(lib),
  };
}

// ---------------------------------------------------------------------------
// 窗口
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1080,
    minHeight: 680,
    frame: false,                 // 完全自绘边框，配合 CSS 实现圆角+阴影
    icon: APP_ICON,               // 任务栏/窗口图标（自定义 .ico）
    // 亚克力（Win10 1809+ / Win11 走 SWCA）与 CSS 毛玻璃（<1809）都需要透明底；其余平台不透明纯白
    transparent: USE_ACRYLIC || IS_WIN10_LEGACY,
    backgroundColor: (USE_ACRYLIC || IS_WIN10_LEGACY) ? '#00000000' : '#fafafa',
    title: 'FairScan Desktop',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // 最大化状态同步到渲染层（用于去掉圆角等细节）
  mainWindow.on('maximize', () => send('win:state', { maximized: true }));
  mainWindow.on('unmaximize', () => send('win:state', { maximized: false }));

  // Windows 10 1809+ / Win11：显示后调用 SetWindowCompositionAttribute 启用亚克力
  // （Electron 22 无 backgroundMaterial，统一走 SWCA；Win7 走 CSS 毛玻璃回退）
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    // 延迟一帧，确保 HWND 已完成创建并可见
    setTimeout(() => {
      const hwndBuf = mainWindow.getNativeWindowHandle();
      if (enableRoundedCorners) {
        enableRoundedCorners(hwndBuf);
      }
      if (USE_ACRYLIC && enableWin10Acrylic && !enableWin10Acrylic(hwndBuf)) {
        // SWCA 亚克力失败：回退不透明纯白窗口，避免“透穿桌面”观感
        mainWindow.setBackgroundColor('#fafafa');
      }
    }, 80);
  });

  // Win10 <1809：无系统亚克力，给页面加 mat-css 类启用 CSS 毛玻璃回退
  if (IS_WIN10_LEGACY) {
    mainWindow.webContents.on('did-finish-load', () => {
      mainWindow.webContents
        .executeJavaScript('document.documentElement.classList.add("mat-css")')
        .catch(() => {});
    });
  }

  // E2E 冒烟测试：E2E=<图片路径> 启动时自动跑完整扫描链路；E2E_OCR=1 追加 OCR 回写验证
  const e2eImg = process.env.E2E;
  mainWindow.loadFile(
    path.join(__dirname, 'renderer', 'index.html'),
    e2eImg
      ? { search: `?e2e=1&img=${encodeURIComponent(e2eImg)}&ocr=${process.env.E2E_OCR === '1' ? '1' : '0'}` }
      : undefined,
  );

  // 开发辅助：渲染进程 console 转发到主进程 stdout
  mainWindow.webContents.on('console-message', (_e, level, message) => {
    console.log(`[renderer:${level}] ${message}`);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopChild();
  });
}

// ---------------------------------------------------------------------------
// JVM 子进程
// ---------------------------------------------------------------------------

function spawnCli(extraArgs, { onLine, onLog, onExit }) {
  const info = cliInfo();
  if (!info.exists) {
    onExit(2, `未找到 JVM 内核: ${info.lib}（先运行 gradlew :desktop:installDist）`);
    return null;
  }
  const args = [
    '-Dfile.encoding=UTF-8',
    '-cp',
    path.join(info.lib, '*'),
    MAIN_CLASS,
    ...extraArgs,
  ];
  const proc = spawn(info.java, args, {
    cwd: PACKAGED ? path.join(process.resourcesPath, 'kernel') : DESKTOP_DIR,
    windowsHide: true,
    encoding: 'utf8',
  });

  const rl = readline.createInterface({ input: proc.stdout });
  rl.on('line', (line) => {
    const t = line.trim();
    if (!t) return;
    try {
      onLine(JSON.parse(t));
    } catch {
      onLog(t);
    }
  });

  let errBuf = '';
  proc.stderr.on('data', (chunk) => {
    errBuf += chunk;
    const lines = errBuf.split(/\r?\n/);
    errBuf = lines.pop() || '';
    for (const l of lines) {
      const t = l.trim();
      if (t) onLog(t);
    }
  });
  proc.on('error', (err) => onExit(3, err.message));
  proc.on('close', (code) => {
    if (errBuf.trim()) onLog(errBuf.trim());
    onExit(code ?? 0);
  });
  return proc;
}

function stopChild() {
  if (child) {
    child.kill();
    child = null;
  }
}

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------

ipcMain.handle('dialog:openFiles', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择图片',
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: '图片', extensions: ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'] },
      { name: '所有文件', extensions: ['*'] },
    ],
  });
  return r.canceled ? [] : r.filePaths;
});

ipcMain.handle('dialog:openDirectory', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择文件夹',
    properties: ['openDirectory'],
  });
  return r.canceled ? null : r.filePaths[0];
});

// 递归扫描目录，返回其中所有图片文件（按路径排序）
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff']);

function walkImages(dir, recursive) {
  const out = [];
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    let entries;
    try { entries = fs.readdirSync(cur, { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      const full = path.join(cur, e.name);
      if (e.isDirectory()) {
        if (recursive) stack.push(full);
      } else if (e.isFile() && IMAGE_EXTS.has(path.extname(e.name).toLowerCase())) {
        out.push(full);
      }
    }
  }
  return out.sort();
}

ipcMain.handle('dir:listImages', (_e, dir, recursive) => {
  if (!dir || !fs.existsSync(dir)) return [];
  return walkImages(dir, !!recursive);
});

ipcMain.handle('dialog:savePdf', async () => {
  const r = await dialog.showSaveDialog(mainWindow, {
    title: '保存 PDF',
    defaultPath: 'fairscan-output.pdf',
    filters: [{ name: 'PDF', extensions: ['pdf'] }],
  });
  return r.canceled ? null : r.filePath;
});

ipcMain.handle('fs:saveText', async (_e, payload) => {
  const { defaultName, content } = payload;
  const r = await dialog.showSaveDialog(mainWindow, {
    title: '保存文件',
    defaultPath: defaultName || 'file.json',
    filters: [{ name: 'JSON', extensions: ['json'] }],
  });
  if (r.canceled || !r.filePath) return { ok: false };
  fs.writeFileSync(r.filePath, content, 'utf8');
  return { ok: true, path: r.filePath };
});

ipcMain.handle('env:cliInfo', (_e, overrideDir) => cliInfo(overrideDir));

// 默认输出目录：文档/Fairsacn扫描结果（不存在则创建）
ipcMain.handle('env:defaultOutDir', () => {
  const dir = path.join(app.getPath('documents'), 'Fairsacn扫描结果');
  try { fs.mkdirSync(dir, { recursive: true }); } catch { /* 忽略 */ }
  return dir;
});

ipcMain.handle('scan:start', (_e, config) => {
  if (child) return { ok: false, error: '已有任务在运行' };
  const args = ['scan', ...config.inputs, '--json', '--model', resolveAsset(config.model), '--out', config.outDir];
  if (config.autoRotate) args.push('--auto-rotate');
  else if (config.rotation) args.push('--rotation', String(config.rotation));
  if (config.mode) args.push('--mode', config.mode);
  if (config.color && config.color !== 'AUTO') args.push('--color', config.color);
  if (config.quality) args.push('--quality', config.quality);
  if (config.quad) args.push('--quad', config.quad.join(','));
  if (config.exportImage === false) args.push('--no-image');
  if (config.pdf) args.push('--pdf');
  if (!config.ocrEnabled) args.push('--no-ocr');
  if (config.dewarp) args.push('--dewarp');
  if (config.removeShadow) args.push('--remove-shadow');
  if (config.lang) args.push('--lang', config.lang);
  if (config.tessdata) args.push('--tessdata', resolveAsset(config.tessdata));
  if (config.threads) args.push('--threads', String(config.threads));

  child = spawnCli(args, {
    onLine: (obj) => send('scan:line', obj),
    onLog: (line) => send('scan:log', { level: 'info', text: line }),
    onExit: (code, message) => {
      if (message) send('scan:log', { level: 'error', text: message });
      send('scan:exit', code);
      child = null;
    },
  });
  return child ? { ok: true } : { ok: false, error: 'JVM 内核不可用' };
});

ipcMain.handle('scan:stop', () => {
  stopChild();
  return true;
});

/** OCR 词条编辑回写：写词条 JSON 到临时目录，调用 pdf 子命令生成新 PDF。 */
ipcMain.handle('ocr:regeneratePdf', async (_e, payload) => {
  const { pagePath, outPath, width, height, words } = payload;
  const wordsFile = path.join(os.tmpdir(), `fairscan-words-${Date.now()}.json`);
  const doc = { width, height, words };
  fs.writeFileSync(wordsFile, JSON.stringify(doc), 'utf8');

  return new Promise((resolve) => {
    const args = ['pdf', pagePath, '--ocr-words', wordsFile, '--out', outPath, '--json'];
    child = spawnCli(args, {
      onLine: (obj) => {
        if (obj.t === 'pdf') resolve({ ok: true, out: obj.out, words: obj.words });
      },
      onLog: () => {},
      onExit: (code, message) => {
        if (code !== 0 && code !== undefined) resolve({ ok: false, error: message || `退出码 ${code}` });
      },
    });
  }).finally(() => {
    try { fs.unlinkSync(wordsFile); } catch { /* ignore */ }
  });
});

ipcMain.handle('shell:showItem', (_e, p) => {
  if (p && fs.existsSync(p)) shell.showItemInFolder(p);
  return true;
});

ipcMain.handle('shell:openPath', (_e, p) => {
  if (p) shell.openPath(p);
  return true;
});

/** 保存摄像头捕获的图片到临时文件，返回文件路径。 */
ipcMain.handle('camera:saveCapture', async (_e, dataUrl) => {
  const base64 = dataUrl.replace(/^data:image\/\w+;base64,/, '');
  const buf = Buffer.from(base64, 'base64');
  const name = `capture-${Date.now()}.jpg`;
  const out = path.join(os.tmpdir(), 'fairscan-camera', name);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
  return { path: out };
});

/** 摄像头实时定位：对单帧调用 CLI detect，返回归一化 quad（独立进程，不与 scan 冲突）。 */
ipcMain.handle('camera:detect', (_e, payload) => {
  const { path: imgPath, model, mode, threads } = payload || {};
  if (!imgPath) return Promise.resolve({ t: 'detect', detected: false, error: '缺少图片路径' });
  return new Promise((resolve) => {
    const args = ['detect', imgPath, '--json', '--model', resolveAsset(model)];
    if (mode) args.push('--mode', mode);
    if (threads) args.push('--threads', String(threads));
    const proc = spawnCli(args, {
      onLine: (obj) => {
        if (obj && obj.t === 'detect') resolve(obj);
      },
      onLog: () => {},
      onExit: (code, message) => {
        if (code !== 0 && code !== undefined) {
          resolve({ t: 'detect', file: imgPath, detected: false, error: message || `退出码 ${code}` });
        }
      },
    });
    if (!proc) resolve({ t: 'detect', file: imgPath, detected: false, error: 'JVM 内核不可用' });
  });
});

ipcMain.handle('app:quit', () => {
  app.quit();
  return true;
});

// 自绘标题栏窗口控制
ipcMain.handle('win:minimize', () => { mainWindow?.minimize(); return true; });
ipcMain.handle('win:maximize', () => {
  if (!mainWindow) return true;
  if (mainWindow.isMaximized()) mainWindow.unmaximize();
  else mainWindow.maximize();
  return true;
});
ipcMain.handle('win:close', () => { mainWindow?.close(); return true; });

// ---------------------------------------------------------------------------

app.whenReady().then(() => {
  // 白色亚克力：强制浅色主题，亚克力材质随主题渲染为白色
  nativeTheme.themeSource = 'light';
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopChild();
  if (process.platform !== 'darwin') app.quit();
});
