/*
 * FairScan Desktop 渲染层逻辑
 */
'use strict';

const API = window.fairscan;

// ---------------------------------------------------------------------------
// 元素引用
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const els = {
  cliStatus: $('cliStatus'),
  fileCount: $('fileCount'),
  dropZone: $('dropZone'), dropHint: $('dropHint'),
  fileList: $('fileList'),
  btnAddFiles: $('btnAddFiles'), btnAddDir: $('btnAddDir'), btnClearFiles: $('btnClearFiles'),
  previewStage: $('previewStage'), previewImg: $('previewImg'),
  quadOverlay: $('quadOverlay'), quadPolygon: $('quadPolygon'),
  quadHandles: [$('qh0'), $('qh1'), $('qh2'), $('qh3')],
  previewEmpty: $('previewEmpty'), previewInfo: $('previewInfo'), previewTitle: $('previewTitle'),
  btnViewSrc: $('btnViewSrc'), btnViewResult: $('btnViewResult'),
  btnEditQuad: $('btnEditQuad'), btnRerunQuad: $('btnRerunQuad'),
  paramMode: $('paramMode'), paramRotation: $('paramRotation'), paramColor: $('paramColor'),
  paramQuality: $('paramQuality'), paramThreads: $('paramThreads'),
  paramLang: $('paramLang'),
  paramExportImage: $('paramExportImage'), paramExportPdf: $('paramExportPdf'),
  paramDewarp: $('paramDewarp'),
  paramRemoveShadow: $('paramRemoveShadow'),
  paramOutDir: $('paramOutDir'), paramModel: $('paramModel'),
  paramTessdata: $('paramTessdata'), paramCliDir: $('paramCliDir'),
  btnBrowseOut: $('btnBrowseOut'), btnBrowseTess: $('btnBrowseTess'), btnBrowseCli: $('btnBrowseCli'),
  btnScan: $('btnScan'), btnStop: $('btnStop'),
  progressBar: $('progressBar'), statusLine: $('statusLine'),
  btnToggleLog: $('btnToggleLog'),
  logPanel: $('logPanel'),
  btnOpenOut: $('btnOpenOut'),
  btnOpenOcr: $('btnOpenOcr'), ocrDialog: $('ocrDialog'), ocrClose: $('ocrClose'),
  ocrFileLabel: $('ocrFileLabel'), ocrTable: $('ocrTable'),
  ocrEmpty: $('ocrEmpty'), btnSaveWords: $('btnSaveWords'), btnRegenPdf: $('btnRegenPdf'),
  btnSettings: $('btnSettings'), settingsDialog: $('settingsDialog'),
  settingsClose: $('settingsClose'), settingsClose2: $('settingsClose2'),
  btnCamera: $('btnCamera'), cameraDialog: $('cameraDialog'),
  cameraClose: $('cameraClose'), cameraVideo: $('cameraVideo'),
  cameraPlaceholder: $('cameraPlaceholder'), cameraError: $('cameraError'),
  cameraQuad: $('cameraQuad'), cameraQuadPolygon: $('cameraQuadPolygon'),
  cameraAutoLocate: $('cameraAutoLocate'),
  btnCapture: $('btnCapture'),
  paramSubdirs: $('paramSubdirs'),
  winMin: $('winMin'), winMax: $('winMax'), winClose: $('winClose'),
  toast: $('toast'),
};

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const STATUS = { IDLE: 'idle', RUNNING: 'running', DONE: 'done', ERROR: 'error' };

const state = {
  files: [],                 // {path, name, status, quad(归一化 8), result, words, srcW, srcH}
  selected: -1,
  running: false,
  view: 'src',               // src | result
  editQuad: true,
  cliInfo: null,
  lastWords: null,           // 最近一次 ocr 行（含 width/height）
  wordsDirty: false,
  dragIndex: -1,
  e2eOcr: false,
  cameraStream: null,        // MediaStream 摄像头流
  camDetectBusy: false,      // 实时定位是否进行中（防止重叠）
  camDetectTimer: null,      // 实时定位定时器
  camLastQuad: null,         // 最近一次实时定位到的归一化 quad（8 数组）
};

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
function fileUrl(p) {
  // encodeURI 编码非 ASCII（中文等），否则 file:// URL 在部分目录/文件名下无法加载
  return 'file:///' + encodeURI(p.replace(/\\/g, '/')).replace(/#/g, '%23');
}

function toast(msg, isErr) {
  els.toast.innerHTML = '';
  const icon = document.createElement('svg');
  icon.setAttribute('viewBox', '0 0 16 16');
  icon.style.cssText = 'width:14px;height:14px;fill:currentColor;vertical-align:text-bottom;margin-right:6px;';
  icon.innerHTML = isErr
    ? '<circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.2"/><path d="M5 5l6 6M11 5L5 11" stroke="currentColor" stroke-width="1.2"/>'
    : '<circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.2"/><path d="M4 8l3 3 5-6" fill="none" stroke="currentColor" stroke-width="1.5"/>';
  const span = document.createElement('span');
  span.textContent = msg;
  els.toast.append(icon, span);
  els.toast.classList.toggle('err', !!isErr);
  els.toast.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.remove('show'), 2600);
}

function log(text, cls) {
  const div = document.createElement('div');
  div.className = 'line ' + (cls || 'info');
  div.textContent = text;
  els.logPanel.appendChild(div);
  while (els.logPanel.children.length > 2000) els.logPanel.removeChild(els.logPanel.firstChild);
  els.logPanel.scrollTop = els.logPanel.scrollHeight;
}

function setStatus(text) { els.statusLine.textContent = text; }

function setProgress(pct) { els.progressBar.style.width = Math.max(0, Math.min(100, pct)) + '%'; }

function setRunning(run) {
  state.running = run;
  els.btnScan.disabled = run;
  els.btnStop.disabled = !run;
}

// ---------------------------------------------------------------------------
// 摄像头
// ---------------------------------------------------------------------------
async function openCamera() {
  try {
    els.cameraError.textContent = '';
    els.cameraPlaceholder.style.display = '';
    els.cameraVideo.style.display = 'none';
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    els.cameraVideo.srcObject = state.cameraStream;
    els.cameraVideo.style.display = '';
    els.cameraPlaceholder.style.display = 'none';
    els.cameraDialog.classList.remove('hidden');
    if (els.cameraAutoLocate.checked) startCamAutoLocate();
  } catch (e) {
    els.cameraError.textContent = '摄像头访问失败: ' + e.message;
    els.cameraDialog.classList.remove('hidden');
  }
}

function closeCamera() {
  stopCamAutoLocate();
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((t) => t.stop());
    state.cameraStream = null;
  }
  els.cameraVideo.srcObject = null;
  els.cameraVideo.style.display = 'none';
  els.cameraPlaceholder.style.display = '';
  els.cameraError.textContent = '';
  els.cameraQuad.classList.add('hidden');
  els.cameraDialog.classList.add('hidden');
}

/** 从当前视频帧截取 JPEG dataURL（拍照与实时定位共用）。 */
function grabCameraFrame() {
  const video = els.cameraVideo;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.92);
}

// 摄像头实时自动定位文档（参考原版 LIVE_ANALYSIS：每 ~2s 取一帧检测并叠加四边形）
function startCamAutoLocate() {
  stopCamAutoLocate();
  els.cameraQuad.classList.add('hidden');
  const tick = async () => {
    if (!els.cameraAutoLocate.checked || state.camDetectBusy || state.running) return;
    const video = els.cameraVideo;
    if (!state.cameraStream || !video.videoWidth || !video.videoHeight) return;
    state.camDetectBusy = true;
    try {
      const { path } = await API.saveCameraCapture(grabCameraFrame());
      const model = els.paramModel.value.trim();
      const threads = Math.max(1, Math.min(8, Number(els.paramThreads.value) || 2));
      const r = await API.cameraDetect({ path, model, mode: 'CAPTURE', threads });
      if (r && r.detected && r.quad) {
        state.camLastQuad = r.quad;
        drawCamQuad(r.quad);
      } else {
        state.camLastQuad = null;
        els.cameraQuad.classList.add('hidden');
      }
    } catch (e) {
      // 单帧定位失败不中断预览
    } finally {
      state.camDetectBusy = false;
    }
  };
  state.camDetectTimer = setInterval(tick, 2000);
}

function stopCamAutoLocate() {
  if (state.camDetectTimer) {
    clearInterval(state.camDetectTimer);
    state.camDetectTimer = null;
  }
}

/** video 采用 object-fit: contain，计算视频内容在取景框内的实际显示区域。 */
function camVideoBox() {
  const video = els.cameraVideo;
  const cw = video.clientWidth;
  const ch = video.clientHeight;
  if (!cw || !ch) return null;
  const scale = Math.min(cw / video.videoWidth, ch / video.videoHeight);
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

function drawCamQuad(q) {
  const b = camVideoBox();
  if (!b) return;
  const pts = [
    (b.x + q[0] * b.w), (b.y + q[1] * b.h),
    (b.x + q[2] * b.w), (b.y + q[3] * b.h),
    (b.x + q[4] * b.w), (b.y + q[5] * b.h),
    (b.x + q[6] * b.w), (b.y + q[7] * b.h),
  ];
  els.cameraQuadPolygon.setAttribute('points', pts.join(' '));
  els.cameraQuad.classList.remove('hidden');
}

async function capturePhoto() {
  const video = els.cameraVideo;
  if (!video.videoWidth || !video.videoHeight) {
    toast('摄像头未就绪', true);
    return;
  }
  try {
    const { path } = await API.saveCameraCapture(grabCameraFrame());
    closeCamera();
    if (els.cameraAutoLocate.checked) {
      // 自动定位开启：拍照后立即以 CAPTURE 模式处理（使用实时检测到的四角）
      await scanSinglePhoto(path, video.videoWidth, video.videoHeight);
    } else {
      log(`已拍照: ${path.split(/[\\/]/).pop()}`, 'ok');
      addFiles([path]);
      toast('拍照成功，已添加到文件列表');
    }
  } catch (e) {
    toast('保存照片失败: ' + e.message, true);
  }
}

/** 对单张照片立即执行 CAPTURE 模式扫描。 */
async function scanSinglePhoto(path, srcW, srcH) {
  addFiles([path]);
  if (state.running) return;
  const cfg = scanConfig();
  cfg.inputs = [path];
  cfg.mode = 'CAPTURE'; // 摄像头拍照固定使用 CAPTURE（多阈值 + 兜底裁剪）
  const q = state.camLastQuad;
  if (q && srcW && srcH) {
    // detect 输出归一化坐标，CLI --quad 需要源图像素坐标
    cfg.quad = q.map((v, i) => (i % 2 === 0 ? v * srcW : v * srcH).toFixed(1));
  }
  state.camLastQuad = null;
  setRunning(true);
  setProgress(0);
  setStatus('处理照片…');
  log('—— 开始处理照片（CAPTURE 模式） ——', 'dim');
  const r = await API.scanStart(cfg);
  if (!r.ok) {
    toast(r.error || '启动失败', true);
    setRunning(false);
  }
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------
async function init() {
  console.log('renderer init: start');
  state.cliInfo = await API.cliInfo();
  console.log('renderer init: cliInfo', JSON.stringify(state.cliInfo));
  if (!state.cliInfo) return;
  refreshCliStatus();

  // 默认输出目录：文档/Fairsacn扫描结果
  try {
    const d = await API.defaultOutDir();
    if (d && !els.paramOutDir.value) els.paramOutDir.value = d;
  } catch { /* 忽略 */ }

  els.paramCliDir.addEventListener('change', () => reprobeCli());

  bindFileList();
  bindPreview();
  bindParams();
  bindScan();
  bindOcr();
  bindNav();
  bindDnD();

  API.onScanLine(onScanLine);
  API.onScanLog((o) => log(o.text, o.level === 'error' ? 'error' : 'dim'));
  API.onScanExit(onScanExit);
  API.onWinState((s) => {
    document.body.classList.toggle('maximized', !!s.maximized);
    els.winMax.innerHTML = s.maximized
      ? '<svg viewBox="0 0 16 16"><path d="M3 8h10v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1zm-1-7v9h1V3h9V2z"/></svg>'
      : '<svg viewBox="0 0 16 16"><path d="M3 3h10v10H3zM4 4v8h8V4z"/></svg>';
  });

  // 自绘标题栏窗口控制
  els.winMin.addEventListener('click', () => API.winMinimize());
  els.winMax.addEventListener('click', () => API.winToggleMaximize());
  els.winClose.addEventListener('click', () => API.winClose());

  // 摄像头
  els.btnCamera.addEventListener('click', openCamera);
  els.cameraClose.addEventListener('click', closeCamera);
  els.cameraDialog.addEventListener('click', (e) => {
    if (e.target === els.cameraDialog) closeCamera();
  });
  els.btnCapture.addEventListener('click', capturePhoto);
  els.cameraAutoLocate.addEventListener('change', () => {
    if (!els.cameraDialog.classList.contains('hidden')) {
      if (els.cameraAutoLocate.checked) startCamAutoLocate();
      else stopCamAutoLocate();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!els.cameraDialog.classList.contains('hidden')) closeCamera();
  });

  window.addEventListener('resize', drawQuad);
  log('界面就绪。添加图片后点击「开始扫描」。', 'dim');
}

function refreshCliStatus() {
  const info = state.cliInfo;
  if (!info) {
    els.cliStatus.textContent = 'JVM 内核不可用';
    els.cliStatus.className = 'cli-status bad';
    return;
  }
  if (info.exists) {
    els.cliStatus.textContent = '内核就绪';
    els.cliStatus.className = 'cli-status ok';
  } else {
    els.cliStatus.textContent = '未找到内核';
    els.cliStatus.className = 'cli-status bad';
  }
}

// ---------------------------------------------------------------------------
// 文件列表
// ---------------------------------------------------------------------------
function bindFileList() {
  els.btnAddFiles.addEventListener('click', async () => {
    const paths = await API.openFiles();
    addFiles(paths);
  });
  els.btnAddDir.addEventListener('click', async () => {
    const dir = await API.openDirectory();
    if (!dir) return;
    const recursive = els.paramSubdirs.checked;
    const files = await API.listDirImages(dir, recursive);
    if (!files.length) { toast('该文件夹下没有图片', true); return; }
    addFiles(files);
    log(`已从目录添加 ${files.length} 个图片${recursive ? '（含子目录）' : ''}: ${dir}`, 'dim');
  });
  els.btnClearFiles.addEventListener('click', () => {
    if (state.running) return toast('扫描中，无法清空');
    state.files = [];
    state.selected = -1;
    renderFileList();
    clearPreview();
  });
}

function addFiles(paths) {
  if (!paths || paths.length === 0) return;
  if (state.running) return toast('扫描中，无法添加');
  let added = 0;
  for (const p of paths) {
    const isImg = /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(p);
    const name = p.split(/[\\/]/).pop();
    if (!isImg) { log(`跳过非图片: ${name}`, 'warn'); continue; }
    if (state.files.some((f) => f.path === p)) continue;
    state.files.push({ path: p, name, status: STATUS.IDLE, quad: null, result: null, words: null });
    added++;
  }
  if (added) {
    renderFileList();
    selectFile(state.files.length - 1);
    log(`已添加 ${added} 个文件`, 'dim');
  }
}

function renderFileList() {
  els.fileList.innerHTML = '';
  els.fileCount.textContent = String(state.files.length);
  els.dropHint.style.display = state.files.length ? 'none' : '';
  state.files.forEach((f, i) => {
    const li = document.createElement('li');
    if (i === state.selected) li.classList.add('selected');

    const dot = document.createElement('span');
    dot.className = 'dot ' + f.status;

    const name = document.createElement('span');
    name.className = 'fname';
    name.textContent = f.name;
    name.title = f.path;

    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = f.status === STATUS.DONE ? (f.result && f.result.ocrWords ? `✓ ${f.result.ocrWords}词` : '✓') : (f.status === STATUS.ERROR ? '×' : '');

    const rm = document.createElement('button');
    rm.className = 'remove';
    rm.innerHTML = '<svg viewBox="0 0 16 16"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" stroke-width="1.8"/></svg>';
    rm.addEventListener('click', (e) => {
      e.stopPropagation();
      if (state.running) return;
      state.files.splice(i, 1);
      if (state.selected >= state.files.length) state.selected = state.files.length - 1;
      renderFileList();
      if (state.selected >= 0) loadPreview(); else clearPreview();
    });

    li.append(dot, name, meta, rm);
    li.addEventListener('click', () => selectFile(i));
    els.fileList.appendChild(li);
  });
}

function selectFile(i) {
  state.selected = i;
  renderFileList();
  loadPreview();
}

// ---------------------------------------------------------------------------
// 拖拽添加
// ---------------------------------------------------------------------------
function bindDnD() {
  let depth = 0;
  els.dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; els.dropZone.classList.add('dragover'); });
  els.dropZone.addEventListener('dragover', (e) => e.preventDefault());
  els.dropZone.addEventListener('dragleave', () => { if (--depth <= 0) els.dropZone.classList.remove('dragover'); });
  els.dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); depth = 0; els.dropZone.classList.remove('dragover');
    const paths = [];
    for (const f of e.dataTransfer.files) {
      if (f.path) paths.push(f.path);
    }
    if (paths.length) addFiles(paths);
  });
}

// ---------------------------------------------------------------------------
// 预览 + 四角
// ---------------------------------------------------------------------------
function bindPreview() {
  els.btnViewSrc.addEventListener('click', () => setView('src'));
  els.btnViewResult.addEventListener('click', () => setView('result'));
  els.btnEditQuad.addEventListener('click', () => {
    state.editQuad = !state.editQuad;
    els.btnEditQuad.classList.toggle('active', state.editQuad);
    drawQuad();
  });
  els.btnRerunQuad.addEventListener('click', rerunWithQuad);

  for (const h of els.quadHandles) {
    h.addEventListener('pointerdown', (e) => startQuadDrag(e));
  }
  window.addEventListener('pointermove', (e) => moveQuadDrag(e));
  window.addEventListener('pointerup', endQuadDrag);
}

function setView(v) {
  state.view = v;
  els.btnViewSrc.classList.toggle('active', v === 'src');
  els.btnViewResult.classList.toggle('active', v === 'result');
  loadPreview();
}

function clearPreview() {
  els.previewImg.classList.remove('visible');
  els.previewImg.removeAttribute('src');
  els.quadOverlay.classList.add('hidden');
  els.previewEmpty.style.display = '';
  els.previewInfo.textContent = '';
  els.previewTitle.textContent = '预览';
  els.btnRerunQuad.disabled = true;
}

function selectedFile() {
  return state.selected >= 0 ? state.files[state.selected] : null;
}

function loadPreview() {
  const f = selectedFile();
  if (!f) { clearPreview(); return; }

  els.previewEmpty.style.display = 'none';
  els.previewTitle.textContent = f.name;

  const isResult = state.view === 'result' && f.result;
  const src = isResult ? f.result.page : f.path;

  els.previewImg.onload = () => {
    if (state.view === 'src') {
      f.srcW = els.previewImg.naturalWidth;
      f.srcH = els.previewImg.naturalHeight;
    }
    drawQuad();
    updatePreviewInfo();
  };
  els.previewImg.onerror = () => {
    toast('无法加载图片: ' + src, true);
    els.previewImg.classList.remove('visible');
  };
  els.previewImg.src = fileUrl(src);
  els.previewImg.classList.add('visible');

  // 结果视图下无检测四边形则不显示四角编辑
  const quadAvailable = state.view === 'src' && f.quad;
  els.btnEditQuad.disabled = state.view !== 'src' || !f.quad;
  els.btnRerunQuad.disabled = !(state.view === 'src' && f.quad);
  if (!quadAvailable && state.editQuad) {
    // 保持编辑模式，但仅无 quad 时隐藏
  }
  drawQuad();
}

function updatePreviewInfo() {
  const f = selectedFile();
  if (!f) return;
  const parts = [];
  if (f.result) {
    parts.push(`色彩 ${f.result.colorMode}`);
    parts.push(`推理 ${f.result.inferenceMs}ms`);
    parts.push(`总计 ${f.result.totalMs}ms`);
    if (f.result.ocrWords) parts.push(`OCR ${f.result.ocrWords} 词条`);
    if (f.result.pdf) parts.push('PDF 已生成');
    if (state.view === 'src' && f.quad) parts.push('拖动圆点可调整四角');
  }
  els.previewInfo.textContent = parts.join(' · ') || f.path;
}

/** 计算图片在舞台内的实际显示矩形（object-fit contain 等效）。 */
function displayedRect() {
  const stage = els.previewStage;
  const img = els.previewImg;
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  if (!img.naturalWidth || !img.naturalHeight || sw <= 0 || sh <= 0) return null;
  const scale = Math.min(sw / img.naturalWidth, sh / img.naturalHeight);
  const w = img.naturalWidth * scale;
  const h = img.naturalHeight * scale;
  return { x: (sw - w) / 2, y: (sh - h) / 2, w, h };
}

function drawQuad() {
  const f = selectedFile();
  els.quadOverlay.classList.add('hidden');
  if (!f || !f.quad || state.view !== 'src' || !state.editQuad) {
    return;
  }
  const r = displayedRect();
  if (!r) return;
  const q = f.quad; // 归一化 [tlx,tly,trx,try,brx,bry,blx,bly]
  const pts = [
    [q[0] * r.w + r.x, q[1] * r.h + r.y],
    [q[2] * r.w + r.x, q[3] * r.h + r.y],
    [q[4] * r.w + r.x, q[5] * r.h + r.y],
    [q[6] * r.w + r.x, q[7] * r.h + r.y],
  ];
  els.quadPolygon.setAttribute('points', pts.map((p) => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '));
  for (let i = 0; i < 4; i++) {
    els.quadHandles[i].setAttribute('cx', pts[i][0]);
    els.quadHandles[i].setAttribute('cy', pts[i][1]);
  }
  els.quadOverlay.classList.remove('hidden');
}

// ---- 四角拖拽 ----
function startQuadDrag(e) {
  const f = selectedFile();
  if (!f || !f.quad || state.view !== 'src') return;
  state.dragIndex = Number(e.target.getAttribute('data-i'));
  e.target.setPointerCapture(e.pointerId);
  e.preventDefault();
}

function moveQuadDrag(e) {
  if (state.dragIndex < 0) return;
  const r = displayedRect();
  if (!r) return;
  const nx = Math.min(1, Math.max(0, (e.clientX - els.previewStage.getBoundingClientRect().left - r.x) / r.w));
  const ny = Math.min(1, Math.max(0, (e.clientY - els.previewStage.getBoundingClientRect().top - r.y) / r.h));
  const f = selectedFile();
  f.quad[state.dragIndex * 2] = nx;
  f.quad[state.dragIndex * 2 + 1] = ny;
  drawQuad();
}

function endQuadDrag() {
  if (state.dragIndex >= 0) {
    state.dragIndex = -1;
    els.btnRerunQuad.disabled = !selectedFile() || !selectedFile().quad || state.running;
  }
}

async function rerunWithQuad() {
  const f = selectedFile();
  if (!f || !f.quad || state.running) return;
  if (!f.srcW || !f.srcH) { toast('请等待原图加载完成', true); return; }
  const q = f.quad;
  const quadPx = q.map((v, i) => (i % 2 === 0 ? v * f.srcW : v * f.srcH).toFixed(1));
  toast('正在用调整后的四角重新处理…');
  const cfg = scanConfig();
  cfg.inputs = [f.path];
  cfg.quad = quadPx;
  f.status = STATUS.RUNNING;
  renderFileList();
  setRunning(true);
  setProgress(0);
  await API.scanStart(cfg);
}

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------
function bindParams() {
  els.btnBrowseOut.addEventListener('click', async () => {
    const dir = await API.openDirectory();
    if (dir) els.paramOutDir.value = dir;
  });
  els.btnBrowseTess.addEventListener('click', async () => {
    const dir = await API.openDirectory();
    if (dir) els.paramTessdata.value = dir;
  });
  els.btnBrowseCli.addEventListener('click', async () => {
    const dir = await API.openDirectory();
    if (dir) {
      els.paramCliDir.value = dir;
      await reprobeCli();
    }
  });
}

async function reprobeCli() {
  const dir = els.paramCliDir.value.trim() || undefined;
  state.cliInfo = await API.cliInfo(dir);
  refreshCliStatus();
}

function scanConfig() {
  return {
    inputs: state.files.map((f) => f.path),
    model: els.paramModel.value.trim(),
    outDir: els.paramOutDir.value.trim(),
    rotation: els.paramRotation.value === 'auto' ? 0 : Number(els.paramRotation.value),
    autoRotate: els.paramRotation.value === 'auto',
    mode: els.paramMode.value,
    color: els.paramColor.value,
    quality: els.paramQuality.value,
    exportImage: els.paramExportImage.checked,
    pdf: els.paramExportPdf.checked,
    ocrEnabled: els.paramExportPdf.checked,
    dewarp: els.paramDewarp.checked,
    removeShadow: els.paramRemoveShadow.checked,
    lang: els.paramLang.value.trim(),
    tessdata: els.paramTessdata.value.trim(),
    threads: Math.max(1, Math.min(8, Number(els.paramThreads.value) || 2)),
    quad: null,
  };
}

// ---------------------------------------------------------------------------
// 扫描
// ---------------------------------------------------------------------------
function bindScan() {
  els.btnScan.addEventListener('click', startScan);
  els.btnStop.addEventListener('click', async () => {
    await API.scanStop();
    setStatus('已停止');
  });
}

async function startScan() {
  if (state.running) return;
  if (state.files.length === 0) { toast('请先添加文件', true); return; }
  if (!state.cliInfo || !state.cliInfo.exists) { toast('JVM 内核不可用', true); return; }

  for (const f of state.files) f.status = STATUS.IDLE;
  state.lastWords = null;
  setRunning(true);
  setProgress(0);
  setStatus('启动扫描…');
  log('—— 开始扫描 ——', 'dim');

  const cfg = scanConfig();
  const r = await API.scanStart(cfg);
  if (!r.ok) {
    toast(r.error || '启动失败', true);
    setRunning(false);
  }
}

function onScanLine(obj) {
  const idx = (i, total) => (total ? ((i - 1) / total) * 100 : 0);

  switch (obj.t) {
    case 'config':
      log(`模型: ${obj.model} · 共 ${obj.total} 个文件`, 'dim');
      break;
    case 'start': {
      const f = state.files[obj.index - 1];
      if (f) { f.status = STATUS.RUNNING; renderFileList(); }
      setProgress(idx(obj.index, obj.total));
      setStatus(`[${obj.index}/${obj.total}] ${obj.file} 处理中…`);
      log(`[${obj.index}/${obj.total}] ${obj.file}`, 'info');
      break;
    }
    case 'ocr': {
      // 匹配当前处理文件
      const f = state.files.find((x) => x.name === obj.file);
      if (f) {
        f.words = { width: obj.width, height: obj.height, words: obj.words };
        state.lastWords = f.words;
      }
      break;
    }
    case 'result': {
      const f = state.files.find((x) => x.name === obj.file);
      if (!f) break;
      f.status = STATUS.DONE;
      f.result = obj;
      f.quad = obj.quad || null;
      if (obj.detected) {
        log(`${obj.file}: 检测到文档 ${obj.colorMode} · ${obj.inferenceMs}ms · OCR ${obj.ocrWords} 词条`, 'ok');
      } else {
        log(`${obj.file}: 未检测到文档，整图缩放 · ${obj.inferenceMs}ms`, 'warn');
      }
      renderFileList();
      if (state.selected === state.files.indexOf(f)) { loadPreview(); }
      break;
    }
    case 'error': {
      const f = state.files.find((x) => x.name === obj.file);
      if (f) { f.status = STATUS.ERROR; renderFileList(); }
      log(`错误 ${obj.file}: ${obj.message}`, 'error');
      break;
    }
    case 'summary': {
      setProgress(100);
      setStatus(`完成: 成功 ${obj.success} / 失败 ${obj.failed} · 总耗时 ${(obj.totalMs / 1000).toFixed(1)}s`);
      log(`—— 完成: 成功 ${obj.success}, 失败 ${obj.failed}, 检测到 ${obj.detected} ——`, 'ok');
      break;
    }
    default:
      break;
  }
}

async function onScanExit(code) {
  setRunning(false);
  els.btnRerunQuad.disabled = !(selectedFile() && selectedFile().quad);
  if (code === 0) {
    setStatus('扫描完成');
  } else if (code !== undefined && code !== null) {
    setStatus(`扫描进程退出 (code=${code})`);
    log(`进程退出 code=${code}`, 'error');
  } else {
    setStatus('已停止');
  }
  // 刷新预览（结果可能已更新）
  if (selectedFile()) loadPreview();

  if (isE2E()) {
    const f = selectedFile();
    if (state.e2eOcr && f && f.words && f.result) {
      // OCR 回写链路：编辑首个词条 → 重新生成 PDF
      const edited = f.words.words.map((w, i) =>
        i === 0 ? { ...w, text: 'E2E-EDITED' } : w,
      );
      const outPath = f.result.page.replace(/[\\/][^\\/]+$/, '') + '\\e2e-edited.pdf';
      const r = await API.ocrRegeneratePdf({
        pagePath: f.result.page, outPath,
        width: f.words.width, height: f.words.height, words: edited,
      });
      console.log('E2E_OCR ' + JSON.stringify(r));
    } else {
      console.log(
        'E2E_RESULT ' + JSON.stringify({
          code,
          status: f ? f.status : null,
          detected: f && f.result ? f.result.detected : null,
          colorMode: f && f.result ? f.result.colorMode : null,
          ocrWords: f && f.result ? f.result.ocrWords : 0,
          wordsTable: f && f.words ? f.words.words.length : 0,
          page: f && f.result ? f.result.page : null,
          pdf: f && f.result ? f.result.pdf : null,
        }),
      );
    }
    setTimeout(() => API.quit(), 800);
  }
}

// ---------------------------------------------------------------------------
// E2E 冒烟测试（E2E=<图片路径> 环境变量触发）
// ---------------------------------------------------------------------------
function isE2E() {
  return new URLSearchParams(location.search).get('e2e') === '1';
}

function runE2E() {
  const params = new URLSearchParams(location.search);
  const img = params.get('img');
  state.e2eOcr = params.get('ocr') === '1';
  if (!img) { console.log('E2E_RESULT ' + JSON.stringify({ error: 'no img param' })); API.quit(); return; }
  log(`E2E: 自动添加 ${img}`, 'dim');
  addFiles([img]);
  setTimeout(() => startScan(), 600);
}

// ---------------------------------------------------------------------------
// 导航：日志折叠 / OCR 对话框
// ---------------------------------------------------------------------------
function bindNav() {
  els.btnToggleLog.addEventListener('click', () => {
    const show = els.logPanel.classList.toggle('hidden');
    els.btnToggleLog.classList.toggle('active', !show);
  });
  els.btnOpenOcr.addEventListener('click', openOcrDialog);
  els.ocrClose.addEventListener('click', closeOcrDialog);
  els.ocrDialog.addEventListener('click', (e) => {
    if (e.target === els.ocrDialog) closeOcrDialog();
  });
  els.btnSettings.addEventListener('click', openSettingsDialog);
  els.settingsClose.addEventListener('click', closeSettingsDialog);
  els.settingsClose2.addEventListener('click', closeSettingsDialog);
  els.settingsDialog.addEventListener('click', (e) => {
    if (e.target === els.settingsDialog) closeSettingsDialog();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!els.ocrDialog.classList.contains('hidden')) closeOcrDialog();
    else if (!els.settingsDialog.classList.contains('hidden')) closeSettingsDialog();
  });
  els.btnOpenOut.addEventListener('click', () => {
    const dir = els.paramOutDir.value.trim();
    API.openPath(dir || '.');
  });
}

function openOcrDialog() {
  const f = selectedFile();
  if (!f) { toast('请先选择文件', true); return; }
  populateOcr(f);
  els.ocrDialog.classList.remove('hidden');
}

function closeOcrDialog() {
  els.ocrDialog.classList.add('hidden');
}

function openSettingsDialog() {
  els.settingsDialog.classList.remove('hidden');
}

function closeSettingsDialog() {
  els.settingsDialog.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// OCR 词条编辑
// ---------------------------------------------------------------------------
function bindOcr() {
  els.btnSaveWords.addEventListener('click', saveWordsJson);
  els.btnRegenPdf.addEventListener('click', regeneratePdf);
}

function populateOcr(f) {
  const words = f && f.words && f.words.words ? f.words.words : null;
  const tbody = els.ocrTable.querySelector('tbody');
  tbody.innerHTML = '';
  els.ocrEmpty.style.display = words ? 'none' : '';
  els.ocrTable.style.display = words ? '' : 'none';
  els.btnSaveWords.disabled = !words;
  els.btnRegenPdf.disabled = !words;

  if (!words) {
    els.ocrFileLabel.textContent = '';
    return;
  }
  els.ocrFileLabel.textContent = `${f.name} · ${words.length} 词条 · 双击文本可编辑`;

  let lastLine = null;
  words.forEach((w, i) => {
    if (w.lineBottom !== lastLine) {
      const tr = document.createElement('tr');
      tr.className = 'group-row';
      const td = document.createElement('td');
      td.colSpan = 3;
      td.textContent = `—— 行 ${w.lineBottom} ——`;
      tr.appendChild(td);
      tbody.appendChild(tr);
      lastLine = w.lineBottom;
    }
    const tr = document.createElement('tr');
    const tdText = document.createElement('td');
    const input = document.createElement('input');
    input.className = 'word-text';
    input.value = w.text;
    input.dataset.i = String(i);
    input.addEventListener('input', () => {
      f.words.words[Number(input.dataset.i)] = { ...f.words.words[Number(input.dataset.i)], text: input.value };
      state.wordsDirty = true;
    });
    tdText.appendChild(input);

    const tdPos = document.createElement('td');
    tdPos.className = 'pos';
    tdPos.textContent = `${w.left},${w.top},${w.right},${w.bottom}`;

    const tdH = document.createElement('td');
    tdH.className = 'pos';
    tdH.textContent = String(w.lineHeight);

    tr.append(tdText, tdPos, tdH);
    tbody.appendChild(tr);
  });
}

async function saveWordsJson() {
  const f = selectedFile();
  if (!f || !f.words) return;
  const r = await API.saveText({
    defaultName: f.name.replace(/\.[^.]+$/, '') + '-words.json',
    content: JSON.stringify(f.words, null, 1),
  });
  if (r && r.ok) { toast('词条 JSON 已保存'); state.wordsDirty = false; }
}

async function regeneratePdf() {
  const f = selectedFile();
  if (!f || !f.words || !f.result) return;
  const outPath = await API.savePdf();
  if (!outPath) return;
  const payload = {
    pagePath: f.result.page,
    outPath,
    width: f.words.width,
    height: f.words.height,
    words: f.words.words.map((w) => ({
      text: w.text, left: w.left, top: w.top,
      right: w.right, bottom: w.bottom,
      lineHeight: w.lineHeight, lineBottom: w.lineBottom,
    })),
  };
  toast('正在生成 PDF…');
  const r = await API.ocrRegeneratePdf(payload);
  if (r && r.ok) {
    toast(`PDF 已生成 (${r.words} 词条)`);
    log(`PDF 已生成: ${r.out}`, 'ok');
    state.wordsDirty = false;
  } else {
    toast(r && r.error ? r.error : 'PDF 生成失败', true);
    log(`PDF 生成失败: ${r ? r.error : '未知错误'}`, 'error');
  }
}

// ---------------------------------------------------------------------------
init().then(() => {
  if (isE2E()) runE2E();
});
