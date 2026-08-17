/*
 * FairScan Desktop preload：通过 contextBridge 暴露安全 API。
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('fairscan', {
  openFiles: () => ipcRenderer.invoke('dialog:openFiles'),
  openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
  listDirImages: (dir, recursive) => ipcRenderer.invoke('dir:listImages', dir, recursive),
  savePdf: () => ipcRenderer.invoke('dialog:savePdf'),
  saveText: (payload) => ipcRenderer.invoke('fs:saveText', payload),
  cliInfo: (overrideDir) => ipcRenderer.invoke('env:cliInfo', overrideDir),
  defaultOutDir: () => ipcRenderer.invoke('env:defaultOutDir'),
  scanStart: (config) => ipcRenderer.invoke('scan:start', config),
  scanStop: () => ipcRenderer.invoke('scan:stop'),
  ocrRegeneratePdf: (payload) => ipcRenderer.invoke('ocr:regeneratePdf', payload),
  showItem: (p) => ipcRenderer.invoke('shell:showItem', p),
  openPath: (p) => ipcRenderer.invoke('shell:openPath', p),
  saveCameraCapture: (dataUrl) => ipcRenderer.invoke('camera:saveCapture', dataUrl),
  cameraDetect: (payload) => ipcRenderer.invoke('camera:detect', payload),
  quit: () => ipcRenderer.invoke('app:quit'),
  winMinimize: () => ipcRenderer.invoke('win:minimize'),
  winToggleMaximize: () => ipcRenderer.invoke('win:maximize'),
  winClose: () => ipcRenderer.invoke('win:close'),
  onWinState: (cb) => ipcRenderer.on('win:state', (_e, s) => cb(s)),
  onScanLine: (cb) => ipcRenderer.on('scan:line', (_e, obj) => cb(obj)),
  onScanLog: (cb) => ipcRenderer.on('scan:log', (_e, obj) => cb(obj)),
  onScanExit: (cb) => ipcRenderer.on('scan:exit', (_e, code) => cb(code)),
});
