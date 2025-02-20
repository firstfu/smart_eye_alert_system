// DOM 元素
const videoFeed = document.getElementById("videoFeed");
const alertOverlay = document.getElementById("alertOverlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const systemStatus = document.getElementById("systemStatus");
const alertThreshold = document.getElementById("alertThreshold");
const sensitivity = document.getElementById("sensitivity");

// 系統狀態
let isMonitoring = false;
let stream = null;
let lastEyesOpenTime = Date.now();
let eyesClosedDuration = 0;

// 初始化攝影機
async function initializeCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
    videoFeed.srcObject = stream;
    updateSystemStatus("就緒");
  } catch (error) {
    console.error("無法存取攝影機:", error);
    updateSystemStatus("錯誤：無法存取攝影機");
  }
}

// 更新系統狀態
function updateSystemStatus(status) {
  systemStatus.textContent = status;
}

// 開始監測
function startMonitoring() {
  if (!stream) {
    alert("請先允許存取攝影機");
    return;
  }

  isMonitoring = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  updateSystemStatus("監測中");

  // 這裡將來會加入眼睛偵測的邏輯
  simulateEyeDetection();
}

// 停止監測
function stopMonitoring() {
  isMonitoring = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateSystemStatus("已停止");
  alertOverlay.classList.remove("active");
}

// 模擬眼睛偵測（示範用）
function simulateEyeDetection() {
  if (!isMonitoring) return;

  const threshold = alertThreshold.value * 1000; // 轉換為毫秒
  const currentTime = Date.now();

  // 模擬隨機檢測（實際應用中會替換為真實的眼睛偵測邏輯）
  const eyesClosed = Math.random() > 0.7;

  if (eyesClosed) {
    eyesClosedDuration = currentTime - lastEyesOpenTime;
    if (eyesClosedDuration > threshold) {
      showAlert();
    }
  } else {
    lastEyesOpenTime = currentTime;
    hideAlert();
  }

  // 持續監測
  setTimeout(simulateEyeDetection, 100);
}

// 顯示警示
function showAlert() {
  alertOverlay.classList.add("active");
  // 可以在這裡加入聲音警示
}

// 隱藏警示
function hideAlert() {
  alertOverlay.classList.remove("active");
}

// 事件監聽器
startBtn.addEventListener("click", startMonitoring);
stopBtn.addEventListener("click", stopMonitoring);

// 初始化
initializeCamera();
