import { CameraService } from './camera.js';
import { YoloWorker } from './inference.js';
import { KeypointFilter, RepCounter, jointAngleDeg, normaliseAngle, Phase } from './metrics.js';

// DOM Elements
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('render-canvas');
const ctx = canvasEl.getContext('2d');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const repCountEl = document.getElementById('rep-count');
const angleDisplayEl = document.getElementById('angle-display');
const phaseDisplayEl = document.getElementById('phase-display');
const repCounterPanel = document.getElementById('rep-counter-panel');

// State
let isRunning = false;
const camera = new CameraService(videoEl, canvasEl);
const yolo = new YoloWorker();
const filters = Array.from({ length: 17 }, () => new KeypointFilter());
const repCounter = new RepCounter(0.08, 0.08, 0.0, 1.0); // thresholds for bicep curl

// Exercise config (Bicep Curl)
const config = {
  activeJointA: 6, // Right Shoulder
  activeJointB: 8, // Right Elbow
  activeJointC: 10, // Right Wrist
  minAngleDeg: 30.0,
  maxAngleDeg: 160.0
};

// Skeleton pairs for rendering
const skeletonEdges = [
  [15,13], [13,11], [16,14], [14,12], [11,12], [5,11], [6,12], [5,6], [5,7],
  [6,8], [7,9], [8,10], [1,2], [0,1], [0,2], [1,3], [2,4], [3,5], [4,6]
];

function setStatus(text, type) {
  statusText.textContent = text;
  statusBadge.className = `badge ${type}`;
}

async function init() {
  setStatus('STARTING CAMERA...', 'warning');
  const camStarted = await camera.start();
  if (!camStarted) {
    setStatus('CAMERA ERROR', 'error');
    return;
  }

  setStatus('LOADING AI...', 'warning');
  const aiReady = await yolo.init('/models/yolo11n-pose_int8.onnx');
  if (!aiReady) {
    setStatus('MODEL ERROR', 'error');
    return;
  }

  setStatus('TRACKING', 'ok');
  isRunning = true;
  requestAnimationFrame(drawLoop);
  inferenceLoop();
}

let lastPoses = null;
let isInferencing = false;

function drawLoop() {
  if (!isRunning) return;

  // 1. Draw Camera Feed at 60 FPS
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  // 2. Draw latest skeleton state (even if AI is calculating next frame)
  if (lastPoses && lastPoses.length > 0) {
    const primaryPose = lastPoses[0];
    
    // Filter Keypoints
    const filteredKeypoints = primaryPose.keypoints.map((kp, i) => {
      if (kp.confidence > 0.3) {
        const { x, y } = filters[i].filter(kp.x, kp.y);
        return { x, y, confidence: kp.confidence };
      }
      return kp;
    });

    drawSkeleton(filteredKeypoints, canvasEl.width, canvasEl.height);
    analyzeForm(filteredKeypoints);
  } else {
    setStatus('LOST TARGET', 'error');
  }

  requestAnimationFrame(drawLoop);
}

async function inferenceLoop() {
  if (!isRunning) return;

  if (!isInferencing) {
    isInferencing = true;
    const frameData = camera.getFrameData();
    if (frameData) {
      lastPoses = await yolo.runInference(frameData);
    }
    isInferencing = false;
  }

  // Yield to browser to prevent freezing the UI thread
  setTimeout(inferenceLoop, 10);
}

function drawSkeleton(keypoints, width, height) {
  // Scale from [0, 640] normalized YOLO output space to screen width/height
  const scaleX = width / yolo.inputSize;
  const scaleY = height / yolo.inputSize;

  ctx.lineWidth = 4;
  ctx.lineCap = 'round';

  // Draw Edges
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.6)';
  for (const edge of skeletonEdges) {
    const kp1 = keypoints[edge[0]];
    const kp2 = keypoints[edge[1]];

    if (kp1.confidence > 0.3 && kp2.confidence > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
      ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
      ctx.stroke();
    }
  }

  // Draw Points
  ctx.fillStyle = '#00e5ff';
  for (const kp of keypoints) {
    if (kp.confidence > 0.3) {
      ctx.beginPath();
      ctx.arc(kp.x * scaleX, kp.y * scaleY, 6, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function analyzeForm(keypoints) {
  const kpA = keypoints[config.activeJointA];
  const kpB = keypoints[config.activeJointB];
  const kpC = keypoints[config.activeJointC];

  if (kpA.confidence < 0.3 || kpB.confidence < 0.3 || kpC.confidence < 0.3) {
    setStatus('JOINTS OBSCURED', 'warning');
    return;
  }

  setStatus('TRACKING', 'ok');

  const angle = jointAngleDeg(kpA, kpB, kpC);
  const normSignal = normaliseAngle(angle, config.minAngleDeg, config.maxAngleDeg);

  const prevCount = repCounter.repCount;
  repCounter.update(normSignal);

  // Update HUD
  angleDisplayEl.textContent = `${angle.toFixed(1)}°`;
  phaseDisplayEl.textContent = repCounter.phase;
  
  phaseDisplayEl.className = 'metric-value';
  if (repCounter.phase === Phase.ASCENDING) phaseDisplayEl.classList.add('phase-ascending');
  else if (repCounter.phase === Phase.DESCENDING) phaseDisplayEl.classList.add('phase-descending');
  else phaseDisplayEl.classList.add('phase-idle');

  if (repCounter.repCount > prevCount) {
    repCountEl.textContent = repCounter.repCount;
    // Trigger CSS animation
    repCounterPanel.classList.remove('bump');
    void repCounterPanel.offsetWidth; // trigger reflow
    repCounterPanel.classList.add('bump');
  }
}

// Start
init();
