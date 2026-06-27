import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import { GymMetricEngine } from "./engine.js";

// ─── DOM Elements ──────────────────────────────────────────
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const repCountEl = document.getElementById("rep-count");
const feedbackEl = document.getElementById("form-feedback");
const exerciseSelect = document.getElementById("exercise-select");
const loadingOverlay = document.getElementById("loading-overlay");

// ─── State ─────────────────────────────────────────────────
let poseLandmarker = null;
let lastVideoTime = -1;
let animationFrameId = null;
let currentFacingMode = "user";
let currentZoom = 1;
let currentStream = null;

const engine = new GymMetricEngine();
const videoWrapper = document.getElementById("video-wrapper");
const btnFlipCam = document.getElementById("btn-flip-cam");
const btnZoomIn = document.getElementById("btn-zoom-in");
const btnZoomOut = document.getElementById("btn-zoom-out");
const zoomControls = document.getElementById("zoom-controls");

// Skeleton connections for drawing (MediaPipe Pose 33 landmarks)
const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// ─── Reset engine on exercise change ──────────────────────
exerciseSelect.addEventListener("change", () => {
  engine.repCount = 0;
  engine.state = "START";
  engine.filters = {};
  repCountEl.textContent = "0";
  feedbackEl.textContent = "Switched — start moving!";
  feedbackEl.className = "value has-good";
});

// ─── Initialize MediaPipe PoseLandmarker ──────────────────
async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  loadingOverlay.classList.add("hidden");
}

// ─── Update View Transform ──────────────────────────────────
function updateViewTransform() {
  // Front camera needs mirroring (scaleX(-1))
  if (currentFacingMode === "user") {
    videoWrapper.style.transform = `scaleX(-1) scale(${currentZoom})`;
  } else {
    videoWrapper.style.transform = `scale(${currentZoom})`;
  }
}

// ─── Start Webcam ─────────────────────────────────────────
async function startCamera(facingMode = "user") {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
  }

  currentFacingMode = facingMode;
  currentZoom = 1; // Reset zoom on camera switch
  updateViewTransform();

  // Show zoom controls only for back camera (optional, but good UX)
  if (currentFacingMode === "environment") {
    zoomControls.classList.add("active");
  } else {
    zoomControls.classList.remove("active");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: currentFacingMode },
    audio: false,
  });
  
  currentStream = stream;
  videoEl.srcObject = stream;

  return new Promise((resolve) => {
    videoEl.onloadeddata = () => {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
      resolve();
    };
  });
}

// ─── Controls Event Listeners ─────────────────────────────
btnFlipCam.addEventListener("click", async () => {
  const newMode = currentFacingMode === "user" ? "environment" : "user";
  loadingOverlay.classList.remove("hidden");
  loadingOverlay.querySelector("p").textContent = "Switching camera...";
  try {
    await startCamera(newMode);
  } catch (err) {
    console.error("Camera flip failed:", err);
  }
  loadingOverlay.classList.add("hidden");
});

btnZoomIn.addEventListener("click", () => {
  currentZoom = Math.min(currentZoom + 0.2, 3.0);
  updateViewTransform();
});

btnZoomOut.addEventListener("click", () => {
  currentZoom = Math.max(currentZoom - 0.2, 1.0);
  updateViewTransform();
});

// ─── Draw Skeleton ────────────────────────────────────────
function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (!landmarks || landmarks.length === 0) return;

  const w = canvasEl.width;
  const h = canvasEl.height;

  // Draw connections
  ctx.strokeStyle = "rgba(108, 92, 231, 0.7)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [i, j] of POSE_CONNECTIONS) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (a.visibility < 0.5 || b.visibility < 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.stroke();
  }

  // Draw keypoints
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (lm.visibility < 0.5) continue;
    const x = lm.x * w;
    const y = lm.y * h;

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(108, 92, 231, 0.3)";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#a29bfe";
    ctx.fill();
  }
}

// ─── Detection Loop ───────────────────────────────────────
function detectFrame() {
  if (!poseLandmarker || videoEl.readyState < 2) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  const currentTime = videoEl.currentTime;
  if (currentTime === lastVideoTime) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }
  lastVideoTime = currentTime;

  const startMs = performance.now();
  const result = poseLandmarker.detectForVideo(videoEl, startMs);

  if (result.landmarks && result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];

    // Draw skeleton
    drawSkeleton(landmarks);

    // Evaluate with engine
    const activeExercise = exerciseSelect.value;
    const evaluation = engine.evaluateFrame(landmarks, activeExercise);

    // Update UI
    repCountEl.textContent = evaluation.reps;

    if (evaluation.feedback && evaluation.feedback.length > 0) {
      feedbackEl.textContent = evaluation.feedback;
      feedbackEl.className = "value has-warning";
    } else {
      feedbackEl.textContent = "Good form — keep going! 💪";
      feedbackEl.className = "value has-good";
    }
  } else {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    feedbackEl.textContent = "No pose detected — step into frame";
    feedbackEl.className = "value";
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// ─── Bootstrap ────────────────────────────────────────────
async function main() {
  try {
    await Promise.all([initPoseLandmarker(), startCamera()]);
    feedbackEl.textContent = "Ready — start your exercise!";
    feedbackEl.className = "value has-good";
    detectFrame();
  } catch (err) {
    console.error("Initialization failed:", err);
    loadingOverlay.querySelector("p").textContent =
      "Error: " + err.message;
    loadingOverlay.querySelector(".spinner").style.display = "none";
  }
}

main();
