import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { GymMetricEngine, buildSquatCorrectionGuides } from "./engine.js";

// ─── DOM Elements ──────────────────────────────────────────
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const repCountEl = document.getElementById("rep-count");
const repLabelEl = document.getElementById("rep-label");
const feedbackEl = document.getElementById("form-feedback");
const exerciseSelect = document.getElementById("exercise-select");
const loadingOverlay = document.getElementById("loading-overlay");
const SKELETON_COLOR = "#ffffff";
const BAD_LANDMARK_COLOR = "#ff5252";
const phaseIndicatorEl = document.getElementById("phase-indicator");
const angleIndicatorEl = document.getElementById("angle-indicator");

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

function isMobileDevice() {
  return window.matchMedia("(pointer: coarse)").matches || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

async function lockLandscapeIfPossible() {
  if (!isMobileDevice()) return;
  const orientation = screen.orientation;
  if (!orientation || orientation.type?.startsWith("landscape")) return;
  if (typeof orientation.lock !== "function") return;

  try {
    await orientation.lock("landscape");
  } catch {
    // Ignore browsers that require fullscreen or do not support locking.
  }
}

/**
 * Keep the drawing buffer aligned with the element's displayed size. The
 * camera frame itself is projected in drawSkeleton() using the same
 * object-fit: cover calculation as the video element.
 */
function resizeOverlay() {
  // Use layout size (pre-CSS-transform) so zoom/mirror don't desync the buffer.
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.round(videoWrapper.clientWidth * pixelRatio);
  const height = Math.round(videoWrapper.clientHeight * pixelRatio);

  if (width > 0 && height > 0 && (canvasEl.width !== width || canvasEl.height !== height)) {
    canvasEl.width = width;
    canvasEl.height = height;
  }
}

new ResizeObserver(resizeOverlay).observe(videoWrapper);

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

// ─── Populate exercise selector from engine profiles ──────
function populateExerciseSelect() {
  exerciseSelect.innerHTML = "";
  for (const key of GymMetricEngine.exerciseKeys) {
    const profile = GymMetricEngine.getProfile(key);
    const option = document.createElement("option");
    option.value = key;
    option.textContent = profile.label;
    exerciseSelect.appendChild(option);
  }
}
populateExerciseSelect();

// ─── Reset engine on exercise change ──────────────────────
exerciseSelect.addEventListener("change", () => {
  engine.reset();
  repCountEl.textContent = "0";
  updateRepLabel();
  feedbackEl.textContent = "Switched, get into position!";
  feedbackEl.className = "value has-good";
  if (phaseIndicatorEl) phaseIndicatorEl.textContent = "";
  if (angleIndicatorEl) {
    angleIndicatorEl.textContent = "";
    angleIndicatorEl.className = "angle-indicator";
  }
});

/** Update the rep/hold label based on current exercise type. */
function updateRepLabel() {
  if (!repLabelEl) return;
  const profile = GymMetricEngine.getProfile(exerciseSelect.value);
  repLabelEl.textContent = profile && profile.type === "hold" ? "Hold Time" : "Reps";
}
updateRepLabel();

// ─── Initialize MediaPipe PoseLandmarker ──────────────────
async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
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
  currentZoom = 1;
  updateViewTransform();

  await lockLandscapeIfPossible();

  if (currentFacingMode === "environment") {
    zoomControls.classList.add("active");
  } else {
    zoomControls.classList.remove("active");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1920, min: 1280 },
      height: { ideal: 1080, min: 720 },
      aspectRatio: { ideal: 16 / 9 },
      resizeMode: "crop-and-scale",
      facingMode: currentFacingMode,
    },
    audio: false,
  });

  currentStream = stream;
  videoEl.srcObject = stream;

  return new Promise((resolve) => {
    videoEl.onloadeddata = () => {
      resizeOverlay();
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
function projectPoint(point, canvasWidth, canvasHeight) {
  // Match object-fit: cover using the video's layout box, then scale into the
  // canvas backing store. Avoids skeleton drift when the wrapper is mirrored/zoomed.
  const layoutW = videoEl.clientWidth || canvasWidth;
  const layoutH = videoEl.clientHeight || canvasHeight;
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (!vw || !vh || !layoutW || !layoutH) return { x: 0, y: 0 };

  const videoScale = Math.max(layoutW / vw, layoutH / vh);
  const renderedVideoWidth = vw * videoScale;
  const renderedVideoHeight = vh * videoScale;
  const offsetX = (layoutW - renderedVideoWidth) / 2;
  const offsetY = (layoutH - renderedVideoHeight) / 2;
  const sx = canvasWidth / layoutW;
  const sy = canvasHeight / layoutH;
  return {
    x: (offsetX + point.x * renderedVideoWidth) * sx,
    y: (offsetY + point.y * renderedVideoHeight) * sy,
  };
}

function drawGuideLines(guideLines) {
  if (!guideLines || guideLines.length === 0) return;

  ctx.save();
  ctx.strokeStyle = "#00e676";
  ctx.fillStyle = "#00e676";
  ctx.shadowColor = "rgba(0, 230, 118, 0.55)";
  ctx.shadowBlur = 16;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const guide of guideLines) {
    if (!guide?.from || !guide?.to) continue;
    if (!Number.isFinite(guide.from.x) || !Number.isFinite(guide.from.y)) continue;
    if (!Number.isFinite(guide.to.x) || !Number.isFinite(guide.to.y)) continue;
    const from = projectPoint(guide.from, canvasEl.width, canvasEl.height);
    const to = projectPoint(guide.to, canvasEl.width, canvasEl.height);
    if (!Number.isFinite(from.x) || !Number.isFinite(to.x)) continue;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(to.x, to.y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.restore();
}

function drawSkeleton(landmarks, displayedLandmarks, guideLines = [], formOk = true, badLandmarks = []) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (!landmarks || landmarks.length === 0 || !videoEl.videoWidth || !videoEl.videoHeight) return;

  const visible = new Set(displayedLandmarks);
  const badSet = new Set(badLandmarks || []);
  const project = (point) => projectPoint(point, canvasEl.width, canvasEl.height);

  const colorFor = (indices) => {
    // Selective red for specific bad joints (e.g. curl arm); otherwise whole-body red on form fail.
    if (badSet.size > 0) {
      return indices.some((idx) => badSet.has(idx)) ? BAD_LANDMARK_COLOR : SKELETON_COLOR;
    }
    return formOk ? SKELETON_COLOR : BAD_LANDMARK_COLOR;
  };

  // Draw connections
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [i, j] of POSE_CONNECTIONS) {
    if (!visible.has(i) || !visible.has(j)) continue;
    const a = landmarks[i];
    const b = landmarks[j];
    if (a.visibility < 0.5 || b.visibility < 0.5) continue;
    const color = colorFor([i, j]);
    const start = project(a);
    const end = project(b);
    ctx.strokeStyle = color + "b3";
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }

  // Draw keypoints
  for (let i = 0; i < landmarks.length; i++) {
    if (!visible.has(i)) continue;
    const lm = landmarks[i];
    if (lm.visibility < 0.5) continue;
    const { x, y } = project(lm);
    const color = colorFor([i]);

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = color + "4d";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }

  if (guideLines && guideLines.length > 0) {
    drawGuideLines(guideLines);
  }
}

/** Format seconds into mm:ss display. */
function formatHoldTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
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
  let result;
  try {
    result = poseLandmarker.detectForVideo(videoEl, startMs);
  } catch (error) {
    console.warn("Pose detection skipped frame due to error:", error);
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  if (result.landmarks && result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];
    const worldLandmarks = result.worldLandmarks && result.worldLandmarks.length > 0
      ? result.worldLandmarks[0]
      : null;

    // Evaluate with engine — never let a form-check error kill the skeleton loop.
    const activeExercise = exerciseSelect.value;
    let evaluation;
    try {
      evaluation = engine.evaluateFrame(landmarks, activeExercise, worldLandmarks);
    } catch (error) {
      console.warn("Form evaluation skipped frame due to error:", error);
      evaluation = {
        reps: engine.repCount,
        feedback: "Tracking hiccup, keep going",
        feedbackType: "info",
        holdTime: null,
        phase: "",
        cameraAngle: null,
        angleOk: true,
        activeArm: engine._activeArm,
        primaryAngle: null,
        formOk: true,
        formIssues: [],
        formGuideLines: [],
        badLandmarks: [],
        calibrationState: null,
      };
    }

    // Draw only joints used by the selected exercise. This makes the overlay
    // easier to read and avoids presenting unrelated limbs as active inputs.

    // Squat: always rebuild green legs here from the drawn landmarks so they
    // cannot be dropped by the engine path. Same skeleton joints, slid apart.
    let guideLines = evaluation.formGuideLines || [];
    const badLandmarks = evaluation.badLandmarks || [];
    if (activeExercise === "squat" && badLandmarks.length > 0) {
      const squatGuides = buildSquatCorrectionGuides(landmarks);
      if (squatGuides.length > 0) guideLines = squatGuides;
    }

    drawSkeleton(
      landmarks,
      GymMetricEngine.getDisplayLandmarks(activeExercise, evaluation.activeArm),
      guideLines,
      evaluation.formOk,
      badLandmarks
    );

    // Update rep count or hold time
    if (evaluation.holdTime !== null) {
      repCountEl.textContent = formatHoldTime(evaluation.holdTime);
    } else {
      repCountEl.textContent = evaluation.reps;
    }

    // Update phase indicator
    if (phaseIndicatorEl && evaluation.phase) {
      const phaseLabel = evaluation.phase.replace(/_/g, " ").toLowerCase();
      const curlAngle = activeExercise === "bicep_curl" && Number.isFinite(evaluation.primaryAngle)
        ? ` · ${evaluation.primaryAngle}°`
        : "";
      phaseIndicatorEl.textContent = phaseLabel + curlAngle;
    }

    // Update camera angle indicator
    if (angleIndicatorEl && evaluation.cameraAngle) {
      const angleLabel = "Angle: " + evaluation.cameraAngle;
      angleIndicatorEl.textContent = angleLabel;
      if (evaluation.angleOk) {
        angleIndicatorEl.className = "angle-indicator is-ok";
      } else {
        angleIndicatorEl.className = "angle-indicator is-bad";
      }
    } else if (angleIndicatorEl) {
      angleIndicatorEl.textContent = "";
      angleIndicatorEl.className = "angle-indicator";
    }

    // Update feedback with type-aware styling.
    // Push-up / plank: trust the engine sticky coach (no formIssues override flicker).
    if (evaluation.feedback && evaluation.feedback.length > 0) {
      feedbackEl.textContent = evaluation.feedback;
      if (evaluation.feedbackType === "warning") {
        feedbackEl.className = "value has-warning";
      } else {
        feedbackEl.className = "value has-good";
      }
    } else {
      feedbackEl.textContent = "Good form, keep going! 💪";
      feedbackEl.className = "value has-good";
    }
  } else {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    feedbackEl.textContent = "No pose detected, step into frame";
    feedbackEl.className = "value";
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// ─── Bootstrap ────────────────────────────────────────────
async function main() {
  try {
    await Promise.all([initPoseLandmarker(), startCamera()]);
    feedbackEl.textContent = "Ready, start your exercise!";
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
