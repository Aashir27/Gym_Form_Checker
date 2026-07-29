import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { GymMetricEngine } from "./engine.js";

// ─── DOM Elements ──────────────────────────────────────────
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const repCountEl = document.getElementById("rep-count");
const repLabelEl = document.getElementById("rep-label");
const feedbackEl = document.getElementById("form-feedback");
const exerciseSelect = document.getElementById("exercise-select");
const colorPicker = document.getElementById("skeleton-color");
const loadingOverlay = document.getElementById("loading-overlay");
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

/**
 * Keep the drawing buffer aligned with the element's displayed size. The
 * camera frame itself is projected in drawSkeleton() using the same
 * object-fit: cover calculation as the video element.
 */
function resizeOverlay() {
  const bounds = canvasEl.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.round(bounds.width * pixelRatio);
  const height = Math.round(bounds.height * pixelRatio);

  if (canvasEl.width !== width || canvasEl.height !== height) {
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
  feedbackEl.textContent = "Switched; get into position!";
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

  if (currentFacingMode === "environment") {
    zoomControls.classList.add("active");
  } else {
    zoomControls.classList.remove("active");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1920, min: 1280 },
      height: { ideal: 1080, min: 720 },
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
  const videoScale = Math.max(
    canvasWidth / videoEl.videoWidth,
    canvasHeight / videoEl.videoHeight
  );
  const renderedVideoWidth = videoEl.videoWidth * videoScale;
  const renderedVideoHeight = videoEl.videoHeight * videoScale;
  const offsetX = (canvasWidth - renderedVideoWidth) / 2;
  const offsetY = (canvasHeight - renderedVideoHeight) / 2;
  return {
    x: offsetX + point.x * renderedVideoWidth,
    y: offsetY + point.y * renderedVideoHeight,
  };
}

function drawGuideLines(guideLines) {
  if (!guideLines || guideLines.length === 0) return;

  ctx.save();
  ctx.strokeStyle = "#00e676";
  ctx.fillStyle = "#00e676";
  ctx.shadowColor = "rgba(0, 230, 118, 0.45)";
  ctx.shadowBlur = 14;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";

  for (const guide of guideLines) {
    if (!guide?.from || !guide?.to) continue;
    const from = projectPoint(guide.from, canvasEl.width, canvasEl.height);
    const to = projectPoint(guide.to, canvasEl.width, canvasEl.height);
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(to.x, to.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.restore();
}

function drawSkeleton(landmarks, displayedLandmarks, guideLines = [], formOk = true) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (!landmarks || landmarks.length === 0 || !videoEl.videoWidth || !videoEl.videoHeight) return;

  const visible = new Set(displayedLandmarks);
  const project = (point) => projectPoint(point, canvasEl.width, canvasEl.height);
  const baseColor = colorPicker.value;
  const connectionColor = formOk ? baseColor : "#ff5252";

  // Draw connections
  ctx.strokeStyle = connectionColor + "b3";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [i, j] of POSE_CONNECTIONS) {
    if (!visible.has(i) || !visible.has(j)) continue;
    const a = landmarks[i];
    const b = landmarks[j];
    if (a.visibility < 0.5 || b.visibility < 0.5) continue;
    const start = project(a);
    const end = project(b);
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

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = connectionColor + "4d";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = connectionColor;
    ctx.fill();
  }

  if (!formOk) {
    drawGuideLines(guideLines);
  }
}

/** Format seconds into mm:ss display. */
function formatHoldTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Draw calibration overlay UI on the canvas.
 * - During push_up / plank exercises, shows:
 *   - Orientation-warning prompt ("Please turn side-on…")
 *   - 3-second progress bar + "Get into your best … position and hold still"
 *   - Rejection banner + restart hint
 */
function drawCalibrationOverlay(calState, activeExercise) {
  if (!calState || calState.state === "calibrated" || (activeExercise !== "push_up" && activeExercise !== "plank")) {
    return;
  }

  const w = canvasEl.width;
  const h = canvasEl.height;
  if (!w || !h) return;

  const cx = w / 2;
  const barOuterW = Math.min(w * 0.55, 480);
  const barOuterH = 22;
  const barY = h / 2 - 20;

  ctx.save();

  // Backdrop (subtle dark box behind text
  ctx.fillStyle = "rgba(10, 10, 20, 0.55)";
  const padX = 28;
  const padY = 22;
  const boxH = 150;
  const boxW = Math.max(barOuterW + padX * 2, 460);
  const boxX = cx - boxW / 2;
  const boxY = barY - 60;
  ctx.beginPath();
  roundRect(ctx, boxX, boxY, boxW, boxH, 18);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Title
  const msg = calState.rejectionReason || calState.prompt || "Calibrating…";
  const isRejection = !!calState.rejectionReason;
  const isOrient = calState.state === "orientation_check";
  ctx.font = `600 ${Math.max(14, Math.round(w * 0.02))}px Inter, -apple-system, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if (isRejection) {
    ctx.fillStyle = "#ffab40";
    ctx.shadowColor = "rgba(255,171,64,0.35)";
    ctx.shadowBlur = 16;
  } else if (isOrient) {
    ctx.fillStyle = "#ffab40";
    ctx.shadowColor = "rgba(255,171,64,0.3)";
    ctx.shadowBlur = 12;
  } else {
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = "rgba(108,92,231,0.3)";
    ctx.shadowBlur = 10;
  }
  ctx.fillText(msg, cx, barY - 28);

  // Subtitle
  ctx.shadowBlur = 0;
  ctx.font = `500 ${Math.max(11, Math.round(w * 0.014))}px Inter, -apple-system, sans-serif`;
  if (isRejection) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillText("Window will restart automatically", cx, barY - 2);
  } else if (isOrient) {
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("Side profile view needed for accurate 3D form tracking", cx, barY - 2);
  } else {
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("Session will begin automatically", cx, barY - 2);
  }

  // Progress bar background
  const barX = cx - barOuterW / 2;
  const barR = barOuterH / 2;
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath();
  roundRect(ctx, barX, barY, barOuterW, barOuterH, barR);
  ctx.fill();

  // Progress fill
  const progress = typeof calState.progressPct === "number" ? calState.progressPct : 0;
  const innerW = Math.max(0, Math.min(1, progress) * barOuterW);
  const grad = ctx.createLinearGradient(barX, 0, barX + barOuterW, 0);
  if (isRejection || isOrient) {
    grad.addColorStop(0, "#ffab40");
    grad.addColorStop(1, "#ff5252");
  } else {
    grad.addColorStop(0, "#6c5ce7");
    grad.addColorStop(0.5, "#00e676");
    grad.addColorStop(1, "#69f0ae");
  }
  ctx.fillStyle = grad;
  ctx.beginPath();
  roundRect(ctx, barX, barY, innerW, barOuterH, barR);
  ctx.fill();

  // Percentage label on bar
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.font = `700 ${Math.max(10, Math.round(w * 0.012))}px Inter, sans-serif`;
  const pctLabel = Math.round(progress * 100) + "%";
  ctx.fillText(pctLabel, cx, barY + barOuterH / 2);

  // Tag row (Hip/neck baseline preview
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = `500 ${Math.max(10, Math.round(w * 0.012))}px Inter, sans-serif`;
  ctx.fillText("3D world-landmark calibration • personal baseline", cx, barY + barOuterH + 20);

  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, r, r, Math.PI + Math.PI / 2);
  ctx.arcTo(x + w, y + h, r, Math.PI / 2, 0);
  ctx.arcTo(x, y + h, r, 0, -Math.PI / 2);
  ctx.arcTo(x, y, r, -Math.PI / 2, Math.PI);
  ctx.closePath();
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

    // Evaluate with engine
    const activeExercise = exerciseSelect.value;
    const evaluation = engine.evaluateFrame(landmarks, activeExercise, worldLandmarks);

    // Draw only joints used by the selected exercise. This makes the overlay
    // easier to read and avoids presenting unrelated limbs as active inputs.
    const calState = evaluation.calibrationState;
    const isInCalibration = calState && calState.state !== "calibrated" && (activeExercise === "push_up" || activeExercise === "plank");

    drawSkeleton(
      landmarks,
      GymMetricEngine.getDisplayLandmarks(activeExercise, evaluation.activeArm),
      evaluation.formGuideLines,
      isInCalibration ? true : evaluation.formOk
    );

    drawCalibrationOverlay(calState, activeExercise);

    // Update rep count or hold time
    if (!isInCalibration) {
      if (evaluation.holdTime !== null) {
        repCountEl.textContent = formatHoldTime(evaluation.holdTime);
      } else {
        repCountEl.textContent = evaluation.reps;
      }
    } else {
      if (activeExercise === "plank") {
        repCountEl.textContent = formatHoldTime(0);
      } else {
        repCountEl.textContent = evaluation.reps;
      }
    }

    // Update phase indicator
    if (phaseIndicatorEl && evaluation.phase) {
      const phaseLabel = evaluation.phase.replace(/_/g, " ").toLowerCase();
      const curlAngle = activeExercise === "bicep_curl" && Number.isFinite(evaluation.primaryAngle)
        ? ` · ${evaluation.primaryAngle}°`
        : "";
      if (isInCalibration) {
        phaseIndicatorEl.textContent = "";
      } else {
        phaseIndicatorEl.textContent = phaseLabel + curlAngle;
      }
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

    // Update feedback with type-aware styling
    if (isInCalibration && calState) {
      const msg = calState.rejectionReason || calState.prompt || "Calibrating…";
      if (calState.rejectionReason) {
        feedbackEl.textContent = msg;
        feedbackEl.className = "value has-warning";
      } else if (calState.state === "orientation_check") {
        feedbackEl.textContent = msg;
        feedbackEl.className = "value has-warning";
      } else {
        feedbackEl.textContent = msg;
        feedbackEl.className = "value has-good";
      }
    } else if ((activeExercise === "push_up" || activeExercise === "plank") && evaluation.formIssues && evaluation.formIssues.length > 0) {
      const correctionMsg = evaluation.formIssues.length === 1
        ? evaluation.formIssues[0]
        : evaluation.formIssues.join(" + ");
      feedbackEl.textContent = `Fix form: ${correctionMsg}`;
      feedbackEl.className = "value has-warning";
    } else if (evaluation.feedback && evaluation.feedback.length > 0) {
      feedbackEl.textContent = evaluation.feedback;
      if (evaluation.feedbackType === "warning") {
        feedbackEl.className = "value has-warning";
      } else {
        feedbackEl.className = "value has-good";
      }
    } else {
      feedbackEl.textContent = "Good form; keep going! 💪";
      feedbackEl.className = "value has-good";
    }
  } else {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    drawCalibrationOverlay(null, "");
    feedbackEl.textContent = "No pose detected; step into frame";
    feedbackEl.className = "value";
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

// ─── Bootstrap ────────────────────────────────────────────
async function main() {
  try {
    await Promise.all([initPoseLandmarker(), startCamera()]);
    feedbackEl.textContent = "Ready; start your exercise!";
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
