/**
 * EMAFilter. Exponential Moving Average for low-pass jitter smoothing.
 * Rock-steady smoothing with minimal overhead.
 */
export class EMAFilter {
  constructor(alpha = 0.5) {
    this.alpha = alpha;
    this.xPrev = null;
    this.yPrev = null;
  }

  filter(x, y) {
    if (this.xPrev === null || this.yPrev === null) {
      this.xPrev = x;
      this.yPrev = y;
      return { x, y };
    }
    const xHat = (this.alpha * x) + ((1 - this.alpha) * this.xPrev);
    const yHat = (this.alpha * y) + ((1 - this.alpha) * this.yPrev);
    this.xPrev = xHat;
    this.yPrev = yHat;
    return { x: xHat, y: yHat };
  }
}

/* ───────────────────────────────────────────────────────────────
 * MediaPipe PoseLandmarker index reference (33 landmarks)
 * ───────────────────────────────────────────────────────────────
 *  0  nose              11 left_shoulder     23 left_hip
 *  1  left_eye_inner    12 right_shoulder    24 right_hip
 *  2  left_eye          13 left_elbow        25 left_knee
 *  3  left_eye_outer    14 right_elbow       26 right_knee
 *  4  right_eye_inner   15 left_wrist        27 left_ankle
 *  5  right_eye         16 right_wrist       28 right_ankle
 *  6  right_eye_outer   17 left_pinky        29 left_heel
 *  7  left_ear          18 right_pinky       30 right_heel
 *  8  right_ear         19 left_index        31 left_foot_index
 *  9  mouth_left        20 right_index       32 right_foot_index
 * 10  mouth_right       21 left_thumb
 *                       22 right_thumb
 * ────────────────────────────────────────────────────────────── */

const LM = {
  NOSE: 0,
  L_EYE: 2,     R_EYE: 5,
  L_EAR: 7,     R_EAR: 8,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13,    R_ELBOW: 14,
  L_WRIST: 15,    R_WRIST: 16,
  L_HIP: 23,      R_HIP: 24,
  L_KNEE: 25,     R_KNEE: 26,
  L_ANKLE: 27,    R_ANKLE: 28,
  L_HEEL: 29,     R_HEEL: 30,
  L_FOOT: 31,     R_FOOT: 32,
};

/* ───────────────────────────────────────────────────────────────
 * Geometry helpers
 * ────────────────────────────────────────────────────────────── */

/** 
 * Scale-Invariant Vector Math using Math.atan2.
 * Calculates absolute internal angle between 0 and 180 degrees (2D).
 */
export function calculateAngle(a, b, c) {
  let radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * (180.0 / Math.PI));
  if (angle > 180.0) {
    angle = 360.0 - angle;
  }
  return angle;
}

/**
 * 3D world-landmark angle calculation.
 * Computes the absolute internal angle at vertex b formed by segments
 * a→b and b→c, using the full 3D vectors from MediaPipe worldLandmarks.
 * This uses the dot-product method which is inherently scale- and
 * position-invariant, making it robust to camera distance and pose.
 */
export function calculateAngle3D(a, b, c) {
  const ax = a.x - b.x, ay = a.y - b.y, az = a.z - b.z;
  const bx = c.x - b.x, by = c.y - b.y, bz = c.z - b.z;
  const dot = ax * bx + ay * by + az * bz;
  const magA = Math.hypot(ax, ay, az);
  const magB = Math.hypot(bx, by, bz);
  if (magA < 1e-8 || magB < 1e-8) return 0;
  const cos = clamp(dot / (magA * magB), -1, 1);
  return Math.acos(cos) * (180.0 / Math.PI);
}

/** 3D Euclidean distance between two world landmarks. */
function dist3D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** Midpoint of two 3D landmarks. */
function mid3D(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
  };
}

/** Signed tilt of segment a→b from vertical (positive = tilted right). */
function tiltFromVertical(a, b) {
  return Math.atan2(b.x - a.x, -(b.y - a.y)) * (180 / Math.PI);
}

/** Midpoint of two landmarks. */
function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, visibility: Math.min(a.visibility, b.visibility) };
}

/** Euclidean distance between two landmarks (normalised coords). */
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Project point p onto the line segment defined by a and b. */
function projectPointOnSegment(p, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = p.x - a.x;
  const apy = p.y - a.y;
  const abLenSq = abx * abx + aby * aby;
  let t = abLenSq > 0 ? (apx * abx + apy * aby) / abLenSq : 0;
  t = clamp(t, 0, 1);
  return { x: a.x + abx * t, y: a.y + aby * t };
}

/** Project point p onto the infinite line through a and b (no endpoint clamp). */
function projectPointOnLine(p, a, b) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = p.x - a.x;
  const apy = p.y - a.y;
  const abLenSq = abx * abx + aby * aby;
  const t = abLenSq > 0 ? (apx * abx + apy * aby) / abLenSq : 0;
  return { x: a.x + abx * t, y: a.y + aby * t, visibility: p.visibility ?? 1 };
}

/**
 * Signed hip offset from the shoulder→ankle body line, in % of image height.
 * Positive = hip below the line (toward floor / sag). Negative = hip above (pike).
 * Uses projected-line Y so a diagonal side-view plank is scored correctly
 * (unlike a simple average of shoulder/ankle Y).
 */
function hipOffsetFromBodyLine(shoulder, hip, ankle) {
  if (!shoulder || !hip || !ankle) return 0;
  const proj = projectPointOnLine(hip, shoulder, ankle);
  return (hip.y - proj.y) * 100;
}

/** Visibility check. */
function vis(pt, threshold = 0.4) {
  return pt && pt.visibility >= threshold;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function point(x, y, visibility = 1) {
  return { x, y, visibility };
}

function addPoint(a, b) {
  return point(a.x + b.x, a.y + b.y, Math.min(a.visibility ?? 1, b.visibility ?? 1));
}

function subtractPoint(a, b) {
  return point(a.x - b.x, a.y - b.y, Math.min(a.visibility ?? 1, b.visibility ?? 1));
}

function scalePoint(a, scalar) {
  return point(a.x * scalar, a.y * scalar, a.visibility ?? 1);
}

function lengthPoint(a) {
  return Math.hypot(a.x, a.y);
}

function normalizePoint(a) {
  const len = lengthPoint(a);
  if (len < 1e-6) return point(0, 0, a.visibility ?? 1);
  return point(a.x / len, a.y / len, a.visibility ?? 1);
}

function average(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getLandmark(points, idx) {
  const candidate = points[idx];
  return vis(candidate) ? candidate : null;
}

function selectVisibleSide(points, previousSide = null) {
  const scoreSide = (side) => {
    const shoulder = getLandmark(points, side === "left" ? LM.L_SHOULDER : LM.R_SHOULDER);
    const hip = getLandmark(points, side === "left" ? LM.L_HIP : LM.R_HIP);
    const knee = getLandmark(points, side === "left" ? LM.L_KNEE : LM.R_KNEE);
    const ankle = getLandmark(points, side === "left" ? LM.L_ANKLE : LM.R_ANKLE);
    const ear = getLandmark(points, side === "left" ? LM.L_EAR : LM.R_EAR);
    const nose = getLandmark(points, LM.NOSE);

    const bodyScore = [shoulder, hip, knee, ankle].reduce((sum, item) => sum + (item ? item.visibility : 0), 0);
    const headScore = ear ? ear.visibility * 1.1 : (nose ? nose.visibility * 0.4 : 0);
    return bodyScore * 1.2 + headScore;
  };

  const leftScore = scoreSide("left");
  const rightScore = scoreSide("right");

  if (leftScore === 0 && rightScore === 0) return null;
  // Stronger stickiness — left/right profile was flipping and breaking plank/push-up.
  if (previousSide === "left" && leftScore >= rightScore - 0.35) return "left";
  if (previousSide === "right" && rightScore >= leftScore - 0.35) return "right";
  return leftScore >= rightScore ? "left" : "right";
}

function pickSidePoints(points, side) {
  if (!side) return null;
  const isLeft = side === "left";
  return {
    side,
    head: getLandmark(points, isLeft ? LM.L_EAR : LM.R_EAR) || getLandmark(points, LM.NOSE),
    shoulder: getLandmark(points, isLeft ? LM.L_SHOULDER : LM.R_SHOULDER),
    elbow: getLandmark(points, isLeft ? LM.L_ELBOW : LM.R_ELBOW),
    wrist: getLandmark(points, isLeft ? LM.L_WRIST : LM.R_WRIST),
    hip: getLandmark(points, isLeft ? LM.L_HIP : LM.R_HIP),
    knee: getLandmark(points, isLeft ? LM.L_KNEE : LM.R_KNEE),
    ankle: getLandmark(points, isLeft ? LM.L_ANKLE : LM.R_ANKLE),
  };
}

function makeGuidePoint(anchor, reference, direction, lengthFactor = 1) {
  const offset = subtractPoint(reference, anchor);
  const distance = Math.max(lengthPoint(offset) * lengthFactor, 0.0001);
  return addPoint(anchor, scalePoint(direction, distance));
}

/** Ideal arm path for a curl: same shape, but elbow pulled onto the body line. */
function buildCurlInlineGuides(shoulder, elbow, wrist, hip) {
  if (!shoulder || !elbow || !wrist || !hip) return [];
  const torsoDir = normalizePoint(subtractPoint(hip, shoulder));
  if (lengthPoint(torsoDir) < 1e-6) return [];

  const upperLen = dist(shoulder, elbow);
  const idealElbow = addPoint(shoulder, scalePoint(torsoDir, upperLen));
  const forearmLen = dist(elbow, wrist);
  const forearmDir = normalizePoint(subtractPoint(wrist, elbow));
  const idealWrist = lengthPoint(forearmDir) > 1e-6
    ? addPoint(idealElbow, scalePoint(forearmDir, forearmLen))
    : addPoint(idealElbow, scalePoint(torsoDir, forearmLen));

  return [
    { from: shoulder, to: idealElbow, type: "arm" },
    { from: idealElbow, to: idealWrist, type: "arm" },
  ];
}

/**
 * Green correction guides for push-up / plank side view.
 * Hip: straight shoulder → idealHip → ankle (hips on the body line).
 * Neck: ideal ear on the hip→shoulder extension (neutral head).
 */
function buildStraightnessGuides(sidePoints, { hipBad = false, neckBad = false } = {}) {
  if (!sidePoints || !sidePoints.shoulder || !sidePoints.hip) return [];

  const guides = [];
  const { shoulder, hip, ankle, head } = sidePoints;

  if (hipBad && ankle) {
    // Ideal hip sits on the shoulder→ankle line (same depth as real hip along that line).
    const idealHip = projectPointOnLine(hip, shoulder, ankle);
    idealHip.visibility = hip.visibility ?? 1;
    guides.push({ from: shoulder, to: idealHip, type: "hip" });
    guides.push({ from: idealHip, to: ankle, type: "hip" });
  }

  if (neckBad && head) {
    // Ideal ear continues the hip→shoulder line past the shoulder (neutral neck).
    const up = normalizePoint(subtractPoint(shoulder, hip));
    if (lengthPoint(up) > 1e-6) {
      const earLen = Math.max(dist(shoulder, head), 0.04);
      const idealEar = addPoint(shoulder, scalePoint(up, earLen));
      guides.push({ from: idealEar, to: shoulder, type: "neck" });
      guides.push({ from: shoulder, to: hip, type: "neck" });
    }
  }

  return guides;
}

function debounceFormState(state, rawCorrect, goodFrames = 3, badFrames = 2) {
  if (!state) return rawCorrect;

  if (!state.initialized) {
    state.initialized = true;
    state.isCorrect = rawCorrect;
    state.goodCount = rawCorrect ? goodFrames : 0;
    state.badCount = rawCorrect ? 0 : badFrames;
    return state.isCorrect;
  }

  if (rawCorrect) {
    state.goodCount = (state.goodCount || 0) + 1;
    state.badCount = 0;
    if (state.goodCount >= goodFrames) {
      state.isCorrect = true;
    }
  } else {
    state.badCount = (state.badCount || 0) + 1;
    state.goodCount = 0;
    if (state.badCount >= badFrames) {
      state.isCorrect = false;
    }
  }

  return !!state.isCorrect;
}

/** Sticky boolean flag — hard to turn on, easy/slow to clear (or vice versa). */
function debounceFlag(state, key, rawValue, enterFrames = 8, exitFrames = 10) {
  const enterKey = `${key}Enter`;
  const exitKey = `${key}Exit`;
  const valueKey = `${key}Value`;

  if (state[valueKey] === undefined) {
    state[valueKey] = !!rawValue;
    state[enterKey] = rawValue ? enterFrames : 0;
    state[exitKey] = rawValue ? 0 : exitFrames;
    return state[valueKey];
  }

  if (rawValue) {
    state[enterKey] = (state[enterKey] || 0) + 1;
    state[exitKey] = 0;
    if (state[enterKey] >= enterFrames) state[valueKey] = true;
  } else {
    state[exitKey] = (state[exitKey] || 0) + 1;
    state[enterKey] = 0;
    if (state[exitKey] >= exitFrames) state[valueKey] = false;
  }
  return !!state[valueKey];
}

function getSmoothedMetric(state, key, value, windowSize = 6) {
  if (!state[key]) state[key] = [];
  state[key].push(value);
  if (state[key].length > windowSize) state[key].shift();
  return average(state[key]);
}

function analyzeSideViewForm(landmarks, state = {}, options = {}) {
  const {
    hipTolerance = 14,
    neckTolerance = 14,
    worldLandmarks = null,
    // "knee" = push-up style; "ankle" = full body line (better for plank side view)
    hipEndpoint = "knee",
    goodFrames = 3,
    badFrames = 2,
    smoothWindow = 6,
    // Plank: neck must also fail an image-space check before we paint / guide it.
    lenientNeck = false,
    // Prefer perpendicular offset from shoulder–ankle line (side-view accurate).
    useLineOffset = false,
    offsetTolerance = 5,
    enterIssueFrames = 3,
    exitIssueFrames = 2,
  } = options;

  const side = selectVisibleSide(landmarks, state.visibleSide);
  // When we cannot measure form yet, stay "correct" with no red paint — otherwise
  // an empty badLandmarks list turns the ENTIRE skeleton red and looks frozen.
  const unscored = (extra = {}) => ({
    isCorrect: true,
    hipAngle: null,
    neckAngle: null,
    issues: [],
    visibleSide: side,
    guideLines: [],
    badLandmarks: [],
    rawIsCorrect: true,
    ...extra,
  });

  if (!side) {
    return unscored({ issues: ["Need a clearer side view"], visibleSide: null });
  }

  const points = pickSidePoints(landmarks, side);
  const endPt = hipEndpoint === "ankle" ? points.ankle : points.knee;
  const missing = !points.shoulder || !points.hip || !endPt || !points.ankle || !points.head;
  if (missing) {
    return unscored({
      issues: ["Need clearer visibility of shoulder, hip, ankle, and head landmarks"],
    });
  }

  // Prefer 3D world landmarks for the body-line / neck angles.
  const isLeft = side === "left";
  let hipAngleRaw;
  let neckAngleRaw;
  let usedWorld = false;
  if (worldLandmarks && worldLandmarks.length >= 33) {
    const wShoulder = getLandmark(worldLandmarks, isLeft ? LM.L_SHOULDER : LM.R_SHOULDER);
    const wHip = getLandmark(worldLandmarks, isLeft ? LM.L_HIP : LM.R_HIP);
    const wEnd = hipEndpoint === "ankle"
      ? getLandmark(worldLandmarks, isLeft ? LM.L_ANKLE : LM.R_ANKLE)
      : getLandmark(worldLandmarks, isLeft ? LM.L_KNEE : LM.R_KNEE);
    const wHead = getLandmark(worldLandmarks, isLeft ? LM.L_EAR : LM.R_EAR)
      || getLandmark(worldLandmarks, LM.NOSE);
    if (wShoulder && wHip && wEnd && wHead) {
      hipAngleRaw = calculateAngle3D(wShoulder, wHip, wEnd);
      neckAngleRaw = calculateAngle3D(wHead, wShoulder, wHip);
      usedWorld = true;
    }
  }
  if (!usedWorld) {
    hipAngleRaw = calculateAngle(points.shoulder, points.hip, endPt);
    neckAngleRaw = calculateAngle(points.head, points.shoulder, points.hip);
  }

  const hipAngle = getSmoothedMetric(state, "hipAngles", hipAngleRaw, smoothWindow);
  const neckAngle = getSmoothedMetric(state, "neckAngles", neckAngleRaw, smoothWindow);

  // Angle deviation from straight (180°) + signed offset from shoulder–ankle line.
  const angleDeviation = Math.abs(hipAngle - 180);
  const neckDeviation = Math.abs(neckAngle - 180);
  const hipLineOffsetRaw = hipOffsetFromBodyLine(points.shoulder, points.hip, points.ankle);
  const hipLineOffset = getSmoothedMetric(state, "hipOffsets", hipLineOffsetRaw, smoothWindow);

  // Line-offset mode (plank): hips must sit near the shoulder–ankle line.
  // Angle-only mode (push-up): keep previous interior-angle check.
  const hipBadRaw = useLineOffset
    ? Math.abs(hipLineOffset) > offsetTolerance || angleDeviation > hipTolerance
    : angleDeviation > hipTolerance;

  // Side-view neck: angle alone is noisy (head sits slightly off the torso line).
  // Lenient mode also requires a clear image-space droop / crane.
  let neckBadRaw = neckDeviation > neckTolerance;
  if (lenientNeck) {
    const headDrop = points.head.y - points.shoulder.y;
    const clearlyOff = headDrop > 0.045 || headDrop < -0.06;
    neckBadRaw = neckBadRaw && clearlyOff;
  }

  // Sticky per-issue flags so green guides / red paint don't flicker every frame.
  const hipBad = debounceFlag(state, "hipBad", hipBadRaw, enterIssueFrames, exitIssueFrames);
  const neckBad = debounceFlag(state, "neckBad", neckBadRaw, enterIssueFrames, exitIssueFrames);

  const issues = [];
  if (hipBad) {
    if (hipLineOffset > 1.5) issues.push("hips sagging");
    else if (hipLineOffset < -1.5) issues.push("hips piking");
    else issues.push("hips not in a straight line");
  }
  if (neckBad) {
    if (points.head.y > points.shoulder.y + 0.03) issues.push("head dropping");
    else if (points.head.y < points.shoulder.y - 0.04) issues.push("head craning up");
    else issues.push("neck not neutral");
  }

  const rawIsCorrect = !hipBadRaw && !neckBadRaw;
  const isCorrect = debounceFormState(state, !hipBad && !neckBad, goodFrames, badFrames);

  // Only draw guides for sticky (debounced) faults — stops "random" green flashes.
  const guideLines = (hipBad || neckBad)
    ? buildStraightnessGuides(points, { hipBad, neckBad })
    : [];

  const earIdx = isLeft ? LM.L_EAR : LM.R_EAR;
  const badLandmarks = [];
  if (hipBad) {
    badLandmarks.push(
      isLeft ? LM.L_SHOULDER : LM.R_SHOULDER,
      isLeft ? LM.L_HIP : LM.R_HIP,
      isLeft ? LM.L_ANKLE : LM.R_ANKLE
    );
    if (hipEndpoint === "knee") {
      badLandmarks.push(isLeft ? LM.L_KNEE : LM.R_KNEE);
    }
  }
  if (neckBad) {
    badLandmarks.push(
      earIdx,
      isLeft ? LM.L_SHOULDER : LM.R_SHOULDER,
      isLeft ? LM.L_HIP : LM.R_HIP
    );
  }

  state.visibleSide = side;
  state.lastHipAngle = hipAngle;
  state.lastNeckAngle = neckAngle;
  state.lastIssues = issues;
  state.lastGuideLines = guideLines;
  state.lastBadLandmarks = badLandmarks;

  return {
    isCorrect,
    hipAngle: Math.round(hipAngle),
    neckAngle: Math.round(neckAngle),
    issues,
    visibleSide: side,
    guideLines,
    badLandmarks: [...new Set(badLandmarks)],
    rawIsCorrect,
    bodyDeviation: angleDeviation,
    neckDeviation,
    hipLineOffset,
    usedWorld,
    points,
  };
}

export function checkPushupForm(landmarks, state = {}, options = {}) {
  // Same body-line scoring as plank (shoulder–ankle offset). Invariant to the
  // torso rising/lowering during the rep — only cares that hips stay inline.
  return analyzeSideViewForm(landmarks, state, {
    hipTolerance: 28,
    neckTolerance: 42,
    hipEndpoint: "ankle",
    useLineOffset: true,
    offsetTolerance: 4.5,
    lenientNeck: true,
    goodFrames: 10,
    badFrames: 12,
    smoothWindow: 10,
    enterIssueFrames: 10,
    exitIssueFrames: 14,
    ...options,
  });
}

export function checkPlankForm(landmarks, state = {}, options = {}) {
  // Side plank: score hips against the shoulder–ankle line (not average Y).
  // Slightly looser than before so left-side foreshortening doesn't false-fail.
  return analyzeSideViewForm(landmarks, state, {
    hipTolerance: 32,
    neckTolerance: 45,
    hipEndpoint: "ankle",
    useLineOffset: true,
    offsetTolerance: 5.5,
    lenientNeck: true,
    goodFrames: 8,
    badFrames: 10,
    smoothWindow: 10,
    enterIssueFrames: 8,
    exitIssueFrames: 12,
    ...options,
  });
}

/* ───────────────────────────────────────────────────────────────
 * Push-up / Plank: Pre-Calibration Orientation Validation
 *
 * Runs before the 3-second calibration window to ensure the user
 * is positioned correctly (side-on) with high-confidence landmarks.
 * ─────────────────────────────────────────────────────────────── */

/**
 * Validate that the user is positioned correctly for push-up / plank
 * calibration. Checks:
 *   1. Shoulder / hip / knee visibility confidence on the active side
 *   2. 3D horizontal separation between left & right shoulders
 *      (insufficient separation = user is facing camera, not side-on)
 *
 * @param {Array} landmarks 2D image-space landmarks
 * @param {Array} worldLandmarks 3D world-space landmarks (meters)
 * @param {number} confThreshold Min visibility confidence (default 0.4)
 * @param {number} minShoulderSepMeters Min 3D x-axis separation (default 0.12m = ~4.7in)
 * @returns {{ pass: boolean, message: string|null }}
 */
export function validatePushupPlankOrientation(landmarks, worldLandmarks, confThreshold = 0.4, minShoulderSepMeters = 0.12) {
  if (!landmarks || landmarks.length < 33) {
    return { pass: false, message: "Please turn side-on to the camera" };
  }

  const side = selectVisibleSide(landmarks, null);
  if (!side) {
    return { pass: false, message: "Please turn side-on to the camera" };
  }

  const isLeft = side === "left";
  const shoulder = getLandmark(landmarks, isLeft ? LM.L_SHOULDER : LM.R_SHOULDER);
  const hip = getLandmark(landmarks, isLeft ? LM.L_HIP : LM.R_HIP);
  const knee = getLandmark(landmarks, isLeft ? LM.L_KNEE : LM.R_KNEE);
  const lShoulder = getLandmark(landmarks, LM.L_SHOULDER);
  const rShoulder = getLandmark(landmarks, LM.R_SHOULDER);

  if (!shoulder || !hip || !knee || !lShoulder || !rShoulder) {
    return { pass: false, message: "Please turn side-on to the camera" };
  }

  if (
    (shoulder.visibility ?? 0) < confThreshold ||
    (hip.visibility ?? 0) < confThreshold ||
    (knee.visibility ?? 0) < confThreshold ||
    (lShoulder.visibility ?? 0) < confThreshold ||
    (rShoulder.visibility ?? 0) < confThreshold
  ) {
    return { pass: false, message: "Please turn side-on to the camera" };
  }

  if (worldLandmarks && worldLandmarks.length >= 33) {
    const wLSh = worldLandmarks[LM.L_SHOULDER];
    const wRSh = worldLandmarks[LM.R_SHOULDER];
    if (wLSh && wRSh) {
      const shoulderSeparationX = Math.abs(wLSh.x - wRSh.x);
      if (shoulderSeparationX < minShoulderSepMeters) {
        return { pass: false, message: "Please turn side-on to the camera" };
      }
    }
  } else {
    const shoulderSeparationX = Math.abs(lShoulder.x - rShoulder.x);
    if (shoulderSeparationX < 0.05) {
      return { pass: false, message: "Please turn side-on to the camera" };
    }
  }

  return { pass: true, message: null };
}

/* ───────────────────────────────────────────────────────────────
 * Push-up / Plank: 3-Second Calibration Manager
 *
 * Collects hip & neck angles from stable 3D world landmarks over a
 * 3-second window, averages them for session-specific baselines,
 * and runs sanity / variance / confidence checks before accepting.
 * ─────────────────────────────────────────────────────────────── */

const CALIBRATION_WINDOW_MS = 3000;
const CALIBRATION_REJECT_DISPLAY_MS = 900;
const CALIBRATION_MIN_SAMPLES = 15;
const CALIBRATION_HIP_MIN = 160;
const CALIBRATION_HIP_MAX = 200;
const CALIBRATION_NECK_MIN = 150;
const CALIBRATION_NECK_MAX = 190;
const CALIBRATION_MAX_HIP_VAR = 10;
const CALIBRATION_MAX_NECK_VAR = 10;

export class PushupPlankCalibrator {
  constructor() {
    this.reset();
  }

  reset() {
    this.state = "idle"; // idle | orientation_check | calibrating | calibrated | rejected
    this.startedAt = 0;
    this.rejectedAt = 0;
    this.samples = []; // { hipAngle, neckAngle, confOK }
    this.lastRejectionReason = null;
    this.baselineHipAngle = null;
    this.baselineNeckAngle = null;
  }

  start() {
    this.state = "calibrating";
    this.startedAt = performance.now();
    this.rejectedAt = 0;
    this.samples = [];
    this.lastRejectionReason = null;
  }

  /** Compute std-dev of a number array. */
  _stddev(arr) {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
  }

  /**
   * Tick per frame during calibration.
   * @param {number} now performance.now()
   * @param {Array} landmarks 2D landmarks
   * @param {Array} worldLandmarks 3D world landmarks
   * @param {object} formState reusable smoothing state
   * @param {"push_up"|"plank"} exercise for context (does not alter math)
   * @returns {object} { state, progressPct, prompt, rejectionReason, baseline? }
   */
  tick(now, landmarks, worldLandmarks, formState, exercise = "push_up") {
    if (this.state === "idle") {
      this.start();
    } else if (this.state === "rejected") {
      if (this.rejectedAt === 0) {
        this.rejectedAt = now;
      }
      if (now - this.rejectedAt >= CALIBRATION_REJECT_DISPLAY_MS) {
        this.start();
      } else {
        const remaining = 1 - (now - this.rejectedAt) / CALIBRATION_REJECT_DISPLAY_MS;
        return {
          state: "rejected",
          progressPct: clamp(remaining, 0, 1),
          prompt: null,
          rejectionReason: this.lastRejectionReason,
        };
      }
    }

    if (this.state === "calibrated") {
      return {
        state: "calibrated",
        progressPct: 1,
        prompt: null,
        rejectionReason: null,
        baselineHipAngle: this.baselineHipAngle,
        baselineNeckAngle: this.baselineNeckAngle,
      };
    }

    const elapsed = now - this.startedAt;
    const progressPct = clamp(elapsed / CALIBRATION_WINDOW_MS, 0, 1);

    const orientationCheck = validatePushupPlankOrientation(landmarks, worldLandmarks);
    if (!orientationCheck.pass) {
      return {
        state: "orientation_check",
        progressPct: 0,
        prompt: orientationCheck.message,
        rejectionReason: null,
      };
    }

    const side = selectVisibleSide(landmarks, formState.visibleSide);
    if (side) {
      const isLeft = side === "left";
      let hipAngleRaw = null;
      let neckAngleRaw = null;
      let confOK = false;

      if (worldLandmarks && worldLandmarks.length >= 33) {
        const wSh = getLandmark(worldLandmarks, isLeft ? LM.L_SHOULDER : LM.R_SHOULDER);
        const wHip = getLandmark(worldLandmarks, isLeft ? LM.L_HIP : LM.R_HIP);
        const wKnee = getLandmark(worldLandmarks, isLeft ? LM.L_KNEE : LM.R_KNEE);
        const wHead = getLandmark(worldLandmarks, isLeft ? LM.L_EAR : LM.R_EAR) || getLandmark(worldLandmarks, LM.NOSE);
        if (wSh && wHip && wKnee && wHead) {
          hipAngleRaw = calculateAngle3D(wSh, wHip, wKnee);
          neckAngleRaw = calculateAngle3D(wHead, wSh, wHip);
          confOK =
            (wSh.visibility ?? 0) >= 0.4 &&
            (wHip.visibility ?? 0) >= 0.4 &&
            (wKnee.visibility ?? 0) >= 0.4 &&
            (wHead.visibility ?? 0) >= 0.4;
        }
      }

      if (hipAngleRaw === null || neckAngleRaw === null) {
        const points = pickSidePoints(landmarks, side);
        if (points.shoulder && points.hip && points.knee && points.head) {
          hipAngleRaw = calculateAngle(points.shoulder, points.hip, points.knee);
          neckAngleRaw = calculateAngle(points.head, points.shoulder, points.hip);
          confOK =
            (points.shoulder.visibility ?? 0) >= 0.4 &&
            (points.hip.visibility ?? 0) >= 0.4 &&
            (points.knee.visibility ?? 0) >= 0.4 &&
            (points.head.visibility ?? 0) >= 0.4;
        }
      }

      if (hipAngleRaw !== null && neckAngleRaw !== null) {
        this.samples.push({ hipAngle: hipAngleRaw, neckAngle: neckAngleRaw, confOK });
      }
    }

    if (elapsed >= CALIBRATION_WINDOW_MS) {
      const confSamples = this.samples.filter((s) => s.confOK);
      if (confSamples.length < CALIBRATION_MIN_SAMPLES) {
        this.state = "rejected";
        this.rejectedAt = 0;
        this.lastRejectionReason = "Calibration looks off, check your position and try again";
        return {
          state: "rejected",
          progressPct: 1,
          prompt: null,
          rejectionReason: this.lastRejectionReason,
        };
      }

      const avgHip = confSamples.reduce((s, v) => s + v.hipAngle, 0) / confSamples.length;
      const avgNeck = confSamples.reduce((s, v) => s + v.neckAngle, 0) / confSamples.length;
      const hipVariance = this._stddev(confSamples.map((s) => s.hipAngle));
      const neckVariance = this._stddev(confSamples.map((s) => s.neckAngle));

      const hipInRange = avgHip >= CALIBRATION_HIP_MIN && avgHip <= CALIBRATION_HIP_MAX;
      const neckInRange = avgNeck >= CALIBRATION_NECK_MIN && avgNeck <= CALIBRATION_NECK_MAX;
      const hipStable = hipVariance <= CALIBRATION_MAX_HIP_VAR;
      const neckStable = neckVariance <= CALIBRATION_MAX_NECK_VAR;

      if (!hipInRange || !neckInRange) {
        this.state = "rejected";
        this.rejectedAt = 0;
        this.lastRejectionReason = "Calibration looks off, check your position and try again";
        return {
          state: "rejected",
          progressPct: 1,
          prompt: null,
          rejectionReason: this.lastRejectionReason,
        };
      }

      if (!hipStable || !neckStable) {
        this.state = "rejected";
        this.rejectedAt = 0;
        this.lastRejectionReason = "Calibration looks off, hold still and try again";
        return {
          state: "rejected",
          progressPct: 1,
          prompt: null,
          rejectionReason: this.lastRejectionReason,
        };
      }

      this.baselineHipAngle = Math.round(avgHip);
      this.baselineNeckAngle = Math.round(avgNeck);
      this.state = "calibrated";
      return {
        state: "calibrated",
        progressPct: 1,
        prompt: null,
        rejectionReason: null,
        baselineHipAngle: this.baselineHipAngle,
        baselineNeckAngle: this.baselineNeckAngle,
      };
    }

    const posName = exercise === "plank" ? "plank" : "pushup-up";
    return {
      state: "calibrating",
      progressPct,
      prompt: `Get into your best ${posName} position and hold still`,
      rejectionReason: null,
    };
  }
}

/* ───────────────────────────────────────────────────────────────
 * Camera Angle Detection
 *
 * We infer how the user is oriented relative to the camera by
 * comparing left/right landmark separation (shoulder width in
 * image space). A side-on view compresses shoulder width to near
 * zero; a front-on view maximises it.
 *
 * viewAngle:
 *   "side"   : user turned ~90 deg (profile view)
 *   "angled" : user turned ~30-60 deg (three-quarter view)
 *   "front"  : user facing camera directly
 *
 * We also detect body orientation (upright vs horizontal) for
 * floor exercises like push-ups and planks.
 * ────────────────────────────────────────────────────────────── */

/**
 * @param {Array} pts - smoothed landmarks
 * @returns {{ viewAngle: "side"|"angled"|"front", isHorizontal: boolean, shoulderWidthRatio: number, confidence: number }}
 */
function detectCameraAngle(pts) {
  const lSh = pts[LM.L_SHOULDER];
  const rSh = pts[LM.R_SHOULDER];
  const lHip = pts[LM.L_HIP];
  const rHip = pts[LM.R_HIP];

  if (!vis(lSh) || !vis(rSh) || !vis(lHip) || !vis(rHip)) {
    return { viewAngle: "front", isHorizontal: false, shoulderWidthRatio: 0.5, confidence: 0 };
  }

  // Use horizontal shoulder separation, rather than full 2D distance. In a
  // profile view the two shoulders can be vertically offset, but their
  // horizontal separation should remain small. Full Euclidean distance was
  // therefore classifying many true side views as angled/front.
  const shoulderWidth = Math.abs(lSh.x - rSh.x);
  const torsoHeight = dist(mid(lSh, rSh), mid(lHip, rHip));
  const ratio = torsoHeight > 0.01 ? shoulderWidth / torsoHeight : 0.5;

  // A literal profile view is rarely a perfect 90° turn in a real webcam.
  // This deliberately accepts a slight three-quarter turn as "side" so a
  // valid curl is not rejected merely because one shoulder is a little ahead.
  let viewAngle;
  if (ratio < 0.45) viewAngle = "side";
  else if (ratio < 0.85) viewAngle = "angled";
  else viewAngle = "front";

  // Body orientation: is the person lying down / in plank position?
  const midShoulder = mid(lSh, rSh);
  const midHip = mid(lHip, rHip);
  const torsoAngleFromHorizontal = Math.abs(Math.atan2(midShoulder.y - midHip.y, midShoulder.x - midHip.x) * (180 / Math.PI));
  const isHorizontal = torsoAngleFromHorizontal < 45 || torsoAngleFromHorizontal > 135;

  return { viewAngle, isHorizontal, shoulderWidthRatio: ratio, confidence: 1 };
}

function armIsVisible(points) {
  return points.every((point) => vis(point));
}

/**
 * Pick the arm closest to / most reliably seen by the camera, then keep that
 * choice for the rep. Re-selecting on every frame caused the tracker to swap
 * between a moving arm and a stationary, partly-hidden arm during side curls.
 */
function chooseCurlArm(p, previousArm) {
  const left = [p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]];
  const right = [p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]];
  const leftVisible = armIsVisible(left);
  const rightVisible = armIsVisible(right);

  const score = (arm) => {
    const visibility = arm.reduce((sum, point) => sum + point.visibility, 0) / arm.length;
    const projectedLength = dist(arm[0], arm[1]) + dist(arm[1], arm[2]);
    // MediaPipe's z value is lower for a point closer to the camera. It is a
    // tie-breaker only; visibility and visible arm length remain dominant.
    const depth = arm.reduce((sum, point) => sum + (Number.isFinite(point.z) ? -point.z : 0), 0) / arm.length;
    return visibility * 2 + projectedLength + depth * 0.1;
  };

  const leftScore = leftVisible ? score(left) : -Infinity;
  const rightScore = rightVisible ? score(right) : -Infinity;

  if (!leftVisible && !rightVisible) return null;
  if (previousArm === "left" && leftVisible && leftScore >= rightScore - 0.2) return "left";
  if (previousArm === "right" && rightVisible && rightScore >= leftScore - 0.2) return "right";
  if (leftVisible && !rightVisible) return "left";
  if (rightVisible && !leftVisible) return "right";

  return score(left) >= score(right) ? "left" : "right";
}

/* ───────────────────────────────────────────────────────────────
 * Velocity Tracker
 *
 * Tracks the rate of change of the primary angle to distinguish
 * deliberate movement from noise/drift. Reps require the primary
 * angle to move at a meaningful velocity during key transitions.
 * ────────────────────────────────────────────────────────────── */
class VelocityTracker {
  constructor(windowSize = 5) {
    this.windowSize = windowSize;
    this.history = []; // { value, time }
  }

  push(value, time) {
    this.history.push({ value, time });
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }
  }

  /** Degrees per second (positive = increasing angle). */
  get velocity() {
    if (this.history.length < 2) return 0;
    const first = this.history[0];
    const last = this.history[this.history.length - 1];
    const dt = last.time - first.time;
    if (dt < 0.001) return 0;
    return (last.value - first.value) / dt;
  }

  /** Absolute velocity (speed). */
  get speed() {
    return Math.abs(this.velocity);
  }

  reset() {
    this.history = [];
  }
}

/**
 * Knee valgus / flare vs the hip→ankle line in the frontal plane.
 * Prefer world landmarks so forward knee travel (z) is ignored.
 * Positive = knee medial of the hip–ankle line (cave-in).
 * midX = body midline (avg hip or ankle x) so left/right work in world + image space.
 */
function squatKneeMedialOffset(hip, knee, ankle, midX) {
  if (!hip || !knee || !ankle || !Number.isFinite(midX)) return 0;
  const ySpan = ankle.y - hip.y;
  const t = Math.abs(ySpan) > 1e-4
    ? clamp((knee.y - hip.y) / ySpan, 0, 1)
    : 0.5;
  const expectedX = hip.x + t * (ankle.x - hip.x);
  // + when knee moves from the hip–ankle line toward the body midline.
  const towardMid = Math.sign(midX - expectedX) || 1;
  return towardMid * (knee.x - expectedX);
}

/** Plain {x,y,visibility} copy — MediaPipe landmark objects are unsafe to reuse. */
function squatPlainPt(lm) {
  if (!lm) return null;
  const x = Number(lm.x);
  const y = Number(lm.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y, visibility: Number(lm.visibility ?? 1) };
}

/**
 * Green squat correction legs from the SAME landmarks the skeleton uses.
 * Exact hip→knee→ankle clones, slid sideways so they sit beside the real legs.
 * Exported so the draw loop can call this directly (never drop green guides).
 */
export function buildSquatCorrectionGuides(landmarks) {
  if (!landmarks || landmarks.length < 33) return [];

  const lHip = squatPlainPt(landmarks[LM.L_HIP]);
  const rHip = squatPlainPt(landmarks[LM.R_HIP]);
  const lKnee = squatPlainPt(landmarks[LM.L_KNEE]);
  const rKnee = squatPlainPt(landmarks[LM.R_KNEE]);
  const lAnkle = squatPlainPt(landmarks[LM.L_ANKLE]);
  const rAnkle = squatPlainPt(landmarks[LM.R_ANKLE]);
  if (!lHip || !rHip || !lKnee || !rKnee || !lAnkle || !rAnkle) return [];

  const kneeMid = (lKnee.x + rKnee.x) / 2;
  const half = Math.max(Math.abs(rAnkle.x - lAnkle.x) / 2, 0.06);
  const leftIsLowerX = lAnkle.x <= rAnkle.x;
  const targetLX = leftIsLowerX ? kneeMid - half : kneeMid + half;
  const targetRX = leftIsLowerX ? kneeMid + half : kneeMid - half;

  let dxL = targetLX - lKnee.x;
  let dxR = targetRX - rKnee.x;
  const MIN = 0.045;
  if (Math.abs(dxL) < MIN) dxL = (leftIsLowerX ? -1 : 1) * MIN;
  if (Math.abs(dxR) < MIN) dxR = (leftIsLowerX ? 1 : -1) * MIN;

  const shift = (p, dx) => ({ x: p.x + dx, y: p.y, visibility: p.visibility });
  const leg = (hip, knee, ankle, dx) => {
    const h = shift(hip, dx);
    const k = shift(knee, dx);
    const a = shift(ankle, dx);
    return [
      { from: h, to: k, type: "leg" },
      { from: k, to: a, type: "leg" },
    ];
  };

  return [
    ...leg(lHip, lKnee, lAnkle, dxL),
    ...leg(rHip, rKnee, rAnkle, dxR),
  ];
}

/**
 * Squat form overlay flags → red landmarks + green spaced leg copies.
 */
function buildSquatLateralOverlay(guidePts, flags) {
  if (!guidePts || !flags) {
    return { formOk: true, badLandmarks: [], guideLines: [], flags: null };
  }

  const hasCave = !!(flags.leftCave || flags.rightCave);
  const hasFlare = !!(flags.leftFlare || flags.rightFlare);
  // Hip-shift alone never paints red without green — skip visual for it.
  const hasFlags = hasCave || hasFlare;

  if (!hasFlags) {
    return { formOk: true, badLandmarks: [], guideLines: [], flags };
  }

  const badLandmarks = hasCave
    ? [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE, LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]
    : [
      ...(flags.leftFlare ? [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE] : []),
      ...(flags.rightFlare ? [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE] : []),
    ];

  // Build from whatever points we were given (caller should pass raw landmarks).
  const asList = [
    guidePts.lHip, guidePts.rHip, guidePts.lKnee, guidePts.rKnee,
    guidePts.lAnkle, guidePts.rAnkle,
  ];
  // Fake a 33-slot list for the shared builder.
  const fake = new Array(33);
  fake[LM.L_HIP] = guidePts.lHip;
  fake[LM.R_HIP] = guidePts.rHip;
  fake[LM.L_KNEE] = guidePts.lKnee;
  fake[LM.R_KNEE] = guidePts.rKnee;
  fake[LM.L_ANKLE] = guidePts.lAnkle;
  fake[LM.R_ANKLE] = guidePts.rAnkle;
  const guideLines = asList.every(Boolean) ? buildSquatCorrectionGuides(fake) : [];

  return {
    formOk: false,
    badLandmarks,
    guideLines,
    flags,
  };
}

/* ───────────────────────────────────────────────────────────────
 * Exercise Profiles  (camera-angle-aware, strict rep counting)
 *
 * Each profile adds:
 *   preferredAngles    : which camera view(s) work best
 *   angleHint          : guidance message shown when angle is wrong
 *   bodyOrientation    : "upright" | "horizontal" | "any"
 *   validateAngle(cam) : returns true if camera angle is acceptable
 *   minROM             : minimum range-of-motion (degrees) to count
 *   minRepDurationMs   : minimum time for a full rep cycle
 *   minVelocity        : min deg/s during active movement phases
 *   stabilityChecks(a) : extra guards that BLOCK a rep if they fail
 * ────────────────────────────────────────────────────────────── */

const EXERCISE_PROFILES = {

  /* ══════════════════════════════════════════════════════════
   *  SQUAT  (~45° three-quarter view)
   *  Depth/reps from hip–knee clearance. Form (knee cave / flare /
   *  hip shift) only while loaded — never while standing tall.
   * ══════════════════════════════════════════════════════════ */
  squat: {
    label: "Squat",
    type: "rep",
    preferredAngles: ["angled", "front", "side"],
    angleHint: "Turn a bit more toward the camera (~45°) so both legs stay visible",
    bodyOrientation: "upright",
    requiredLandmarks: [
      LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE,
      LM.L_SHOULDER, LM.R_SHOULDER,
    ],

    /**
     * primary = hip–knee vertical clearance in meters.
     * Standing → large positive; parallel / below → ~0 or negative.
     *
     * Knee cave = frontal-plane medial drift of the knee vs hip→ankle
     * (world X when available so forward knee travel is ignored).
     */
    computeAngles(p, engine, worldLandmarks = null) {
      const useWorld = !!(
        worldLandmarks
        && worldLandmarks.length >= 33
        && worldLandmarks[LM.L_HIP] && worldLandmarks[LM.R_HIP]
        && worldLandmarks[LM.L_KNEE] && worldLandmarks[LM.R_KNEE]
        && worldLandmarks[LM.L_ANKLE] && worldLandmarks[LM.R_ANKLE]
      );

      const src = useWorld ? worldLandmarks : p;
      const lHip = src[LM.L_HIP];
      const rHip = src[LM.R_HIP];
      const lKnee = src[LM.L_KNEE];
      const rKnee = src[LM.R_KNEE];
      const lAnkle = src[LM.L_ANKLE];
      const rAnkle = src[LM.R_ANKLE];

      const midHipY = (lHip.y + rHip.y) / 2;
      const midKneeY = (lKnee.y + rKnee.y) / 2;
      const sep = Math.abs(midKneeY - midHipY);
      const kneesBelowHips2D =
        ((p[LM.L_KNEE].y + p[LM.R_KNEE].y) / 2) > ((p[LM.L_HIP].y + p[LM.R_HIP].y) / 2);
      let clearance = kneesBelowHips2D ? sep : -sep;
      if (!useWorld) {
        const scale = Math.max(dist(p[LM.L_SHOULDER], p[LM.L_HIP]), 0.15);
        clearance = clearance / scale * 0.25;
      }

      const ALPHA = 0.40;
      if (!engine._squatDepthSmoothed || !Number.isFinite(engine._squatDepthSmoothed)) {
        engine._squatDepthSmoothed = clearance;
      } else {
        engine._squatDepthSmoothed =
          ALPHA * clearance + (1 - ALPHA) * engine._squatDepthSmoothed;
      }

      // Form metrics in the same space as depth (world preferred).
      const midHipX = (lHip.x + rHip.x) / 2;
      const midAnkleX = (lAnkle.x + rAnkle.x) / 2;
      const midX = (midHipX + midAnkleX) / 2;
      const leftMedial = squatKneeMedialOffset(lHip, lKnee, lAnkle, midX);
      const rightMedial = squatKneeMedialOffset(rHip, rKnee, rAnkle, midX);
      const hipLateralOffset = midHipX - midAnkleX;
      const stanceWidth = Math.max(Math.abs(lAnkle.x - rAnkle.x), useWorld ? 0.08 : 0.05);
      const kneeSpan = Math.abs(lKnee.x - rKnee.x);
      // World + image ratios — angled webcams foreshorten; take the more collapsed.
      const worldRatio = kneeSpan / stanceWidth;
      const imgKneeSpan = Math.abs(p[LM.L_KNEE].x - p[LM.R_KNEE].x);
      const imgAnkleSpan = Math.max(Math.abs(p[LM.L_ANKLE].x - p[LM.R_ANKLE].x), 0.05);
      const imageRatio = imgKneeSpan / imgAnkleSpan;
      const kneeAnkleWidthRatio = Math.min(worldRatio, imageRatio);

      return {
        primary: engine._squatDepthSmoothed,
        hipKneeClearance: engine._squatDepthSmoothed,
        usingWorld: useWorld,
        leftMedial,
        rightMedial,
        hipLateralOffset,
        stanceWidth,
        kneeAnkleWidthRatio,
        // Image-space points for drawing green guides on the skeleton.
        guidePts: {
          lHip: p[LM.L_HIP],
          rHip: p[LM.R_HIP],
          lKnee: p[LM.L_KNEE],
          rKnee: p[LM.R_KNEE],
          lAnkle: p[LM.L_ANKLE],
          rAnkle: p[LM.R_ANKLE],
        },
      };
    },

    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    // Clearance thresholds in meters. Standing ≈ 0.15–0.30 m above knees.
    thresholds: {
      standClearance: 0.10,
      enterBottom: 0.045,
      leaveBottom: 0.07,
      // Form overlays only once hips have dropped below this (not standing).
      formActiveClearance: 0.085,
    },
    minTransitionMs: 140,
    minROM: 0.08,
    minRepDurationMs: 600,
    minVelocity: 0.03,
    minRepFrames: 4,

    // Angled view: allow some foreshortening; still catch real cave-ins.
    lateral: {
      minKneeAnkleRatio: 0.62,
      kneeFlareOut: 0.48,
      hipShift: 0.45,
      warnFrames: 10,
    },

    validateAngle() {
      // Never hard-block reps on camera angle. Soft tip only in checkForm.
      return true;
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      const c = a.primary;
      switch (phase) {
        case "READY":
          return c > t.standClearance ? "DESCENDING" : null;
        case "DESCENDING":
          return c < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":
          return c > t.leaveBottom ? "ASCENDING" : null;
        case "ASCENDING":
          return c > t.standClearance ? "COMPLETE" : null;
        default:
          return null;
      }
    },

    stabilityChecks() {
      // Lateral issues are coached visually; they do not void the rep.
      return { pass: true };
    },

    checkForm(a, phase, cam, engine) {
      const emptyOverlay = { formOk: true, badLandmarks: [], guideLines: [], flags: null };

      // Soft tip only for an extreme profile (far leg nearly invisible).
      if (
        cam
        && cam.viewAngle === "side"
        && (cam.shoulderWidthRatio ?? 1) < 0.22
        && phase === "READY"
      ) {
        a.squatOverlay = emptyOverlay;
        return { type: "info", msg: this.angleHint };
      }

      const state = engine._formState.squat || (engine._formState.squat = {
        cave: 0, leftFlare: 0, rightFlare: 0,
      });

      // Correction lines / cave checks only while actually squatting (hips down).
      const isLoaded = (a.primary ?? 1) < this.thresholds.formActiveClearance;
      if (!isLoaded) {
        state.cave = 0;
        state.leftFlare = 0;
        state.rightFlare = 0;
        a.squatOverlay = emptyOverlay;
        if (phase === "READY" || phase === "DESCENDING" || phase === "ASCENDING") {
          return { type: "good", msg: "Stand tall, ready to squat" };
        }
        return null;
      }

      const lat = this.lateral;
      const stance = Math.max(a.stanceWidth || 0.12, a.usingWorld ? 0.10 : 0.05);
      const flareThresh = lat.kneeFlareOut * stance;
      const widthRatio = a.kneeAnkleWidthRatio ?? 1;

      const tick = (key, active) => {
        if (active) state[key] = Math.min(state[key] + 1, 40);
        else state[key] = Math.max(state[key] - 2, 0);
        return state[key] >= lat.warnFrames;
      };

      const caveActive = tick("cave", widthRatio < lat.minKneeAnkleRatio);
      const leftMed = a.leftMedial ?? 0;
      const rightMed = a.rightMedial ?? 0;
      const leftFlare = tick("leftFlare", widthRatio > 1.2 && leftMed < -flareThresh);
      const rightFlare = tick("rightFlare", widthRatio > 1.2 && rightMed < -flareThresh);

      a.squatOverlay = buildSquatLateralOverlay(a.guidePts, {
        leftCave: caveActive,
        rightCave: caveActive,
        leftFlare,
        rightFlare,
        hipShift: false,
      });

      if (caveActive) {
        return { type: "warning", msg: "Knees caving in, push them out over your toes" };
      }
      if (leftFlare || rightFlare) {
        return { type: "warning", msg: "Knees flaring out, track them over your toes" };
      }

      if (phase === "DESCENDING") {
        return { type: "info", msg: "Keep your chest up, sit down between your heels" };
      }
      if (phase === "BOTTOM") {
        return { type: "good", msg: "Good depth — drive up, knees out" };
      }
      if (phase === "ASCENDING") {
        return { type: "good", msg: "Drive up through your mid-foot!" };
      }
      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  BICEP CURL
   *  Best from side view; elbow flexion is a sagittal-plane motion
   * ══════════════════════════════════════════════════════════ */
  bicep_curl: {
    label: "Bicep Curl",
    type: "rep",
    preferredAngles: ["side"],
    angleHint: "Turn sideways to camera, curls are best tracked from the side",
    bodyOrientation: "upright",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP],
    requiredLandmarkGroups: [
      [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST],
      [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST],
    ],

    computeAngles(p, engine) {
      const activeArm = chooseCurlArm(p, engine._activeArm);
      engine._activeArm = activeArm;

      // Never throw — a crash here stops the whole draw loop / skeleton.
      if (!activeArm) {
        return { primary: 180, elbowAwayNorm: 0, peakElbowAway: 0, activeArm: null };
      }

      const shoulder = p[activeArm === "left" ? LM.L_SHOULDER : LM.R_SHOULDER];
      const elbow = p[activeArm === "left" ? LM.L_ELBOW : LM.R_ELBOW];
      const wrist = p[activeArm === "left" ? LM.L_WRIST : LM.R_WRIST];
      const hip = p[activeArm === "left" ? LM.L_HIP : LM.R_HIP];
      if (!shoulder || !elbow || !wrist || !hip) {
        return { primary: 180, elbowAwayNorm: 0, peakElbowAway: 0, activeArm };
      }

      const curlAngle = calculateAngle(shoulder, elbow, wrist);

      // Distance of elbow from the shoulder→hip body line, in torso lengths.
      const torsoLen = Math.max(dist(shoulder, hip), 0.08);
      const ySpan = hip.y - shoulder.y;
      const t = Math.abs(ySpan) > 1e-4
        ? clamp((elbow.y - shoulder.y) / ySpan, -0.2, 1.2)
        : 0;
      const bodyX = shoulder.x + t * (hip.x - shoulder.x);
      const rawAway = Math.abs(elbow.x - bodyX) / torsoLen;

      if (!engine._curlBaseline || engine._curlBaseline.arm !== activeArm) {
        engine._curlBaseline = {
          arm: activeArm,
          smoothedAway: rawAway,
          peakAway: rawAway,
        };
      } else {
        const ALPHA = 0.30;
        engine._curlBaseline.smoothedAway =
          ALPHA * rawAway + (1 - ALPHA) * engine._curlBaseline.smoothedAway;
        engine._curlBaseline.peakAway = Math.max(
          engine._curlBaseline.peakAway,
          engine._curlBaseline.smoothedAway
        );
      }

      return {
        primary: curlAngle,
        elbowAwayNorm: engine._curlBaseline.smoothedAway,
        peakElbowAway: engine._curlBaseline.peakAway,
        activeArm,
      };
    },

    phases: ["READY", "CONTRACTING", "EXTENDING", "COMPLETE"],
    thresholds: {},
    minTransitionMs: 80,
    minROM: 65,
    minRepDurationMs: 450,
    minVelocity: 0,
    minRepFrames: 4,

    // Elbow must stay close to the body line (torso-length units).
    elbowInline: {
      warn: 0.10,
      fail: 0.14,
    },

    validateAngle(cam) {
      return cam.viewAngle === "side";
    },

    transitionRules(a, phase) {
      switch (phase) {
        case "READY":
          return a.primary > 140 ? "CONTRACTING" : null;
        case "CONTRACTING":
          return a.primary < 75 ? "EXTENDING" : null;
        case "EXTENDING":
          return a.primary > 140 ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a, engine) {
      // Reject the rep if the elbow left the body line at any point this cycle.
      const peak = a.peakElbowAway ?? a.elbowAwayNorm ?? 0;
      const cb = engine._curlBaseline;
      if (cb) cb.peakAway = 0;

      if (peak > this.elbowInline.fail) {
        return {
          pass: false,
          reason: "Elbow is away from your body, keep it inline",
        };
      }
      return { pass: true };
    },

    checkForm(a, phase, cam, engine) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }

      // Elbow anchoring — short hysteresis so the cue kicks in quickly.
      const state = engine._formState.bicep_curl || (engine._formState.bicep_curl = {
        awayFrames: 0,
      });
      const away = a.elbowAwayNorm ?? 0;

      if (away > this.elbowInline.warn) {
        state.awayFrames = Math.min(state.awayFrames + 1, 40);
      } else {
        state.awayFrames = Math.max(state.awayFrames - 2, 0);
      }

      if (state.awayFrames >= 6) {
        return { type: "warning", msg: "Elbow is away from your body, keep it inline" };
      }

      const isExtending = engine._velocityTracker.velocity > 0;

      if (phase === "CONTRACTING" && isExtending && a.primary > 75 && a.primary < 125) {
        return { type: "warning", msg: "Curl all the way up to finish the rep!" };
      }

      if (phase === "EXTENDING" && !isExtending && a.primary > 75 && a.primary < 140) {
        return { type: "warning", msg: "Extend your arm fully at the bottom." };
      }

      if (phase === "READY" || phase === "EXTENDING") return { type: "good", msg: "Extend arm fully downward." };
      if (phase === "CONTRACTING") return { type: "good", msg: "Curl up for full contraction!" };

      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  PUSH-UP
   *  Best from side view; shows elbow bend, body line, hip alignment
   * ══════════════════════════════════════════════════════════ */
  push_up: {
    label: "Push-Up",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Place the camera on the floor beside you — true side / profile view",
    bodyOrientation: "horizontal",
    // Side view often hides the far arm; accept either full side.
    // Slightly looser groups: body line OR arm is enough to start tracking.
    requiredLandmarks: [],
    requiredLandmarkGroups: [
      [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST],
      [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST],
    ],
    trackingLandmarks: [
      LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST,
      LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE,
      LM.L_EAR, LM.R_EAR, LM.L_KNEE, LM.R_KNEE, LM.NOSE,
    ],

    computeAngles(p, engine) {
      // Softer visibility for side-view near arm (left side was dropping out at 0.4).
      const armVis = (pt) => vis(pt, 0.28);
      const bodyVis = (pt) => vis(pt, 0.32);

      const leftArmOk = armVis(p[LM.L_SHOULDER]) && armVis(p[LM.L_ELBOW]) && armVis(p[LM.L_WRIST]);
      const rightArmOk = armVis(p[LM.R_SHOULDER]) && armVis(p[LM.R_ELBOW]) && armVis(p[LM.R_WRIST]);
      const leftBodyOk = bodyVis(p[LM.L_SHOULDER]) && bodyVis(p[LM.L_HIP]) && bodyVis(p[LM.L_ANKLE]);
      const rightBodyOk = bodyVis(p[LM.R_SHOULDER]) && bodyVis(p[LM.R_HIP]) && bodyVis(p[LM.R_ANKLE]);

      const scoreArm = (side) => {
        const isLeft = side === "left";
        const sh = p[isLeft ? LM.L_SHOULDER : LM.R_SHOULDER];
        const el = p[isLeft ? LM.L_ELBOW : LM.R_ELBOW];
        const wr = p[isLeft ? LM.L_WRIST : LM.R_WRIST];
        if (!armVis(sh) || !armVis(el) || !armVis(wr)) return -Infinity;
        // Prefer the arm that looks more extended in image space (clearer side profile).
        const len = dist(sh, el) + dist(el, wr);
        const conf = (sh.visibility + el.visibility + wr.visibility) / 3;
        return conf * 2 + len;
      };

      const state = engine._formState.pushupTrack || (engine._formState.pushupTrack = {
        side: null,
        lastElbow: 155,
        readyFrames: 0,
        armed: false,
      });

      const leftScore = scoreArm("left");
      const rightScore = scoreArm("right");
      let side = state.side;
      // Sticky side — only swap if the other arm is clearly better (fixes left-side flicker).
      if (side === "left" && leftScore > -Infinity && leftScore >= rightScore - 0.35) {
        side = "left";
      } else if (side === "right" && rightScore > -Infinity && rightScore >= leftScore - 0.35) {
        side = "right";
      } else if (leftScore > -Infinity || rightScore > -Infinity) {
        side = leftScore >= rightScore ? "left" : "right";
      } else if (leftBodyOk) {
        side = "left";
      } else if (rightBodyOk) {
        side = "right";
      } else {
        side = state.side;
      }
      if (side) state.side = side;

      const isLeft = side === "left";
      const shoulder = side ? p[isLeft ? LM.L_SHOULDER : LM.R_SHOULDER] : null;
      const elbow = side ? p[isLeft ? LM.L_ELBOW : LM.R_ELBOW] : null;
      const wrist = side ? p[isLeft ? LM.L_WRIST : LM.R_WRIST] : null;
      const hip = side ? p[isLeft ? LM.L_HIP : LM.R_HIP] : null;
      const ankle = side ? p[isLeft ? LM.L_ANKLE : LM.R_ANKLE] : null;

      // Arms drive the rep counter. Never invent a fake 180° when the arm is missing
      // (that was auto-advancing READY → DESCENDING while walking into frame).
      let armVisible = false;
      let primary = state.lastElbow;
      if (shoulder && elbow && wrist && armVis(shoulder) && armVis(elbow) && armVis(wrist)) {
        primary = calculateAngle(shoulder, elbow, wrist);
        state.lastElbow = primary;
        armVisible = true;
      }

      let bodyLine = 180;
      let hipDeviation = 0;
      const bodyOk = !!(shoulder && hip && ankle && bodyVis(shoulder) && bodyVis(hip) && bodyVis(ankle));
      if (bodyOk) {
        bodyLine = calculateAngle(shoulder, hip, ankle);
        hipDeviation = hipOffsetFromBodyLine(shoulder, hip, ankle);
      }

      // Stance gate: must look like a floor push-up, not standing / walking in.
      const horizontal = !!(engine._cameraAngle && engine._cameraAngle.isHorizontal);
      const wristUnder = !!(shoulder && wrist && armVisible && wrist.y > shoulder.y + 0.015);
      const torsoFlat = bodyOk && Math.abs(shoulder.y - ankle.y) < 0.42;
      const inPushupStance = armVisible && (horizontal || torsoFlat) && wristUnder && bodyLine > 120;

      if (inPushupStance && primary > 125) {
        state.readyFrames = Math.min(state.readyFrames + 1, 40);
        if (state.readyFrames >= 10) state.armed = true;
      } else if (!inPushupStance) {
        state.readyFrames = Math.max(0, state.readyFrames - 3);
        if (state.readyFrames < 4) state.armed = false;
      }

      if (side) engine._activeArm = side;

      return {
        primary,
        bodyLine,
        hipDeviation,
        activeArm: side,
        armVisible,
        inPushupStance,
        armed: state.armed,
        shoulderHeight: shoulder?.y ?? 0.5,
      };
    },

    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    thresholds: {
      // Side/left foreshortening compresses elbow angles — use gentler gates.
      // ROM check still requires a real up→down→up movement.
      startExtended: 135,
      enterBottom: 152,
      leaveBottom: 156,
      extendBack: 142,
      // Body-line form (line offset, not ground angle)
      minBodyLine: 145,
      warnSag: 6.5,
      warnPike: -6.5,
    },
    minTransitionMs: 100,
    minROM: 18,
    minRepDurationMs: 500,
    minVelocity: 3,
    minRepFrames: 2,

    validateAngle() {
      return true;
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      // No phase progress without a real arm reading — blocks walk-in false reps.
      if (!a.armVisible) return null;
      // Must settle in a push-up stance before the first descent can start.
      if (!a.armed && (phase === "READY" || phase === null)) return null;

      switch (phase) {
        case "READY":      return a.primary > t.startExtended ? "DESCENDING" : null;
        case "DESCENDING": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":     return a.primary > t.leaveBottom ? "ASCENDING" : null;
        case "ASCENDING":  return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      // Void the rep if it finished while clearly not in a push-up stance.
      if (!a.inPushupStance && !a.armed) {
        return { pass: false, reason: "Get into push-up position first" };
      }
      return { pass: true };
    },

    checkForm(a, phase, cam, engine) {
      // Sticky coach patterned on plank — form uses body-line offset, not floor tilt.
      const coach = engine._formState.pushupCoach || (engine._formState.pushupCoach = {
        candidateMsg: null,
        candidateType: null,
        candidateCount: 0,
        active: null,
        lockedGoodUntil: 0,
      });
      const t = this.thresholds;
      const now = performance.now();

      let next;
      if (!a.armed || !a.inPushupStance) {
        next = { type: "info", msg: "Get into push-up position, body straight, then begin" };
      } else if (a.hipDeviation > t.warnSag + 1.5) {
        next = { type: "warning", msg: "Hips sagging, tighten your core" };
      } else if (a.hipDeviation < t.warnPike - 1.5) {
        next = { type: "warning", msg: "Hips too high, flatten out" };
      } else if (a.bodyLine < t.minBodyLine - 12) {
        next = { type: "warning", msg: "Keep your body in a straight line" };
      } else if (phase === "BOTTOM" && a.primary > 155) {
        next = { type: "info", msg: "Go a bit lower, chest toward the floor" };
      } else if (phase === "BOTTOM") {
        next = { type: "good", msg: "Good depth, keep hips in line" };
      } else if (phase === "ASCENDING") {
        next = { type: "good", msg: "Push up strong, keep hips in line" };
      } else if (phase === "DESCENDING") {
        next = { type: "good", msg: "Lower with control, keep hips in line" };
      } else {
        next = { type: "good", msg: "Keep hips in line with your body, begin your push-up" };
      }

      const inRep = phase === "DESCENDING" || phase === "BOTTOM" || phase === "ASCENDING" || phase === "READY";
      if (
        inRep
        && coach.active
        && coach.active.type === "good"
        && next.type === "good"
        && next.msg === coach.active.msg
      ) {
        coach.lockedGoodUntil = Math.max(coach.lockedGoodUntil || 0, now + 2500);
        coach.candidateCount = 0;
        return coach.active;
      }
      // Soft-lock good form: ignore brief warning blips mid-rep.
      if (
        inRep
        && coach.active
        && coach.active.type === "good"
        && next.type === "warning"
        && now < (coach.lockedGoodUntil || 0)
      ) {
        coach.candidateCount = 0;
        return coach.active;
      }

      // Phase tips (good/info) may update a bit faster than hard warnings.
      const stickyFrames = next.type === "warning" ? 40 : (next.type === "info" ? 18 : 12);
      if (!coach.active) {
        coach.active = next;
        coach.candidateMsg = next.msg;
        coach.candidateType = next.type;
        coach.candidateCount = 0;
        if (next.type === "good") coach.lockedGoodUntil = now + 2500;
        return coach.active;
      }
      if (next.msg === coach.active.msg) {
        coach.candidateCount = 0;
        if (next.type === "good") {
          coach.lockedGoodUntil = Math.max(coach.lockedGoodUntil || 0, now + 2500);
        }
        return coach.active;
      }
      if (next.msg === coach.candidateMsg) {
        coach.candidateCount += 1;
      } else {
        coach.candidateMsg = next.msg;
        coach.candidateType = next.type;
        coach.candidateCount = 1;
      }
      if (coach.candidateCount >= stickyFrames) {
        coach.active = { type: coach.candidateType, msg: coach.candidateMsg };
        coach.candidateCount = 0;
        if (coach.active.type === "good") coach.lockedGoodUntil = now + 2500;
      }
      return coach.active;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  PLANK (hold exercise)
   *  Side view reveals body line; front view shows hip sag
   * ══════════════════════════════════════════════════════════ */
  plank: {
    label: "Plank",
    type: "hold",
    preferredAngles: ["side", "angled"],
    angleHint: "Place the camera on the floor beside you — true side / profile view",
    bodyOrientation: "horizontal",
    requiredLandmarks: [],
    requiredLandmarkGroups: [
      // Need an arm support point so lying flat on the floor doesn't count as a plank.
      [LM.L_SHOULDER, LM.L_HIP, LM.L_ANKLE, LM.L_ELBOW],
      [LM.R_SHOULDER, LM.R_HIP, LM.R_ANKLE, LM.R_ELBOW],
      [LM.L_SHOULDER, LM.L_HIP, LM.L_ANKLE, LM.L_WRIST],
      [LM.R_SHOULDER, LM.R_HIP, LM.R_ANKLE, LM.R_WRIST],
    ],
    trackingLandmarks: [
      LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE,
      LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST,
      LM.L_EAR, LM.R_EAR, LM.L_KNEE, LM.R_KNEE, LM.NOSE,
    ],

    computeAngles(p, engine) {
      // Softer visibility + sticky side (left profile was flickering / dropping out).
      const soft = (pt) => vis(pt, 0.28);
      const track = engine._formState.plankTrack || (engine._formState.plankTrack = {
        side: null,
        lastBodyLine: 170,
        lastHipDev: 0,
      });

      const scoreSide = (side) => {
        const isLeft = side === "left";
        const sh = p[isLeft ? LM.L_SHOULDER : LM.R_SHOULDER];
        const hip = p[isLeft ? LM.L_HIP : LM.R_HIP];
        const ank = p[isLeft ? LM.L_ANKLE : LM.R_ANKLE];
        const el = p[isLeft ? LM.L_ELBOW : LM.R_ELBOW];
        const wr = p[isLeft ? LM.L_WRIST : LM.R_WRIST];
        if (!soft(sh) || !soft(hip) || !soft(ank)) return -Infinity;
        const conf = (sh.visibility + hip.visibility + ank.visibility) / 3;
        const len = dist(sh, hip) + dist(hip, ank);
        const armBonus = (soft(el) ? 0.35 : 0) + (soft(wr) ? 0.25 : 0);
        return conf * 2 + len + armBonus;
      };

      const leftScore = scoreSide("left");
      const rightScore = scoreSide("right");
      let side = track.side;
      if (side === "left" && leftScore > -Infinity && leftScore >= rightScore - 0.4) {
        side = "left";
      } else if (side === "right" && rightScore > -Infinity && rightScore >= leftScore - 0.4) {
        side = "right";
      } else if (leftScore > -Infinity || rightScore > -Infinity) {
        side = leftScore >= rightScore ? "left" : "right";
      }
      if (side) track.side = side;

      // Keep form-check side in sync so red/green guides use the same limb.
      const formState = engine._formState.plank || (engine._formState.plank = {});
      if (side) formState.visibleSide = side;

      if (!side) {
        return {
          primary: track.lastBodyLine,
          bodyLine: track.lastBodyLine,
          hipDeviation: track.lastHipDev,
          formAllowsHold: false,
          inPlankStance: false,
          sideVisible: false,
        };
      }

      const isLeft = side === "left";
      const shoulder = p[isLeft ? LM.L_SHOULDER : LM.R_SHOULDER];
      const hip = p[isLeft ? LM.L_HIP : LM.R_HIP];
      const ankle = p[isLeft ? LM.L_ANKLE : LM.R_ANKLE];
      const elbow = p[isLeft ? LM.L_ELBOW : LM.R_ELBOW];
      const wrist = p[isLeft ? LM.L_WRIST : LM.R_WRIST];

      if (!soft(shoulder) || !soft(hip) || !soft(ankle)) {
        return {
          primary: track.lastBodyLine,
          bodyLine: track.lastBodyLine,
          hipDeviation: track.lastHipDev,
          formAllowsHold: false,
          inPlankStance: false,
          sideVisible: false,
          activeArm: side,
        };
      }

      const bodyLine = calculateAngle(shoulder, hip, ankle);
      const hipDeviation = hipOffsetFromBodyLine(shoulder, hip, ankle);
      track.lastBodyLine = bodyLine;
      track.lastHipDev = hipDeviation;

      const horizontal = !!(engine._cameraAngle && engine._cameraAngle.isHorizontal);
      const torsoFlat = Math.abs(shoulder.y - ankle.y) < 0.42;

      // True plank = body supported on arms. Lying flat on the floor has shoulders
      // at roughly the same height as elbows/wrists — that must NOT count.
      const elbowSupport = soft(elbow) && shoulder.y < elbow.y - 0.035;
      const wristSupport = soft(wrist) && shoulder.y < wrist.y - 0.045;
      const armSupporting = elbowSupport || wristSupport;

      // Lying prone: torso on the floor, arm not propping the shoulders up.
      const lyingFlat = !armSupporting
        || (Math.abs(shoulder.y - hip.y) < 0.03 && Math.abs(hip.y - ankle.y) < 0.045 && !armSupporting);

      const inPlankStance = !lyingFlat
        && armSupporting
        && (horizontal || torsoFlat)
        && bodyLine > 120;

      // Timer may run only in a real supported plank with inline hips.
      const t = this.thresholds;
      const formAllowsHold = inPlankStance
        && bodyLine >= t.minBodyLine - 8
        && Math.abs(hipDeviation) <= t.maxHipDev + 1.2;

      return {
        primary: bodyLine,
        bodyLine,
        hipDeviation,
        formAllowsHold,
        inPlankStance,
        sideVisible: true,
        armSupporting,
        activeArm: side,
      };
    },

    phases: ["NOT_IN_POSITION", "HOLDING"],
    // Moderate inline band; leave HOLDING if form clearly breaks (so timer pauses).
    thresholds: {
      minBodyLine: 145,
      maxHipDev: 6.0,
      warnSag: 6.0,
      warnPike: -6.0,
    },
    minTransitionMs: 400,
    holdTimeDisplay: true,

    validateAngle() {
      return true;
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      // Must be in a real plank stance — blocks walk-in / standing false holds.
      const enterOk = a.inPlankStance
        && a.sideVisible
        && a.primary > t.minBodyLine
        && Math.abs(a.hipDeviation) < t.maxHipDev;
      // Tighter stay band than before — incorrect form should leave HOLDING.
      const stayOk = a.sideVisible
        && a.inPlankStance
        && a.primary > t.minBodyLine - 14
        && Math.abs(a.hipDeviation) < t.maxHipDev + 2.5;
      switch (phase) {
        case "NOT_IN_POSITION": return enterOk ? "HOLDING" : null;
        case "HOLDING":         return stayOk ? null : "NOT_IN_POSITION";
        default: return null;
      }
    },

    checkForm(a, phase, cam, engine) {
      const coach = engine._formState.plankCoach || (engine._formState.plankCoach = {
        candidateMsg: null,
        candidateType: null,
        candidateCount: 0,
        active: null,
        lockedGoodUntil: 0,
      });
      const t = this.thresholds;
      const now = performance.now();
      const GOOD = { type: "good", msg: "Keep hips in line with your body, hold steady!" };

      let next;
      if (!a.sideVisible || !a.armSupporting) {
        next = { type: "info", msg: "Get into a plank on your hands or forearms, not lying flat" };
      } else if (!a.inPlankStance || phase !== "HOLDING") {
        next = { type: "info", msg: "Get into plank position, straight body line" };
      } else if (a.hipDeviation > t.warnSag + 1.2) {
        next = { type: "warning", msg: "Hips sagging, tighten your core" };
      } else if (a.hipDeviation < t.warnPike - 1.2) {
        next = { type: "warning", msg: "Hips too high, flatten out" };
      } else if (a.primary < t.minBodyLine - 10) {
        next = { type: "warning", msg: "Straighten your body, maintain alignment" };
      } else {
        next = GOOD;
      }

      // While holding a correct plank, keep the good message locked for several
      // seconds so comments don't thrash while the timer is running.
      if (
        phase === "HOLDING"
        && coach.active
        && coach.active.type === "good"
        && next.type === "good"
      ) {
        coach.lockedGoodUntil = Math.max(coach.lockedGoodUntil || 0, now + 4000);
        coach.candidateCount = 0;
        coach.candidateMsg = GOOD.msg;
        return coach.active;
      }
      if (
        phase === "HOLDING"
        && coach.active
        && coach.active.type === "good"
        && next.type !== "good"
        && a.formAllowsHold
        && now < (coach.lockedGoodUntil || 0)
      ) {
        // Ignore brief warning blips only while form is still hold-legal.
        coach.candidateCount = 0;
        return coach.active;
      }
      // Clear the good-lock immediately when form no longer allows the timer.
      if (!a.formAllowsHold && coach.active?.type === "good" && next.type !== "good") {
        coach.lockedGoodUntil = 0;
      }

      // Hard to leave a message: good switches in quickly, warnings need a long hold.
      const stickyFrames = next.type === "good" ? 10 : (next.type === "warning" ? 20 : 45);
      if (!coach.active) {
        coach.active = next;
        coach.candidateMsg = next.msg;
        coach.candidateType = next.type;
        coach.candidateCount = 0;
        if (next.type === "good" && phase === "HOLDING") {
          coach.lockedGoodUntil = now + 4000;
        }
        return coach.active;
      }
      if (next.msg === coach.active.msg) {
        coach.candidateMsg = next.msg;
        coach.candidateCount = 0;
        if (next.type === "good" && phase === "HOLDING") {
          coach.lockedGoodUntil = Math.max(coach.lockedGoodUntil || 0, now + 4000);
        }
        return coach.active;
      }
      if (next.msg === coach.candidateMsg) {
        coach.candidateCount += 1;
      } else {
        coach.candidateMsg = next.msg;
        coach.candidateType = next.type;
        coach.candidateCount = 1;
      }
      if (coach.candidateCount >= stickyFrames) {
        coach.active = { type: coach.candidateType, msg: coach.candidateMsg };
        coach.candidateCount = 0;
        if (coach.active.type === "good" && phase === "HOLDING") {
          coach.lockedGoodUntil = now + 4000;
        }
      }
      return coach.active;
    },
  },
};

/* ───────────────────────────────────────────────────────────────
 * GymMetricEngine
 *
 * Camera-angle-aware, phase-based, multi-signal rep counter
 * with conservative counting bias (prefer missing a rep over
 * counting a false positive).
 *
 * Anti-false-positive layers:
 *  1. Camera angle validation; wrong angle blocks counting
 *  2. Landmark visibility gating
 *  3. Phase-based state machine (must complete full cycle)
 *  4. Min time-in-phase (hysteresis)
 *  5. Min rep duration (full cycle timing)
 *  6. Min range-of-motion check
 *  7. Min velocity check (deliberate movement)
 *  8. Stability checks (exercise-specific posture gates)
 *  9. Frame-count debounce between reps
 * 10. Angular smoothing via OneEuroFilter
 * ────────────────────────────────────────────────────────────── */
export class GymMetricEngine {
  constructor() {
    this.repCount = 0;
    this.filters = {};
    this.currentPhase = null;
    this.phaseEnteredAt = 0;
    this.repCycleStartedAt = 0;   // when the current rep cycle began
    this.peakAngle = null;        // track ROM: max angle in cycle
    this.valleyAngle = null;      // track ROM: min angle in cycle
    this.holdStartTime = 0;
    this.totalHoldTime = 0;
    this._holdPausedTotal = 0;    // plank: banked seconds while form was good
    this._lastExercise = null;
    this._feedbackCooldown = 0;
    this._lastFeedback = null;
    this._noRepFrames = 0;
    this._velocityTracker = new VelocityTracker(6);
    this._cameraAngle = null;
    this._angleWarningShown = false;
    this._stabilityFailCount = 0;  // consecutive frames of stability failure
    this._activeArm = null;
    this._maxCycleSpeed = 0;
    this._formState = {};
    this._calibrator = new PushupPlankCalibrator();
    this._calibrationState = null; // last calibrator tick result for push_up/plank
  }

  static get exerciseKeys() {
    return Object.keys(EXERCISE_PROFILES);
  }

  static getProfile(key) {
    return EXERCISE_PROFILES[key] || null;
  }

  /** Landmark subset shown in the overlay for the selected exercise. */
  static getDisplayLandmarks(key, activeArm = null) {
    const profile = EXERCISE_PROFILES[key];
    if (!profile) return [];

    if (key === "bicep_curl") {
      const torso = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP];
      return [...torso, LM.L_ELBOW, LM.L_WRIST, LM.R_ELBOW, LM.R_WRIST];
    }

    // Push-up / plank: show tracked body line (either side) + ears/knees for form paint.
    if (key === "push_up" || key === "plank") {
      const groups = (profile.requiredLandmarkGroups || []).flat();
      const tracked = profile.trackingLandmarks || [];
      return [...new Set([...profile.requiredLandmarks, ...groups, ...tracked])];
    }

    return profile.requiredLandmarks;
  }

  reset() {
    this.repCount = 0;
    this.filters = {};
    this.currentPhase = null;
    this.phaseEnteredAt = 0;
    this.repCycleStartedAt = 0;
    this.peakAngle = null;
    this.valleyAngle = null;
    this.holdStartTime = 0;
    this.totalHoldTime = 0;
    this._holdPausedTotal = 0;
    this._lastExercise = null;
    this._feedbackCooldown = 0;
    this._lastFeedback = null;
    this._noRepFrames = 0;
    this._velocityTracker.reset();
    this._cameraAngle = null;
    this._angleWarningShown = false;
    this._stabilityFailCount = 0;
    this._activeArm = null;
    this._maxCycleSpeed = 0;
    this._formState = {};
    this._calibrator.reset();
    this._calibrationState = null;
    this._curlBaseline = null;
    this._squatDepthSmoothed = null;
  }

  smoothPoint(idx, pt) {
    if (!this.filters[idx]) {
      this.filters[idx] = new EMAFilter(0.5); // alpha = 0.5 as requested
    }
    const smoothed = this.filters[idx].filter(pt.x, pt.y);
    return {
      x: smoothed.x,
      y: smoothed.y,
      z: pt.z,
      visibility: pt.visibility,
    };
  }

  /**
   * Main per-frame evaluation.
   * @param {Array} landmarks 2D image-space landmarks
   * @param {string} activeExercise exercise key
   * @param {Array} [worldLandmarks] 3D world-space landmarks (meters, hip-centered)
   * @returns {{ reps, feedback, feedbackType, holdTime, phase, cameraAngle, angleOk, calibrationState }}
   */
  evaluateFrame(landmarks, activeExercise, worldLandmarks = null) {
    const now = performance.now();
    const timeSec = now / 1000;

    if (activeExercise !== this._lastExercise) {
      this.reset();
      this._lastExercise = activeExercise;
    }

    const profile = EXERCISE_PROFILES[activeExercise];
    if (!profile || !landmarks || landmarks.length < 33) {
      return this._result(profile, "");
    }

    // ── Filter and Smooth Required Landmarks Only ─────────
    const pts = [];
    const cameraAngleIndices = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP];
    const alternativeGroups = profile.requiredLandmarkGroups || [];
    const alternativeIndices = alternativeGroups.flat();
    const trackingIndices = profile.trackingLandmarks || [];
    const indicesToProcess = new Set([
      ...profile.requiredLandmarks,
      ...alternativeIndices,
      ...trackingIndices,
      ...cameraAngleIndices,
    ]);
    
    for (const idx of indicesToProcess) {
      if (landmarks[idx]) {
        pts[idx] = this.smoothPoint(idx, landmarks[idx]);
      }
    }

    // ── Visibility gate ───────────────────────────────────
    const allRequiredVisible = profile.requiredLandmarks.every(idx => vis(pts[idx]));
    const oneAlternativeVisible = alternativeGroups.length === 0 || alternativeGroups.some(
      (group) => group.every((idx) => vis(pts[idx]))
    );
    if (!allRequiredVisible || !oneAlternativeVisible) {
      return this._result(profile, "Can't see key joints, adjust your position", "warning");
    }

    // ── Camera angle detection ────────────────────────────
    this._cameraAngle = detectCameraAngle(pts);
    const angleOk = profile.validateAngle ? profile.validateAngle(this._cameraAngle) : true;

    // ── Push-up / Plank: form coaching (no calibration gate — it blocked reps) ──
    let calibrationState = null;
    let formCheck = null;
    if (activeExercise === "push_up" || activeExercise === "plank") {
      const fsKey = activeExercise;
      const formState = this._formState[fsKey] || (this._formState[fsKey] = {});
      // Keep calibrator marked ready so any leftover UI stays quiet.
      calibrationState = {
        state: "calibrated",
        progressPct: 1,
        prompt: null,
        rejectionReason: null,
      };
      this._calibrationState = calibrationState;

      const formOpts = { worldLandmarks };
      if (activeExercise === "push_up") {
        formCheck = checkPushupForm(pts, formState, formOpts);
      } else {
        formCheck = checkPlankForm(pts, formState, formOpts);
      }
    } else {
      formCheck = {
        isCorrect: true,
        hipAngle: null,
        neckAngle: null,
        issues: [],
        guideLines: [],
        badLandmarks: [],
        rawIsCorrect: true,
      };
    }
    const formOk = formCheck ? formCheck.isCorrect : true;

    // ── Compute angles ────────────────────────────────────
    // Third arg (worldLandmarks) is used by squat; other profiles ignore it.
    const angles = profile.computeAngles(pts, this, worldLandmarks);

    // ── Track velocity ────────────────────────────────────
    this._velocityTracker.push(angles.primary, timeSec);
    this._maxCycleSpeed = Math.max(this._maxCycleSpeed, this._velocityTracker.speed);

    // ── Track ROM within current rep cycle ─────────────────
    if (this.peakAngle === null) {
      this.peakAngle = angles.primary;
      this.valleyAngle = angles.primary;
    } else {
      this.peakAngle = Math.max(this.peakAngle, angles.primary);
      this.valleyAngle = Math.min(this.valleyAngle, angles.primary);
    }

    // ── Initialize phase ──────────────────────────────────
    if (this.currentPhase === null) {
      this.currentPhase = profile.phases[0];
      this.phaseEnteredAt = now;
      this.repCycleStartedAt = now;
    }

    // ── State machine transitions ─────────────────────────
    const timeInPhase = now - this.phaseEnteredAt;
    const nextPhase = profile.transitionRules.call(profile, angles, this.currentPhase);

    let repJustCounted = false;

    if (nextPhase !== null && timeInPhase >= profile.minTransitionMs) {
      if (nextPhase === "COMPLETE" && profile.type === "rep") {
        // ── STRICT REP VALIDATION ───────────────────────
        const repDuration = now - this.repCycleStartedAt;
        const rom = (this.peakAngle !== null && this.valleyAngle !== null)
          ? this.peakAngle - this.valleyAngle : 0;
        const hasMinROM = !profile.minROM || rom >= profile.minROM;
        const hasMinDuration = !profile.minRepDurationMs || repDuration >= profile.minRepDurationMs;
        const hasMinVelocity = !profile.minVelocity || this._maxCycleSpeed >= profile.minVelocity;
        const hasMinFrames = this._noRepFrames >= (profile.minRepFrames ?? 4);
        const stabilityResult = profile.stabilityChecks ? profile.stabilityChecks(angles, this) : { pass: true };
        const isAngleOk = angleOk;
        // Push-up / plank: form is coached with red segments; do not void the rep.
        const isFormOk = true;

        // All checks must pass to count the rep
        if (hasMinROM && hasMinDuration && hasMinVelocity && hasMinFrames && stabilityResult.pass && isAngleOk && isFormOk) {
          this.repCount++;
          repJustCounted = true;
          this._noRepFrames = 0;
          this._stabilityFailCount = 0;
        } else {
          // Do not count; but still let the cycle continue
          // If stability failed, note it for feedback
          if (!stabilityResult.pass) {
            this._stabilityFailCount++;
            this._lastFeedback = { type: "warning", msg: stabilityResult.reason };
            this._feedbackCooldown = now + 1200;
          }
          if (!hasMinROM) {
            this._lastFeedback = { type: "warning", msg: "Partial rep, go through full range of motion" };
            this._feedbackCooldown = now + 1200;
          }
          if (!hasMinVelocity) {
            this._lastFeedback = { type: "warning", msg: "Move through the rep with control" };
            this._feedbackCooldown = now + 1200;
          }
          if (!isAngleOk) {
            this._lastFeedback = { type: "info", msg: profile.angleHint || "Adjust camera angle" };
            this._feedbackCooldown = now + 1500;
          }
          if (!isFormOk) {
            const message = formCheck?.issues?.length
              ? `Fix form: ${formCheck.issues.join(" and ")}`
              : "Fix your body line before counting reps";
            this._lastFeedback = { type: "warning", msg: message };
            this._feedbackCooldown = now + 1200;
          }
        }

        // Cycle back to first active phase regardless
        this.currentPhase = profile.phases[1];
        this.repCycleStartedAt = now;
        this.peakAngle = angles.primary;
        this.valleyAngle = angles.primary;
        this._maxCycleSpeed = 0;
      } else if (nextPhase !== "COMPLETE") {
        this.currentPhase = nextPhase;
      }
      this.phaseEnteredAt = now;
    }

    if (!repJustCounted) {
      this._noRepFrames++;
    }

    // ── Hold time tracking ────────────────────────────────
    let holdTime = null;
    if (profile.type === "hold") {
      // Plank: only accumulate time while form is actually good (pause on sag/pike).
      const holdFormOk = activeExercise === "plank"
        ? !!(angles.formAllowsHold && formOk)
        : formOk;

      if (this.currentPhase === "HOLDING" && holdFormOk) {
        if (this.holdStartTime === 0) this.holdStartTime = now;
        this.totalHoldTime = this._holdPausedTotal + (now - this.holdStartTime) / 1000;
      } else {
        // Pause (or leave HOLDING): bank time so far, do not keep counting.
        if (this.holdStartTime !== 0) {
          this._holdPausedTotal += (now - this.holdStartTime) / 1000;
          this.holdStartTime = 0;
        }
        this.totalHoldTime = this._holdPausedTotal;
      }
      holdTime = this.totalHoldTime;
    }

    // ── Form feedback ─────────────────────────────────────
    let feedbackResult = profile.checkForm(angles, this.currentPhase, this._cameraAngle, this);

    if (activeExercise === "plank" || activeExercise === "push_up") {
      // Sticky coach for plank / push-up — don't let frame jitter rewrite the text.
      const lockMs = activeExercise === "plank" ? 3500 : 2200;
      const refreshMs = activeExercise === "plank" ? 2500 : 1600;
      if (feedbackResult) {
        if (!this._lastFeedback) {
          this._lastFeedback = feedbackResult;
          this._feedbackCooldown = now + lockMs;
        } else if (feedbackResult.msg === this._lastFeedback.msg) {
          this._lastFeedback = feedbackResult;
          if (feedbackResult.type === "good" || this.currentPhase === "HOLDING") {
            this._feedbackCooldown = Math.max(this._feedbackCooldown, now + refreshMs);
          }
        } else if (now >= this._feedbackCooldown) {
          this._lastFeedback = feedbackResult;
          this._feedbackCooldown = now + lockMs;
        }
        feedbackResult = this._lastFeedback;
      } else if (this._lastFeedback) {
        feedbackResult = this._lastFeedback;
      }
    } else if (feedbackResult !== null && feedbackResult !== undefined) {
      this._lastFeedback = feedbackResult;
      this._feedbackCooldown = now + 700;
    } else if (now < this._feedbackCooldown && this._lastFeedback) {
      feedbackResult = this._lastFeedback;
    }

    const feedback = feedbackResult ? feedbackResult.msg : "Good form, keep going! 💪";
    const feedbackType = feedbackResult ? feedbackResult.type : "good";

    // Bicep curl: paint the active arm red + show ideal arm guide when elbow drifts.
    let effectiveFormOk = formOk;
    let formGuideLines = formCheck?.guideLines || [];
    let badLandmarks = [];

    // Push-up / plank: selective red segments + green straight-line guides.
    // Only mark form bad when we have specific joints to paint — never whole-body red.
    if ((activeExercise === "push_up" || activeExercise === "plank") && formCheck) {
      badLandmarks = formCheck.badLandmarks || [];
      formGuideLines = formCheck.guideLines || [];
      // Clear paint/guides when sticky form is good (debounced upstream).
      if (formCheck.isCorrect) {
        badLandmarks = [];
        formGuideLines = [];
        effectiveFormOk = true;
      } else {
        effectiveFormOk = badLandmarks.length === 0 ? true : formCheck.isCorrect;
      }
    }

    if (activeExercise === "bicep_curl") {
      const curlState = this._formState.bicep_curl;
      const arm = angles.activeArm;
      if (curlState && curlState.awayFrames >= 6 && arm) {
        effectiveFormOk = false;
        const isLeft = arm === "left";
        badLandmarks = isLeft
          ? [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST]
          : [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST];
        formGuideLines = buildCurlInlineGuides(
          pts[isLeft ? LM.L_SHOULDER : LM.R_SHOULDER],
          pts[isLeft ? LM.L_ELBOW : LM.R_ELBOW],
          pts[isLeft ? LM.L_WRIST : LM.R_WRIST],
          pts[isLeft ? LM.L_HIP : LM.R_HIP]
        );
      }
    }


    // Squat: red legs + green spaced copies from RAW landmarks (match skeleton).
    if (activeExercise === "squat" && angles.squatOverlay) {
      const flags = angles.squatOverlay.flags;
      if (flags && (flags.leftCave || flags.rightCave || flags.leftFlare || flags.rightFlare || flags.hipShift)) {
        const drawn = buildSquatLateralOverlay(
          {
            lHip: landmarks[LM.L_HIP],
            rHip: landmarks[LM.R_HIP],
            lKnee: landmarks[LM.L_KNEE],
            rKnee: landmarks[LM.R_KNEE],
            lAnkle: landmarks[LM.L_ANKLE],
            rAnkle: landmarks[LM.R_ANKLE],
          },
          flags
        );
        effectiveFormOk = drawn.formOk;
        badLandmarks = drawn.badLandmarks;
        formGuideLines = drawn.guideLines;
      } else {
        effectiveFormOk = true;
        badLandmarks = [];
        formGuideLines = [];
      }
    }


    return {
      reps: this.repCount,
      feedback,
      feedbackType,
      holdTime,
      phase: this.currentPhase,
      cameraAngle: this._cameraAngle ? this._cameraAngle.viewAngle : null,
      angleOk,
      activeArm: this._activeArm,
      primaryAngle: Math.round(angles.primary),
      formOk: effectiveFormOk,
      formIssues: formCheck?.issues || [],
      formHipAngle: formCheck?.hipAngle ?? null,
      formNeckAngle: formCheck?.neckAngle ?? null,
      formGuideLines,
      badLandmarks,
      visibleSide: formCheck?.visibleSide || null,
      calibrationState,
    };
  }

  /** Helper to build a return value without repeating boilerplate. */
  _result(profile, feedback, type = "neutral") {
    return {
      reps: this.repCount,
      feedback,
      feedbackType: type,
      holdTime: profile && profile.type === "hold" ? this.totalHoldTime : null,
      phase: this.currentPhase || "",
      cameraAngle: this._cameraAngle ? this._cameraAngle.viewAngle : null,
      angleOk: true,
      activeArm: this._activeArm,
      primaryAngle: null,
      formOk: true,
      formIssues: [],
      formHipAngle: null,
      formNeckAngle: null,
      formGuideLines: [],
      badLandmarks: [],
      visibleSide: null,
    };
  }
}
