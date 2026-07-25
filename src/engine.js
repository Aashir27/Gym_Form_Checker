/**
 * EMAFilter — Exponential Moving Average for low-pass jitter smoothing.
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
 * Calculates absolute internal angle between 0 and 180 degrees.
 */
export function calculateAngle(a, b, c) {
  let radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * (180.0 / Math.PI));
  if (angle > 180.0) {
    angle = 360.0 - angle;
  }
  return angle;
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

/** Visibility check. */
function vis(pt, threshold = 0.4) {
  return pt && pt.visibility >= threshold;
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
 *   "side"   — user turned ~90° (profile view)
 *   "angled" — user turned ~30-60° (three-quarter view)
 *   "front"  — user facing camera directly
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

  // Shoulder width in image space vs torso height
  const shoulderWidth = dist(lSh, rSh);
  const torsoHeight = dist(mid(lSh, rSh), mid(lHip, rHip));
  const ratio = torsoHeight > 0.01 ? shoulderWidth / torsoHeight : 0.5;

  // Typical ratios: front ~0.7-1.0, angled ~0.3-0.7, side <0.3
  let viewAngle;
  if (ratio < 0.25) viewAngle = "side";
  else if (ratio < 0.55) viewAngle = "angled";
  else viewAngle = "front";

  // Body orientation: is the person lying down / in plank position?
  const midShoulder = mid(lSh, rSh);
  const midHip = mid(lHip, rHip);
  const torsoAngleFromHorizontal = Math.abs(Math.atan2(midShoulder.y - midHip.y, midShoulder.x - midHip.x) * (180 / Math.PI));
  const isHorizontal = torsoAngleFromHorizontal < 45 || torsoAngleFromHorizontal > 135;

  return { viewAngle, isHorizontal, shoulderWidthRatio: ratio, confidence: 1 };
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

/* ───────────────────────────────────────────────────────────────
 * Exercise Profiles  (camera-angle-aware, strict rep counting)
 *
 * Each profile adds:
 *   preferredAngles    — which camera view(s) work best
 *   angleHint          — guidance message shown when angle is wrong
 *   bodyOrientation    — "upright" | "horizontal" | "any"
 *   validateAngle(cam) — returns true if camera angle is acceptable
 *   minROM             — minimum range-of-motion (degrees) to count
 *   minRepDurationMs   — minimum time for a full rep cycle
 *   minVelocity        — min deg/s during active movement phases
 *   stabilityChecks(a) — extra guards that BLOCK a rep if they fail
 * ────────────────────────────────────────────────────────────── */

const EXERCISE_PROFILES = {

  /* ══════════════════════════════════════════════════════════
   *  SQUAT
   *  Best viewed from side/angled — need to see knee bend & hip hinge
   * ══════════════════════════════════════════════════════════ */
  squat: {
    label: "Squat",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Stand sideways to camera for best tracking",
    bodyOrientation: "upright",
    requiredLandmarks: [LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE, LM.L_SHOULDER, LM.R_SHOULDER],

    computeAngles(p) {
      const lKnee = calculateAngle(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = calculateAngle(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      const lHip  = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip  = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const torsoTilt = Math.abs(tiltFromVertical(
        mid(p[LM.L_HIP], p[LM.R_HIP]),
        mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER])
      ));
      // Knee tracking over toes — compare knee x to ankle x (side view)
      const kneeOverToe = Math.abs(
        ((p[LM.L_KNEE].x + p[LM.R_KNEE].x) / 2) - ((p[LM.L_ANKLE].x + p[LM.R_ANKLE].x) / 2)
      ) * 100;
      return {
        primary: (lKnee + rKnee) / 2,
        hipAngle: (lHip + rHip) / 2,
        kneeDiff: Math.abs(lKnee - rKnee),
        torsoTilt,
        kneeOverToe,
      };
    },

    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    thresholds: { startStanding: 158, enterBottom: 108, standBack: 158 },
    minTransitionMs: 180,
    minROM: 45,            // must bend at least 45° from standing
    minRepDurationMs: 800, // a full squat can't happen in <0.8s
    minVelocity: 15,       // deg/s — must move intentionally

    validateAngle(cam) {
      // Squats work from side or angled view; front view is unreliable
      // because knee flexion is ambiguous when viewed head-on
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":      return a.primary > t.startStanding ? "DESCENDING" : null;
        case "DESCENDING": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":     return a.primary > t.enterBottom + 12 ? "ASCENDING" : null;
        case "ASCENDING":  return a.primary > t.standBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      // Block rep if torso is excessively tilted (probably a bow, not a squat)
      if (a.torsoTilt > 55) return { pass: false, reason: "Excessive forward lean — not a valid squat" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      // Camera angle guidance takes priority
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (phase === "BOTTOM" && a.primary < 55)        return { type: "warning", msg: "Too deep — risk of knee strain" };
      if (phase === "BOTTOM" && a.primary > 120)        return { type: "warning", msg: "Too shallow — go deeper for a full rep" };
      if (a.torsoTilt > 40)                             return { type: "warning", msg: "Chest too forward — stay upright" };
      if (a.kneeDiff > 18)                              return { type: "warning", msg: "Uneven knees — push both sides evenly" };
      if (phase === "BOTTOM" && a.hipAngle < 70)        return { type: "warning", msg: "Hips too far back — sit down, not back" };
      if (phase === "BOTTOM" && a.primary >= 55 && a.primary <= 100)
                                                        return { type: "good", msg: "Good depth — parallel or below 🔥" };
      if (phase === "ASCENDING")                        return { type: "good", msg: "Drive up through your heels!" };
      if (phase === "READY")                            return { type: "good", msg: "Stand tall — ready to squat" };
      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  BICEP CURL
   *  Best from side view — elbow flexion is a sagittal-plane motion
   * ══════════════════════════════════════════════════════════ */
  bicep_curl: {
    label: "Bicep Curl",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Turn sideways to camera — curls are best tracked from the side",
    bodyOrientation: "upright",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST, LM.L_HIP, LM.R_HIP],

    computeAngles(p) {
      // 1. Dynamic Arm Visibility Picker (The Orientation Fix)
      const lVis = (p[LM.L_SHOULDER]?.visibility || 0) + (p[LM.L_ELBOW]?.visibility || 0) + (p[LM.L_WRIST]?.visibility || 0);
      const rVis = (p[LM.R_SHOULDER]?.visibility || 0) + (p[LM.R_ELBOW]?.visibility || 0) + (p[LM.R_WRIST]?.visibility || 0);
      const activeArm = lVis > rVis ? 'left' : 'right';

      // 2. Scale-Invariant Joint Angle Tracking (The Distance Fix)
      let curlAngle, driftAngle;
      if (activeArm === 'left') {
        curlAngle = calculateAngle(p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]);
        driftAngle = calculateAngle(p[LM.L_HIP], p[LM.L_SHOULDER], p[LM.L_ELBOW]);
      } else {
        curlAngle = calculateAngle(p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]);
        driftAngle = calculateAngle(p[LM.R_HIP], p[LM.R_SHOULDER], p[LM.R_ELBOW]);
      }

      // Track direction locally for error conditions
      if (!this.lastAngle) this.lastAngle = curlAngle;
      this.isExtending = curlAngle > this.lastAngle;
      this.lastAngle = curlAngle;

      return { primary: curlAngle, driftAngle };
    },

    // 3. Two-State Repetition Engine (The Count Fix)
    // Map user's stage "up"/"down" to our engine's phases:
    // UP (waiting for > 160) -> DOWN (waiting for < 35) -> COMPLETE (increments rep, sets back to UP)
    phases: ["UP", "UP", "DOWN", "COMPLETE"],
    thresholds: {},
    minTransitionMs: 0,
    minROM: 0,
    minRepDurationMs: 0,
    minVelocity: 0,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      switch (phase) {
        case "UP":
          // "When curlAngle > 160, the arm is fully extended. Set stage = 'down'."
          return a.primary > 160 ? "DOWN" : null;
        case "DOWN":
          // "When curlAngle < 35 AND stage === 'down', increment rep and flip stage = 'up'."
          return a.primary < 35 ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      
      // 4. Specific Form Error Conditions

      // Error A: "If the user is mid-rep and driftAngle > 30..."
      if (a.driftAngle > 30) {
        return { type: "warning", msg: "Keep your elbow tucked at your side! Don't swing your arm." };
      }

      // Error B: "If the user reverses direction and goes back down before hitting the < 35 contraction target..."
      // (They are in "DOWN" phase trying to contract, but arm starts extending again before reaching 35)
      if (phase === "DOWN" && this.isExtending && a.primary > 35 && a.primary < 130) {
        return { type: "warning", msg: "Curl all the way up to finish the rep!" };
      }

      // Error C: "If curlAngle drops but doesn't reach > 160 at the bottom..."
      // (They are in "UP" phase trying to extend, but arm starts contracting again before reaching 160)
      if (phase === "UP" && !this.isExtending && a.primary > 50 && a.primary < 160) {
        return { type: "warning", msg: "Extend your arm fully at the bottom." };
      }

      // Contextual state feedback
      if (phase === "UP") return { type: "good", msg: "Extend arm fully downward." };
      if (phase === "DOWN") return { type: "good", msg: "Curl up for full contraction!" };

      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  PUSH-UP
   *  Best from side view — shows elbow bend, body line, hip alignment
   * ══════════════════════════════════════════════════════════ */
  push_up: {
    label: "Push-Up",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Place camera to the side — push-ups need a profile view",
    bodyOrientation: "horizontal",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST, LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE],

    computeAngles(p) {
      const lElbow = calculateAngle(p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]);
      const rElbow = calculateAngle(p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]);
      const primary = (lElbow + rElbow) / 2;
      // Body line: shoulder-hip-ankle should be ~180°
      const lBody = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      // Hip sag/pike
      const hipY = (p[LM.L_HIP].y + p[LM.R_HIP].y) / 2;
      const shoulderY = (p[LM.L_SHOULDER].y + p[LM.R_SHOULDER].y) / 2;
      const ankleY = (p[LM.L_ANKLE].y + p[LM.R_ANKLE].y) / 2;
      const expectedHipY = (shoulderY + ankleY) / 2;
      const hipDeviation = (hipY - expectedHipY) * 100;
      // Shoulder vertical travel (how much does the body move up/down)
      const shoulderHeight = shoulderY;
      return { primary, bodyLine, hipDeviation, shoulderHeight };
    },

    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    thresholds: { startExtended: 158, enterBottom: 95, leaveBottom: 108, extendBack: 158 },
    minTransitionMs: 200,
    minROM: 50,
    minRepDurationMs: 900,
    minVelocity: 12,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":      return a.primary > t.startExtended ? "DESCENDING" : null;
        case "DESCENDING": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":     return a.primary > t.leaveBottom ? "ASCENDING" : null;
        case "ASCENDING":  return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      // Body must be roughly straight — no pike or extreme sag
      if (a.bodyLine < 140) return { pass: false, reason: "Body not straight — maintain plank position" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (a.hipDeviation > 10)                 return { type: "warning", msg: "Hips sagging — tighten core" };
      if (a.hipDeviation < -12)                return { type: "warning", msg: "Hips too high — flatten out your body" };
      if (a.bodyLine < 150)                    return { type: "warning", msg: "Maintain a straight body line" };
      if (phase === "BOTTOM" && a.primary > 110) return { type: "warning", msg: "Go lower — chest toward floor" };
      if (phase === "BOTTOM" && a.primary <= 95) return { type: "good", msg: "Great depth! Chest near floor 🔥" };
      if (phase === "ASCENDING")               return { type: "good", msg: "Push strong — full lockout!" };
      if (phase === "READY")                   return { type: "good", msg: "Good plank position — begin!" };
      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  REVERSE LUNGE
   *  Best from side/angled — need to see front + back leg bend
   * ══════════════════════════════════════════════════════════ */
  reverse_lunge: {
    label: "Reverse Lunge",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Stand sideways to camera for clear leg tracking",
    bodyOrientation: "upright",
    requiredLandmarks: [LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE, LM.L_SHOULDER, LM.R_SHOULDER],

    computeAngles(p) {
      const lKnee = calculateAngle(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = calculateAngle(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      const primary = Math.min(lKnee, rKnee); // lunging leg
      const secondary = Math.max(lKnee, rKnee); // front leg
      const torsoTilt = Math.abs(tiltFromVertical(
        mid(p[LM.L_HIP], p[LM.R_HIP]),
        mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER])
      ));
      // Hip level check
      const hipTilt = Math.abs(p[LM.L_HIP].y - p[LM.R_HIP].y) * 100;
      // Stance width (how far apart feet are — should be meaningful for a lunge)
      const stanceWidth = Math.abs(p[LM.L_ANKLE].y - p[LM.R_ANKLE].y) * 100;
      return { primary, secondary, torsoTilt, hipTilt, stanceWidth };
    },

    phases: ["READY", "STEPPING_BACK", "BOTTOM", "RETURNING", "COMPLETE"],
    thresholds: { startStanding: 158, enterBottom: 102, standBack: 158 },
    minTransitionMs: 220,
    minROM: 48,
    minRepDurationMs: 1200,
    minVelocity: 12,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":         return (a.primary > t.startStanding && a.secondary > t.startStanding) ? "STEPPING_BACK" : null;
        case "STEPPING_BACK": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":        return a.primary > t.enterBottom + 12 ? "RETURNING" : null;
        case "RETURNING":     return a.primary > t.standBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      if (a.torsoTilt > 30) return { pass: false, reason: "Torso too tilted — stay upright through the lunge" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (a.torsoTilt > 22)                return { type: "warning", msg: "Stay upright — don't lean forward" };
      if (a.hipTilt > 6)                   return { type: "warning", msg: "Keep hips level and square" };
      if (phase === "BOTTOM" && a.primary > 115) return { type: "warning", msg: "Step deeper — bend rear knee more" };
      if (phase === "BOTTOM" && a.secondary < 70) return { type: "warning", msg: "Front knee too far forward" };
      if (phase === "BOTTOM" && a.primary <= 102) return { type: "good", msg: "Good lunge depth — rear knee near floor!" };
      if (phase === "RETURNING")            return { type: "good", msg: "Drive back up to standing!" };
      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  GLUTE BRIDGE
   *  Best from side view — lying on back, hip extension is sagittal
   * ══════════════════════════════════════════════════════════ */
  glute_bridge: {
    label: "Glute Bridge",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Lie down with camera to your side for best tracking",
    bodyOrientation: "horizontal",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE],

    computeAngles(p) {
      const lHip = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const primary = (lHip + rHip) / 2;
      const lKnee = calculateAngle(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = calculateAngle(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      const kneeAngle = (lKnee + rKnee) / 2;
      const hipDiff = Math.abs(lHip - rHip);
      return { primary, kneeAngle, hipDiff };
    },

    phases: ["READY", "LIFTING", "TOP", "LOWERING", "COMPLETE"],
    thresholds: { startLow: 108, enterTop: 158, leaveTop: 148, lowerBack: 108 },
    minTransitionMs: 250,
    minROM: 40,
    minRepDurationMs: 1000,
    minVelocity: 10,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":    return a.primary < t.startLow ? "LIFTING" : null;
        case "LIFTING":  return a.primary > t.enterTop ? "TOP" : null;
        case "TOP":      return a.primary < t.leaveTop ? "LOWERING" : null;
        case "LOWERING": return a.primary < t.lowerBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      if (a.hipDiff > 20) return { pass: false, reason: "Hips tilting unevenly — don't count" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (a.hipDiff > 14)                   return { type: "warning", msg: "Keep hips level — don't tilt" };
      if (phase === "TOP" && a.primary < 152) return { type: "warning", msg: "Squeeze higher — full hip extension" };
      if (phase === "TOP" && a.primary >= 160) return { type: "good", msg: "Great hip extension! 🍑" };
      if (phase === "LOWERING")              return { type: "good", msg: "Lower with control — don't drop" };
      return null;
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  DEAD BUG
   *  Best from front view (lying on back, camera above/in front)
   *  or side view
   * ══════════════════════════════════════════════════════════ */
  dead_bug: {
    label: "Dead Bug",
    type: "rep",
    preferredAngles: ["side", "angled", "front"],
    angleHint: "Lie on your back — camera from front or side works",
    bodyOrientation: "horizontal",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_WRIST, LM.R_WRIST],

    computeAngles(p) {
      const lHip = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const maxExtension = Math.max(lHip, rHip);
      const minExtension = Math.min(lHip, rHip);
      const lArm = calculateAngle(p[LM.L_HIP], p[LM.L_SHOULDER], p[LM.L_WRIST]);
      const rArm = calculateAngle(p[LM.R_HIP], p[LM.R_SHOULDER], p[LM.R_WRIST]);
      const armExtension = Math.max(lArm, rArm);
      const hipDiff = Math.abs(p[LM.L_HIP].y - p[LM.R_HIP].y) * 100;
      return { primary: maxExtension, minExtension, armExtension, hipDiff };
    },

    phases: ["READY", "EXTENDING", "EXTENDED", "RETURNING", "COMPLETE"],
    thresholds: { startTucked: 98, enterExtended: 148, returnTucked: 98 },
    minTransitionMs: 300,
    minROM: 40,
    minRepDurationMs: 1200,
    minVelocity: 8,

    validateAngle(cam) {
      // Dead bug works from any angle as long as limbs are visible
      return true;
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":     return a.primary < t.startTucked ? "EXTENDING" : null;
        case "EXTENDING": return a.primary > t.enterExtended ? "EXTENDED" : null;
        case "EXTENDED":  return a.primary < t.enterExtended - 12 ? "RETURNING" : null;
        case "RETURNING": return a.primary < t.returnTucked ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      if (a.hipDiff > 7) return { pass: false, reason: "Core unstable — hips rocking too much" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (a.hipDiff > 5)                                 return { type: "warning", msg: "Keep hips still — don't rock" };
      if (phase === "EXTENDED" && a.armExtension < 120)   return { type: "warning", msg: "Reach arm overhead fully" };
      if (phase === "EXTENDED" && a.primary >= 148)       return { type: "good", msg: "Full extension — great control! 🎯" };
      if (phase === "RETURNING")                          return { type: "good", msg: "Move with control — slow and steady" };
      return null;
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
    angleHint: "Place camera to the side to see your body alignment",
    bodyOrientation: "horizontal",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE],

    computeAngles(p) {
      const lBody = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      const hipY = (p[LM.L_HIP].y + p[LM.R_HIP].y) / 2;
      const shoulderY = (p[LM.L_SHOULDER].y + p[LM.R_SHOULDER].y) / 2;
      const ankleY = (p[LM.L_ANKLE].y + p[LM.R_ANKLE].y) / 2;
      const expectedHipY = (shoulderY + ankleY) / 2;
      const hipDeviation = (hipY - expectedHipY) * 100;
      return { primary: bodyLine, hipDeviation };
    },

    phases: ["NOT_IN_POSITION", "HOLDING"],
    thresholds: { minBodyLine: 158, maxHipDev: 9 },
    minTransitionMs: 600,
    holdTimeDisplay: true,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      const inPosition = a.primary > t.minBodyLine && Math.abs(a.hipDeviation) < t.maxHipDev;
      switch (phase) {
        case "NOT_IN_POSITION": return inPosition ? "HOLDING" : null;
        case "HOLDING":         return !inPosition ? "NOT_IN_POSITION" : null;
        default: return null;
      }
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (phase !== "HOLDING") return { type: "info", msg: "Get into plank position — straight body line" };
      if (a.hipDeviation > 7)  return { type: "warning", msg: "Hips sagging — tighten your core" };
      if (a.hipDeviation < -8) return { type: "warning", msg: "Hips too high — flatten out" };
      if (a.primary < 155)     return { type: "warning", msg: "Straighten your body — maintain alignment" };
      return { type: "good", msg: "Solid plank — hold steady! 💎" };
    },
  },

  /* ══════════════════════════════════════════════════════════
   *  MOUNTAIN CLIMBER
   *  Side/angled view to see knee drive and plank alignment
   * ══════════════════════════════════════════════════════════ */
  mountain_climber: {
    label: "Mountain Climber",
    type: "rep",
    preferredAngles: ["side", "angled"],
    angleHint: "Camera to the side — need to see knee drive clearly",
    bodyOrientation: "horizontal",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE],

    computeAngles(p) {
      const lHip = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const primary = Math.min(lHip, rHip);   // driving leg
      const secondary = Math.max(lHip, rHip); // extended leg
      const lBody = calculateAngle(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = calculateAngle(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      return { primary, secondary, bodyLine };
    },

    phases: ["READY", "DRIVING", "TUCKED", "EXTENDING", "COMPLETE"],
    thresholds: { startExtended: 152, enterTucked: 98, extendBack: 148 },
    minTransitionMs: 120,   // fast exercise
    minROM: 40,
    minRepDurationMs: 400,  // mountain climbers are fast
    minVelocity: 25,

    validateAngle(cam) {
      return cam.viewAngle === "side" || cam.viewAngle === "angled";
    },

    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":     return a.secondary > t.startExtended ? "DRIVING" : null;
        case "DRIVING":   return a.primary < t.enterTucked ? "TUCKED" : null;
        case "TUCKED":    return a.primary > t.enterTucked + 12 ? "EXTENDING" : null;
        case "EXTENDING": return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },

    stabilityChecks(a) {
      if (a.bodyLine < 130) return { pass: false, reason: "Plank position lost — hips too high" };
      return { pass: true };
    },

    checkForm(a, phase, cam) {
      if (cam && !this.validateAngle(cam)) {
        return { type: "info", msg: this.angleHint };
      }
      if (a.bodyLine < 140)                          return { type: "warning", msg: "Keep hips down — maintain plank" };
      if (phase === "TUCKED" && a.primary > 108)     return { type: "warning", msg: "Drive knee higher — full tuck" };
      if (phase === "TUCKED" && a.primary <= 98)     return { type: "good", msg: "Great drive! 🔥" };
      if (phase === "EXTENDING")                     return { type: "good", msg: "Keep the pace — stay controlled!" };
      return null;
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
 *  1. Camera angle validation — wrong angle blocks counting
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
    this._lastExercise = null;
    this._feedbackCooldown = 0;
    this._lastFeedback = null;
    this._noRepFrames = 0;
    this._velocityTracker = new VelocityTracker(6);
    this._cameraAngle = null;
    this._angleWarningShown = false;
    this._stabilityFailCount = 0;  // consecutive frames of stability failure
  }

  static get exerciseKeys() {
    return Object.keys(EXERCISE_PROFILES);
  }

  static getProfile(key) {
    return EXERCISE_PROFILES[key] || null;
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
    this._lastExercise = null;
    this._feedbackCooldown = 0;
    this._lastFeedback = null;
    this._noRepFrames = 0;
    this._velocityTracker.reset();
    this._cameraAngle = null;
    this._angleWarningShown = false;
    this._stabilityFailCount = 0;
  }

  smoothPoint(idx, pt) {
    if (!this.filters[idx]) {
      this.filters[idx] = new EMAFilter(0.5); // alpha = 0.5 as requested
    }
    const smoothed = this.filters[idx].filter(pt.x, pt.y);
    return {
      x: smoothed.x,
      y: smoothed.y,
      visibility: pt.visibility,
    };
  }

  /**
   * Main per-frame evaluation.
   * @returns {{ reps, feedback, feedbackType, holdTime, phase, cameraAngle, angleOk }}
   */
  evaluateFrame(landmarks, activeExercise) {
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
    // Only process landmarks relevant to the current exercise and camera tracking
    const cameraAngleIndices = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP];
    const indicesToProcess = new Set([...profile.requiredLandmarks, ...cameraAngleIndices]);
    
    for (const idx of indicesToProcess) {
      if (landmarks[idx]) {
        pts[idx] = this.smoothPoint(idx, landmarks[idx]);
      }
    }

    // ── Visibility gate ───────────────────────────────────
    const allVisible = profile.requiredLandmarks.every(idx => vis(pts[idx]));
    if (!allVisible) {
      return this._result(profile, "Can't see key joints — adjust your position", "warning");
    }

    // ── Camera angle detection ────────────────────────────
    this._cameraAngle = detectCameraAngle(pts);
    const angleOk = profile.validateAngle ? profile.validateAngle(this._cameraAngle) : true;

    // ── Compute angles ────────────────────────────────────
    const angles = profile.computeAngles(pts);

    // ── Track velocity ────────────────────────────────────
    this._velocityTracker.push(angles.primary, timeSec);

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
        const hasMinFrames = this._noRepFrames >= 8;
        const stabilityResult = profile.stabilityChecks ? profile.stabilityChecks(angles) : { pass: true };
        const isAngleOk = angleOk;

        // All checks must pass to count the rep
        if (hasMinROM && hasMinDuration && hasMinFrames && stabilityResult.pass && isAngleOk) {
          this.repCount++;
          repJustCounted = true;
          this._noRepFrames = 0;
          this._stabilityFailCount = 0;
        } else {
          // Don't count — but still let the cycle continue
          // If stability failed, note it for feedback
          if (!stabilityResult.pass) {
            this._stabilityFailCount++;
            this._lastFeedback = { type: "warning", msg: stabilityResult.reason };
            this._feedbackCooldown = now + 1200;
          }
          if (!hasMinROM) {
            this._lastFeedback = { type: "warning", msg: "Partial rep — go through full range of motion" };
            this._feedbackCooldown = now + 1200;
          }
          if (!isAngleOk) {
            this._lastFeedback = { type: "info", msg: profile.angleHint || "Adjust camera angle" };
            this._feedbackCooldown = now + 1500;
          }
        }

        // Cycle back to first active phase regardless
        this.currentPhase = profile.phases[1];
        this.repCycleStartedAt = now;
        this.peakAngle = angles.primary;
        this.valleyAngle = angles.primary;
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
      if (this.currentPhase === "HOLDING") {
        if (this.holdStartTime === 0) {
          this.holdStartTime = now;
        }
        this.totalHoldTime = (now - this.holdStartTime) / 1000;
      } else {
        this.holdStartTime = 0;
      }
      holdTime = this.totalHoldTime;
    }

    // ── Form feedback ─────────────────────────────────────
    let feedbackResult = profile.checkForm(angles, this.currentPhase, this._cameraAngle);

    if (feedbackResult !== null && feedbackResult !== undefined) {
      this._lastFeedback = feedbackResult;
      this._feedbackCooldown = now + 700;
    } else if (now < this._feedbackCooldown && this._lastFeedback) {
      feedbackResult = this._lastFeedback;
    }

    const feedback = feedbackResult ? feedbackResult.msg : "Good form — keep going! 💪";
    const feedbackType = feedbackResult ? feedbackResult.type : "good";

    return {
      reps: this.repCount,
      feedback,
      feedbackType,
      holdTime,
      phase: this.currentPhase,
      cameraAngle: this._cameraAngle ? this._cameraAngle.viewAngle : null,
      angleOk,
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
    };
  }
}
