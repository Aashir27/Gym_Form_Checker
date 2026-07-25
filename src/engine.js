/**
 * OneEuroFilter — low-pass filter for real-time signal smoothing.
 * Reduces jitter in landmark coordinates while preserving fast movements.
 */
export class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }

  filter(x, t) {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = t;
      this.dxPrev = 0;
      return x;
    }
    const te = t - this.tPrev;
    if (te <= 0) return this.xPrev;
    const alphaD = 1.0 / (1.0 + 1.0 / (2.0 * Math.PI * this.dcutoff * te));
    const dx = (x - this.xPrev) / te;
    const dxHat = alphaD * dx + (1.0 - alphaD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const alpha = 1.0 / (1.0 + 1.0 / (2.0 * Math.PI * cutoff * te));
    const xHat = alpha * x + (1.0 - alpha) * this.xPrev;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    this.tPrev = t;
    return xHat;
  }
}

/* ───────────────────────────────────────────────────────────────
 * MediaPipe PoseLandmarker index reference
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

// Landmark indices as readable constants
const LM = {
  NOSE: 0,
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
function angleDeg(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.sqrt(v1.x ** 2 + v1.y ** 2) * Math.sqrt(v2.x ** 2 + v2.y ** 2);
  if (mag === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
}

/** Signed angle of segment (a→b) vs vertical, positive = tilted right. */
function tiltFromVertical(a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.atan2(dx, -dy) * (180 / Math.PI);
}

/** Midpoint of two landmarks. */
function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, visibility: Math.min(a.visibility, b.visibility) };
}

/** Euclidean distance between two landmarks (normalised coords). */
function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/** Check if a landmark is visible enough. */
function vis(pt, threshold = 0.4) {
  return pt && pt.visibility >= threshold;
}

/* ───────────────────────────────────────────────────────────────
 * Exercise Profiles
 *
 * Each profile defines:
 *   label            — display name
 *   type             — "rep" (counted) or "hold" (timed)
 *   requiredLandmarks — indices that must be visible
 *   computeAngles(pts) — returns { primary, secondary?, extras }
 *   phases           — ordered state names, last = completion
 *   transitionRules(angles, phase) — returns next phase or null
 *   minTransitionMs  — minimum time in a phase before leaving
 *   checkForm(angles, phase) — { type: "good"|"warning", msg }
 * ────────────────────────────────────────────────────────────── */

const EXERCISE_PROFILES = {
  /* ── SQUAT ─────────────────────────────────────────────── */
  squat: {
    label: "Squat",
    type: "rep",
    requiredLandmarks: [LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE, LM.L_SHOULDER, LM.R_SHOULDER],
    computeAngles(p) {
      const lKnee = angleDeg(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = angleDeg(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      const lHip  = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip  = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const torsoTilt = Math.abs(tiltFromVertical(mid(p[LM.L_HIP], p[LM.R_HIP]), mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER])));
      return {
        primary: (lKnee + rKnee) / 2,
        hipAngle: (lHip + rHip) / 2,
        kneeDiff: Math.abs(lKnee - rKnee),
        torsoTilt,
      };
    },
    // Phases: READY → DESCENDING → BOTTOM → ASCENDING → COMPLETE (rep++)
    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    thresholds: { startStanding: 155, enterBottom: 110, deepEnough: 105, standBack: 155 },
    minTransitionMs: 150,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":       return a.primary > t.startStanding ? "DESCENDING" : null;
        case "DESCENDING":  return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":      return a.primary > t.enterBottom + 10 ? "ASCENDING" : null;
        case "ASCENDING":   return a.primary > t.standBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (phase === "BOTTOM" && a.primary < 60)       return { type: "warning", msg: "Too deep — keep thighs parallel" };
      if (phase === "BOTTOM" && a.primary > 115)       return { type: "warning", msg: "Too shallow — go deeper" };
      if (a.torsoTilt > 45)                            return { type: "warning", msg: "Chest too forward — stay upright" };
      if (a.kneeDiff > 20)                             return { type: "warning", msg: "Uneven knees — push evenly" };
      if (phase === "BOTTOM" && a.primary >= 60 && a.primary <= 105)
                                                       return { type: "good", msg: "Good depth 🔥" };
      if (phase === "ASCENDING")                        return { type: "good", msg: "Drive up strong!" };
      return null;
    },
  },

  /* ── BICEP CURL ────────────────────────────────────────── */
  bicep_curl: {
    label: "Bicep Curl",
    type: "rep",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST, LM.L_HIP, LM.R_HIP],
    computeAngles(p) {
      const lElbow = angleDeg(p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]);
      const rElbow = angleDeg(p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]);
      // Pick the arm that has the smaller (more curled) angle as the active arm
      const primary = Math.min(lElbow, rElbow);
      const secondary = Math.max(lElbow, rElbow);
      // Shoulder swing: upper arm should stay close to torso
      const lShoulderSwing = angleDeg(p[LM.L_HIP], p[LM.L_SHOULDER], p[LM.L_ELBOW]);
      const rShoulderSwing = angleDeg(p[LM.R_HIP], p[LM.R_SHOULDER], p[LM.R_ELBOW]);
      const shoulderSwing = Math.min(lShoulderSwing, rShoulderSwing);
      const torsoTilt = Math.abs(tiltFromVertical(mid(p[LM.L_HIP], p[LM.R_HIP]), mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER])));
      return { primary, secondary, shoulderSwing, torsoTilt };
    },
    phases: ["READY", "CURLING", "TOP", "LOWERING", "COMPLETE"],
    thresholds: { startExtended: 140, enterTop: 55, topCeiling: 60, extendBack: 140 },
    minTransitionMs: 200,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":    return a.primary > t.startExtended ? "CURLING" : null;
        case "CURLING":  return a.primary < t.enterTop ? "TOP" : null;
        case "TOP":      return a.primary > t.topCeiling + 15 ? "LOWERING" : null;
        case "LOWERING": return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.torsoTilt > 20)             return { type: "warning", msg: "Don't lean — keep torso upright" };
      if (a.shoulderSwing > 55)          return { type: "warning", msg: "Keep elbows tucked — no swinging" };
      if (phase === "TOP" && a.primary > 65) return { type: "warning", msg: "Curl higher for full range" };
      if (phase === "TOP" && a.primary <= 55) return { type: "good", msg: "Great squeeze at the top 💪" };
      if (phase === "LOWERING")          return { type: "good", msg: "Control the negative" };
      return null;
    },
  },

  /* ── PUSH-UP ───────────────────────────────────────────── */
  push_up: {
    label: "Push-Up",
    type: "rep",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_ELBOW, LM.R_ELBOW, LM.L_WRIST, LM.R_WRIST, LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE],
    computeAngles(p) {
      const lElbow = angleDeg(p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]);
      const rElbow = angleDeg(p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]);
      const primary = (lElbow + rElbow) / 2;
      // Body alignment: shoulder-hip-ankle should be ~180 for good plank
      const lBody = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      // Hip sag/pike detection
      const hipY = (p[LM.L_HIP].y + p[LM.R_HIP].y) / 2;
      const shoulderY = (p[LM.L_SHOULDER].y + p[LM.R_SHOULDER].y) / 2;
      const ankleY = (p[LM.L_ANKLE].y + p[LM.R_ANKLE].y) / 2;
      const expectedHipY = (shoulderY + ankleY) / 2;
      const hipDeviation = (hipY - expectedHipY) * 100; // positive = hips sagging (below line)
      return { primary, bodyLine, hipDeviation };
    },
    phases: ["READY", "DESCENDING", "BOTTOM", "ASCENDING", "COMPLETE"],
    thresholds: { startExtended: 155, enterBottom: 100, bottomFloor: 95, extendBack: 155 },
    minTransitionMs: 200,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":      return a.primary > t.startExtended ? "DESCENDING" : null;
        case "DESCENDING": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":     return a.primary > t.enterBottom + 10 ? "ASCENDING" : null;
        case "ASCENDING":  return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.hipDeviation > 8)                return { type: "warning", msg: "Hips sagging — tighten core" };
      if (a.hipDeviation < -10)              return { type: "warning", msg: "Hips too high — flatten body" };
      if (a.bodyLine < 150)                  return { type: "warning", msg: "Keep body in a straight line" };
      if (phase === "BOTTOM" && a.primary > 110) return { type: "warning", msg: "Go lower — chest toward floor" };
      if (phase === "BOTTOM" && a.primary <= 100) return { type: "good", msg: "Great depth! 🔥" };
      if (phase === "ASCENDING")             return { type: "good", msg: "Push strong!" };
      return null;
    },
  },

  /* ── REVERSE LUNGE ─────────────────────────────────────── */
  reverse_lunge: {
    label: "Reverse Lunge",
    type: "rep",
    requiredLandmarks: [LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE, LM.L_SHOULDER, LM.R_SHOULDER],
    computeAngles(p) {
      const lKnee = angleDeg(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = angleDeg(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      // The lunging leg has the smaller knee angle
      const primary = Math.min(lKnee, rKnee);
      const secondary = Math.max(lKnee, rKnee);
      // Torso upright check
      const torsoTilt = Math.abs(tiltFromVertical(mid(p[LM.L_HIP], p[LM.R_HIP]), mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER])));
      // Hip level check (are hips dropping to one side?)
      const hipTilt = Math.abs(p[LM.L_HIP].y - p[LM.R_HIP].y) * 100;
      return { primary, secondary, torsoTilt, hipTilt };
    },
    phases: ["READY", "STEPPING_BACK", "BOTTOM", "RETURNING", "COMPLETE"],
    thresholds: { startStanding: 155, enterBottom: 105, standBack: 155 },
    minTransitionMs: 200,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":        return a.primary > t.startStanding && a.secondary > t.startStanding ? "STEPPING_BACK" : null;
        case "STEPPING_BACK": return a.primary < t.enterBottom ? "BOTTOM" : null;
        case "BOTTOM":       return a.primary > t.enterBottom + 10 ? "RETURNING" : null;
        case "RETURNING":    return a.primary > t.standBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.torsoTilt > 25)               return { type: "warning", msg: "Stay upright — don't lean forward" };
      if (a.hipTilt > 5)                  return { type: "warning", msg: "Keep hips level" };
      if (phase === "BOTTOM" && a.primary > 115) return { type: "warning", msg: "Step deeper — more knee bend" };
      if (phase === "BOTTOM" && a.secondary < 75) return { type: "warning", msg: "Front knee too far forward" };
      if (phase === "BOTTOM" && a.primary <= 105) return { type: "good", msg: "Good lunge depth!" };
      if (phase === "RETURNING")           return { type: "good", msg: "Drive back up!" };
      return null;
    },
  },

  /* ── GLUTE BRIDGE ──────────────────────────────────────── */
  glute_bridge: {
    label: "Glute Bridge",
    type: "rep",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE],
    computeAngles(p) {
      // Hip angle: shoulder-hip-knee
      const lHip = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      const primary = (lHip + rHip) / 2;
      // Knee angle for reference
      const lKnee = angleDeg(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]);
      const rKnee = angleDeg(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]);
      const kneeAngle = (lKnee + rKnee) / 2;
      // Hip symmetry
      const hipDiff = Math.abs(lHip - rHip);
      return { primary, kneeAngle, hipDiff };
    },
    phases: ["READY", "LIFTING", "TOP", "LOWERING", "COMPLETE"],
    // Lying down: hip angle is small (~90). At top of bridge: hip angle opens toward ~170.
    thresholds: { startLow: 110, enterTop: 155, topCeiling: 150, lowerBack: 110 },
    minTransitionMs: 250,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":    return a.primary < t.startLow ? "LIFTING" : null;
        case "LIFTING":  return a.primary > t.enterTop ? "TOP" : null;
        case "TOP":      return a.primary < t.topCeiling - 5 ? "LOWERING" : null;
        case "LOWERING": return a.primary < t.lowerBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.hipDiff > 15)                 return { type: "warning", msg: "Keep hips level — don't tilt" };
      if (phase === "TOP" && a.primary < 150) return { type: "warning", msg: "Squeeze higher — full extension" };
      if (phase === "TOP" && a.primary >= 160) return { type: "good", msg: "Great hip extension! 🍑" };
      if (phase === "LOWERING")            return { type: "good", msg: "Lower with control" };
      return null;
    },
  },

  /* ── DEAD BUG ──────────────────────────────────────────── */
  dead_bug: {
    label: "Dead Bug",
    type: "rep",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_WRIST, LM.R_WRIST],
    computeAngles(p) {
      // Measure how extended each leg is (hip angle: shoulder-hip-knee)
      const lHip = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      // For dead bug, we track the leg that extends the most
      const maxExtension = Math.max(lHip, rHip);
      const minExtension = Math.min(lHip, rHip);
      // Arm overhead check (shoulder angle)
      const lArm = angleDeg(p[LM.L_HIP], p[LM.L_SHOULDER], p[LM.L_WRIST]);
      const rArm = angleDeg(p[LM.R_HIP], p[LM.R_SHOULDER], p[LM.R_WRIST]);
      const armExtension = Math.max(lArm, rArm);
      // Core stability: hips should stay level
      const hipDiff = Math.abs(p[LM.L_HIP].y - p[LM.R_HIP].y) * 100;
      return { primary: maxExtension, minExtension, armExtension, hipDiff };
    },
    phases: ["READY", "EXTENDING", "EXTENDED", "RETURNING", "COMPLETE"],
    thresholds: { startTucked: 100, enterExtended: 145, returnTucked: 100 },
    minTransitionMs: 300,
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":     return a.primary < t.startTucked ? "EXTENDING" : null;
        case "EXTENDING": return a.primary > t.enterExtended ? "EXTENDED" : null;
        case "EXTENDED":  return a.primary < t.enterExtended - 10 ? "RETURNING" : null;
        case "RETURNING": return a.primary < t.returnTucked ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.hipDiff > 5)                  return { type: "warning", msg: "Keep hips still — don't rock" };
      if (phase === "EXTENDED" && a.armExtension < 120) return { type: "warning", msg: "Reach arm overhead fully" };
      if (phase === "EXTENDED" && a.primary >= 145)     return { type: "good", msg: "Full extension! 🎯" };
      if (phase === "RETURNING")           return { type: "good", msg: "Move with control" };
      return null;
    },
  },

  /* ── PLANK (hold exercise) ─────────────────────────────── */
  plank: {
    label: "Plank",
    type: "hold",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_ANKLE, LM.R_ANKLE, LM.L_ELBOW, LM.R_ELBOW],
    computeAngles(p) {
      // Body alignment: shoulder-hip-ankle
      const lBody = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      // Hip sag/pike
      const hipY = (p[LM.L_HIP].y + p[LM.R_HIP].y) / 2;
      const shoulderY = (p[LM.L_SHOULDER].y + p[LM.R_SHOULDER].y) / 2;
      const ankleY = (p[LM.L_ANKLE].y + p[LM.R_ANKLE].y) / 2;
      const expectedHipY = (shoulderY + ankleY) / 2;
      const hipDeviation = (hipY - expectedHipY) * 100;
      // Is the person roughly horizontal? (shoulders and ankles at similar y)
      const bodyAngleFromHorizontal = Math.abs(shoulderY - ankleY) * 100;
      return { primary: bodyLine, hipDeviation, bodyAngleFromHorizontal };
    },
    // Hold exercises just track hold time, no phases for reps
    phases: ["NOT_IN_POSITION", "HOLDING"],
    thresholds: { minBodyLine: 155, maxHipDev: 10 },
    minTransitionMs: 500,
    holdTimeDisplay: true,
    transitionRules(a, phase) {
      const t = this.thresholds;
      const inPosition = a.primary > t.minBodyLine && Math.abs(a.hipDeviation) < t.maxHipDev;
      switch (phase) {
        case "NOT_IN_POSITION": return inPosition ? "HOLDING" : null;
        case "HOLDING":         return !inPosition ? "NOT_IN_POSITION" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (phase !== "HOLDING") return { type: "warning", msg: "Get into plank position" };
      if (a.hipDeviation > 8)  return { type: "warning", msg: "Hips sagging — tighten core" };
      if (a.hipDeviation < -8) return { type: "warning", msg: "Hips too high — flatten out" };
      if (a.primary < 155)     return { type: "warning", msg: "Straighten your body line" };
      return { type: "good", msg: "Solid plank — hold it! 💎" };
    },
  },

  /* ── MOUNTAIN CLIMBER ──────────────────────────────────── */
  mountain_climber: {
    label: "Mountain Climber",
    type: "rep",
    requiredLandmarks: [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE],
    computeAngles(p) {
      // Track hip-knee tucking. When a knee drives forward, hip angle decreases.
      const lHip = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]);
      const rHip = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]);
      // The driving leg has the smaller hip angle
      const primary = Math.min(lHip, rHip);
      const secondary = Math.max(lHip, rHip);
      // Body line check (similar to plank/push-up)
      const lBody = angleDeg(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_ANKLE]);
      const rBody = angleDeg(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_ANKLE]);
      const bodyLine = (lBody + rBody) / 2;
      return { primary, secondary, bodyLine };
    },
    phases: ["READY", "DRIVING", "TUCKED", "EXTENDING", "COMPLETE"],
    thresholds: { startExtended: 150, enterTucked: 100, extendBack: 145 },
    minTransitionMs: 100, // faster exercise
    transitionRules(a, phase) {
      const t = this.thresholds;
      switch (phase) {
        case "READY":     return a.secondary > t.startExtended ? "DRIVING" : null;
        case "DRIVING":   return a.primary < t.enterTucked ? "TUCKED" : null;
        case "TUCKED":    return a.primary > t.enterTucked + 10 ? "EXTENDING" : null;
        case "EXTENDING": return a.primary > t.extendBack ? "COMPLETE" : null;
        default: return null;
      }
    },
    checkForm(a, phase) {
      if (a.bodyLine < 140)               return { type: "warning", msg: "Keep hips down — stay in plank" };
      if (phase === "TUCKED" && a.primary > 110) return { type: "warning", msg: "Drive knee higher" };
      if (phase === "TUCKED" && a.primary <= 100) return { type: "good", msg: "Great drive! 🔥" };
      if (phase === "EXTENDING")           return { type: "good", msg: "Keep the pace!" };
      return null;
    },
  },
};

/* ───────────────────────────────────────────────────────────────
 * GymMetricEngine — phase-based rep counter & form checker
 *
 * State machine per exercise walks through ordered phases.
 * A rep is counted when the machine cycles back to the first
 * "active" phase after completing all intermediate phases.
 *
 * Hysteresis/debounce is enforced via:
 *  - minTransitionMs per profile
 *  - minimum range-of-motion checks built into transition rules
 *  - landmark visibility gating
 *  - angular smoothing via OneEuroFilter
 * ────────────────────────────────────────────────────────────── */
export class GymMetricEngine {
  constructor() {
    this.repCount = 0;
    this.filters = {};
    this.currentPhase = null;
    this.phaseEnteredAt = 0;     // timestamp when we entered current phase (ms)
    this.holdStartTime = 0;      // for hold exercises
    this.totalHoldTime = 0;      // accumulated hold time in seconds
    this._lastExercise = null;
    this._feedbackCooldown = 0;  // prevent feedback flicker
    this._lastFeedback = null;
    this._noRepFrames = 0;       // frames since last rep (extra debounce)
  }

  /** Get all exercise profile keys. */
  static get exerciseKeys() {
    return Object.keys(EXERCISE_PROFILES);
  }

  /** Get the profile for an exercise. */
  static getProfile(key) {
    return EXERCISE_PROFILES[key] || null;
  }

  /** Reset all state (called when exercise changes). */
  reset() {
    this.repCount = 0;
    this.filters = {};
    this.currentPhase = null;
    this.phaseEnteredAt = 0;
    this.holdStartTime = 0;
    this.totalHoldTime = 0;
    this._lastExercise = null;
    this._feedbackCooldown = 0;
    this._lastFeedback = null;
    this._noRepFrames = 0;
  }

  /** Smooth a landmark point through a OneEuroFilter. */
  smoothPoint(idx, pt, time) {
    if (!this.filters[idx]) {
      this.filters[idx] = {
        x: new OneEuroFilter(1.2, 0.01),
        y: new OneEuroFilter(1.2, 0.01),
      };
    }
    return {
      x: this.filters[idx].x.filter(pt.x, time),
      y: this.filters[idx].y.filter(pt.y, time),
      visibility: pt.visibility,
    };
  }

  /**
   * Main per-frame evaluation.
   * @param {Array} landmarks  — 33 MediaPipe landmarks
   * @param {string} activeExercise — key into EXERCISE_PROFILES
   * @returns {{ reps: number, feedback: string, feedbackType: string, holdTime: number|null, phase: string }}
   */
  evaluateFrame(landmarks, activeExercise) {
    const now = performance.now();
    const timeSec = now / 1000;

    // Reset if exercise changed
    if (activeExercise !== this._lastExercise) {
      this.reset();
      this._lastExercise = activeExercise;
    }

    const profile = EXERCISE_PROFILES[activeExercise];
    if (!profile || !landmarks || landmarks.length < 33) {
      return { reps: this.repCount, feedback: "", feedbackType: "neutral", holdTime: null, phase: "" };
    }

    // ── Smooth all landmarks ──────────────────────────────
    const pts = [];
    for (let i = 0; i < landmarks.length; i++) {
      pts[i] = this.smoothPoint(i, landmarks[i], timeSec);
    }

    // ── Visibility gate ───────────────────────────────────
    const allVisible = profile.requiredLandmarks.every(idx => vis(pts[idx]));
    if (!allVisible) {
      return {
        reps: this.repCount,
        feedback: "Can't see key joints — adjust camera",
        feedbackType: "warning",
        holdTime: profile.type === "hold" ? this.totalHoldTime : null,
        phase: this.currentPhase || "",
      };
    }

    // ── Compute angles ────────────────────────────────────
    const angles = profile.computeAngles(pts);

    // ── Initialize phase on first valid frame ─────────────
    if (this.currentPhase === null) {
      this.currentPhase = profile.phases[0];
      this.phaseEnteredAt = now;
    }

    // ── State machine transitions ─────────────────────────
    const timeInPhase = now - this.phaseEnteredAt;
    const nextPhase = profile.transitionRules.call(profile, angles, this.currentPhase);

    let repJustCounted = false;

    if (nextPhase !== null && timeInPhase >= profile.minTransitionMs) {
      // Extra debounce: require at least 5 frames between reps
      const canCountRep = this._noRepFrames >= 5;

      if (nextPhase === "COMPLETE" && profile.type === "rep" && canCountRep) {
        this.repCount++;
        repJustCounted = true;
        this._noRepFrames = 0;
        // Cycle back to the second phase (skip READY, go to first active phase)
        this.currentPhase = profile.phases[1];
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
    let feedbackResult = profile.checkForm(angles, this.currentPhase);

    // Cooldown to prevent flicker (hold feedback for at least 600ms)
    if (feedbackResult !== null && feedbackResult !== undefined) {
      this._lastFeedback = feedbackResult;
      this._feedbackCooldown = now + 600;
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
    };
  }
}
