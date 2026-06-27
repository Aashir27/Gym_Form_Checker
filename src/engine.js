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

/**
 * GymMetricEngine — rep counter & form checker.
 *
 * Joint index mapping (MediaPipe PoseLandmarker):
 *   Hip=24  Knee=26  Ankle=28  Shoulder=12  Elbow=14  Wrist=16
 *
 * State machine:  START → GOING_DOWN → GOING_UP → (rep++) → GOING_DOWN …
 */
export class GymMetricEngine {
  constructor() {
    this.repCount = 0;
    this.state = "START";
    this.filters = {};
    this.exerciseProfiles = {
      squat: {
        primary: "knee_angle",
        valley: 100,
        peak: 160,
        checkForm: (angles) =>
          angles.knee_angle < 65
            ? "Too deep! Keep thighs parallel."
            : "",
      },
      bicep_curl: {
        primary: "elbow_angle",
        valley: 50,
        peak: 145,
        checkForm: (angles) => "",
      },
    };
  }

  calculateAngle(p1, p2, p3) {
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    if (mag1 * mag2 === 0) return 0;
    return Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
  }

  smoothPoint(idx, pt, time) {
    if (!this.filters[idx]) {
      this.filters[idx] = {
        x: new OneEuroFilter(),
        y: new OneEuroFilter(),
      };
    }
    return {
      x: this.filters[idx].x.filter(pt.x, time),
      y: this.filters[idx].y.filter(pt.y, time),
      visibility: pt.visibility,
    };
  }

  evaluateFrame(landmarks, activeExercise) {
    const time = performance.now() / 1000;
    const profile = this.exerciseProfiles[activeExercise];
    if (!profile || !landmarks || landmarks.length < 33) {
      return { reps: this.repCount, feedback: "" };
    }

    const hip = this.smoothPoint(24, landmarks[24], time);
    const knee = this.smoothPoint(26, landmarks[26], time);
    const ankle = this.smoothPoint(28, landmarks[28], time);
    const shoulder = this.smoothPoint(12, landmarks[12], time);
    const elbow = this.smoothPoint(14, landmarks[14], time);
    const wrist = this.smoothPoint(16, landmarks[16], time);

    const angles = {
      knee_angle: this.calculateAngle(hip, knee, ankle),
      elbow_angle: this.calculateAngle(shoulder, elbow, wrist),
    };

    const currentVal = angles[profile.primary];
    let feedback = profile.checkForm(angles);

    if (this.state === "START" && currentVal > profile.peak - 10) {
      this.state = "GOING_DOWN";
    } else if (this.state === "GOING_DOWN" && currentVal < profile.valley) {
      this.state = "GOING_UP";
    } else if (this.state === "GOING_UP" && currentVal > profile.peak - 10) {
      this.repCount++;
      this.state = "GOING_DOWN";
    }

    return { reps: this.repCount, feedback };
  }
}
