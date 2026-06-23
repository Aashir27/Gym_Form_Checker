/**
 * One Euro Filter and Repetition Counter logic.
 * Ported from the original Dart implementation.
 */

class LowPassFilter {
  constructor(initVal) {
    this.hatXPrev = initVal;
    this.initialized = false;
  }

  filter(x, alpha) {
    if (!this.initialized) {
      this.initialized = true;
      this.hatXPrev = x;
      return x;
    }
    const hatX = alpha * x + (1.0 - alpha) * this.hatXPrev;
    this.hatXPrev = hatX;
    return hatX;
  }

  get lastValue() {
    return this.hatXPrev;
  }
}

export class OneEuroFilter {
  constructor(freq = 30.0, minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;

    this.xFilter = new LowPassFilter(0.0);
    this.dxFilter = new LowPassFilter(0.0);
    this.initialized = false;
  }

  alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  filter(value) {
    let dx = 0.0;
    if (!this.initialized) {
      this.initialized = true;
    } else {
      dx = (value - this.xFilter.lastValue) * this.freq;
    }

    const edx = this.dxFilter.filter(dx, this.alpha(this.dCutoff));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this.alpha(cutoff));
  }

  reset() {
    this.initialized = false;
    this.xFilter = new LowPassFilter(0.0);
    this.dxFilter = new LowPassFilter(0.0);
  }
}

export class KeypointFilter {
  constructor(freq = 30.0, minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.fx = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
  }

  filter(rawX, rawY) {
    return {
      x: this.fx.filter(rawX),
      y: this.fy.filter(rawY)
    };
  }

  reset() {
    this.fx.reset();
    this.fy.reset();
  }
}

// --- Rep Counter ---

export const Phase = Object.freeze({
  IDLE: 'IDLE',
  DESCENDING: 'DESCENDING',
  ASCENDING: 'ASCENDING'
});

export class RepCounter {
  constructor(valleyRiseThreshold = 0.08, peakDropThreshold = 0.08, minSignal = 0.0, maxSignal = 1.0) {
    this.valleyRiseThreshold = valleyRiseThreshold;
    this.peakDropThreshold = peakDropThreshold;
    this.minSignal = minSignal;
    this.maxSignal = maxSignal;

    this.reset();
  }

  reset() {
    this.repCount = 0;
    this.runningMin = Infinity;
    this.runningMax = -Infinity;
    this.lastValley = null;
    this.lastPeak = null;
    this.valleyConfirmed = false;
    this.peakConfirmed = false;
    this.phase = Phase.IDLE;
  }

  update(signal) {
    if (signal < this.minSignal || signal > this.maxSignal) return false;

    if (signal < this.runningMin) this.runningMin = signal;
    if (signal > this.runningMax) this.runningMax = signal;

    let completedRep = false;

    // Valley detection
    if (signal >= this.runningMin + this.valleyRiseThreshold) {
      if (!this.valleyConfirmed) {
        this.lastValley = this.runningMin;
        this.valleyConfirmed = true;
        this.peakConfirmed = false;
        this.phase = Phase.ASCENDING;
        this.runningMax = signal; 
      }
    }

    // Peak detection
    if (this.valleyConfirmed && signal <= this.runningMax - this.peakDropThreshold) {
      if (!this.peakConfirmed) {
        this.lastPeak = this.runningMax;
        this.peakConfirmed = true;
        this.phase = Phase.DESCENDING;
        this.runningMin = signal;
      }
    }

    // Rep completion (valley -> peak -> valley)
    if (this.valleyConfirmed && this.peakConfirmed && signal >= this.runningMin + this.valleyRiseThreshold) {
      this.repCount++;
      completedRep = true;
      this.lastValley = this.runningMin;
      this.peakConfirmed = false;
      this.phase = Phase.ASCENDING;
      this.runningMax = signal;
    }

    return completedRep;
  }
}

// --- Math Helpers ---

export function jointAngleDeg(a, b, c) {
  const abX = a.x - b.x, abY = a.y - b.y;
  const cbX = c.x - b.x, cbY = c.y - b.y;
  const dot = abX * cbX + abY * cbY;
  const magA = Math.sqrt(abX * abX + abY * abY);
  const magC = Math.sqrt(cbX * cbX + cbY * cbY);
  if (magA === 0 || magC === 0) return 0.0;
  
  let val = dot / (magA * magC);
  val = Math.max(-1.0, Math.min(1.0, val));
  return Math.acos(val) * 180.0 / Math.PI;
}

export function normaliseAngle(angle, minDeg = 0, maxDeg = 180) {
  let norm = (angle - minDeg) / (maxDeg - minDeg);
  return Math.max(0.0, Math.min(1.0, norm));
}
