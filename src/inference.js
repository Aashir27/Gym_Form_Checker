import * as ort from 'onnxruntime-web';

/**
 * YOLO11-Pose inference using ONNX Runtime Web.
 */
export class YoloWorker {
  constructor() {
    this.session = null;
    this.inputSize = 640;
    
    // Disable webgl/wasm threads if it crashes, but webgl is fast for vision
    ort.env.wasm.numThreads = 1;
  }

  async init(modelUrl) {
    try {
      // Try webgl for hardware acceleration, fallback to wasm
      this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgl', 'wasm'] });
      console.log('ONNX Session initialized.');
      return true;
    } catch (err) {
      console.error('Failed to load ONNX model:', err);
      return false;
    }
  }

  async runInference(imageData) {
    if (!this.session) return null;

    // 1. Preprocess: Resize to 640x640 and normalize to [0,1], CHW format
    const tensor = this.preprocess(imageData);

    // 2. Run inference
    const feeds = { images: tensor };
    const results = await this.session.run(feeds);
    
    // 3. Postprocess: Output is [1, 56, 8400]
    const output = results.output0; // Name depends on YOLO export, usually 'output0'
    const poses = this.decode(output.data, output.dims);
    
    return poses;
  }

  preprocess(imageData) {
    const { width, height, data } = imageData;
    const float32Data = new Float32Array(3 * this.inputSize * this.inputSize);
    
    // Simple nearest neighbor resize and CHW packing
    const scaleX = width / this.inputSize;
    const scaleY = height / this.inputSize;

    let rOffset = 0;
    let gOffset = this.inputSize * this.inputSize;
    let bOffset = 2 * this.inputSize * this.inputSize;

    for (let y = 0; y < this.inputSize; y++) {
      for (let x = 0; x < this.inputSize; x++) {
        const srcX = Math.floor(x * scaleX);
        const srcY = Math.floor(y * scaleY);
        const i = (srcY * width + srcX) * 4;

        // YOLO expects RGB float32 [0-1]
        float32Data[rOffset++] = data[i] / 255.0;
        float32Data[gOffset++] = data[i + 1] / 255.0;
        float32Data[bOffset++] = data[i + 2] / 255.0;
      }
    }

    return new ort.Tensor('float32', float32Data, [1, 3, this.inputSize, this.inputSize]);
  }

  decode(tensorData, dims) {
    // dims: [1, 56, 8400]
    const cols = dims[2];
    const poses = [];

    const confThreshold = 0.35;

    for (let c = 0; c < cols; c++) {
      const conf = tensorData[4 * cols + c];
      if (conf < confThreshold) continue;

      const cx = tensorData[0 * cols + c];
      const cy = tensorData[1 * cols + c];
      const w = tensorData[2 * cols + c];
      const h = tensorData[3 * cols + c];

      const keypoints = [];
      for (let k = 0; k < 17; k++) {
        const base = (5 + k * 3) * cols + c;
        keypoints.push({
          x: tensorData[base],
          y: tensorData[base + 1],
          confidence: tensorData[base + 2]
        });
      }

      poses.push({ cx, cy, w, h, confidence: conf, keypoints });
    }

    return this.nms(poses, 0.45);
  }

  nms(poses, iouThreshold) {
    poses.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    const suppressed = new Array(poses.length).fill(false);

    for (let i = 0; i < poses.length; i++) {
      if (suppressed[i]) continue;
      kept.push(poses[i]);
      for (let j = i + 1; j < poses.length; j++) {
        if (!suppressed[j] && this.iou(poses[i], poses[j]) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  iou(a, b) {
    const aX1 = a.cx - a.w / 2, aY1 = a.cy - a.h / 2;
    const aX2 = a.cx + a.w / 2, aY2 = a.cy + a.h / 2;
    const bX1 = b.cx - b.w / 2, bY1 = b.cy - b.h / 2;
    const bX2 = b.cx + b.w / 2, bY2 = b.cy + b.h / 2;

    const ix1 = Math.max(aX1, bX1);
    const iy1 = Math.max(aY1, bY1);
    const ix2 = Math.min(aX2, bX2);
    const iy2 = Math.min(aY2, bY2);

    const interW = Math.max(0, ix2 - ix1);
    const interH = Math.max(0, iy2 - iy1);
    const inter = interW * interH;
    
    const union = a.w * a.h + b.w * b.h - inter;
    return union <= 0 ? 0 : inter / union;
  }
}
