import * as ort from 'onnxruntime-web';

/**
 * YOLO11-Pose inference using ONNX Runtime Web.
 */
export class YoloWorker {
  constructor() {
    this.session = null;
    this.inputSize = 640;
    
    // Enable multi-threading based on hardware
    ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
  }

  async init(modelUrl) {
    try {
      // Prioritize modern WebGPU if available, then fallback to WebGL and WASM
      this.session = await ort.InferenceSession.create(modelUrl, { 
        executionProviders: ['webgpu', 'webgl', 'wasm'] 
      });
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
    const { data } = imageData; // Assuming data is strictly 640x640 from the optimized camera grab
    const float32Data = new Float32Array(3 * this.inputSize * this.inputSize);
    
    let rOffset = 0;
    let gOffset = this.inputSize * this.inputSize;
    let bOffset = 2 * this.inputSize * this.inputSize;

    // Extremely fast 1D array loop (avoids expensive 2D coordinate math in JS)
    for (let i = 0; i < data.length; i += 4) {
      float32Data[rOffset++] = data[i] / 255.0;
      float32Data[gOffset++] = data[i + 1] / 255.0;
      float32Data[bOffset++] = data[i + 2] / 255.0;
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
