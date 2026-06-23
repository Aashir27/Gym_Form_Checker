/**
 * Camera Handler for Web
 */

export class CameraService {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    
    // Create an offscreen canvas specifically for fast GPU resizing to 640x640
    this.aiCanvas = document.createElement('canvas');
    this.aiCanvas.width = 640;
    this.aiCanvas.height = 640;
    this.aiCtx = this.aiCanvas.getContext('2d', { willReadFrequently: true });

    this.stream = null;
    this.isActive = false;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      this.video.srcObject = this.stream;
      
      // Wait for video to be ready
      await new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          this.canvas.width = this.video.videoWidth;
          this.canvas.height = this.video.videoHeight;
          resolve();
        };
      });
      
      this.video.play();
      this.isActive = true;
      return true;
    } catch (err) {
      console.error('Failed to start camera:', err);
      return false;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    this.isActive = false;
  }

  /**
   * Draws the current video frame to the canvas and returns the ImageData.
   */
  getFrameData() {
    if (!this.isActive) return null;
    // Draw the video resized directly to 640x640 using the browser's fast hardware scaler
    this.aiCtx.drawImage(this.video, 0, 0, 640, 640);
    return this.aiCtx.getImageData(0, 0, 640, 640);
  }
}
