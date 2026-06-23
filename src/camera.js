/**
 * Camera Handler for Web
 */

export class CameraService {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
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
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }
}
