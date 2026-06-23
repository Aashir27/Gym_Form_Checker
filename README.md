# GymForm AI — Offline Flutter Pose Estimation

YOLO11-Pose Nano (INT8 ONNX) · `flutter_onnxruntime` · NNAPI/CoreML · Zero-copy YUV FFI

---

## PART 1 — Local Host Machine Dependencies

### Build Toolchain

| Requirement | Version / Notes |
|---|---|
| Flutter SDK | ≥ 3.22 (stable channel) |
| Dart SDK | ≥ 3.3 (bundled with Flutter) |
| Android NDK | r26b+ (`$ANDROID_HOME/ndk/<ver>`) |
| Android SDK | API 26+ (minSdk), API 35 (targetSdk) |
| CMake | ≥ 3.22 (via Android SDK Manager or system) |
| Xcode | ≥ 15.2 (macOS only, iOS builds) |
| CocoaPods | ≥ 1.14 (macOS/iOS only) |
| Python | 3.10 – 3.12 |
| pip packages | `ultralytics`, `onnx`, `onnxruntime`, `onnxsim`, `numpy` |

### Android ABI Targets
```
armeabi-v7a   (32-bit ARM — legacy devices)
arm64-v8a     (64-bit ARM — primary target, NNAPI accelerated)
x86_64        (emulator)
```

### Python: PyTorch → INT8 ONNX Conversion
```python
# Requirements: pip install ultralytics onnx onnxsim onnxruntime
from ultralytics import YOLO
import onnx
from onnxsim import simplify

# 1. Export FP32 ONNX from official YOLO11-Pose Nano weights
model = YOLO("yolo11n-pose.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    simplify=True,
    dynamic=False,
    half=False,         # keep FP32 for calibration
    nms=False,          # handle NMS in Dart for max flexibility
)

# 2. Static INT8 quantisation via ONNXRuntime quantization tools
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np

class DummyCalibReader(CalibrationDataReader):
    """Replace with real gym frame samples for production accuracy."""
    def __init__(self, n=50):
        self.data = iter([
            {"images": np.random.rand(1, 3, 640, 640).astype(np.float32)}
            for _ in range(n)
        ])
    def get_next(self):
        return next(self.data, None)

quantize_static(
    model_input="yolo11n-pose.onnx",
    model_output="yolo11n-pose_int8.onnx",
    calibration_data_reader=DummyCalibReader(n=100),
    quant_type=QuantType.QInt8,
    per_channel=False,          # per-tensor for NNAPI compat
    reduce_range=False,
    extra_options={"ActivationSymmetric": True},
)

# 3. Verify & simplify output
m = onnx.load("yolo11n-pose_int8.onnx")
m_sim, ok = simplify(m)
assert ok, "Simplification failed"
onnx.save(m_sim, "assets/models/yolo11n-pose_int8.onnx")
print("INT8 model saved →", m_sim.graph.node[0].op_type)
```

> **Copy output** to `assets/models/yolo11n-pose_int8.onnx` before `flutter run`.

---

## PART 2 — Feature-First Skeleton Directory Tree

```
gym_form_checker/
├── assets/
│   ├── models/
│   │   └── yolo11n-pose_int8.onnx          ← INT8 ONNX model (not in VCS)
│   └── config/
│       └── model_config.json               ← thresholds, keypoint labels, skeleton
│
├── native/
│   └── yuv_to_rgb/
│       ├── CMakeLists.txt
│       └── src/
│           └── yuv_converter.c             ← zero-copy YUV420→RGB float32 FFI
│
├── android/
│   └── app/
│       ├── build.gradle                    ← NDK abiFilters, cmake linkage
│       └── src/main/
│           ├── AndroidManifest.xml         ← CAMERA, NNAPI permissions
│           └── jniLibs/                    ← pre-built .so if not cmake-built
│
├── ios/
│   └── Runner/
│       ├── Info.plist                      ← NSCameraUsageDescription
│       └── Frameworks/                     ← CoreML .mlpackage if bundled
│
├── lib/
│   ├── main.dart
│   │
│   ├── core/
│   │   ├── filtering/
│   │   │   └── one_euro_filter.dart        ← 1€ adaptive low-pass + KeypointFilter
│   │   └── math/
│   │       ├── angle.dart                  ← jointAngleDeg(), normaliseAngle()
│   │       ├── normalization.dart          ← pixel→[0,1] keypoint helpers
│   │       └── math.dart                   ← barrel export
│   │
│   ├── services/
│   │   └── camera/
│   │       └── camera_service.dart         ← CameraController lifecycle + YUV420 stream
│   │
│   ├── features/
│   │   ├── inference/
│   │   │   ├── yolo_worker.dart            ← OrtSession lifecycle + pose decode + NMS
│   │   │   ├── yuv_ffi_bridge.dart         ← dart:ffi ↔ native YUV converter
│   │   │   └── pose_pipeline.dart          ← orchestrates camera→FFI→ONNX→filter
│   │   │
│   │   └── metrics/
│   │       ├── rep_counter.dart            ← peak/valley state machine
│   │       ├── form_analyser.dart          ← joint angle thresholds per exercise
│   │       └── exercise_config.dart        ← exercise definitions (squat, curl, etc.)
│   │
│   └── ui/
│       ├── screens/
│       │   ├── home_screen.dart
│       │   └── workout_screen.dart
│       └── widgets/
│           ├── skeleton_painter.dart       ← CustomPainter drawing COCO edges
│           ├── rep_counter_overlay.dart    ← HUD: rep count, phase indicator
│           └── confidence_badge.dart       ← per-keypoint visibility indicator
│
└── pubspec.yaml
```

---

## PART 3 — Base Workspace Files

See source files at:

- [`pubspec.yaml`](pubspec.yaml)
- [`lib/core/filtering/one_euro_filter.dart`](lib/core/filtering/one_euro_filter.dart)
- [`lib/features/inference/yolo_worker.dart`](lib/features/inference/yolo_worker.dart)
- [`lib/features/metrics/rep_counter.dart`](lib/features/metrics/rep_counter.dart)
- [`lib/services/camera/camera_service.dart`](lib/services/camera/camera_service.dart)
- [`lib/core/math/angle.dart`](lib/core/math/angle.dart)
- [`assets/config/model_config.json`](assets/config/model_config.json)

---

## Quick Start

```bash
# 1. Get packages
flutter pub get

# 2. Generate Riverpod providers
dart run build_runner build --delete-conflicting-outputs

# 3. Run on device (ensure model file is placed first)
flutter run --release
```

### Android: Enable NNAPI
In `android/app/build.gradle` ensure:
```groovy
android {
    defaultConfig {
        minSdk 26          // NNAPI minimum
        ndk { abiFilters "arm64-v8a", "armeabi-v7a", "x86_64" }
    }
}
```

### iOS: Enable CoreML
In `ios/Runner/Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Required for real-time gym form analysis.</string>
```