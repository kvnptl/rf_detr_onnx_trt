# RF-DETR

### Setup
- **Package Manager**: UV (all dependencies)
- **Python**: 3.12
- **CUDA**: 12.6
- **PyTorch**: 2.8.0+cu128
- **TensorRT**: tensorrt-cu12 10.14.1
- **OpenCV**: opencv-python 4.11.0
- **ONNX Runtime**: onnxruntime-gpu 1.17+

## Sample Commands

### 1: ONNX Image Inference
```bash
uv run inference.py --model models/rf-detr-nano.onnx --image data/dog.jpeg --output test_onnx_img.jpg
```

### 2: ONNX Video Inference
```bash
uv run inference_video.py --model models/rf-detr-nano.onnx --video data/cars.mp4 --output data/test_onnx_video.mp4 --threshold 0.6
```

### 3: ONNX to TensorRT Conversion
```bash
./convert_to_trt.sh --onnx models/rf-detr-nano.onnx --engine models/rf-detr-nano-test.trt --fp16
``` 

### 4: TRT Image Inference
```bash
./run_trt_inference.sh --model models/rf-detr-nano-test.trt --image data/dog.jpeg --output test_trt_img.jpg
```

### 5: TRT Video Inference
```bash
./run_trt_video_inference.sh --model models/rf-detr-nano-test.trt --video data/cars.mp4 --output data/test_trt_video.mp4 --threshold 0.6
```

**Note:** TensorRT's main advantage is single-image latency, not video throughput (which is I/O bound).