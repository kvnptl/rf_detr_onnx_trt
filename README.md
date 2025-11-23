# RF-DETR with ONNX and TensorRT

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)


This repository contains code to load an ONNX version of RF-DETR and perform inference, including drawing the results on images. It demonstrates how to convert a PyTorch model to ONNX format and inference with minimal dependencies. And also how to convert the ONNX model to TensorRT for faster inference.

RF-DETR is a transformer-based object detection and instance segmentation architecture developed by Roboflow. For more details on the model, please refer to the impressive work by the Roboflow team [here](https://github.com/roboflow/rf-detr/tree/main).

| Roboflow | ONNX Runtime Inference<p> (Object detection) | ONNX Runtime Inference <p> (Instance segmentation) |
|----------------------|-----------------------------|-----------------------------|
| <p align="center"><img src="assets/official_repo.png" width="100%"></p> | <p align="center"><img src="assets/object_detection.jpg" width="74%"></p> | <p align="center"><img src="assets/instance_segmentation.jpg" width="74%"></p> |

## Installation

First, clone the repository:

```bash
git clone --depth 1 https://github.com/PierreMarieCurie/rf-detr-onnx.git
```
Then, install the required dependencies.
<details open>
  <summary>Using uv (recommanded) </summary><br>
  
  If not installed, just run (on macOS and Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
> Check [Astral documentation](https://docs.astral.sh/uv/getting-started/installation) if you need alternative installation methods.

Then:
```bash
uv sync --extra export-tools
```
If you only want to use the inference scripts without converting your own model, you don’t need the `rfdetr` dependencies, so just run:
```bash
uv sync
```
</details>
<details>
  <summary>Not using uv (not recommanded)</summary><br>

```bash
pip install --upgrade .
```
Make sure to install Python 3.9+ on your local or virtual environment.
</details>

## Model to ONNX format

### Downloading from Hugging-face

Roboflow provides pre-trained RF-DETR models on the [COCO](https://cocodataset.org/#home) and [Objects365](https://www.objects365.org/overview.html) datasets. We have already converted some of these models to the ONNX format for you, which you can directly download from [Hugging Face](https://huggingface.co/PierreMarieCurie/rf-detr-onnx).

Note that this corresponds to [rf-detr version 1.3.0](https://github.com/roboflow/rf-detr/tree/1.3.0):
- **Object detection**:
    - [rf-detr-base](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base-coco.onnx), [rf-detr-large](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-large.onnx), [rf-detr-nano](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx), [rf-detr-small](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-small.onnx) and [rf-detr-medium](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-medium.onnx): different checkpoints trained on COCO dataset
    - [rf-detr-base-o365](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base-o365.onnx): base checkpoint trained on Objects365 dataset
- **Instance segmentation**
    - [rf-detr-seg-preview](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-preview.onnx): trained on COCO dataset

### Converting 

If you want to export your own fine-tuned RF-DETR model, we provide a script to help you do it:
``` bash
uv run export.py --checkpoint path/to/your/file.pth
```
You don’t need to specify the architecture (Nano, Small, Medium, Base, Large), it is detected automatically.
<details>
  <summary>Additionnal conversion parameters</summary><br>

```bash
uv run export.py -h
```
Use the `--model-name` argument to specify the output ONNX file, and add the `--no-simplify` flag if you want to skip simplification.
</details>

## Inference Script Example

Below is an example showing how to perform inference on a single image:

``` Python
from rfdetr_onnx import RFDETR_ONNX

# Get model and image
image_path = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
model_path = "rf-detr-base.onnx"

# Initialize the model
model = RFDETR_ONNX(model_path)

# Run inference and get detections
_, labels, boxes, masks = model.predict(image_path)

# Draw and display the detections
model.save_detections(image_path, boxes, labels, masks, "output.jpg")
```

Alternatively, we provide a script to help you do it:
``` bash
uv run inference.py --model path/to/your/model.onnx --image path/to/your/image
```
<details>
  <summary>Additionnal inference parameters</summary><br>

```bash
uv run inference.py -h
```
Use the `--threshold` argument to specify the confidence threshold and the `--max_number_boxes` argument to limit the maximum number of bounding boxes. Also, add `--output` option to specify the output file name and extension if needed (default: output.jpg)
</details>

## TensorRT Deployment (Optional)

For high-performance inference on NVIDIA GPUs, you can convert the ONNX model to TensorRT:

### Requirements
- TensorRT must be installed on your system (matching your CUDA version)
- ONNX library will be automatically installed by the script

### Convert ONNX to TensorRT

**Easy method** (recommended):
```bash
./convert_to_trt.sh --onnx models/rf-detr-nano.onnx --engine models/rf-detr-nano.trt --fp16 --max-batch-size 8
```

**Manual method**:
```bash
uv sync --extra deploy-tools
CUDA_MODULE_LOADING=LAZY PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages:$PYTHONPATH \
  python3 onnx_to_trt.py \
  --onnx models/rf-detr-nano.onnx \
  --engine models/rf-detr-nano.trt \
  --fp16 \
  --max-batch-size 8
```

**Arguments:**
- `--onnx`: Path to your RF-DETR ONNX model
- `--engine`: Output path for the TensorRT engine file
- `--fp16`: (Optional) Enable FP16 precision for faster inference
- `--max-batch-size`: (Optional) Maximum batch size for optimization (default: 32)

**Notes:**
- The script automatically detects the model's input resolution (384, 512, 576, or 560)
- Handles both fixed and dynamic batch sizes automatically
- TensorRT engines are GPU-specific and must be rebuilt for different GPUs
- FP16 mode provides ~2x speedup on modern NVIDIA GPUs with Tensor Cores
- For fixed batch models, the `--max-batch-size` parameter is ignored

### Run Inference with TensorRT

Once you've converted your model to TensorRT, run inference:

```bash
./run_trt_inference.sh --model models/rf-detr-nano.trt --image data/dog.jpeg --output output_trt.jpg
```

**Arguments:**
- `--model`: Path to your TensorRT engine file (.trt)
- `--image`: Path or URL to the input image
- `--output`: Path to save the output image (default: output_trt.jpg)
- `--threshold`: Confidence threshold for filtering detections (default: 0.5)
- `--max_number_boxes`: Maximum number of boxes to return (default: 300)

**Comparison:**

| Feature | ONNX Runtime | TensorRT |
|---------|--------------|----------|
| **Command** | `uv run inference.py --model model.onnx ...` | `./run_trt_inference.sh --model model.trt ...` |
| **Model Format** | `.onnx` | `.trt` |
| **Speed** | Good | Faster (2-3x) |
| **Portability** | Cross-platform | GPU-specific |
| **Precision** | FP32 | FP16/FP32 |

## License

This repository is licensed under the MIT License. See [license file](LICENSE) for more details.

However, some parts of the code are derived from third-party software licensed under the Apache License 2.0. Below are the details:

- RF-DETR pretrained weights and all rfdetr package in export.py (Copyright 2025 Roboflow): [link](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/detr.py#L42)
- _postprocess method of RFDETR_ONNX class in rfdetr_onnx.py.models.lwdetr.py (Copyright 2025 Roboflow): [link](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/models/lwdetr.py#L708) 

Apache License 2.0 reference: https://www.apache.org/licenses/LICENSE-2.0

## Acknowledgements
- Thanks to the **Roboflow** team and everyone involved in the development of RF-DETR, particularly for sharing a state-of-the-art model under a permissive free software license.