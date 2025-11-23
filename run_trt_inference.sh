#!/bin/bash
# Wrapper script to run RF-DETR TensorRT inference (PyTorch backend)
# All dependencies from UV - no system packages

# Set up environment variables
export CUDA_MODULE_LOADING=LAZY
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Run the inference script with UV (all packages from venv)
uv run --extra trt-inference python inference_trt.py "$@"

