#!/bin/bash
# Wrapper script to convert RF-DETR ONNX models to TensorRT
# All dependencies from UV - no system packages

# Set up environment variables
export CUDA_MODULE_LOADING=LAZY
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Run the conversion script with UV (needs both onnx and tensorrt)
# Use deploy-tools for onnx, but need trt-inference for tensorrt
uv run --extra trt-inference python onnx_to_trt.py "$@"

