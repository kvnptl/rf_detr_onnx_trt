#!/usr/bin/env python3
"""
Convert RF-DETR ONNX model to TensorRT engine using Python API
"""
import argparse
import os
import sys

try:
    import tensorrt as trt
except ImportError:
    print("Error: tensorrt module not found. Install with: pip install tensorrt")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("Error: onnx module not found. Install with: pip install onnx")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def get_onnx_input_shape(onnx_file_path):
    """
    Read ONNX model and extract input shape
    
    Returns:
        tuple: (input_name, batch_size, channels, height, width)
    """
    model = onnx.load(onnx_file_path)
    
    # Get first input (RF-DETR has single input named 'input')
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    
    # Extract shape dimensions
    dims = input_tensor.type.tensor_type.shape.dim
    shape = [d.dim_value if d.dim_value > 0 else -1 for d in dims]
    
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got {len(shape)}D: {shape}")
    
    batch_size, channels, height, width = shape
    
    print(f"Detected input: {input_name}")
    print(f"  Shape: [batch={batch_size}, channels={channels}, height={height}, width={width}]")
    
    return input_name, batch_size, channels, height, width


def build_engine(onnx_file_path, engine_file_path, fp16=False, max_batch_size=32):
    """
    Build TensorRT engine from RF-DETR ONNX file
    
    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        max_batch_size: Maximum batch size for optimization
    """
    print(f"Building TensorRT engine from {onnx_file_path}")
    print(f"FP16 mode: {fp16}")
    print(f"Max batch size: {max_batch_size}")
    
    # Get input shape from ONNX model
    input_name, batch_size, channels, height, width = get_onnx_input_shape(onnx_file_path)
    
    if height <= 0 or width <= 0:
        raise ValueError(f"Dynamic spatial dimensions not supported. Got height={height}, width={width}")
    
    # Check if batch dimension is dynamic or fixed
    is_dynamic_batch = (batch_size <= 0)
    
    if is_dynamic_batch:
        print(f"Using resolution: {height}x{width} with dynamic batch size")
    else:
        print(f"Using resolution: {height}x{width} with fixed batch size: {batch_size}")
        max_batch_size = batch_size  # Override max_batch_size for fixed models
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"Loading ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"Successfully parsed ONNX file")
    print(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set memory pool limit (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    
    # Enable FP16 if requested
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("Warning: FP16 not supported on this platform, using FP32")
    
    # Build optimization profile for batch size
    profile = builder.create_optimization_profile()
    
    if is_dynamic_batch:
        # Dynamic batch: create profile with min/opt/max
        min_batch = 1
        opt_batch = max(1, max_batch_size // 2)
        max_batch = max_batch_size
    else:
        # Fixed batch: all three must match the fixed size
        min_batch = opt_batch = max_batch = batch_size
    
    profile.set_shape(
        input_name,
        min=(min_batch, channels, height, width),
        opt=(opt_batch, channels, height, width),
        max=(max_batch, channels, height, width)
    )
    
    print(f"Optimization profile:")
    print(f"  Min: [{min_batch}, {channels}, {height}, {width}]")
    print(f"  Opt: [{opt_batch}, {channels}, {height}, {width}]")
    print(f"  Max: [{max_batch}, {channels}, {height}, {width}]")
    
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    # Save engine to file
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Successfully created TensorRT engine: {engine_file_path}")
    print(f"Engine size: {os.path.getsize(engine_file_path) / (1024**2):.2f} MB")
    
    return serialized_engine


def main():
    parser = argparse.ArgumentParser(description='Convert RF-DETR ONNX to TensorRT engine')
    parser.add_argument('--onnx', required=True, help='Path to RF-DETR ONNX model')
    parser.add_argument('--engine', required=True, help='Path to save TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--max-batch-size', type=int, default=32, help='Maximum batch size (default: 32)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX file not found: {args.onnx}")
        sys.exit(1)
    
    build_engine(args.onnx, args.engine, args.fp16, args.max_batch_size)


if __name__ == '__main__':
    main()

