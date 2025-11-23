# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
# ------------------------------------------------------------------------

import argparse
from rfdetr_trt import RFDETR_TRT, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a RF-DETR TensorRT engine."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the TensorRT engine file (e.g., rf-detr-nano.trt)"
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path or URL to the input image"
    )
    parser.add_argument(
        "--output",
        default="output_trt.jpg",
        type=str,
        help="Path to save the output image with detections (default: output_trt.jpg)"
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        type=float,
        help=f"Confidence threshold for filtering detections (default: {DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--max_number_boxes",
        default=DEFAULT_MAX_NUMBER_BOXES,
        type=int,
        help=f"Maximum number of boxes to return (default: {DEFAULT_MAX_NUMBER_BOXES})"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize the TensorRT model
    print(f"Loading TensorRT engine: {args.model}")
    model = RFDETR_TRT(args.model)

    # Run inference and get detections
    print(f"Running inference on: {args.image}")
    scores, labels, boxes, masks = model.predict(args.image, args.threshold, args.max_number_boxes)
    
    print(f"Detected {len(boxes)} objects")

    # Draw and save detections
    model.save_detections(args.image, boxes, labels, masks, scores, args.output)
    print(f"Detections saved to: {args.output}")

if __name__ == "__main__":
    main()

