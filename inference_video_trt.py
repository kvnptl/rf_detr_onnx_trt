# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
# ------------------------------------------------------------------------

# Import cv2 FIRST to avoid conflicts
import cv2
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from rfdetr_trt import RFDETR_TRT, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES
from tqdm import tqdm

def process_video(model, video_path, output_path, threshold, max_number_boxes, skip_frames=0):
    """
    Process video file and save output with detections using TensorRT.
    
    Args:
        model: RFDETR_TRT model instance
        video_path: Path to input video
        output_path: Path to save output video
        threshold: Confidence threshold
        max_number_boxes: Maximum number of boxes
        skip_frames: Process every Nth frame (0 = process all frames)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                out.write(frame)
                frame_count += 1
                pbar.update(1)
                continue
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference
            scores, labels, boxes, masks = model.predict_from_image(
                pil_image, threshold, max_number_boxes
            )
            
            # Draw detections on frame
            frame_with_detections = model.draw_detections(
                pil_image, boxes, labels, masks, scores
            )
            
            # Convert back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(frame_bgr)
            
            frame_count += 1
            processed_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\nProcessed {processed_count}/{total_frames} frames")
    print(f"Output saved to: {output_path}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a RF-DETR TensorRT engine on video files."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the TensorRT engine file (e.g., rf-detr-nano.trt)"
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Path to save the output video (default: input_trt_output.mp4)"
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
    parser.add_argument(
        "--skip_frames",
        default=0,
        type=int,
        help="Process every Nth frame (0 = process all frames, 1 = process every other frame, etc.)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="CUDA device to use (default: cuda:0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_trt_output{video_path.suffix}")
    
    # Initialize the model
    print("Loading TensorRT engine...")
    model = RFDETR_TRT(args.model, device=args.device)
    
    # Process video
    process_video(
        model,
        args.video,
        args.output,
        args.threshold,
        args.max_number_boxes,
        args.skip_frames
    )

if __name__ == "__main__":
    main()

