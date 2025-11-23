# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
# ------------------------------------------------------------------------

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from rfdetr_onnx_tracking import RFDETR_ONNX, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES
from tqdm import tqdm
import supervision as sv

def get_video_rotation(video_path):
    """
    Get the rotation metadata from a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Rotation angle in degrees (0, 90, 180, 270) or None
    """
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'side_data=rotation', 
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            # Get first rotation value
            rotation = int(float(result.stdout.strip().split('\n')[0]))
            # Normalize to 0-360 range
            rotation = rotation % 360
            if rotation < 0:
                rotation += 360
            return rotation if rotation in [0, 90, 180, 270] else None
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None

def rotate_frame(frame, rotation):
    """
    Rotate frame based on rotation angle.
    
    Args:
        frame: numpy array (OpenCV frame)
        rotation: rotation angle (0, 90, 180, 270)
        
    Returns:
        Rotated frame
    """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def process_video(model, video_path, output_path, threshold, max_number_boxes, skip_frames=0):
    """
    Process video file with object tracking and save output with detections.
    
    Args:
        model: RFDETR_ONNX model instance
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
    
    # Check for rotation metadata
    rotation = get_video_rotation(video_path)
    # if rotation and rotation != 0:
    #     print(f"Video has rotation metadata: {rotation}° - will auto-correct")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust dimensions if rotation is 90 or 270 degrees
    if rotation in [90, 270]:
        width, height = height, width
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize ByteTrack tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=threshold,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=fps
    )
    print(f"Initialized ByteTrack tracker")
    
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
            
            # Apply rotation correction if needed
            if rotation and rotation != 0:
                frame = rotate_frame(frame, rotation)
            
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
            
            # Convert to supervision Detections format
            detections = model.to_supervision_detections(scores, labels, boxes, masks)
            
            # Update tracker with detections
            detections = tracker.update_with_detections(detections)
            
            # Draw tracked detections on frame
            frame_with_detections = model.draw_tracked_detections(pil_image, detections)
            
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
        description="Run inference with a RF-DETR ONNX model on video files."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the ONNX model file (e.g., rf-detr-base.onnx)"
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
        help="Path to save the output video (default: input_output.mp4)"
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
        "--cpu",
        action="store_true",
        help="Force CPU execution instead of GPU (default: use GPU if available)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_output{video_path.suffix}")
    
    # Initialize the model
    print("Loading model...")
    model = RFDETR_ONNX(args.model, use_gpu=not args.cpu)
    
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

