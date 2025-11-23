# Object Tracking with RF-DETR and Supervision

This guide explains how to use the object tracking integration with RF-DETR ONNX models using the [Roboflow Supervision](https://github.com/roboflow/supervision) library.

## Installation

The project uses `uv` for dependency management. Install the dependencies with:

```bash
uv sync
```

If you need to add supervision to a different project:

```bash
uv add supervision
```

The `supervision` package (>=0.27.0) is already included in the project dependencies.

## Features

The tracking integration provides:

- **Object Tracking**: Maintains consistent IDs for objects across video frames using ByteTrack
- **Track Visualization**: Displays tracker IDs alongside class names and confidence scores
- **Mask Support**: Works with both detection and instance segmentation models
- **Supervision Integration**: Leverages supervision's powerful annotators and tracking algorithms

## Usage

### Basic Video Tracking

Process a video with object tracking:

```bash
uv run python inference_video_tracking.py \
    --model rf-detr-base.onnx \
    --video input_video.mp4 \
    --output tracked_output.mp4
```

### Advanced Options

```bash
uv run python inference_video_tracking.py \
    --model rf-detr-base.onnx \
    --video input_video.mp4 \
    --output tracked_output.mp4 \
    --threshold 0.3 \
    --max_number_boxes 100 \
    --skip_frames 0 \
    --cpu
```

**Arguments:**
- `--model`: Path to ONNX model file (required)
- `--video`: Path to input video file (required)
- `--output`: Path to save output video (default: `{input}_output.mp4`)
- `--threshold`: Confidence threshold for detections (default: 0.5)
- `--max_number_boxes`: Maximum number of boxes per frame (default: 300)
- `--skip_frames`: Process every Nth frame (0 = all frames, 1 = every other frame, etc.)
- `--cpu`: Force CPU execution instead of GPU

## How It Works

### 1. Detection to Supervision Format

The `RFDETR_ONNX` class now includes a method to convert detections to supervision's format:

```python
# Run detection
scores, labels, boxes, masks = model.predict_from_image(image, threshold=0.5)

# Convert to supervision Detections
detections = model.to_supervision_detections(scores, labels, boxes, masks)
```

### 2. ByteTrack Tracker

The tracker maintains object identities across frames:

```python
import supervision as sv

# Initialize tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30
)

# Update tracker with new detections
detections = tracker.update_with_detections(detections)
```

### 3. Enhanced Visualization

The tracked detections are visualized with:
- Bounding boxes
- Instance segmentation masks (if available)
- Class names with confidence scores
- **Tracker IDs** (e.g., `#42 person 0.95`)

```python
# Draw tracked detections
annotated_frame = model.draw_tracked_detections(image, detections)
```

## Customization

### Adjust Tracker Parameters

Modify the ByteTrack initialization in `inference_video_tracking.py`:

```python
tracker = sv.ByteTrack(
    track_activation_threshold=0.5,  # Minimum confidence to start tracking
    lost_track_buffer=30,             # Frames to keep lost tracks
    minimum_matching_threshold=0.8,   # IOU threshold for matching
    frame_rate=30                     # Video frame rate
)
```

### Custom Visualization

Use supervision's annotators directly for custom visualizations:

```python
import supervision as sv

# Initialize annotators with custom settings
box_annotator = sv.BoxAnnotator(
    thickness=4,
    color=sv.ColorPalette.DEFAULT
)

label_annotator = sv.LabelAnnotator(
    text_thickness=2,
    text_scale=0.8,
    text_padding=10
)

# Annotate frame
frame = box_annotator.annotate(scene=frame, detections=detections)
frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
```

## Example Output

The output video will show:
- Each detected object with a bounding box
- A unique tracker ID that persists across frames (e.g., `#1`, `#2`, `#3`)
- Class name and confidence score
- Instance segmentation masks (if using segmentation model)

Example label format: `#42 person 0.95`

## Performance Tips

1. **Skip Frames**: Use `--skip_frames` to process fewer frames for faster processing:
   ```bash
   uv run python inference_video_tracking.py --model model.onnx --video input.mp4 --skip_frames 1
   ```

2. **Adjust Confidence**: Lower threshold tracks more objects but may include false positives:
   ```bash
   uv run python inference_video_tracking.py --model model.onnx --video input.mp4 --threshold 0.3
   ```

3. **Limit Detections**: Reduce `max_number_boxes` for better performance:
   ```bash
   uv run python inference_video_tracking.py --model model.onnx --video input.mp4 --max_number_boxes 50
   ```

4. **GPU Acceleration**: Ensure CUDA is available and don't use `--cpu` flag

## Troubleshooting

### Import Errors

If you see "Import 'supervision' could not be resolved":
```bash
uv sync
```

### Tracking Quality Issues

- **Lost tracks**: Increase `lost_track_buffer` parameter
- **ID switches**: Increase `minimum_matching_threshold` parameter
- **Slow tracking**: Decrease `max_number_boxes` or increase `skip_frames`

### GPU Issues

If GPU is not detected, make sure your `pyproject.toml` has `onnxruntime-gpu` (not `onnxruntime`):
```bash
uv sync
```

## API Reference

### RFDETR_ONNX Methods

#### `to_supervision_detections(scores, labels, boxes, masks=None)`
Converts model outputs to supervision Detections format.

**Args:**
- `scores`: Confidence scores array
- `labels`: Class labels array
- `boxes`: Bounding boxes in xyxy format
- `masks`: Optional segmentation masks

**Returns:** `supervision.Detections` object

#### `draw_tracked_detections(image, detections)`
Draws tracked detections with IDs on image.

**Args:**
- `image`: PIL Image object
- `detections`: supervision.Detections with tracker_id

**Returns:** PIL Image with annotations

## Resources

- [Supervision Documentation](https://supervision.roboflow.com)
- [Supervision GitHub](https://github.com/roboflow/supervision)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [RF-DETR Repository](https://github.com/roboflow/rf-detr)

## License

This implementation is provided under the MIT License. See the LICENSE file for details.

