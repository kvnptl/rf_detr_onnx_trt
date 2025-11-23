# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
#
# Portions of this file are adapted from RF-DETR
# Copyright (c) Roboflow
# Licensed under the Apache License, Version 2.0
# See https://www.apache.org/licenses/LICENSE-2.0 for details.
# ------------------------------------------------------------------------

import io
import requests
import onnxruntime as ort
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import supervision as sv

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_NUMBER_BOXES = 300

def open_image(path):
    # Check if the path is a URL (starts with 'http://' or 'https://')
    if path.startswith('http://') or path.startswith('https://'):
        img = Image.open(io.BytesIO(requests.get(path).content))
    # If it's a local file path, open the image directly
    else:
        if os.path.exists(path):
            img = Image.open(path)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")
    return img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

def generate_color_from_class_id(class_id):
    """
    Generate a consistent color for each class ID using a deterministic method.
    Same class ID will always get the same color.
    """
    # Use class_id as seed for deterministic color generation
    np.random.seed(class_id * 123)  # Multiply by prime to spread values
    color_rgb = tuple(np.random.randint(50, 256, size=3).tolist())  # Avoid very dark colors
    color_rgba = color_rgb + (100,)  # Add alpha for masks
    np.random.seed(None)  # Reset seed to avoid affecting other random operations
    return color_rgb, color_rgba

class RFDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    
    # COCO class names (using actual COCO category IDs with gaps, same as RF-DETR)
    COCO_CLASSES = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }

    def __init__(self, onnx_model_path, use_gpu=True):
        try:
            # Set up execution providers (GPU first, then CPU fallback)
            providers = []
            if use_gpu:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # Load the ONNX model and initialize the ONNX Runtime session
            self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            # Print which provider is being used
            print(f"Using execution provider: {self.ort_session.get_providers()[0]}")

            # Get input shape
            input_info = self.ort_session.get_inputs()[0]
            self.input_height, self.input_width = input_info.shape[2:]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{onnx_model_path}'. "
                f"Ensure the path is correct and the model is a valid ONNX file."
            ) from e

    def _preprocess(self, image):
        """Preprocess the input image for inference."""
        
        # Resize the image to the model's input size
        image = image.resize((self.input_width, self.input_height))

        # Convert image to numpy array and normalize pixel values
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)

        # Change dimensions from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _post_process(self, outputs, origin_height, origin_width, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """
        Post-process the model's output to extract bounding boxes and class information.
        Inspired by the PostProcess class in rfdetr/lwdetr.py: https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/models/lwdetr.py#L701
        """
        # Get masks if instance segmentation
        if len(outputs) == 3:  
            masks = outputs[2]
        else:
            masks = None
        
        # Apply sigmoid activation
        prob = sigmoid(outputs[1]) 
        
        # Get detections with highest confidence and limit to max_number_boxes
        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]
        
        # Convert boxes from cxcywh to xyxyn format and scale to image size (i.e xyxyn -> xyxy)
        boxes = box_cxcywh_to_xyxyn(boxes)
        boxes[..., [0, 2]] *= origin_width
        boxes[..., [1, 3]] *= origin_height
        
        # Resize the masks to the original image size if available
        if masks is not None:
            new_w, new_h = origin_width, origin_height
            masks = np.stack([
                np.array(Image.fromarray(img).resize((new_w, new_h)))
                for img in masks
            ], axis=0)
            masks = (masks > 0).astype(np.uint8) * 255 
        
        # Filter detections based on the confidence threshold
        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]
        
        return scores, labels, boxes, masks

    def predict(self, image_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """Run the model inference and return the raw outputs."""
        
        # Load the image
        image = open_image(image_path).convert('RGB')
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_image = self._preprocess(image)

        # Get input name from the model
        input_name = self.ort_session.get_inputs()[0].name

        # Run the model
        outputs = self.ort_session.run(None, {input_name: input_image})
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def predict_from_image(self, image, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """
        Run the model inference on a PIL Image object and return the raw outputs.
        
        Args:
            image: PIL Image object in RGB format
            confidence_threshold: Confidence threshold for filtering detections
            max_number_boxes: Maximum number of boxes to return
            
        Returns:
            Tuple of (scores, labels, boxes, masks)
        """
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_image = self._preprocess(image)

        # Get input name from the model
        input_name = self.ort_session.get_inputs()[0].name

        # Run the model
        outputs = self.ort_session.run(None, {input_name: input_image})
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def to_supervision_detections(self, scores, labels, boxes, masks=None):
        """
        Convert model outputs to supervision Detections format for tracking.
        
        Args:
            scores: Numpy array of confidence scores
            labels: Numpy array of class labels
            boxes: Numpy array of bounding boxes in xyxy format
            masks: Optional numpy array of segmentation masks
            
        Returns:
            supervision.Detections object
        """
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=labels.astype(int),
            mask=masks if masks is not None else None
        )
        return detections

    def draw_detections(self, image, boxes, labels, masks, scores):
        """
        Draw bounding boxes, masks and class labels on a PIL Image object.
        
        Args:
            image: PIL Image object
            boxes: Numpy array of bounding boxes
            labels: Numpy array of class labels
            masks: Numpy array of segmentation masks (or None)
            scores: Numpy array of confidence scores
            
        Returns:
            PIL Image with detections drawn
        """
        # Convert to RGBA for compositing
        base = image.convert("RGBA")
        result = base.copy()

        # Generate consistent colors for each unique label based on class ID
        label_colors = {}
        for label in np.unique(labels):
            color_rgb, color_rgba = generate_color_from_class_id(int(label))
            label_colors[label] = {'rgb': color_rgb, 'rgba': color_rgba}

        # Loop over all masks
        if masks is not None:
            for i in range(masks.shape[0]):
                label = labels[i]
                color = label_colors[label]['rgba']

                # --- Draw mask ---
                mask_overlay = Image.fromarray(masks[i]).convert("L")
                mask_overlay = ImageOps.autocontrast(mask_overlay)
                overlay_color = Image.new("RGBA", base.size, color)
                overlay_masked = Image.new("RGBA", base.size)
                overlay_masked.paste(overlay_color, (0, 0), mask_overlay)
                result = Image.alpha_composite(result, overlay_masked)

        # Convert to RGB for drawing boxes and text
        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        
        # Try to load a better font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Loop over boxes and draw
        for i, box in enumerate(boxes.astype(int)):
            label = labels[i]
            score = scores[i]
            
            # Get class name from COCO dictionary
            class_name = self.COCO_CLASSES.get(label, f"class_{label}")
            
            # Use consistent color for this class
            box_color = label_colors[label]['rgb']
            draw.rectangle(box.tolist(), outline=box_color, width=4)

            # Create label text with class name and confidence
            label_text = f"{class_name}: {score:.2f}"
            
            # Get text bounding box for background
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background rectangle for text
            background_box = [box[0], box[1] - text_height - 8, box[0] + text_width + 10, box[1]]
            draw.rectangle(background_box, fill=box_color)
            
            # Draw label text on top of background
            draw.text((box[0] + 5, box[1] - text_height - 5), label_text, fill=(255, 255, 255), font=font)

        return result_rgb

    def draw_tracked_detections(self, image, detections):
        """
        Draw bounding boxes, masks, class labels, and tracker IDs on a PIL Image object.
        Uses supervision's native annotators for better tracking visualization.
        
        Args:
            image: PIL Image object
            detections: supervision.Detections object with tracker_id
            
        Returns:
            PIL Image with tracked detections drawn
        """
        # Convert PIL to numpy array (RGB)
        frame = np.array(image)
        
        # Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8, text_padding=10)
        
        # Draw masks if available
        if detections.mask is not None:
            mask_annotator = sv.MaskAnnotator()
            frame = mask_annotator.annotate(scene=frame, detections=detections)
        
        # Draw boxes
        frame = box_annotator.annotate(scene=frame, detections=detections)
        
        # Create labels with class name, confidence, and tracker ID
        labels = []
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            # Add tracker ID if available
            if detections.tracker_id is not None:
                tracker_id = detections.tracker_id[i]
                label = f"#{tracker_id} {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            labels.append(label)
        
        # Draw labels
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Convert back to PIL Image
        return Image.fromarray(frame)

    def save_detections(self, image_path, boxes, labels, masks, scores, save_image_path):
        """
        Draw bounding boxes, masks and class labels on the original image and save it.
        
        Args:
            image_path: Path to input image
            boxes: Numpy array of bounding boxes
            labels: Numpy array of class labels
            masks: Numpy array of segmentation masks (or None)
            scores: Numpy array of confidence scores
            save_image_path: Path to save output image
        """
        # Load base image
        image = open_image(image_path).convert("RGB")
        
        # Draw detections
        result = self.draw_detections(image, boxes, labels, masks, scores)
        
        # Save
        result.save(save_image_path)