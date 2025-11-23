# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
#
# TensorRT inference using PyTorch backend (inspired by DEIM approach)
# ------------------------------------------------------------------------

import io
import requests
import collections
from collections import OrderedDict
import tensorrt as trt
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Import shared utilities from rfdetr_onnx
from rfdetr_onnx import (
    sigmoid, box_cxcywh_to_xyxyn, generate_color_from_class_id,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES, open_image
)


class RFDETR_TRT:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    
    # COCO class names (using actual COCO category IDs with gaps, same as RF-DETR)
    COCO_CLASSES = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
        11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
        21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
        27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
        34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
        39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
        43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
        48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
        53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
        58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
        63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
        70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
        76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
        80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
        85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
        89: "hair drier", 90: "toothbrush",
    }

    def __init__(self, engine_path, device='cuda:0', max_batch_size=1, verbose=False):
        """
        Initialize TensorRT inference engine with PyTorch backend
        
        Args:
            engine_path: Path to TensorRT engine file (.trt)
            device: CUDA device to use (default: 'cuda:0')
            max_batch_size: Maximum batch size (default: 1)
            verbose: Enable verbose logging (default: False)
        """
        self.engine_path = engine_path
        self.device = device
        self.max_batch_size = max_batch_size
        
        # Initialize CUDA through PyTorch first to avoid TensorRT CUDA init issues
        if torch.cuda.is_available():
            torch.cuda.init()
            # Create a dummy tensor to ensure CUDA context is initialized
            _ = torch.zeros(1).to(device)
        else:
            raise RuntimeError("CUDA is not available. TensorRT requires CUDA.")
        
        # Setup TensorRT logger
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        
        try:
            # Load TensorRT engine
            print(f"Loading TensorRT engine from: {engine_path}")
            self.engine = self._load_engine(engine_path)
            self.context = self.engine.create_execution_context()
            
            # Setup bindings (input/output tensors)
            self.bindings = self._get_bindings(self.engine, self.context, self.max_batch_size, self.device)
            self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
            
            # Get input/output names
            self.input_names = self._get_input_names()
            self.output_names = self._get_output_names()
            
            # Store input shape info
            input_shape = self.bindings[self.input_names[0]].shape
            self.batch_size = input_shape[0]
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            
            print(f"TensorRT engine loaded successfully")
            print(f"  Input: {self.input_names[0]} {input_shape}")
            print(f"  Outputs: {[(n, self.bindings[n].shape) for n in self.output_names]}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorRT engine from '{engine_path}'. "
                f"Ensure the file exists and is a valid TensorRT engine."
            ) from e

    def _load_engine(self, path):
        """Load TensorRT engine from file"""
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_input_names(self):
        """Get input tensor names"""
        names = []
        for name in self.engine:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def _get_output_names(self):
        """Get output tensor names"""
        names = []
        for name in self.engine:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def _get_bindings(self, engine, context, max_batch_size=1, device='cuda:0'):
        """
        Create bindings for input/output tensors
        Uses PyTorch tensors for memory management (inspired by DEIM)
        """
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for name in engine:
            shape = list(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            # Handle dynamic batch size
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            # Allocate PyTorch tensor on GPU
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

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

        # Add batch dimension and convert to torch tensor
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        return image

    def _post_process(self, outputs, origin_height, origin_width, 
                     confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
                     max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """
        Post-process the model's output to extract bounding boxes and class information.
        """
        # Convert outputs to numpy for post-processing
        outputs_np = [out.cpu().numpy() for out in outputs]
        
        # Get masks if instance segmentation
        if len(outputs_np) == 3:  
            masks = outputs_np[2]
        else:
            masks = None
        
        # Apply sigmoid activation
        prob = sigmoid(outputs_np[1]) 
        
        # Get detections with highest confidence and limit to max_number_boxes
        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs_np[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]
        
        # Convert boxes from cxcywh to xyxyn format and scale to image size
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

    def _infer(self, input_tensor):
        """
        Run inference using TensorRT with PyTorch backend
        
        Args:
            input_tensor: PyTorch tensor on GPU
            
        Returns:
            List of output tensors
        """
        # Create input blob
        blob = {self.input_names[0]: input_tensor}
        
        # Update bindings if input shape changed
        for n in self.input_names:
            if list(self.bindings[n].shape) != list(blob[n].shape):
                self.context.set_input_shape(n, list(blob[n].shape))
                self.bindings[n] = self.bindings[n]._replace(shape=list(blob[n].shape))
            
            # Copy input data to binding
            self.bindings[n].data.copy_(blob[n])
        
        # Execute inference
        self.context.execute_v2(list(self.bindings_addr.values()))
        
        # Get outputs from bindings (they're updated in-place)
        outputs = [self.bindings[n].data for n in self.output_names]
        
        return outputs

    def predict(self, image_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
                max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
        """Run the model inference and return the raw outputs."""
        # Load the image
        image = open_image(image_path).convert('RGB')
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self._infer(input_tensor)
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, 
                                 confidence_threshold, max_number_boxes)

    def predict_from_image(self, image, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
                          max_number_boxes=DEFAULT_MAX_NUMBER_BOXES):
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
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self._infer(input_tensor)
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, 
                                 confidence_threshold, max_number_boxes)

    def synchronize(self):
        """Synchronize CUDA operations"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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

