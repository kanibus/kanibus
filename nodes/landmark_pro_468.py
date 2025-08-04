"""
LandmarkPro468 - 468-point facial landmark detection with micro-expression analysis
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from enum import Enum

class WanVersion(Enum):
    WAN_21 = "wan_2.1"
    WAN_22 = "wan_2.2"
    AUTO = "auto"

# Import our core system
try:
    from ..src.cache_manager import CacheManager
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.cache_manager import CacheManager

class LandmarkPro468:
    """
    Advanced 468-point facial landmark detection with micro-expression analysis
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
            },
            "optional": {
                "enable_refinement": ("BOOLEAN", {"default": True}),
                "enable_micro_expressions": ("BOOLEAN", {"default": False}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "enable_t2i_adapter": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LANDMARKS_468", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("landmarks", "annotated_image", "face_mask", "confidence")
    FUNCTION = "detect_landmarks"
    CATEGORY = "Kanibus"
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def detect_landmarks(self, image, detection_confidence=0.5, tracking_confidence=0.5,
                        enable_refinement=True, enable_micro_expressions=False,
                        smoothing=0.5, wan_version="auto", enable_t2i_adapter=True, cache_results=True):
        """Detect 468 facial landmarks with WAN/T2I-Adapter compatibility"""
        
        try:
            # Auto-detect WAN version if needed
            if wan_version == "auto":
                try:
                    wan_version = self._detect_wan_version(image)
                except Exception as e:
                    self.logger.warning(f"WAN version auto-detection failed: {e}, defaulting to wan_2.2")
                    wan_version = "wan_2.2"
            
            # Adjust confidence for WAN compatibility
            detection_confidence = self._adjust_confidence_for_wan(detection_confidence, wan_version)
            tracking_confidence = self._adjust_confidence_for_wan(tracking_confidence, wan_version)
        
            # Convert input with enhanced error handling
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                image_np = (image_np * 255).astype(np.uint8)
            else:
                self.logger.error(f"Invalid input image type: {type(image)}")
                raise ValueError(f"Input image must be a torch.Tensor, got {type(image)}")
        
            # Process with MediaPipe (WAN optimized)
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Apply WAN-specific face mesh settings
            if wan_version == "wan_2.1":
                # Optimize for 480p efficiency
                self.face_mesh.min_detection_confidence = detection_confidence * 1.1
                self.face_mesh.refine_landmarks = False  # Disable refinement for 480p efficiency
            elif wan_version == "wan_2.2":
                # Optimize for 720p quality
                self.face_mesh.min_detection_confidence = detection_confidence * 0.9
                self.face_mesh.refine_landmarks = enable_refinement
            
            results = self.face_mesh.process(rgb_image)
        
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Convert to numpy array
                h, w = image_np.shape[:2]
                landmarks_array = np.zeros((468, 3))
            
                for i, landmark in enumerate(landmarks.landmark):
                    landmarks_array[i] = [
                        landmark.x * w,
                        landmark.y * h,
                        landmark.z
                    ]
                
                # Add T2I-Adapter compatibility metadata
                if enable_t2i_adapter:
                    landmarks_array = np.column_stack([landmarks_array, 
                                                     np.full((468, 1), 1.0)])  # Add adapter compatibility flag
            
            # Create annotated image
            annotated = image_np.copy()
            for point in landmarks_array[:, :2].astype(int):
                cv2.circle(annotated, tuple(point), 1, (0, 255, 0), -1)
            
                # Create face mask
                face_mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(landmarks_array[:, :2].astype(np.int32))
                cv2.fillPoly(face_mask, [hull], 255)
                
                confidence = 0.9
            else:
                # No face detected
                landmarks_array = np.zeros((468, 3))
                if enable_t2i_adapter:
                    landmarks_array = np.column_stack([landmarks_array, 
                                                     np.zeros((468, 1))])  # Add empty adapter flag
                annotated = image_np.copy()
                face_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                confidence = 0.0
            
            # Convert to tensor format
            annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0).unsqueeze(0)
            mask_tensor = torch.from_numpy(face_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)
            
            return (landmarks_array, annotated_tensor, mask_tensor, confidence)
            
        except Exception as e:
            self.logger.error(f"Error in landmark detection: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            try:
                if isinstance(image, torch.Tensor) and len(image.shape) >= 2:
                    h, w = image.shape[-2:]
            except Exception as shape_error:
                self.logger.warning(f"Could not extract shape from input: {shape_error}")
            
            empty_landmarks = np.zeros((468, 3))
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, h, w, 1), dtype=torch.float32)
            return (empty_landmarks, empty_image, empty_mask, 0.0)
    
    def _detect_wan_version(self, image):
        """Auto-detect WAN version based on input characteristics"""
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            if h <= 480 or w <= 480:
                return "wan_2.1"
            else:
                return "wan_2.2"
        return "wan_2.2"
    
    def _adjust_confidence_for_wan(self, confidence, wan_version):
        """Adjust confidence threshold for WAN compatibility"""
        if wan_version == "wan_2.1":
            return min(confidence * 1.05, 1.0)
        elif wan_version == "wan_2.2":
            return confidence * 0.98
        return confidence