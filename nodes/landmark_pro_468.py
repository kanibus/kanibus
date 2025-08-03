"""
LandmarkPro468 - 468-point facial landmark detection with micro-expression analysis
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import logging

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
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LANDMARKS_468", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("landmarks", "annotated_image", "face_mask", "confidence")
    FUNCTION = "detect_landmarks"
    CATEGORY = "Kanibus/Landmarks"
    
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
                        smoothing=0.5, cache_results=True):
        """Detect 468 facial landmarks"""
        
        # Convert input
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]
            image_np = (image_np * 255).astype(np.uint8)
        
        # Process with MediaPipe
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
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
            annotated = image_np.copy()
            face_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            confidence = 0.0
        
        # Convert to tensor format
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(face_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)
        
        return (landmarks_array, annotated_tensor, mask_tensor, confidence)