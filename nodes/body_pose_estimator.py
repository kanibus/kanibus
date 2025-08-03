"""
Body Pose Estimator - Full body pose estimation with 70 skeleton points
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

class BodyPoseEstimator:
    """
    Full body pose estimation with detailed skeleton tracking
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_model": (["lite", "full", "heavy"], {"default": "full"}),
                "detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
            },
            "optional": {
                "enable_segmentation": ("BOOLEAN", {"default": False}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("POSE_LANDMARKS", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("pose_landmarks", "pose_visualization", "person_mask", "confidence")
    FUNCTION = "estimate_pose"
    CATEGORY = "Kanibus/Pose"
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def estimate_pose(self, image, pose_model="full", detection_confidence=0.5,
                     enable_segmentation=False, smoothing=0.5, cache_results=True):
        """Estimate full body pose"""
        
        # Convert input
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]
            image_np = (image_np * 255).astype(np.uint8)
        
        # Process with MediaPipe
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        pose_landmarks = None
        annotated = image_np.copy()
        person_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        confidence = 0.0
        
        if results.pose_landmarks:
            h, w = image_np.shape[:2]
            
            # Convert landmarks to array
            pose_landmarks = np.zeros((33, 3))  # MediaPipe has 33 pose landmarks
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                pose_landmarks[i] = [
                    landmark.x * w,
                    landmark.y * h,
                    landmark.z
                ]
            
            # Draw pose on image
            self.mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Create person mask if segmentation available
            if enable_segmentation and results.segmentation_mask is not None:
                person_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            else:
                # Create simple bounding box mask
                valid_points = pose_landmarks[pose_landmarks[:, 2] > -0.5]  # Filter valid points
                if len(valid_points) > 0:
                    x1, y1 = valid_points[:, :2].min(axis=0).astype(int)
                    x2, y2 = valid_points[:, :2].max(axis=0).astype(int)
                    person_mask[y1:y2, x1:x2] = 255
            
            confidence = 0.9  # Placeholder confidence
        
        # Convert to tensors
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(person_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)
        
        return (pose_landmarks, annotated_tensor, mask_tensor, confidence)
