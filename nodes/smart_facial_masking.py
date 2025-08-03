"""
Smart Facial Masking - AI masking with semantic segmentation and part exclusion
"""

import torch
import numpy as np
import cv2
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

class SmartFacialMasking:
    """
    Intelligent facial masking with semantic segmentation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_mode": (["full_face", "eyes_only", "mouth_only", "custom"], {"default": "full_face"}),
                "feather_amount": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "face_landmarks": ("LANDMARKS_468",),
                "exclude_eyes": ("BOOLEAN", {"default": False}),
                "exclude_mouth": ("BOOLEAN", {"default": False}),
                "exclude_eyebrows": ("BOOLEAN", {"default": False}),
                "dilation": ("INT", {"default": 2, "min": 0, "max": 10}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("face_mask", "masked_image", "exclusion_mask", "coverage_ratio")
    FUNCTION = "create_mask"
    CATEGORY = "Kanibus/Masking"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def create_mask(self, image, mask_mode="full_face", feather_amount=5.0,
                   face_landmarks=None, exclude_eyes=False, exclude_mouth=False,
                   exclude_eyebrows=False, dilation=2, cache_results=True):
        """Create intelligent facial mask"""
        
        # Convert input
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]
            image_np = (image_np * 255).astype(np.uint8)
        
        h, w = image_np.shape[:2]
        
        # Create base mask (placeholder - would use actual face detection)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple elliptical face mask for demonstration
        center_x, center_y = w // 2, h // 2
        axes = (w // 4, h // 3)
        cv2.ellipse(face_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        
        # Create exclusion mask
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        if exclude_eyes:
            # Exclude eye regions
            eye_size = min(w, h) // 10
            cv2.circle(exclusion_mask, (center_x - w//6, center_y - h//8), eye_size, 255, -1)
            cv2.circle(exclusion_mask, (center_x + w//6, center_y - h//8), eye_size, 255, -1)
        
        if exclude_mouth:
            # Exclude mouth region
            mouth_w, mouth_h = w//8, h//12
            cv2.ellipse(exclusion_mask, (center_x, center_y + h//6), (mouth_w, mouth_h), 0, 0, 360, 255, -1)
        
        # Apply exclusions
        face_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclusion_mask))
        
        # Apply dilation
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
            face_mask = cv2.dilate(face_mask, kernel, iterations=1)
        
        # Apply feathering (Gaussian blur)\n        if feather_amount > 0:\n            face_mask = cv2.GaussianBlur(face_mask, (int(feather_amount*2)+1, int(feather_amount*2)+1), feather_amount/3)\n        \n        # Create masked image\n        mask_3d = np.stack([face_mask] * 3, axis=2) / 255.0\n        masked_image = image_np * mask_3d\n        \n        # Calculate coverage ratio\n        coverage_ratio = np.sum(face_mask > 128) / (h * w)\n        \n        # Convert to tensors\n        face_mask_tensor = torch.from_numpy(face_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)\n        masked_image_tensor = torch.from_numpy(masked_image.astype(np.float32) / 255.0).unsqueeze(0)\n        exclusion_mask_tensor = torch.from_numpy(exclusion_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)\n        \n        return (face_mask_tensor, masked_image_tensor, exclusion_mask_tensor, coverage_ratio)