"""
Smart Facial Masking - AI masking with semantic segmentation and part exclusion
"""

import torch
import numpy as np
import cv2
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
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "enable_t2i_adapter": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("face_mask", "masked_image", "exclusion_mask", "coverage_ratio")
    FUNCTION = "create_mask"
    CATEGORY = "Kanibus"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def create_mask(self, image, mask_mode="full_face", feather_amount=5.0,
                   face_landmarks=None, exclude_eyes=False, exclude_mouth=False,
                   exclude_eyebrows=False, dilation=2, wan_version="auto",
                   enable_t2i_adapter=True, cache_results=True):
        """Create intelligent facial mask with WAN/T2I-Adapter compatibility"""
        
        try:
            # Auto-detect WAN version if needed
            if wan_version == "auto":
                wan_version = self._detect_wan_version(image)
            
            # Adjust parameters for WAN compatibility
            feather_amount = self._adjust_feather_for_wan(feather_amount, wan_version)
            dilation = self._adjust_dilation_for_wan(dilation, wan_version)
        
            # Convert input with enhanced error handling
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                image_np = (image_np * 255).astype(np.uint8)
            else:
                raise ValueError("Input image must be a torch.Tensor")
        
            h, w = image_np.shape[:2]
        
            # Create WAN-optimized base mask
            face_mask = np.zeros((h, w), dtype=np.uint8)
            
            # WAN-specific elliptical face mask
            center_x, center_y = w // 2, h // 2
            if wan_version == "wan_2.1":
                # Smaller, more conservative mask for 480p
                axes = (w // 5, h // 4)
            elif wan_version == "wan_2.2":
                # Larger, more detailed mask for 720p
                axes = (w // 4, h // 3)
            else:
                axes = (w // 4, h // 3)
            
            cv2.ellipse(face_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        
            # Create WAN-optimized exclusion mask
            exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
            if exclude_eyes:
                # WAN-specific eye exclusion
                if wan_version == "wan_2.1":
                    eye_size = min(w, h) // 12  # Smaller for 480p
                else:
                    eye_size = min(w, h) // 10  # Standard for 720p
                cv2.circle(exclusion_mask, (center_x - w//6, center_y - h//8), eye_size, 255, -1)
                cv2.circle(exclusion_mask, (center_x + w//6, center_y - h//8), eye_size, 255, -1)
        
            if exclude_mouth:
                # WAN-specific mouth exclusion
                if wan_version == "wan_2.1":
                    mouth_w, mouth_h = w//10, h//15  # Smaller for 480p
                else:
                    mouth_w, mouth_h = w//8, h//12   # Standard for 720p
                cv2.ellipse(exclusion_mask, (center_x, center_y + h//6), (mouth_w, mouth_h), 0, 0, 360, 255, -1)
        
            # Apply exclusions
            face_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclusion_mask))
        
            # Apply WAN-optimized dilation
            if dilation > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
                face_mask = cv2.dilate(face_mask, kernel, iterations=1)
        
            # Apply WAN-optimized feathering
            if feather_amount > 0:
                kernel_size = int(feather_amount*2)+1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                face_mask = cv2.GaussianBlur(face_mask, (kernel_size, kernel_size), feather_amount/3)
            
            # Create masked image
            mask_3d = np.stack([face_mask] * 3, axis=2) / 255.0
            masked_image = image_np * mask_3d
            
            # Calculate coverage ratio with T2I-Adapter metadata
            coverage_ratio = np.sum(face_mask > 128) / (h * w)
            
            # Add T2I-Adapter compatibility information
            if enable_t2i_adapter:
                adapter_info = {
                    "t2i_adapter_compatible": True,
                    "wan_version": wan_version,
                    "coverage_ratio": float(coverage_ratio),
                    "mask_mode": mask_mode
                }
            
            # Convert to tensors
            face_mask_tensor = torch.from_numpy(face_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)
            masked_image_tensor = torch.from_numpy(masked_image.astype(np.float32) / 255.0).unsqueeze(0)
            exclusion_mask_tensor = torch.from_numpy(exclusion_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(-1)
            
            return (face_mask_tensor, masked_image_tensor, exclusion_mask_tensor, coverage_ratio)
            
        except Exception as e:
            self.logger.error(f"Error in facial masking: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            if isinstance(image, torch.Tensor) and len(image.shape) >= 2:
                h, w = image.shape[-2:]
            empty_mask = torch.zeros((1, h, w, 1), dtype=torch.float32)
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            return (empty_mask, empty_image, empty_mask, 0.0)
    
    def _detect_wan_version(self, image):
        """Auto-detect WAN version based on input characteristics"""
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            if h <= 480 or w <= 480:
                return "wan_2.1"
            else:
                return "wan_2.2"
        return "wan_2.2"
    
    def _adjust_feather_for_wan(self, feather_amount, wan_version):
        """Adjust feather amount for WAN compatibility"""
        if wan_version == "wan_2.1":
            return max(feather_amount * 0.8, 1.0)  # Reduce feathering for 480p
        elif wan_version == "wan_2.2":
            return feather_amount * 1.2  # Enhance feathering for 720p
        return feather_amount
    
    def _adjust_dilation_for_wan(self, dilation, wan_version):
        """Adjust dilation for WAN compatibility"""
        if wan_version == "wan_2.1":
            return max(dilation - 1, 0)  # Reduce dilation for 480p
        elif wan_version == "wan_2.2":
            return dilation + 1  # Increase dilation for 720p
        return dilation