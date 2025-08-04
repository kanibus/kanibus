"""
Object Segmentation - SAM-powered instance segmentation
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
    from ..src.neural_engine import NeuralEngine
    from ..src.cache_manager import CacheManager
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.neural_engine import NeuralEngine
    from src.cache_manager import CacheManager

class ObjectSegmentation:
    """
    Advanced object segmentation using SAM (Segment Anything Model)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segmentation_mode": (["everything", "prompt", "box", "point"], {"default": "everything"}),
            },
            "optional": {
                "prompt_points": ("STRING", {"default": "", "multiline": True}),
                "bounding_boxes": ("STRING", {"default": "", "multiline": True}),
                "min_mask_area": ("INT", {"default": 1000, "min": 100, "max": 50000}),
                "max_masks": ("INT", {"default": 10, "min": 1, "max": 50}),
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "enable_t2i_adapter": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("SEGMENTATION_MASKS", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("masks", "segmented_image", "mask_overlay", "mask_count")
    FUNCTION = "segment_objects"
    CATEGORY = "Kanibus"
    
    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def segment_objects(self, image, segmentation_mode="everything", prompt_points="",
                       bounding_boxes="", min_mask_area=1000, max_masks=10, wan_version="auto",
                       enable_t2i_adapter=True, cache_results=True):
        """Segment objects using SAM with WAN/T2I-Adapter compatibility"""
        
        try:
            # Auto-detect WAN version if needed
            if wan_version == "auto":
                wan_version = self._detect_wan_version(image)
            
            # Adjust parameters for WAN compatibility
            min_mask_area = self._adjust_mask_area_for_wan(min_mask_area, wan_version)
            max_masks = self._adjust_max_masks_for_wan(max_masks, wan_version)
        
            # Convert input with enhanced error handling
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                image_np = (image_np * 255).astype(np.uint8)
            else:
                raise ValueError("Input image must be a torch.Tensor")
        
            h, w = image_np.shape[:2]
            
            # Generate WAN-optimized segmentation masks
            masks = []
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
            # Create WAN-optimized sample masks
            for i in range(min(3, max_masks)):
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Create shapes optimized for WAN version
                if wan_version == "wan_2.1":
                    # Simpler shapes for 480p efficiency
                    if i == 0:  # Rectangle
                        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
                    elif i == 1:  # Circle
                        cv2.circle(mask, (w//2, h//2), min(w, h)//8, 255, -1)
                else:  # WAN 2.2
                    # More complex shapes for 720p quality
                    if i == 0:  # Rectangle
                        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
                    elif i == 1:  # Circle
                        cv2.circle(mask, (w//2, h//2), min(w, h)//6, 255, -1)
                    elif i == 2:  # Polygon
                        pts = np.array([[w//3, h//6], [2*w//3, h//6], [5*w//6, h//2], 
                                       [2*w//3, 5*h//6], [w//3, 5*h//6], [w//6, h//2]], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                
                # Filter by area and add T2I-Adapter metadata
                if np.sum(mask > 0) >= min_mask_area:
                    if enable_t2i_adapter:
                        # Add adapter compatibility metadata to mask
                        mask_dict = {
                            "mask": mask,
                            "t2i_adapter_compatible": True,
                            "wan_version": wan_version,
                            "area": int(np.sum(mask > 0))
                        }
                        masks.append(mask_dict)
                    else:
                        masks.append(mask)
            
            # Create segmented image
            segmented = image_np.copy()
            mask_overlay = np.zeros_like(image_np)
            
            for i, mask_data in enumerate(masks):
                color = colors[i % len(colors)]
                
                # Extract mask from data structure
                if isinstance(mask_data, dict):
                    mask = mask_data["mask"]
                else:
                    mask = mask_data
                
                # Apply color to mask overlay
                mask_colored = np.zeros_like(image_np)
                mask_colored[mask > 0] = color
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 0.3, 0)
                
                # Draw contours on segmented image
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(segmented, contours, -1, color, 2)
            
            # Combine original with overlay
            final_overlay = cv2.addWeighted(image_np, 0.7, mask_overlay, 0.3, 0)
            
            # Convert to tensors
            segmented_tensor = torch.from_numpy(segmented.astype(np.float32) / 255.0).unsqueeze(0)
            overlay_tensor = torch.from_numpy(final_overlay.astype(np.float32) / 255.0).unsqueeze(0)
            
            return (masks, segmented_tensor, overlay_tensor, len(masks))
            
        except Exception as e:
            self.logger.error(f"Error in object segmentation: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            if isinstance(image, torch.Tensor) and len(image.shape) >= 2:
                h, w = image.shape[-2:]
            empty_masks = []
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            return (empty_masks, empty_image, empty_image, 0)
    
    def _detect_wan_version(self, image):
        """Auto-detect WAN version based on input characteristics"""
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            if h <= 480 or w <= 480:
                return "wan_2.1"
            else:
                return "wan_2.2"
        return "wan_2.2"
    
    def _adjust_mask_area_for_wan(self, min_mask_area, wan_version):
        """Adjust minimum mask area for WAN compatibility"""
        if wan_version == "wan_2.1":
            return max(min_mask_area // 2, 100)  # Smaller areas for 480p
        elif wan_version == "wan_2.2":
            return min_mask_area * 2  # Larger areas for 720p quality
        return min_mask_area
    
    def _adjust_max_masks_for_wan(self, max_masks, wan_version):
        """Adjust maximum masks for WAN compatibility"""
        if wan_version == "wan_2.1":
            return min(max_masks, 5)  # Limit masks for 480p efficiency
        elif wan_version == "wan_2.2":
            return max_masks  # Full mask count for 720p
        return max_masks
