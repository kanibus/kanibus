"""
Normal Map Generator - Convert depth to surface normal maps
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

class NormalMapGenerator:
    """
    Generate surface normal maps from depth maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
            },
            "optional": {
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 10}),
                "invert_y": ("BOOLEAN", {"default": False}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("normal_map", "normal_visualization", "detail_strength")
    FUNCTION = "generate_normal_map"
    CATEGORY = "Kanibus/Depth"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def generate_normal_map(self, depth_map, strength=1.0, blur_radius=1, invert_y=False, cache_results=True):
        """Generate normal map from depth"""
        
        # Convert input
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.cpu().numpy()
            if depth_np.ndim == 4:
                depth_np = depth_np[0]
            if depth_np.shape[-1] > 1:
                depth_np = depth_np[:, :, 0]  # Use first channel
        
        # Apply blur if requested
        if blur_radius > 0:
            depth_np = cv2.GaussianBlur(depth_np, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # Calculate gradients
        grad_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=3) * strength
        grad_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=3) * strength
        
        if invert_y:
            grad_y = -grad_y
        
        # Create normal vectors
        normal = np.zeros((depth_np.shape[0], depth_np.shape[1], 3))
        normal[:, :, 0] = -grad_x  # X component (red)
        normal[:, :, 1] = -grad_y  # Y component (green)
        normal[:, :, 2] = 1.0      # Z component (blue)
        
        # Normalize vectors
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (norm + 1e-8)
        
        # Convert to 0-1 range
        normal_map = (normal + 1.0) / 2.0
        
        # Create visualization (more saturated)
        normal_vis = normal_map.copy()
        normal_vis = np.clip(normal_vis * 1.2, 0, 1)
        
        # Calculate detail strength
        detail_strength = np.std(grad_x) + np.std(grad_y)
        
        # Convert to tensors
        normal_tensor = torch.from_numpy(normal_map.astype(np.float32)).unsqueeze(0)
        vis_tensor = torch.from_numpy(normal_vis.astype(np.float32)).unsqueeze(0)
        
        return (normal_tensor, vis_tensor, detail_strength)
