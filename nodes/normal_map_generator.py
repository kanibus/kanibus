"""
Normal Map Generator - Convert depth to surface normal maps
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
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "enable_t2i_adapter": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("normal_map", "normal_visualization", "detail_strength")
    FUNCTION = "generate_normal_map"
    CATEGORY = "Kanibus"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def generate_normal_map(self, depth_map, strength=1.0, blur_radius=1, invert_y=False, 
                           wan_version="auto", enable_t2i_adapter=True, cache_results=True):
        """Generate normal map from depth with WAN/T2I-Adapter compatibility"""
        
        try:
            # Auto-detect WAN version if needed
            if wan_version == "auto":
                wan_version = self._detect_wan_version(depth_map)
            
            # Adjust strength for WAN compatibility
            strength = self._adjust_strength_for_wan(strength, wan_version)
        
            # Convert input with enhanced error handling
            if isinstance(depth_map, torch.Tensor):
                depth_np = depth_map.cpu().numpy()
                if depth_np.ndim == 4:
                    depth_np = depth_np[0]
                if depth_np.shape[-1] > 1:
                    depth_np = depth_np[:, :, 0]  # Use first channel
            else:
                raise ValueError("Input depth_map must be a torch.Tensor")
        
            # Apply WAN-specific blur settings
            if blur_radius > 0:
                # Adjust blur for WAN compatibility
                if wan_version == "wan_2.1":
                    # Reduce blur for 480p efficiency
                    effective_blur = max(1, blur_radius - 1)
                elif wan_version == "wan_2.2":
                    # Enhanced blur for 720p quality
                    effective_blur = blur_radius + 1
                else:
                    effective_blur = blur_radius
                
                depth_np = cv2.GaussianBlur(depth_np, (effective_blur*2+1, effective_blur*2+1), 0)
        
            # Calculate gradients with WAN optimization
            if wan_version == "wan_2.1":
                # Use smaller kernel for 480p efficiency
                ksize = 3
            elif wan_version == "wan_2.2":
                # Use larger kernel for 720p quality
                ksize = 5
            else:
                ksize = 3
            
            grad_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=ksize) * strength
            grad_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=ksize) * strength
        
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
            
            # Add T2I-Adapter compatibility metadata
            if enable_t2i_adapter:
                # Add adapter compatibility information to tensor metadata
                normal_map_meta = {
                    "t2i_adapter_compatible": True,
                    "wan_version": wan_version,
                    "detail_strength": float(detail_strength)
                }
        
        # Convert to tensors
        normal_tensor = torch.from_numpy(normal_map.astype(np.float32)).unsqueeze(0)
        vis_tensor = torch.from_numpy(normal_vis.astype(np.float32)).unsqueeze(0)
        
            return (normal_tensor, vis_tensor, detail_strength)
            
        except Exception as e:
            self.logger.error(f"Error in normal map generation: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            if isinstance(depth_map, torch.Tensor) and len(depth_map.shape) >= 2:
                h, w = depth_map.shape[-2:]
            empty_normal = torch.full((1, h, w, 3), 0.5, dtype=torch.float32)  # Neutral normal map
            empty_vis = torch.full((1, h, w, 3), 0.5, dtype=torch.float32)
            return (empty_normal, empty_vis, 0.0)
    
    def _detect_wan_version(self, depth_map):
        """Auto-detect WAN version based on input characteristics"""
        if isinstance(depth_map, torch.Tensor):
            h, w = depth_map.shape[-2:]
            if h <= 480 or w <= 480:
                return "wan_2.1"
            else:
                return "wan_2.2"
        return "wan_2.2"
    
    def _adjust_strength_for_wan(self, strength, wan_version):
        """Adjust strength for WAN compatibility"""
        if wan_version == "wan_2.1":
            return strength * 0.9  # Slightly reduce strength for 480p
        elif wan_version == "wan_2.2":
            return strength * 1.1  # Enhance strength for 720p
        return strength
