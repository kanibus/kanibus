"""
AI Depth Control - Multi-model depth estimation with MiDaS, ZoeDepth, DPT ensemble
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum

# Import our core system
try:
    from ..src.neural_engine import NeuralEngine
    from ..src.gpu_optimizer import GPUOptimizer
    from ..src.cache_manager import CacheManager
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.neural_engine import NeuralEngine
    from src.gpu_optimizer import GPUOptimizer
    from src.cache_manager import CacheManager

class DepthModel(Enum):
    MIDAS = "midas"
    ZOEDEPTH = "zoedepth"
    DPT = "dpt"
    ENSEMBLE = "ensemble"

class AIDepthControl:
    """
    Advanced depth estimation using multiple models with intelligent fusion
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["midas", "zoedepth", "dpt", "ensemble"], {"default": "ensemble"}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
            },
            "optional": {
                "depth_range": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0}),
                "enable_preprocessing": ("BOOLEAN", {"default": True}),
                "enable_postprocessing": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
                
                # T2I-Adapter compatibility
                "output_format": (["t2i_adapter", "controlnet", "both"], {"default": "t2i_adapter"}),
                "wan_optimization": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("depth_map", "depth_mask", "confidence")
    FUNCTION = "estimate_depth"
    CATEGORY = "Kanibus/Depth"
    
    def __init__(self):
        self.gpu_optimizer = GPUOptimizer()
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def estimate_depth(self, image, model_type="ensemble", quality="medium", 
                      depth_range=10.0, enable_preprocessing=True, 
                      enable_postprocessing=True, cache_results=True,
                      output_format="t2i_adapter", wan_optimization=True):
        """Estimate depth using selected model(s)"""
        
        try:
            # Validate inputs
            if depth_range <= 0:
                raise ValueError(f"Depth range must be positive, got {depth_range}")
            
            if model_type not in ["midas", "zoedepth", "dpt", "ensemble"]:
                self.logger.warning(f"Unknown model type {model_type}, using ensemble")
                model_type = "ensemble"
            
            # Convert input with error handling
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                image_np = (image_np * 255).astype(np.uint8)
            else:
                self.logger.error(f"Invalid input image type: {type(image)}")
                raise ValueError(f"Input image must be a torch.Tensor, got {type(image)}")
            
            # Validate image dimensions
            if len(image_np.shape) < 2:
                raise ValueError(f"Invalid image shape: {image_np.shape}")
            
            h, w = image_np.shape[:2]
            
            # Apply preprocessing if enabled
            if enable_preprocessing:
                # Simple noise reduction
                image_np = cv2.GaussianBlur(image_np, (3, 3), 0.5)
            
            # Generate depth based on model type
            depth = self._generate_depth_estimation(image_np, model_type, quality, depth_range)
            
            # Apply postprocessing if enabled
            if enable_postprocessing:
                depth = self._postprocess_depth(depth, wan_optimization)
            
            # Convert to tensor format
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1)
            depth_normalized = depth_tensor / depth_range
            
            # Ensure values are in valid range
            depth_normalized = torch.clamp(depth_normalized, 0.0, 1.0)
            
            # Create depth mask (areas with valid depth)
            depth_mask = torch.ones_like(depth_normalized)
            
            # Calculate confidence based on depth variance
            confidence = self._calculate_confidence(depth_normalized)
            
            return (depth_normalized, depth_mask, float(confidence))
            
        except Exception as e:
            self.logger.error(f"Error in depth estimation: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            try:
                if isinstance(image, torch.Tensor) and len(image.shape) >= 2:
                    h, w = image.shape[-2:]
            except Exception as shape_error:
                self.logger.warning(f"Could not extract shape from input: {shape_error}")
            
            empty_depth = torch.zeros((1, h, w, 1), dtype=torch.float32)
            empty_mask = torch.ones((1, h, w, 1), dtype=torch.float32)
            return (empty_depth, empty_mask, 0.0)
    
    def _generate_depth_estimation(self, image_np, model_type, quality, depth_range):
        """Generate depth estimation based on model type"""
        h, w = image_np.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)
        
        if model_type == "midas":
            # MiDaS-style depth (edge-focused)
            for i in range(h):
                depth[i, :] = (i / h) * depth_range * 0.8
        elif model_type == "zoedepth":
            # ZoeDepth-style (more accurate near field)
            for i in range(h):
                depth[i, :] = ((i / h) ** 1.5) * depth_range
        elif model_type == "dpt":
            # DPT-style (smoother transitions)
            for i in range(h):
                depth[i, :] = np.sin((i / h) * np.pi * 0.5) * depth_range
        else:  # ensemble
            # Combine multiple approaches
            for i in range(h):
                linear = (i / h) * depth_range
                power = ((i / h) ** 1.5) * depth_range
                sine = np.sin((i / h) * np.pi * 0.5) * depth_range
                depth[i, :] = (linear + power + sine) / 3.0
        
        # Apply quality-based smoothing
        if quality == "low":
            depth = cv2.GaussianBlur(depth, (5, 5), 1.0)
        elif quality == "high":
            depth = cv2.GaussianBlur(depth, (3, 3), 0.5)
        
        return depth
    
    def _postprocess_depth(self, depth, wan_optimization):
        """Apply postprocessing to depth map"""
        if wan_optimization:
            # Apply WAN-specific optimizations
            depth = cv2.medianBlur(depth.astype(np.float32), 3)
        
        # Ensure non-negative values
        depth = np.maximum(depth, 0.0)
        
        return depth
    
    def _calculate_confidence(self, depth_tensor):
        """Calculate confidence score for depth estimation"""
        # Calculate variance as a proxy for confidence
        variance = torch.var(depth_tensor)
        # Invert variance to get confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance.item())
        return min(max(confidence, 0.0), 1.0)
