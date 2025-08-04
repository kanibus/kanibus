"""
Temporal Smoother - Frame consistency optimization for video processing
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
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

class TemporalSmoother:
    """
    Temporal smoothing for video consistency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frame": ("IMAGE",),
                "smoothing_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "buffer_size": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
            "optional": {
                "previous_frames": ("IMAGE",),
                "frame_weights": (["linear", "exponential", "gaussian"], {"default": "exponential"}),
                "motion_compensation": ("BOOLEAN", {"default": True}),
                "adaptive_smoothing": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
                
                # WAN optimization
                "wan_version": (["auto", "wan_2.1", "wan_2.2"], {"default": "auto"}),
                "temporal_consistency_mode": (["standard", "enhanced", "ultra"], {"default": "enhanced"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("smoothed_frame", "motion_visualization", "motion_amount", "smoothing_applied")
    FUNCTION = "smooth_temporal"
    CATEGORY = "Kanibus/Processing"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=20)
        self.motion_buffer: Deque[np.ndarray] = deque(maxlen=10)
        self.logger = logging.getLogger(__name__)
        
    def smooth_temporal(self, current_frame, smoothing_strength=0.7, buffer_size=5,
                       previous_frames=None, frame_weights="exponential",
                       motion_compensation=True, adaptive_smoothing=True, cache_results=True):
        """Apply temporal smoothing to frame sequence"""
        
        # Convert input
        if isinstance(current_frame, torch.Tensor):
            current_np = current_frame.cpu().numpy()
            if current_np.ndim == 4:
                current_np = current_np[0]
            current_np = (current_np * 255).astype(np.uint8)
        
        # Add to buffer
        self.frame_buffer.append(current_np.copy())
        
        # If not enough frames for smoothing, return current frame
        if len(self.frame_buffer) < 2:
            smoothed = current_np.copy()
            motion_vis = np.zeros_like(current_np)
            motion_amount = 0.0
            smoothing_applied = 0.0
        else:
            # Calculate motion between frames
            prev_frame = self.frame_buffer[-2]
            motion_amount = self._calculate_motion(prev_frame, current_np)
            
            # Adaptive smoothing based on motion
            if adaptive_smoothing:
                # More smoothing for low motion, less for high motion
                adaptive_strength = smoothing_strength * (1.0 - min(motion_amount / 50.0, 1.0))
            else:
                adaptive_strength = smoothing_strength
            
            # Apply temporal smoothing
            smoothed = self._apply_smoothing(
                current_np, adaptive_strength, buffer_size, frame_weights, motion_compensation
            )
            
            # Create motion visualization
            motion_vis = self._create_motion_visualization(prev_frame, current_np)
            
            smoothing_applied = adaptive_strength
        
        # Convert back to tensors
        smoothed_tensor = torch.from_numpy(smoothed.astype(np.float32) / 255.0).unsqueeze(0)
        motion_vis_tensor = torch.from_numpy(motion_vis.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (smoothed_tensor, motion_vis_tensor, motion_amount, smoothing_applied)
    
    def _calculate_motion(self, prev_frame, current_frame):
        """Calculate motion between frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) if len(current_frame.shape) == 3 else current_frame
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Calculate motion amount
        motion_amount = np.mean(diff)
        
        return motion_amount
    
    def _apply_smoothing(self, current_frame, strength, buffer_size, weight_type, motion_comp):
        """Apply temporal smoothing"""
        if len(self.frame_buffer) < 2:
            return current_frame
        
        # Get frames to use for smoothing
        frames_to_use = list(self.frame_buffer)[-buffer_size:]
        
        if len(frames_to_use) < 2:
            return current_frame
        
        # Generate weights
        weights = self._generate_weights(len(frames_to_use), weight_type)
        
        # Apply weighted average
        smoothed = np.zeros_like(current_frame, dtype=np.float32)
        total_weight = 0.0
        
        for i, (frame, weight) in enumerate(zip(frames_to_use, weights)):
            if motion_comp and i < len(frames_to_use) - 1:
                # Apply simple motion compensation (shift detection)
                aligned_frame = self._align_frame(frame, current_frame)
            else:
                aligned_frame = frame
            
            smoothed += aligned_frame.astype(np.float32) * weight
            total_weight += weight
        
        smoothed = smoothed / total_weight
        
        # Blend with current frame based on strength
        final_smoothed = (
            current_frame.astype(np.float32) * (1.0 - strength) +
            smoothed * strength
        )
        
        return np.clip(final_smoothed, 0, 255).astype(np.uint8)
    
    def _generate_weights(self, num_frames, weight_type):
        """Generate weights for temporal averaging"""
        if weight_type == "linear":
            weights = np.linspace(0.1, 1.0, num_frames)
        elif weight_type == "exponential":
            weights = np.exp(np.linspace(-2, 0, num_frames))
        elif weight_type == "gaussian":
            x = np.linspace(-2, 2, num_frames)
            weights = np.exp(-(x ** 2) / 2)
        else:
            weights = np.ones(num_frames)
        
        return weights / np.sum(weights)
    
    def _align_frame(self, frame, reference):
        """Simple frame alignment using template matching"""
        # Convert to grayscale for alignment
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) if len(reference.shape) == 3 else reference
        
        # Simple phase correlation for shift detection
        try:
            shift = cv2.phaseCorrelate(frame_gray.astype(np.float32), ref_gray.astype(np.float32))[0]
            
            # Apply shift if reasonable
            if abs(shift[0]) < 10 and abs(shift[1]) < 10:
                M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                return aligned
        except:
            pass
        
        return frame  # Return original if alignment fails
    
    def _create_motion_visualization(self, prev_frame, current_frame):
        """Create visualization of motion between frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) if len(current_frame.shape) == 3 else current_frame
        
        # Calculate optical flow
        try:
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10),
                None
            )[0]
            
            # Create motion visualization
            motion_vis = np.zeros_like(current_frame)
            
            if flow is not None and len(flow) > 0:
                # Draw flow vectors
                for point in flow:
                    if point is not None:
                        x, y = point.ravel().astype(int)
                        if 0 <= x < motion_vis.shape[1] and 0 <= y < motion_vis.shape[0]:
                            cv2.circle(motion_vis, (x, y), 2, (0, 255, 0), -1)
        except:
            # Fallback to simple difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_vis = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        return motion_vis
    
    def clear_buffer(self):
        """Clear frame buffer"""
        self.frame_buffer.clear()
        self.motion_buffer.clear()
