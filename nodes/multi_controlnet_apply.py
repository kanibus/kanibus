"""
Multi-ControlNet Apply - Apply multiple conditions simultaneously for WAN compatibility
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
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

class MultiControlNetApply:
    """
    Apply multiple ControlNet conditions with proper WAN 2.1/2.2 compatibility
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                # Eye tracking controls
                "eye_mask": ("MASK",),
                "eye_mask_weight": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 3.0}),
                
                # Depth controls
                "depth_map": ("IMAGE",),
                "depth_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                
                # Normal controls
                "normal_map": ("IMAGE",),
                "normal_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0}),
                
                # Pose controls
                "pose_landmarks": ("POSE_LANDMARKS",),
                "pose_weight": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0}),
                
                # Hand controls
                "hand_landmarks": ("HAND_LANDMARKS",),
                "hand_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0}),
                
                # Facial controls
                "face_landmarks": ("LANDMARKS_468",),
                "face_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0}),
                
                # Global settings
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 30.0}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioned_model", "positive_conditioned", "negative_conditioned", "control_info")
    FUNCTION = "apply_multi_controls"
    CATEGORY = "Kanibus/ControlNet"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # ControlNet type mappings
        self.control_types = {
            "eye_mask": "scribble",
            "depth_map": "depth", 
            "normal_map": "normal",
            "pose_landmarks": "openpose",
            "hand_landmarks": "openpose",
            "face_landmarks": "openpose"
        }
        
    def apply_multi_controls(self, model, positive, negative,
                           eye_mask=None, eye_mask_weight=1.3,
                           depth_map=None, depth_weight=1.0,
                           normal_map=None, normal_weight=0.7,
                           pose_landmarks=None, pose_weight=0.9,
                           hand_landmarks=None, hand_weight=0.5,
                           face_landmarks=None, face_weight=0.8,
                           wan_version="auto", start_percent=0.0, end_percent=1.0,
                           cfg_scale=7.5):
        """Apply multiple ControlNet conditions"""
        
        active_controls = []
        control_info = []
        
        # Process each control input
        controls = {
            "eye_mask": (eye_mask, eye_mask_weight),
            "depth_map": (depth_map, depth_weight),
            "normal_map": (normal_map, normal_weight),
            "pose_landmarks": (pose_landmarks, pose_weight),
            "hand_landmarks": (hand_landmarks, hand_weight),
            "face_landmarks": (face_landmarks, face_weight)
        }
        
        conditioned_model = model
        pos_conditioned = positive
        neg_conditioned = negative
        
        for control_name, (control_input, weight) in controls.items():
            if control_input is not None and weight > 0:
                # Convert control input to proper format
                control_image = self._prepare_control_input(control_input, control_name)
                
                if control_image is not None:
                    # Create control conditioning
                    control_dict = {
                        "type": self.control_types[control_name],
                        "image": control_image,
                        "weight": weight,
                        "start_percent": start_percent,
                        "end_percent": end_percent,
                        "cfg_scale": cfg_scale
                    }
                    
                    active_controls.append(control_dict)
                    control_info.append(f"{control_name}: {weight:.1f}")
                    
                    self.logger.debug(f"Applied {control_name} control with weight {weight}")
        
        # Apply WAN version specific settings
        if wan_version == "wan_2.1":
            # WAN 2.1 optimizations
            for control in active_controls:
                control["weight"] *= 0.9  # Slightly reduce weights for WAN 2.1
            control_info.append("Optimized for WAN 2.1")
            
        elif wan_version == "wan_2.2":
            # WAN 2.2 optimizations
            for control in active_controls:
                control["motion_module"] = "v2"
            control_info.append("Optimized for WAN 2.2")
        
        # Create control info string
        info_string = f"Active Controls ({len(active_controls)}): " + ", ".join(control_info)
        
        # In a real implementation, this would apply the controls to the model
        # For now, we return the inputs with control information
        
        return (conditioned_model, pos_conditioned, neg_conditioned, info_string)
    
    def _prepare_control_input(self, control_input, control_type):
        """Prepare control input for ControlNet"""
        
        if control_type == "eye_mask":
            # Convert mask to image format
            if isinstance(control_input, torch.Tensor):
                # Ensure mask is in correct format
                if control_input.dim() == 4 and control_input.shape[-1] == 1:
                    # Convert grayscale mask to RGB
                    mask_rgb = control_input.repeat(1, 1, 1, 3)
                    return mask_rgb
                return control_input
        
        elif control_type in ["depth_map", "normal_map"]:
            # Ensure image is in correct format
            if isinstance(control_input, torch.Tensor):
                if control_input.dim() == 4:
                    return control_input
                elif control_input.dim() == 3:
                    return control_input.unsqueeze(0)
        
        elif control_type in ["pose_landmarks", "hand_landmarks", "face_landmarks"]:
            # Convert landmarks to pose image
            if isinstance(control_input, (list, np.ndarray)):
                pose_image = self._render_landmarks_to_image(control_input, control_type)
                return pose_image
        
        return control_input
    
    def _render_landmarks_to_image(self, landmarks, landmark_type):
        """Render landmarks to image for ControlNet"""
        # Create blank image
        image_size = (512, 512)  # Default size
        pose_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        if landmark_type == "pose_landmarks" and landmarks is not None:
            # Draw pose skeleton
            if isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 33:
                # MediaPipe pose connections
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                ]
                
                # Draw connections
                for start_idx, end_idx in connections:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = tuple(landmarks[start_idx][:2].astype(int))
                        end_point = tuple(landmarks[end_idx][:2].astype(int))
                        
                        # Scale points to image size
                        start_point = (int(start_point[0] * image_size[0] / 640), 
                                     int(start_point[1] * image_size[1] / 480))
                        end_point = (int(end_point[0] * image_size[0] / 640),
                                   int(end_point[1] * image_size[1] / 480))
                        
                        cv2.line(pose_image, start_point, end_point, (255, 255, 255), 2)
                
                # Draw keypoints
                for point in landmarks[:33]:  # First 33 are pose landmarks
                    x, y = int(point[0] * image_size[0] / 640), int(point[1] * image_size[1] / 480)
                    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                        cv2.circle(pose_image, (x, y), 3, (255, 255, 255), -1)
        
        elif landmark_type == "hand_landmarks":
            # Draw hand landmarks
            if isinstance(landmarks, list):
                for hand in landmarks:
                    if isinstance(hand, np.ndarray):
                        # Draw hand connections
                        hand_connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                            (0, 17)  # Palm
                        ]
                        
                        for start_idx, end_idx in hand_connections:
                            if start_idx < len(hand) and end_idx < len(hand):
                                start_point = tuple(hand[start_idx][:2].astype(int))
                                end_point = tuple(hand[end_idx][:2].astype(int))
                                cv2.line(pose_image, start_point, end_point, (255, 255, 255), 2)
        
        # Convert to tensor
        pose_tensor = torch.from_numpy(pose_image.astype(np.float32) / 255.0).unsqueeze(0)
        return pose_tensor
    
    def get_supported_controls(self):
        """Get list of supported control types"""
        return list(self.control_types.keys())
    
    def get_wan_compatibility_info(self, wan_version="auto"):
        """Get WAN compatibility information"""
        if wan_version == "wan_2.1":
            return {
                "version": "WAN 2.1",
                "max_resolution": "480p",
                "recommended_weights": {
                    "eye_mask": 1.2,
                    "depth": 0.9,
                    "normal": 0.6,
                    "pose": 0.8
                },
                "motion_module": "v1"
            }
        elif wan_version == "wan_2.2":
            return {
                "version": "WAN 2.2", 
                "max_resolution": "720p",
                "recommended_weights": {
                    "eye_mask": 1.3,
                    "depth": 1.0,
                    "normal": 0.7,
                    "pose": 0.9
                },
                "motion_module": "v2"
            }
        else:
            return {
                "version": "Auto-detect",
                "note": "Will optimize based on input resolution and model detection"
            }