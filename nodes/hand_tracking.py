"""
Hand Tracking - 21-point hand tracking with gesture recognition
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
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

class HandTracking:
    """
    Advanced hand tracking with gesture recognition
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_hands": ("INT", {"default": 2, "min": 1, "max": 4}),
                "detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
            },
            "optional": {
                "enable_gestures": ("BOOLEAN", {"default": True}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "wan_version": (["wan_2.1", "wan_2.2", "auto"], {"default": "auto"}),
                "enable_t2i_adapter": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HAND_LANDMARKS", "IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("hand_landmarks", "annotated_image", "gestures", "confidence")
    FUNCTION = "track_hands"
    CATEGORY = "Kanibus"
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
    def track_hands(self, image, max_hands=2, detection_confidence=0.5,
                   enable_gestures=True, smoothing=0.5, wan_version="auto",
                   enable_t2i_adapter=True, cache_results=True):
        """Track hands and detect gestures with WAN/T2I-Adapter compatibility"""
        
        try:
            # Auto-detect WAN version if needed
            if wan_version == "auto":
                wan_version = self._detect_wan_version(image)
            
            # Adjust detection confidence for WAN compatibility
            detection_confidence = self._adjust_confidence_for_wan(detection_confidence, wan_version)
        
            # Convert input with enhanced error handling
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                image_np = (image_np * 255).astype(np.uint8)
            else:
                raise ValueError("Input image must be a torch.Tensor")
        
            # Process with MediaPipe (WAN optimized)
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Apply WAN-specific hand detection settings
            if wan_version == "wan_2.1":
                # Optimize for 480p efficiency
                self.hands.min_detection_confidence = detection_confidence * 1.1
                self.hands.max_num_hands = min(max_hands, 2)  # Limit for 480p
            elif wan_version == "wan_2.2":
                # Optimize for 720p quality
                self.hands.min_detection_confidence = detection_confidence * 0.9
                self.hands.max_num_hands = max_hands
            
            results = self.hands.process(rgb_image)
            
            hand_landmarks = []
            gestures = []
            annotated = image_np.copy()
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                h, w = image_np.shape[:2]
            
            for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                # Convert landmarks to array
                hand_points = np.zeros((21, 3))
                for i, landmark in enumerate(landmarks.landmark):
                    hand_points[i] = [
                        landmark.x * w,
                        landmark.y * h,
                        landmark.z
                    ]
                
                # Add T2I-Adapter compatibility metadata
                if enable_t2i_adapter:
                    hand_points = np.column_stack([hand_points, 
                                                 np.full((21, 1), 1.0)])  # Add adapter compatibility flag
                
                hand_landmarks.append(hand_points)
                
                # Draw landmarks
                for point in hand_points[:, :2].astype(int):
                    cv2.circle(annotated, tuple(point), 3, (0, 255, 0), -1)
                
                # Simple gesture recognition
                if enable_gestures:
                    gesture = self._recognize_gesture(hand_points)
                    gestures.append(gesture)
                    
                    # Draw gesture text
                    cv2.putText(annotated, gesture, (10, 30 + hand_idx * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                confidence += 0.8  # Placeholder confidence
            
            confidence = confidence / max(len(hand_landmarks), 1)
            gesture_string = ", ".join(gestures) if gestures else "none"
            
            # Convert to tensor
            annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0).unsqueeze(0)
            
            return (hand_landmarks, annotated_tensor, gesture_string, confidence)
            
        except Exception as e:
            self.logger.error(f"Error in hand tracking: {str(e)}")
            # Return safe defaults
            h, w = 512, 512
            if isinstance(image, torch.Tensor) and len(image.shape) >= 2:
                h, w = image.shape[-2:]
            empty_landmarks = []
            empty_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            return (empty_landmarks, empty_image, "none", 0.0)
    
    def _detect_wan_version(self, image):
        """Auto-detect WAN version based on input characteristics"""
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            if h <= 480 or w <= 480:
                return "wan_2.1"
            else:
                return "wan_2.2"
        return "wan_2.2"
            
    def _adjust_confidence_for_wan(self, confidence, wan_version):
        """Adjust confidence threshold for WAN compatibility"""
        if wan_version == "wan_2.1":
            return min(confidence * 1.05, 1.0)
        elif wan_version == "wan_2.2":
            return confidence * 0.98
        return confidence
    
    def _recognize_gesture(self, hand_points):
        """Simple gesture recognition"""
        # Placeholder gesture recognition
        # Would implement actual gesture classification here
        gestures = ["open_hand", "fist", "pointing", "peace", "thumbs_up"]
        return np.random.choice(gestures)
