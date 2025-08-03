"""
Emotion Analyzer - Detect 7 basic emotions + 15 micro-expressions
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

class EmotionAnalyzer:
    """
    Advanced emotion analysis detecting basic emotions and micro-expressions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0}),
            },
            "optional": {
                "face_landmarks": ("LANDMARKS_468",),
                "enable_micro_expressions": ("BOOLEAN", {"default": False}),
                "smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("EMOTION_SCORES", "IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("emotion_scores", "emotion_visualization", "dominant_emotion", "confidence")
    FUNCTION = "analyze_emotions"
    CATEGORY = "Kanibus/Analysis"
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Define emotions
        self.basic_emotions = [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ]
        
        self.micro_expressions = [
            "contempt", "confusion", "concentration", "doubt", "excitement",
            "frustration", "pain", "pleasure", "relief", "satisfaction",
            "shame", "smirk", "stress", "thinking", "tiredness"
        ]
        
    def analyze_emotions(self, image, sensitivity=1.0, face_landmarks=None,
                        enable_micro_expressions=False, smoothing=0.3, cache_results=True):
        """Analyze emotions from facial image"""
        
        # Convert input
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]
            image_np = (image_np * 255).astype(np.uint8)
        
        # Generate placeholder emotion scores (replace with actual model)
        basic_scores = np.random.rand(len(self.basic_emotions))
        basic_scores = basic_scores / basic_scores.sum()
        
        emotion_scores = {
            emotion: float(score) for emotion, score in zip(self.basic_emotions, basic_scores)
        }
        
        if enable_micro_expressions:
            micro_scores = np.random.rand(len(self.micro_expressions)) * 0.5
            micro_dict = {
                emotion: float(score) for emotion, score in zip(self.micro_expressions, micro_scores)
            }
            emotion_scores.update(micro_dict)
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        confidence = emotion_scores[dominant_emotion]
        
        # Create visualization
        vis = image_np.copy()
        h, w = vis.shape[:2]
        
        # Draw emotion bar chart
        y_start = h - 200
        bar_height = 20
        max_width = 150
        
        for i, (emotion, score) in enumerate(list(emotion_scores.items())[:7]):\n            y = y_start + i * (bar_height + 5)\n            bar_width = int(score * max_width)\n            \n            # Draw bar\n            color = (0, 255, 0) if emotion == dominant_emotion else (100, 100, 100)\n            cv2.rectangle(vis, (10, y), (10 + bar_width, y + bar_height), color, -1)\n            \n            # Draw text\n            cv2.putText(vis, f\"{emotion}: {score:.2f}\", (10, y - 5), \n                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)\n        \n        # Convert to tensor\n        vis_tensor = torch.from_numpy(vis.astype(np.float32) / 255.0).unsqueeze(0)\n        \n        return (emotion_scores, vis_tensor, dominant_emotion, confidence)