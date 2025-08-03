"""
Neural Pupil Tracker - Advanced eye tracking with MediaPipe iris detection
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time

# Import our core system
try:
    from ..src.neural_engine import NeuralEngine, ProcessingConfig
    from ..src.gpu_optimizer import GPUOptimizer
    from ..src.cache_manager import CacheManager
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.neural_engine import NeuralEngine, ProcessingConfig
    from src.gpu_optimizer import GPUOptimizer
    from src.cache_manager import CacheManager

class BlinkState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    OPENING = "opening"

@dataclass
class EyeTrackingResult:
    """Complete eye tracking result"""
    # Pupil positions (normalized 0-1)
    left_pupil: Tuple[float, float]
    right_pupil: Tuple[float, float]
    
    # 3D gaze vectors
    left_gaze_vector: Tuple[float, float, float]
    right_gaze_vector: Tuple[float, float, float]
    
    # Convergence point (3D world coordinates)
    convergence_point: Optional[Tuple[float, float, float]]
    
    # Blink detection
    left_blink_state: BlinkState
    right_blink_state: BlinkState
    left_ear: float  # Eye Aspect Ratio
    right_ear: float
    
    # Pupil dilation
    left_pupil_diameter: float
    right_pupil_diameter: float
    
    # Saccade detection
    saccade_velocity: float  # degrees/second
    is_saccade: bool
    
    # Confidence scores
    left_eye_confidence: float
    right_eye_confidence: float
    
    # Raw landmarks
    left_iris_landmarks: np.ndarray  # Shape: (4, 3)
    right_iris_landmarks: np.ndarray # Shape: (4, 3)
    
    # Temporal info
    timestamp: float
    frame_id: int

class KalmanFilter:
    """6-DOF Kalman filter for smooth eye tracking"""
    
    def __init__(self, dim_x=6, dim_z=2):
        self.dim_x = dim_x  # State: [x, y, vx, vy, ax, ay]
        self.dim_z = dim_z  # Observation: [x, y]
        
        # State vector [x, y, vx, vy, ax, ay]
        self.x = np.zeros(dim_x)
        
        # State covariance matrix
        self.P = np.eye(dim_x) * 1000
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        self.Q[0:2, 0:2] *= 0.01  # Position noise
        self.Q[2:4, 2:4] *= 0.1   # Velocity noise
        self.Q[4:6, 4:6] *= 1.0   # Acceleration noise
        
        # Measurement noise covariance
        self.R = np.eye(dim_z) * 0.1
        
        # State transition matrix (constant acceleration model)
        dt = 1.0/60.0  # Assume 60 FPS
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt], 
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """Update with measurement"""
        y = z - self.H @ self.x  # Residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate"""
        return float(self.x[0]), float(self.x[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return float(self.x[2]), float(self.x[3])

class NeuralPupilTracker:
    """
    Advanced neural pupil tracking using MediaPipe with Kalman filtering,
    saccade detection, blink analysis, and 3D gaze estimation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "smoothing": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "blink_threshold": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 0.5, "step": 0.05}),
                "saccade_threshold": ("FLOAT", {"default": 300.0, "min": 100.0, "max": 1000.0, "step": 50.0}),
            },
            "optional": {
                "previous_result": ("EYE_TRACKING_RESULT",),
                "enable_3d_gaze": ("BOOLEAN", {"default": True}),
                "enable_saccade_detection": ("BOOLEAN", {"default": True}),
                "enable_pupil_dilation": ("BOOLEAN", {"default": True}),
                "cache_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("EYE_TRACKING_RESULT", "IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("tracking_result", "annotated_image", "gaze_visualization", "left_eye_mask", "right_eye_mask")
    FUNCTION = "track_pupils"
    CATEGORY = "Kanibus/Tracking"
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize core systems
        self.gpu_optimizer = GPUOptimizer()
        self.neural_engine = NeuralEngine(ProcessingConfig(
            device=self.gpu_optimizer.get_device().type,
            precision="fp16" if self.gpu_optimizer.selected_gpu and self.gpu_optimizer.selected_gpu.supports_fp16 else "fp32",
            batch_size=1,
            max_workers=2
        ))
        self.cache_manager = CacheManager()
        
        # Kalman filters for smooth tracking
        self.left_eye_filter = KalmanFilter()
        self.right_eye_filter = KalmanFilter()
        
        # Iris landmark indices (MediaPipe)
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [472, 473, 474, 475]
        
        # Eye region landmarks for EAR calculation
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # State tracking
        self.frame_count = 0
        self.last_positions = {"left": None, "right": None}
        self.last_timestamp = None
        self.blink_states = {"left": BlinkState.OPEN, "right": BlinkState.OPEN}
        
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.performance_stats = {
            "total_frames": 0,
            "avg_fps": 0.0,
            "processing_time": 0.0
        }
    
    def _normalize_landmarks(self, landmarks, image_shape):
        """Normalize landmarks to image coordinates"""
        h, w = image_shape[:2]
        normalized = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            normalized.append([x, y, z])
        return np.array(normalized)
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _detect_blink_state(self, ear, threshold, current_state):
        """Detect blink state based on EAR and current state"""
        if current_state == BlinkState.OPEN:
            if ear < threshold:
                return BlinkState.CLOSING
        elif current_state == BlinkState.CLOSING:
            if ear < threshold * 0.7:  # Hysteresis
                return BlinkState.CLOSED
            elif ear > threshold:
                return BlinkState.OPENING
        elif current_state == BlinkState.CLOSED:
            if ear > threshold * 0.7:
                return BlinkState.OPENING
        elif current_state == BlinkState.OPENING:
            if ear > threshold:
                return BlinkState.OPEN
            elif ear < threshold * 0.7:
                return BlinkState.CLOSING
        
        return current_state
    
    def _calculate_pupil_diameter(self, iris_landmarks):
        """Calculate pupil diameter from iris landmarks"""
        # Use horizontal and vertical distances of iris
        horizontal = np.linalg.norm(iris_landmarks[0] - iris_landmarks[2])
        vertical = np.linalg.norm(iris_landmarks[1] - iris_landmarks[3])
        
        # Average diameter (normalized)
        diameter = (horizontal + vertical) / 2.0
        return float(diameter)
    
    def _calculate_gaze_vector(self, iris_landmarks, eye_landmarks):
        """Calculate 3D gaze vector from iris and eye landmarks"""
        # Iris center
        iris_center = np.mean(iris_landmarks[:, :2], axis=0)
        
        # Eye corners for reference frame
        eye_center = np.mean(eye_landmarks[:, :2], axis=0)
        
        # Normalized gaze vector (simplified)
        gaze_offset = iris_center - eye_center
        
        # Convert to 3D vector (assuming z-component based on depth)
        z_component = -0.1  # Looking forward
        gaze_vector = np.array([gaze_offset[0], gaze_offset[1], z_component])
        
        # Normalize
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        return tuple(gaze_vector.astype(float))
    
    def _calculate_convergence_point(self, left_gaze, right_gaze, left_pos, right_pos):
        """Calculate 3D convergence point from binocular gaze vectors"""
        # Simplified convergence calculation
        # In practice, this would require proper 3D head pose estimation
        
        # Average gaze direction
        avg_gaze = np.array([(left_gaze[i] + right_gaze[i]) / 2 for i in range(3)])
        
        # Convergence distance (estimated)
        convergence_distance = 50.0  # cm, could be dynamic
        
        # Eye center (midpoint between pupils)
        eye_center = np.array([(left_pos[0] + right_pos[0]) / 2, (left_pos[1] + right_pos[1]) / 2, 0])
        
        # Convergence point
        convergence_point = eye_center + avg_gaze * convergence_distance
        
        return tuple(convergence_point.astype(float))
    
    def _detect_saccade(self, current_pos, last_pos, dt, threshold):
        """Detect saccadic eye movement"""
        if last_pos is None or dt <= 0:
            return False, 0.0
        
        # Calculate angular velocity (simplified)
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        
        # Convert pixel movement to approximate degrees
        # Assuming 1 pixel ≈ 0.1 degrees (rough approximation)
        angle_change = np.sqrt(dx*dx + dy*dy) * 0.1
        angular_velocity = angle_change / dt
        
        is_saccade = angular_velocity > threshold
        
        return is_saccade, angular_velocity
    
    def _create_eye_mask(self, image_shape, eye_landmarks, dilation=5):
        """Create binary mask for eye region"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Create convex hull around eye landmarks
        points = eye_landmarks[:, :2].astype(np.int32)
        hull = cv2.convexHull(points)
        
        # Fill the mask
        cv2.fillPoly(mask, [hull], 255)
        
        # Dilate mask slightly
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _create_gaze_visualization(self, image, left_pos, right_pos, left_gaze, right_gaze, convergence_point):
        """Create gaze visualization overlay"""
        vis = image.copy()
        h, w = image.shape[:2]
        
        # Draw eye positions
        left_pixel = (int(left_pos[0] * w), int(left_pos[1] * h))
        right_pixel = (int(right_pos[0] * w), int(right_pos[1] * h))
        
        cv2.circle(vis, left_pixel, 5, (0, 255, 0), -1)
        cv2.circle(vis, right_pixel, 5, (0, 255, 0), -1)
        
        # Draw gaze vectors
        gaze_length = 50
        left_end = (
            int(left_pixel[0] + left_gaze[0] * gaze_length),
            int(left_pixel[1] + left_gaze[1] * gaze_length)
        )
        right_end = (
            int(right_pixel[0] + right_gaze[0] * gaze_length),
            int(right_pixel[1] + right_gaze[1] * gaze_length)
        )
        
        cv2.arrowedLine(vis, left_pixel, left_end, (255, 0, 0), 2)
        cv2.arrowedLine(vis, right_pixel, right_end, (255, 0, 0), 2)
        
        # Draw convergence point if available
        if convergence_point:
            conv_pixel = (
                int(convergence_point[0] * w / 100),  # Scale for visualization
                int(convergence_point[1] * h / 100)
            )
            if 0 <= conv_pixel[0] < w and 0 <= conv_pixel[1] < h:
                cv2.circle(vis, conv_pixel, 10, (0, 0, 255), 2)
        
        return vis
    
    def track_pupils(self, image, sensitivity=1.0, smoothing=0.7, blink_threshold=0.25, 
                    saccade_threshold=300.0, previous_result=None, enable_3d_gaze=True,
                    enable_saccade_detection=True, enable_pupil_dilation=True, cache_results=True):
        """Main pupil tracking function"""
        
        start_time = time.time()
        
        # Convert ComfyUI image to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 4:  # Batch dimension
                image_np = image_np[0]
            if image_np.shape[-1] == 3:  # RGB
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # Check cache
        frame_id = f"frame_{self.frame_count}"
        if cache_results:
            cached_result = self.cache_manager.get(f"pupil_tracking_{frame_id}")
            if cached_result is not None:
                return self._format_output(cached_result, image_np)
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            # No face detected - return default result
            default_result = EyeTrackingResult(
                left_pupil=(0.5, 0.5),
                right_pupil=(0.5, 0.5),
                left_gaze_vector=(0.0, 0.0, -1.0),
                right_gaze_vector=(0.0, 0.0, -1.0),
                convergence_point=None,
                left_blink_state=BlinkState.OPEN,
                right_blink_state=BlinkState.OPEN,
                left_ear=0.3,
                right_ear=0.3,
                left_pupil_diameter=1.0,
                right_pupil_diameter=1.0,
                saccade_velocity=0.0,
                is_saccade=False,
                left_eye_confidence=0.0,
                right_eye_confidence=0.0,
                left_iris_landmarks=np.zeros((4, 3)),
                right_iris_landmarks=np.zeros((4, 3)),
                timestamp=time.time(),
                frame_id=self.frame_count
            )
            return self._format_output(default_result, image_np)
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_np = self._normalize_landmarks(face_landmarks.landmark, image_np.shape)
        
        # Extract iris landmarks
        left_iris = landmarks_np[self.LEFT_IRIS]
        right_iris = landmarks_np[self.RIGHT_IRIS]
        
        # Calculate pupil positions (iris centers)
        left_pupil_pos = np.mean(left_iris[:, :2], axis=0)
        right_pupil_pos = np.mean(right_iris[:, :2], axis=0)
        
        # Normalize to 0-1 range
        h, w = image_np.shape[:2]
        left_pupil_norm = (left_pupil_pos[0] / w, left_pupil_pos[1] / h)
        right_pupil_norm = (right_pupil_pos[0] / w, right_pupil_pos[1] / h)
        
        # Apply Kalman filtering for smoothing
        if smoothing > 0:
            self.left_eye_filter.predict()
            self.left_eye_filter.update(left_pupil_norm)
            left_pupil_norm = self.left_eye_filter.get_position()
            
            self.right_eye_filter.predict()
            self.right_eye_filter.update(right_pupil_norm)
            right_pupil_norm = self.right_eye_filter.get_position()
        
        # Extract eye region landmarks for blink detection
        left_eye_landmarks = landmarks_np[self.LEFT_EYE]
        right_eye_landmarks = landmarks_np[self.RIGHT_EYE]
        
        # Calculate Eye Aspect Ratios
        left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks[:6])  # Use first 6 points
        right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks[:6])
        
        # Detect blink states
        left_blink_state = self._detect_blink_state(left_ear, blink_threshold, 
                                                   self.blink_states.get("left", BlinkState.OPEN))
        right_blink_state = self._detect_blink_state(right_ear, blink_threshold,
                                                    self.blink_states.get("right", BlinkState.OPEN))
        
        self.blink_states["left"] = left_blink_state
        self.blink_states["right"] = right_blink_state
        
        # Calculate pupil diameters
        left_diameter = self._calculate_pupil_diameter(left_iris) if enable_pupil_dilation else 1.0
        right_diameter = self._calculate_pupil_diameter(right_iris) if enable_pupil_dilation else 1.0
        
        # Calculate 3D gaze vectors
        left_gaze_vector = self._calculate_gaze_vector(left_iris, left_eye_landmarks) if enable_3d_gaze else (0.0, 0.0, -1.0)
        right_gaze_vector = self._calculate_gaze_vector(right_iris, right_eye_landmarks) if enable_3d_gaze else (0.0, 0.0, -1.0)
        
        # Calculate convergence point
        convergence_point = self._calculate_convergence_point(left_gaze_vector, right_gaze_vector,
                                                            left_pupil_norm, right_pupil_norm) if enable_3d_gaze else None
        
        # Saccade detection
        current_time = time.time()
        dt = current_time - self.last_timestamp if self.last_timestamp else 0.0
        
        is_saccade = False
        saccade_velocity = 0.0
        
        if enable_saccade_detection and self.last_positions["left"] and self.last_positions["right"]:
            left_saccade, left_velocity = self._detect_saccade(left_pupil_norm, self.last_positions["left"], dt, saccade_threshold)
            right_saccade, right_velocity = self._detect_saccade(right_pupil_norm, self.last_positions["right"], dt, saccade_threshold)
            
            is_saccade = left_saccade or right_saccade
            saccade_velocity = max(left_velocity, right_velocity)
        
        # Update tracking state
        self.last_positions["left"] = left_pupil_norm
        self.last_positions["right"] = right_pupil_norm
        self.last_timestamp = current_time
        
        # Calculate confidence scores (simplified)
        left_confidence = min(1.0, np.mean(left_iris[:, 2]) * sensitivity)  # Use z-coordinate as confidence proxy
        right_confidence = min(1.0, np.mean(right_iris[:, 2]) * sensitivity)
        
        # Create result
        tracking_result = EyeTrackingResult(
            left_pupil=left_pupil_norm,
            right_pupil=right_pupil_norm,
            left_gaze_vector=left_gaze_vector,
            right_gaze_vector=right_gaze_vector,
            convergence_point=convergence_point,
            left_blink_state=left_blink_state,
            right_blink_state=right_blink_state,
            left_ear=left_ear,
            right_ear=right_ear,
            left_pupil_diameter=left_diameter,
            right_pupil_diameter=right_diameter,
            saccade_velocity=saccade_velocity,
            is_saccade=is_saccade,
            left_eye_confidence=left_confidence,
            right_eye_confidence=right_confidence,
            left_iris_landmarks=left_iris,
            right_iris_landmarks=right_iris,
            timestamp=current_time,
            frame_id=self.frame_count
        )
        
        # Cache result
        if cache_results:
            self.cache_manager.put(f"pupil_tracking_{frame_id}", tracking_result, priority=2)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats["total_frames"] += 1
        self.performance_stats["processing_time"] += processing_time
        self.performance_stats["avg_fps"] = self.performance_stats["total_frames"] / self.performance_stats["processing_time"]
        
        self.frame_count += 1
        
        return self._format_output(tracking_result, image_np)
    
    def _format_output(self, result, image_np):
        """Format output for ComfyUI"""
        # Create annotated image
        annotated = image_np.copy()
        h, w = image_np.shape[:2]
        
        # Draw pupils
        left_pixel = (int(result.left_pupil[0] * w), int(result.left_pupil[1] * h))
        right_pixel = (int(result.right_pupil[0] * w), int(result.right_pupil[1] * h))
        
        # Color based on blink state
        left_color = (0, 255, 0) if result.left_blink_state == BlinkState.OPEN else (255, 0, 0)
        right_color = (0, 255, 0) if result.right_blink_state == BlinkState.OPEN else (255, 0, 0)
        
        cv2.circle(annotated, left_pixel, 8, left_color, 2)
        cv2.circle(annotated, right_pixel, 8, right_color, 2)
        
        # Draw info text
        info_text = f"FPS: {self.performance_stats['avg_fps']:.1f} | Saccade: {result.saccade_velocity:.0f}°/s"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create gaze visualization
        gaze_vis = self._create_gaze_visualization(
            image_np, result.left_pupil, result.right_pupil,
            result.left_gaze_vector, result.right_gaze_vector, result.convergence_point
        )
        
        # Create eye masks if landmarks available
        if hasattr(result, 'left_iris_landmarks') and len(result.left_iris_landmarks) > 0:
            left_mask = self._create_eye_mask(image_np.shape, result.left_iris_landmarks)
            right_mask = self._create_eye_mask(image_np.shape, result.right_iris_landmarks)
        else:
            left_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            right_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        # Convert to ComfyUI format
        annotated_tensor = torch.from_numpy(annotated.astype(np.float32) / 255.0).unsqueeze(0)
        gaze_vis_tensor = torch.from_numpy(gaze_vis.astype(np.float32) / 255.0).unsqueeze(0)
        left_mask_tensor = torch.from_numpy(left_mask.astype(np.float32) / 255.0).unsqueeze(0)
        right_mask_tensor = torch.from_numpy(right_mask.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result, annotated_tensor, gaze_vis_tensor, left_mask_tensor, right_mask_tensor)