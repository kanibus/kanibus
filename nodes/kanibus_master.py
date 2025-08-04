"""
Kanibus Master - Main orchestrator node that integrates all features
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
import json

# Import our core system
try:
    from ..src.neural_engine import NeuralEngine, ProcessingConfig, ProcessingMode
    from ..src.gpu_optimizer import GPUOptimizer
    from ..src.cache_manager import CacheManager
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.neural_engine import NeuralEngine, ProcessingConfig, ProcessingMode
    from src.gpu_optimizer import GPUOptimizer
    from src.cache_manager import CacheManager

# Import other nodes
from .neural_pupil_tracker import NeuralPupilTracker, EyeTrackingResult
from .video_frame_loader import VideoFrameLoader, VideoMetadata

class ProcessingPipeline(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    ANALYSIS = "analysis"

class WanVersion(Enum):
    WAN_21 = "wan_2.1"
    WAN_22 = "wan_2.2"
    AUTO_DETECT = "auto_detect"

@dataclass
class KanibusConfig:
    """Master configuration for Kanibus system"""
    # Processing settings
    pipeline_mode: ProcessingPipeline = ProcessingPipeline.REAL_TIME
    wan_version: WanVersion = WanVersion.AUTO_DETECT
    target_fps: float = 24.0
    enable_gpu_optimization: bool = True
    
    # Feature toggles
    enable_eye_tracking: bool = True
    enable_face_tracking: bool = True
    enable_body_tracking: bool = False
    enable_object_tracking: bool = False
    enable_depth_estimation: bool = True
    enable_emotion_analysis: bool = False
    enable_hand_tracking: bool = False
    
    # Quality settings
    tracking_quality: str = "high"  # low, medium, high, ultra
    depth_quality: str = "medium"
    temporal_smoothing: float = 0.7
    
    # ControlNet settings
    controlnet_weights: Dict[str, float] = None
    
    # Performance settings
    batch_size: int = 1
    max_workers: int = 4
    memory_limit: float = 0.8
    enable_caching: bool = True
    cache_size_gb: float = 5.0
    
    def __post_init__(self):
        if self.controlnet_weights is None:
            # Updated weights for T2I-Adapters (more efficient than legacy ControlNet)
            self.controlnet_weights = {
                "eye_mask": 1.1,      # Reduced for T2I-Adapter efficiency
                "depth": 0.9,         # T2I-Adapter depth
                "canny": 0.6,         # T2I-Adapter canny (replaces normal)
                "sketch": 1.2,        # T2I-Adapter sketch
                "landmarks": 0.8,     # Reduced for efficiency
                "pose": 0.7,          # T2I-Adapter openpose
                "hands": 0.5
            }
            
            # WAN-specific weight adjustments
            if self.wan_version == WanVersion.WAN_21:
                # Further reduce weights for WAN 2.1 (480p efficiency)
                for key in self.controlnet_weights:
                    self.controlnet_weights[key] *= 0.85
            elif self.wan_version == WanVersion.WAN_22:
                # Optimize weights for WAN 2.2 (720p quality)
                self.controlnet_weights["eye_mask"] = 1.3
                self.controlnet_weights["depth"] = 1.0
                self.controlnet_weights["canny"] = 0.7

@dataclass
class KanibusResult:
    """Complete processing result from Kanibus Master"""
    # Core data
    frame_id: int
    timestamp: float
    processing_time: float
    
    # Eye tracking
    eye_tracking: Optional[EyeTrackingResult] = None
    
    # Control maps
    eye_mask_left: Optional[torch.Tensor] = None
    eye_mask_right: Optional[torch.Tensor] = None
    depth_map: Optional[torch.Tensor] = None
    normal_map: Optional[torch.Tensor] = None
    pose_map: Optional[torch.Tensor] = None
    
    # Facial analysis
    facial_landmarks: Optional[np.ndarray] = None
    emotion_scores: Optional[Dict[str, float]] = None
    
    # Body analysis
    body_pose: Optional[np.ndarray] = None
    hand_poses: Optional[Dict[str, np.ndarray]] = None
    
    # Object tracking
    tracked_objects: Optional[List[Dict]] = None
    
    # Metadata
    confidence_scores: Dict[str, float] = None
    wan_compatibility: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.wan_compatibility is None:
            self.wan_compatibility = {}

class KanibusMaster:
    """
    Master orchestrator node that coordinates all Kanibus features for complete
    eye-tracking ControlNet generation with WAN 2.1/2.2 compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_source": (["image", "video", "webcam"], {"default": "image"}),
                "pipeline_mode": (["real_time", "batch", "streaming", "analysis"], {"default": "real_time"}),
                "wan_version": (["wan_2.1", "wan_2.2", "auto_detect"], {"default": "auto_detect"}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                # Input data
                "image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "video_metadata": ("VIDEO_METADATA",),
                
                # Feature enables
                "enable_eye_tracking": ("BOOLEAN", {"default": True}),
                "enable_face_tracking": ("BOOLEAN", {"default": True}),
                "enable_body_tracking": ("BOOLEAN", {"default": False}),
                "enable_object_tracking": ("BOOLEAN", {"default": False}),
                "enable_depth_estimation": ("BOOLEAN", {"default": True}),
                "enable_emotion_analysis": ("BOOLEAN", {"default": False}),
                "enable_hand_tracking": ("BOOLEAN", {"default": False}),
                
                # Quality settings
                "tracking_quality": (["low", "medium", "high", "ultra"], {"default": "high"}),
                "depth_quality": (["low", "medium", "high"], {"default": "medium"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # T2I-Adapter weights (optimized for efficiency)
                "eye_mask_weight": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.1}),
                "depth_weight": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0, "step": 0.1}),
                "canny_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.1}),
                "sketch_weight": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1}),
                "landmarks_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1}),
                "pose_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.1}),
                "hands_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1}),
                
                # Model preferences
                "model_preference": (["t2i_adapter", "legacy_controlnet", "auto"], {"default": "t2i_adapter"}),
                
                # Performance
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "enable_caching": ("BOOLEAN", {"default": True}),
                "enable_gpu_optimization": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = (
        "KANIBUS_RESULT",      # Complete result
        "IMAGE",               # Processed image
        "MASK",                # Combined eye mask
        "IMAGE",               # Depth map
        "IMAGE",               # Normal map
        "IMAGE",               # Pose visualization
        "CONDITIONING",        # ControlNet conditioning
        "STRING"               # Processing report
    )
    RETURN_NAMES = (
        "kanibus_result",
        "processed_image", 
        "eye_mask",
        "depth_map",
        "normal_map", 
        "pose_visualization",
        "controlnet_conditioning",
        "processing_report"
    )
    FUNCTION = "process"
    CATEGORY = "Kanibus/Master"
    
    def __init__(self):
        # Initialize core systems
        self.gpu_optimizer = GPUOptimizer()
        self.gpu_optimizer.apply_optimizations()
        
        self.neural_engine = NeuralEngine(ProcessingConfig(
            device=self.gpu_optimizer.get_device().type,
            precision=self.gpu_optimizer.get_optimization_settings().get("precision", "fp32"),
            batch_size=self.gpu_optimizer.get_optimization_settings().get("batch_size", 1),
            max_workers=self.gpu_optimizer.get_optimization_settings().get("num_workers", 4),
            mode=ProcessingMode.REALTIME
        ))
        
        self.cache_manager = CacheManager(
            cache_dir="./cache/kanibus_master",
            max_memory_mb=2048,
            max_disk_gb=5.0
        )
        
        # Initialize component nodes
        self.pupil_tracker = NeuralPupilTracker()
        self.video_loader = VideoFrameLoader()
        
        # Placeholder for other nodes (will be implemented)
        self.face_tracker = None
        self.depth_estimator = None
        self.pose_estimator = None
        self.emotion_analyzer = None
        self.hand_tracker = None
        self.object_tracker = None
        
        # State management
        self.current_config = KanibusConfig()
        self.processing_history = []
        self.performance_monitor = {
            "total_frames": 0,
            "avg_processing_time": 0.0,
            "current_fps": 0.0,
            "peak_fps": 0.0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0
        }
        
        # Threading for real-time processing
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("KanibusMaster initialized")
        
        # Print system info
        self.gpu_optimizer.print_system_info()
    
    def _update_config(self, **kwargs) -> KanibusConfig:
        """Update configuration from node inputs"""
        config = KanibusConfig()
        
        # Map string inputs to enums
        pipeline_mode_map = {
            "real_time": ProcessingPipeline.REAL_TIME,
            "batch": ProcessingPipeline.BATCH,
            "streaming": ProcessingPipeline.STREAMING,
            "analysis": ProcessingPipeline.ANALYSIS
        }
        
        wan_version_map = {
            "wan_2.1": WanVersion.WAN_21,
            "wan_2.2": WanVersion.WAN_22,
            "auto_detect": WanVersion.AUTO_DETECT
        }
        
        # Update configuration
        config.pipeline_mode = pipeline_mode_map.get(kwargs.get("pipeline_mode", "real_time"), ProcessingPipeline.REAL_TIME)
        config.wan_version = wan_version_map.get(kwargs.get("wan_version", "auto_detect"), WanVersion.AUTO_DETECT)
        config.target_fps = kwargs.get("target_fps", 24.0)
        
        # Feature toggles
        config.enable_eye_tracking = kwargs.get("enable_eye_tracking", True)
        config.enable_face_tracking = kwargs.get("enable_face_tracking", True)
        config.enable_body_tracking = kwargs.get("enable_body_tracking", False)
        config.enable_object_tracking = kwargs.get("enable_object_tracking", False)
        config.enable_depth_estimation = kwargs.get("enable_depth_estimation", True)
        config.enable_emotion_analysis = kwargs.get("enable_emotion_analysis", False)
        config.enable_hand_tracking = kwargs.get("enable_hand_tracking", False)
        
        # Quality settings
        config.tracking_quality = kwargs.get("tracking_quality", "high")
        config.depth_quality = kwargs.get("depth_quality", "medium")
        config.temporal_smoothing = kwargs.get("temporal_smoothing", 0.7)
        
        # ControlNet weights
        config.controlnet_weights = {
            "eye_mask": kwargs.get("eye_mask_weight", 1.3),
            "depth": kwargs.get("depth_weight", 1.0),
            "normal": kwargs.get("normal_weight", 0.7),
            "landmarks": kwargs.get("landmarks_weight", 0.9),
            "pose": kwargs.get("pose_weight", 0.6),
            "hands": kwargs.get("hands_weight", 0.5)
        }
        
        # Performance settings
        config.batch_size = kwargs.get("batch_size", 1)
        config.enable_caching = kwargs.get("enable_caching", True)
        config.enable_gpu_optimization = kwargs.get("enable_gpu_optimization", True)
        
        return config
    
    def _detect_wan_version(self, config: KanibusConfig, metadata: Optional[VideoMetadata] = None) -> WanVersion:
        """Auto-detect WAN version based on input characteristics"""
        if config.wan_version != WanVersion.AUTO_DETECT:
            return config.wan_version
        
        # Simple heuristic: use resolution to guess WAN version
        if metadata:
            if metadata.height <= 480:
                return WanVersion.WAN_21
            elif metadata.height >= 720:
                return WanVersion.WAN_22
        
        # Default to WAN 2.2 for new projects
        return WanVersion.WAN_22
    
    def _process_single_frame(self, frame: torch.Tensor, config: KanibusConfig, 
                             frame_id: int = 0) -> KanibusResult:
        """Process a single frame through the complete pipeline"""
        start_time = time.time()
        
        result = KanibusResult(
            frame_id=frame_id,
            timestamp=time.time(),
            processing_time=0.0
        )
        
        try:
            # Eye tracking
            if config.enable_eye_tracking:
                eye_result = self.pupil_tracker.track_pupils(
                    image=frame,
                    sensitivity=1.0,
                    smoothing=config.temporal_smoothing,
                    enable_caching=config.enable_caching
                )
                
                result.eye_tracking = eye_result[0]  # tracking_result
                result.eye_mask_left = eye_result[3]   # left_eye_mask
                result.eye_mask_right = eye_result[4]  # right_eye_mask
                result.confidence_scores["eye_tracking"] = (
                    result.eye_tracking.left_eye_confidence + result.eye_tracking.right_eye_confidence
                ) / 2.0
                
                self.logger.debug(f"Eye tracking: {result.confidence_scores['eye_tracking']:.2f} confidence")
            
            # Depth estimation (placeholder - would use actual depth model)
            if config.enable_depth_estimation:
                result.depth_map = self._generate_placeholder_depth(frame)
                result.confidence_scores["depth"] = 0.8
                
                # Generate normal map from depth
                result.normal_map = self._depth_to_normal(result.depth_map)
                result.confidence_scores["normal"] = 0.7
            
            # Facial landmarks (placeholder - would use actual landmark detector)
            if config.enable_face_tracking:
                result.facial_landmarks = self._generate_placeholder_landmarks()
                result.confidence_scores["face_landmarks"] = 0.85
            
            # Body pose (placeholder)
            if config.enable_body_tracking:
                result.body_pose = self._generate_placeholder_pose()
                result.confidence_scores["body_pose"] = 0.7
            
            # Hand tracking (placeholder)
            if config.enable_hand_tracking:
                result.hand_poses = self._generate_placeholder_hands()
                result.confidence_scores["hand_tracking"] = 0.6
            
            # Emotion analysis (placeholder)
            if config.enable_emotion_analysis:
                result.emotion_scores = self._generate_placeholder_emotions()
                result.confidence_scores["emotion"] = 0.75
            
            # Object tracking (placeholder)
            if config.enable_object_tracking:
                result.tracked_objects = self._generate_placeholder_objects()
                result.confidence_scores["object_tracking"] = 0.65
            
            # WAN compatibility info
            wan_version = self._detect_wan_version(config)
            result.wan_compatibility = {
                "version": wan_version.value,
                "resolution_compatible": True,
                "motion_module_compatible": wan_version == WanVersion.WAN_22,
                "recommended_settings": self._get_wan_settings(wan_version)
            }
            
        except Exception as e:
            self.logger.error(f"Processing error for frame {frame_id}: {e}")
            result.confidence_scores["overall"] = 0.0
        
        # Calculate processing time
        result.processing_time = time.time() - start_time
        
        # Update performance monitoring
        self._update_performance_stats(result.processing_time)
        
        return result
    
    def _generate_placeholder_depth(self, frame: torch.Tensor) -> torch.Tensor:
        """Generate placeholder depth map (replace with actual depth estimation)"""
        if frame.dim() == 4:
            batch_size, height, width, channels = frame.shape
        else:
            height, width, channels = frame.shape
            batch_size = 1
        
        # Simple gradient-based fake depth
        depth = torch.zeros((batch_size, height, width, 1), dtype=torch.float32)
        for i in range(height):
            depth[:, i, :, 0] = i / height  # Vertical gradient
        
        return depth
    
    def _depth_to_normal(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Convert depth map to normal map"""
        if depth_map.dim() == 4:
            depth_np = depth_map[0, :, :, 0].numpy()
        else:
            depth_np = depth_map[:, :, 0].numpy()
        
        # Calculate gradients
        grad_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create normal vectors
        normal = np.zeros((depth_np.shape[0], depth_np.shape[1], 3))
        normal[:, :, 0] = -grad_x  # X component
        normal[:, :, 1] = -grad_y  # Y component
        normal[:, :, 2] = 1.0      # Z component
        
        # Normalize
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (norm + 1e-8)
        
        # Convert to 0-1 range and to tensor
        normal = (normal + 1.0) / 2.0
        return torch.from_numpy(normal.astype(np.float32)).unsqueeze(0)
    
    def _generate_placeholder_landmarks(self) -> np.ndarray:
        """Generate placeholder facial landmarks"""
        return np.random.rand(468, 3) * 100  # 468 MediaPipe landmarks
    
    def _generate_placeholder_pose(self) -> np.ndarray:
        """Generate placeholder body pose"""
        return np.random.rand(33, 3) * 100  # 33 pose keypoints
    
    def _generate_placeholder_hands(self) -> Dict[str, np.ndarray]:
        """Generate placeholder hand poses"""
        return {
            "left_hand": np.random.rand(21, 3) * 100,
            "right_hand": np.random.rand(21, 3) * 100
        }
    
    def _generate_placeholder_emotions(self) -> Dict[str, float]:
        """Generate placeholder emotion scores"""
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        scores = np.random.rand(len(emotions))
        scores = scores / scores.sum()  # Normalize to sum to 1
        return {emotion: float(score) for emotion, score in zip(emotions, scores)}
    
    def _generate_placeholder_objects(self) -> List[Dict]:
        """Generate placeholder tracked objects"""
        return [
            {"class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300]},
            {"class": "face", "confidence": 0.85, "bbox": [120, 120, 180, 180]}
        ]
    
    def _get_wan_settings(self, wan_version: WanVersion) -> Dict:
        """Get recommended settings for WAN version"""
        if wan_version == WanVersion.WAN_21:
            return {
                "recommended_resolution": "480p",
                "max_fps": 24,
                "motion_module": "v1",
                "controlnet_scale": 0.8
            }
        else:  # WAN 2.2
            return {
                "recommended_resolution": "720p",
                "max_fps": 30,
                "motion_module": "v2",
                "controlnet_scale": 1.0
            }
    
    def _create_controlnet_conditioning(self, result: KanibusResult, config: KanibusConfig) -> List[Dict]:
        """Create ControlNet conditioning from results"""
        conditioning = []
        
        # Eye mask conditioning
        if result.eye_mask_left is not None and result.eye_mask_right is not None:
            # Combine left and right eye masks
            combined_mask = torch.maximum(result.eye_mask_left, result.eye_mask_right)
            conditioning.append({
                "type": "scribble",
                "image": combined_mask,
                "weight": config.controlnet_weights["eye_mask"],
                "start_percent": 0.0,
                "end_percent": 1.0
            })
        
        # Depth conditioning
        if result.depth_map is not None:
            conditioning.append({
                "type": "depth",
                "image": result.depth_map,
                "weight": config.controlnet_weights["depth"],
                "start_percent": 0.0,
                "end_percent": 1.0
            })
        
        # Normal map conditioning
        if result.normal_map is not None:
            conditioning.append({
                "type": "normal",
                "image": result.normal_map,
                "weight": config.controlnet_weights["normal"],
                "start_percent": 0.0,
                "end_percent": 1.0
            })
        
        # Pose conditioning (placeholder)
        if result.body_pose is not None:
            pose_image = self._render_pose_image(result.body_pose)
            conditioning.append({
                "type": "openpose",
                "image": pose_image,
                "weight": config.controlnet_weights["pose"],
                "start_percent": 0.0,
                "end_percent": 1.0
            })
        
        return conditioning
    
    def _render_pose_image(self, pose_data: np.ndarray) -> torch.Tensor:
        """Render pose data to image (placeholder)"""
        # This would render the pose keypoints as an image
        pose_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return pose_image
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance monitoring statistics"""
        self.performance_monitor["total_frames"] += 1
        
        # Update average processing time
        total_time = self.performance_monitor["avg_processing_time"] * (self.performance_monitor["total_frames"] - 1)
        self.performance_monitor["avg_processing_time"] = (total_time + processing_time) / self.performance_monitor["total_frames"]
        
        # Calculate current FPS
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            self.performance_monitor["current_fps"] = current_fps
            self.performance_monitor["peak_fps"] = max(self.performance_monitor["peak_fps"], current_fps)
        
        # Update GPU stats
        if self.gpu_optimizer.selected_gpu:
            gpu_stats = self.gpu_optimizer.monitor_gpu_usage()
            self.performance_monitor["gpu_utilization"] = gpu_stats.get("utilization", 0.0)
            self.performance_monitor["memory_usage"] = gpu_stats.get("memory_utilization", 0.0)
    
    def _create_processing_report(self, result: KanibusResult, config: KanibusConfig) -> str:
        """Create detailed processing report"""
        report = {
            "frame_info": {
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "processing_time": f"{result.processing_time * 1000:.1f}ms"
            },
            "configuration": {
                "pipeline_mode": config.pipeline_mode.value,
                "wan_version": config.wan_version.value,
                "target_fps": config.target_fps,
                "tracking_quality": config.tracking_quality
            },
            "features_enabled": {
                "eye_tracking": config.enable_eye_tracking,
                "face_tracking": config.enable_face_tracking,
                "body_tracking": config.enable_body_tracking,
                "depth_estimation": config.enable_depth_estimation,
                "emotion_analysis": config.enable_emotion_analysis
            },
            "confidence_scores": result.confidence_scores,
            "performance": {
                "current_fps": f"{self.performance_monitor['current_fps']:.1f}",
                "avg_processing_time": f"{self.performance_monitor['avg_processing_time'] * 1000:.1f}ms",
                "gpu_utilization": f"{self.performance_monitor['gpu_utilization']:.1f}%",
                "memory_usage": f"{self.performance_monitor['memory_usage']:.1f}%"
            },
            "wan_compatibility": result.wan_compatibility
        }
        
        return json.dumps(report, indent=2)
    
    def process(self, input_source: str, pipeline_mode: str = "real_time", 
               wan_version: str = "auto_detect", target_fps: float = 24.0,
               image: Optional[torch.Tensor] = None,
               video_frames: Optional[List[torch.Tensor]] = None,
               video_metadata: Optional[VideoMetadata] = None,
               **kwargs):
        """Main processing function"""
        
        # Update configuration
        config = self._update_config(
            pipeline_mode=pipeline_mode,
            wan_version=wan_version,
            target_fps=target_fps,
            **kwargs
        )
        
        self.current_config = config
        
        try:
            if input_source == "image" and image is not None:
                # Process single image
                result = self._process_single_frame(image, config, 0)
                
                # Create outputs
                processed_image = image  # Would be actual processed image
                combined_eye_mask = torch.maximum(result.eye_mask_left or torch.zeros_like(image[:,:,:,0:1]), 
                                                result.eye_mask_right or torch.zeros_like(image[:,:,:,0:1]))
                depth_map = result.depth_map or torch.zeros_like(image)
                normal_map = result.normal_map or torch.zeros_like(image)
                pose_viz = processed_image  # Would be pose visualization
                
                controlnet_conditioning = self._create_controlnet_conditioning(result, config)
                processing_report = self._create_processing_report(result, config)
                
                return (result, processed_image, combined_eye_mask, depth_map, 
                       normal_map, pose_viz, controlnet_conditioning, processing_report)
                
            elif input_source == "video" and video_frames is not None:
                # Process video frames
                results = []
                for i, frame in enumerate(video_frames):
                    frame_result = self._process_single_frame(frame, config, i)
                    results.append(frame_result)
                
                # Return results for first frame (batch processing would handle differently)
                if results:
                    result = results[0]
                    processed_image = video_frames[0]
                    combined_eye_mask = torch.maximum(result.eye_mask_left or torch.zeros((1,1,1,1)), 
                                                    result.eye_mask_right or torch.zeros((1,1,1,1)))
                    depth_map = result.depth_map or torch.zeros_like(processed_image)
                    normal_map = result.normal_map or torch.zeros_like(processed_image)
                    pose_viz = processed_image
                    
                    controlnet_conditioning = self._create_controlnet_conditioning(result, config)
                    processing_report = self._create_processing_report(result, config)
                    
                    return (result, processed_image, combined_eye_mask, depth_map,
                           normal_map, pose_viz, controlnet_conditioning, processing_report)
            
            else:
                # Webcam or other inputs (placeholder)
                self.logger.warning(f"Input source '{input_source}' not yet implemented")
                
                # Return empty results
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                empty_mask = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
                empty_result = KanibusResult(
                    frame_id=0,
                    timestamp=time.time(),
                    processing_time=0.0
                )
                
                return (empty_result, empty_image, empty_mask, empty_image,
                       empty_image, empty_image, [], "No input provided")
                
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            
            # Return error state
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 512, 512, 1), dtype=torch.float32)
            error_result = KanibusResult(
                frame_id=0,
                timestamp=time.time(),
                processing_time=0.0
            )
            error_report = f"Processing error: {str(e)}"
            
            return (error_result, empty_image, empty_mask, empty_image,
                   empty_image, empty_image, [], error_report)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_monitor.copy()
    
    def reset_performance_stats(self):
        """Reset performance monitoring"""
        self.performance_monitor = {
            "total_frames": 0,
            "avg_processing_time": 0.0,
            "current_fps": 0.0,
            "peak_fps": 0.0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_manager.get_stats()
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache_manager.clear()
        if hasattr(self, 'pupil_tracker'):
            self.pupil_tracker.cache_manager.clear()
        if hasattr(self, 'video_loader'):
            self.video_loader.clear_cache()
        
        self.logger.info("All caches cleared")