"""
Video Frame Loader - Load videos and extract frames with intelligent caching
"""

import torch
import numpy as np
import cv2
import os
import hashlib
import threading
from typing import Dict, List, Tuple, Optional, Any, Generator
import logging
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

# Import our core system
try:
    from ..src.cache_manager import CacheManager
    from ..src.gpu_optimizer import GPUOptimizer
except ImportError:
    # Fallback for development/testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.cache_manager import CacheManager
    from src.gpu_optimizer import GPUOptimizer

class VideoFormat(Enum):
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"

@dataclass
class VideoMetadata:
    """Video file metadata"""
    filepath: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    format: VideoFormat
    file_size: int
    file_hash: str

@dataclass
class FrameInfo:
    """Individual frame information"""
    frame_number: int
    timestamp: float
    frame_data: np.ndarray
    is_keyframe: bool = False
    compression_ratio: float = 1.0

class VideoFrameLoader:
    """
    Advanced video frame loader with intelligent caching, batch processing,
    and GPU-optimized frame extraction for real-time eye-tracking workflows.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "frame_count": ("INT", {"default": -1, "min": -1, "max": 999999}),  # -1 = all frames
                "step": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "target_fps": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 120.0}),  # -1 = original fps
                "resize_width": ("INT", {"default": -1, "min": -1, "max": 4096}),  # -1 = original
                "resize_height": ("INT", {"default": -1, "min": -1, "max": 4096}),  # -1 = original
                "quality": (["original", "high", "medium", "low"], {"default": "high"}),
                "enable_caching": ("BOOLEAN", {"default": True}),
                
                # WAN compatibility settings
                "wan_version": (["auto", "wan_2.1", "wan_2.2"], {"default": "auto"}),
                "optimize_for_wan": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "preload_frames": ("INT", {"default": 32, "min": 0, "max": 256}),
                "color_space": (["RGB", "BGR", "GRAY", "HSV", "LAB"], {"default": "RGB"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO_METADATA", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "metadata", "total_frames", "actual_fps")
    OUTPUT_IS_LIST = (True, False, False, False)  # frames is a list
    FUNCTION = "load_video_frames"
    CATEGORY = "Kanibus/Input"
    
    def __init__(self):
        # Initialize core systems
        self.gpu_optimizer = GPUOptimizer()
        self.cache_manager = CacheManager(
            cache_dir="./cache/video_frames",
            max_memory_mb=4096,  # 4GB for frame caching
            max_disk_gb=20.0     # 20GB disk cache
        )
        
        # Video capture pool for efficient loading
        self.capture_pool: Dict[str, cv2.VideoCapture] = {}
        self.capture_lock = threading.Lock()
        
        # Supported formats
        self.supported_formats = {
            '.mp4': VideoFormat.MP4,
            '.avi': VideoFormat.AVI,
            '.mov': VideoFormat.MOV,
            '.mkv': VideoFormat.MKV,
            '.webm': VideoFormat.WEBM,
            '.flv': VideoFormat.FLV
        }
        
        # Quality settings
        self.quality_settings = {
            "original": {"compression": 0, "interpolation": cv2.INTER_LANCZOS4},
            "high": {"compression": 5, "interpolation": cv2.INTER_CUBIC},
            "medium": {"compression": 15, "interpolation": cv2.INTER_LINEAR},
            "low": {"compression": 30, "interpolation": cv2.INTER_NEAREST}
        }
        
        # Performance monitoring
        self.stats = {
            "frames_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_load_time": 0.0,
            "avg_fps": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("VideoFrameLoader initialized")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for video file for caching"""
        stat = os.stat(filepath)
        # Use file path, size, and modification time for hash
        hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _get_video_capture(self, filepath: str) -> cv2.VideoCapture:
        """Get or create video capture object"""
        with self.capture_lock:
            if filepath not in self.capture_pool:
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video file: {filepath}")
                self.capture_pool[filepath] = cap
            return self.capture_pool[filepath]
    
    def _extract_metadata(self, filepath: str) -> VideoMetadata:
        """Extract video metadata"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        # Check cache first
        file_hash = self._get_file_hash(filepath)
        cached_metadata = self.cache_manager.get(f"metadata_{file_hash}")
        if cached_metadata:
            return cached_metadata
        
        cap = self._get_video_capture(filepath)
        
        # Extract metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        
        # Get codec (fourcc)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        # Determine format from extension
        ext = Path(filepath).suffix.lower()
        video_format = self.supported_formats.get(ext, VideoFormat.MP4)
        
        # File size
        file_size = os.path.getsize(filepath)
        
        metadata = VideoMetadata(
            filepath=filepath,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            format=video_format,
            file_size=file_size,
            file_hash=file_hash
        )
        
        # Cache metadata
        self.cache_manager.put(f"metadata_{file_hash}", metadata, priority=3, ttl=3600)  # 1 hour TTL
        
        return metadata
    
    def _process_frame(self, frame: np.ndarray, resize_width: int, resize_height: int,
                      color_space: str, quality: str) -> np.ndarray:
        """Process individual frame with resize and color conversion"""
        processed = frame.copy()
        
        # Resize if needed
        if resize_width > 0 and resize_height > 0:
            current_h, current_w = processed.shape[:2]
            if current_w != resize_width or current_h != resize_height:
                interpolation = self.quality_settings[quality]["interpolation"]
                processed = cv2.resize(processed, (resize_width, resize_height), interpolation=interpolation)
        
        # Color space conversion
        if color_space == "RGB" and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        elif color_space == "GRAY":
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                processed = np.expand_dims(processed, axis=2)  # Keep 3D for consistency
        elif color_space == "HSV":
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        elif color_space == "LAB":
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        # BGR is default, no conversion needed
        
        return processed
    
    def _optimize_for_wan(self, wan_version: str, width: int, height: int, fps: float) -> tuple:
        """Optimize video parameters for WAN compatibility"""
        if wan_version == "wan_2.1":
            # WAN 2.1 optimization: 480p, 24fps
            optimal_width = 854
            optimal_height = 480
            optimal_fps = 24.0
            
            # Apply optimization if not manually overridden
            if width <= 0:  # Auto-resize
                width = optimal_width
            if height <= 0:  # Auto-resize  
                height = optimal_height
            if fps <= 0:  # Auto-fps
                fps = optimal_fps
                
            return width, height, fps, "WAN 2.1 optimized (854x480@24fps)"
            
        elif wan_version == "wan_2.2":
            # WAN 2.2 optimization: 720p, 30fps
            optimal_width = 1280
            optimal_height = 720
            optimal_fps = 30.0
            
            # Apply optimization if not manually overridden
            if width <= 0:  # Auto-resize
                width = optimal_width
            if height <= 0:  # Auto-resize
                height = optimal_height
            if fps <= 0:  # Auto-fps
                fps = optimal_fps
                
            return width, height, fps, "WAN 2.2 optimized (1280x720@30fps)"
            
        else:
            # Auto-detect based on input resolution
            if width > 0 and height > 0:
                # Use provided resolution to determine WAN version
                if width <= 854 and height <= 480:
                    return self._optimize_for_wan("wan_2.1", width, height, fps)
                else:
                    return self._optimize_for_wan("wan_2.2", width, height, fps)
            
            # Default to WAN 2.2 for better quality
            return self._optimize_for_wan("wan_2.2", width, height, fps)
    
    def _load_frame_batch(self, cap: cv2.VideoCapture, frame_numbers: List[int],
                         metadata: VideoMetadata, resize_width: int, resize_height: int,
                         color_space: str, quality: str, enable_caching: bool) -> List[np.ndarray]:
        """Load a batch of frames efficiently"""
        frames = []
        
        for frame_num in frame_numbers:
            # Check cache first
            if enable_caching:
                cache_key = f"frame_{metadata.file_hash}_{frame_num}_{resize_width}x{resize_height}_{color_space}_{quality}"
                cached_frame = self.cache_manager.get_frame(cache_key)
                if cached_frame is not None:
                    frames.append(cached_frame)
                    self.stats["cache_hits"] += 1
                    continue
            
            # Load frame from video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                self.logger.warning(f"Could not read frame {frame_num}")
                # Use last valid frame or create blank frame
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((metadata.height, metadata.width, 3), dtype=np.uint8)
            
            # Process frame
            processed_frame = self._process_frame(frame, resize_width, resize_height, color_space, quality)
            frames.append(processed_frame)
            
            # Cache processed frame
            if enable_caching:
                self.cache_manager.cache_frame(cache_key, processed_frame, priority=1)
            
            self.stats["cache_misses"] += 1
            self.stats["frames_loaded"] += 1
        
        return frames
    
    def _calculate_frame_selection(self, metadata: VideoMetadata, start_frame: int,
                                 frame_count: int, step: int, target_fps: float) -> List[int]:
        """Calculate which frames to extract"""
        total_frames = metadata.frame_count
        
        # Adjust start frame
        start_frame = max(0, min(start_frame, total_frames - 1))
        
        # Calculate end frame
        if frame_count == -1:
            end_frame = total_frames
        else:
            end_frame = min(start_frame + frame_count, total_frames)
        
        # Generate base frame list with step
        base_frames = list(range(start_frame, end_frame, step))
        
        # Apply FPS adjustment if specified
        if target_fps > 0 and target_fps != metadata.fps:
            # Calculate frame skip to achieve target FPS
            fps_ratio = metadata.fps / target_fps
            
            if fps_ratio > 1:  # Need to skip frames
                adjusted_frames = []
                for i, frame_num in enumerate(base_frames):
                    if i % int(fps_ratio) == 0:
                        adjusted_frames.append(frame_num)
                return adjusted_frames
            elif fps_ratio < 1:  # Need to duplicate frames
                adjusted_frames = []
                duplication_factor = int(target_fps / metadata.fps)
                for frame_num in base_frames:
                    for _ in range(duplication_factor):
                        adjusted_frames.append(frame_num)
                return adjusted_frames
        
        return base_frames
    
    def _preload_frames(self, cap: cv2.VideoCapture, frame_numbers: List[int],
                       metadata: VideoMetadata, preload_count: int, **process_kwargs):
        """Preload frames in background thread"""
        if preload_count <= 0 or len(frame_numbers) <= preload_count:
            return
        
        def preload_worker():
            preload_frames = frame_numbers[:preload_count]
            try:
                self._load_frame_batch(cap, preload_frames, metadata, **process_kwargs)
                self.logger.debug(f"Preloaded {len(preload_frames)} frames")
            except Exception as e:
                self.logger.error(f"Preload error: {e}")
        
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
    
    def load_video_frames(self, video_path: str, start_frame: int = 0, frame_count: int = -1,
                         step: int = 1, target_fps: float = -1.0, resize_width: int = -1,
                         resize_height: int = -1, quality: str = "high", enable_caching: bool = True,
                         wan_version: str = "auto", optimize_for_wan: bool = True,
                         batch_size: int = 8, preload_frames: int = 32, color_space: str = "RGB"):
        """Load video frames with all optimizations"""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not video_path or not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract metadata
            metadata = self._extract_metadata(video_path)
            self.logger.info(f"Loading video: {metadata.width}x{metadata.height}, {metadata.fps:.1f}fps, {metadata.frame_count} frames")
            
            # Get video capture
            cap = self._get_video_capture(video_path)
            
            # Calculate frame selection
            selected_frames = self._calculate_frame_selection(metadata, start_frame, frame_count, step, target_fps)
            
            if not selected_frames:
                raise ValueError("No frames selected for loading")
            
            # Apply WAN optimizations if enabled
            if optimize_for_wan:
                resize_width, resize_height, target_fps, wan_info = self._optimize_for_wan(
                    wan_version, resize_width, resize_height, target_fps
                )
                self.logger.info(f"WAN optimization applied: {wan_info}")
            
            # Determine actual output dimensions
            output_width = resize_width if resize_width > 0 else metadata.width
            output_height = resize_height if resize_height > 0 else metadata.height
            
            # Start preloading frames
            process_kwargs = {
                "resize_width": resize_width,
                "resize_height": resize_height,
                "color_space": color_space,
                "quality": quality,
                "enable_caching": enable_caching
            }
            
            self._preload_frames(cap, selected_frames, metadata, preload_frames, **process_kwargs)
            
            # Load frames in batches
            all_frames = []
            total_batches = (len(selected_frames) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(selected_frames))
                batch_frame_numbers = selected_frames[batch_start:batch_end]
                
                # Load batch
                batch_frames = self._load_frame_batch(
                    cap, batch_frame_numbers, metadata, **process_kwargs
                )
                
                all_frames.extend(batch_frames)
                
                # Progress logging
                if batch_idx % 10 == 0:
                    progress = (batch_idx + 1) / total_batches * 100
                    self.logger.debug(f"Loading progress: {progress:.1f}% ({len(all_frames)} frames)")
            
            # Convert frames to tensor format
            frame_tensors = []
            for frame in all_frames:
                # Normalize to 0-1 range
                frame_normalized = frame.astype(np.float32) / 255.0
                # Convert to tensor (H, W, C)
                frame_tensor = torch.from_numpy(frame_normalized)
                frame_tensors.append(frame_tensor)
            
            # Calculate actual FPS
            actual_fps = target_fps if target_fps > 0 else metadata.fps
            if step > 1:
                actual_fps = actual_fps / step
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats["total_load_time"] += total_time
            self.stats["avg_fps"] = len(all_frames) / total_time
            
            self.logger.info(f"Loaded {len(frame_tensors)} frames in {total_time:.2f}s ({self.stats['avg_fps']:.1f} fps)")
            
            return (frame_tensors, metadata, len(frame_tensors), actual_fps)
            
        except Exception as e:
            self.logger.error(f"Error loading video frames: {e}")
            # Return empty result
            empty_frame = torch.zeros((1, 1, 3), dtype=torch.float32)
            empty_metadata = VideoMetadata(
                filepath=video_path,
                width=1, height=1, fps=1.0, frame_count=0, duration=0.0,
                codec="", format=VideoFormat.MP4, file_size=0, file_hash=""
            )
            return ([empty_frame], empty_metadata, 0, 1.0)
    
    def get_frame_at_time(self, video_path: str, timestamp: float, **kwargs) -> torch.Tensor:
        """Get a single frame at specific timestamp"""
        metadata = self._extract_metadata(video_path)
        frame_number = int(timestamp * metadata.fps)
        frame_number = max(0, min(frame_number, metadata.frame_count - 1))
        
        # Load single frame
        result = self.load_video_frames(
            video_path=video_path,
            start_frame=frame_number,
            frame_count=1,
            **kwargs
        )
        
        frames, _, _, _ = result
        return frames[0] if frames else torch.zeros((1, 1, 3), dtype=torch.float32)
    
    def get_frame_range(self, video_path: str, start_time: float, end_time: float, **kwargs) -> List[torch.Tensor]:
        """Get frames in a time range"""
        metadata = self._extract_metadata(video_path)
        start_frame = int(start_time * metadata.fps)
        end_frame = int(end_time * metadata.fps)
        frame_count = end_frame - start_frame
        
        result = self.load_video_frames(
            video_path=video_path,
            start_frame=start_frame,
            frame_count=frame_count,
            **kwargs
        )
        
        frames, _, _, _ = result
        return frames
    
    def clear_cache(self):
        """Clear frame cache"""
        self.cache_manager.clear()
        self.logger.info("Video frame cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        cache_stats = self.cache_manager.get_stats()
        return {
            **self.stats,
            "cache_hit_rate": cache_stats["hit_rate_percent"],
            "cache_memory_usage": cache_stats["memory_usage_mb"],
            "cache_disk_usage": cache_stats["disk_usage_mb"]
        }
    
    def __del__(self):
        """Cleanup video captures"""
        with self.capture_lock:
            for cap in self.capture_pool.values():
                cap.release()
            self.capture_pool.clear()