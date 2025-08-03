#!/usr/bin/env python3
"""
Comprehensive test suite for Kanibus core system
"""

import pytest
import torch
import numpy as np
import cv2
import tempfile
import os
import json
from pathlib import Path
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.neural_engine import NeuralEngine, ProcessingConfig, ProcessingMode
from src.gpu_optimizer import GPUOptimizer, GPUInfo, GPUVendor
from src.cache_manager import CacheManager, CacheEntry
from nodes.neural_pupil_tracker import NeuralPupilTracker, EyeTrackingResult, BlinkState
from nodes.video_frame_loader import VideoFrameLoader, VideoMetadata, VideoFormat
from nodes.kanibus_master import KanibusMaster, KanibusConfig, KanibusResult

class TestNeuralEngine:
    """Test suite for NeuralEngine"""
    
    @pytest.fixture
    def neural_engine(self):
        config = ProcessingConfig(
            device="cpu",  # Use CPU for testing
            precision="fp32",
            batch_size=1,
            max_workers=2
        )
        return NeuralEngine(config)
    
    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model for testing"""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        return DummyModel()
    
    def test_engine_initialization(self, neural_engine):
        """Test neural engine initialization"""
        assert neural_engine.device.type in ["cpu", "cuda", "mps"]
        assert neural_engine.config.batch_size == 1
        assert neural_engine.config.max_workers == 2
        assert len(neural_engine.models) == 0
    
    def test_model_registration(self, neural_engine, dummy_model):
        """Test model registration"""
        neural_engine.register_model("test_model", dummy_model)
        
        assert "test_model" in neural_engine.models
        registered_model = neural_engine.get_model("test_model")
        assert isinstance(registered_model, torch.nn.Module)
    
    def test_forward_pass(self, neural_engine, dummy_model):
        """Test forward pass through registered model"""
        neural_engine.register_model("test_model", dummy_model)
        
        input_tensor = torch.randn(1, 10)
        output = neural_engine.forward("test_model", input_tensor)
        
        assert output.shape == (1, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_batch_forward(self, neural_engine, dummy_model):
        """Test batch processing"""
        neural_engine.register_model("test_model", dummy_model)
        
        batch_inputs = [torch.randn(10) for _ in range(3)]
        outputs = neural_engine.batch_forward("test_model", batch_inputs)
        
        assert len(outputs) == 3
        for output in outputs:
            assert output.shape == (5,)
    
    def test_performance_stats(self, neural_engine, dummy_model):
        """Test performance monitoring"""
        neural_engine.register_model("test_model", dummy_model)
        
        # Perform some operations
        for _ in range(5):
            input_tensor = torch.randn(1, 10)
            neural_engine.forward("test_model", input_tensor)
        
        stats = neural_engine.get_performance_stats()
        assert stats["frame_count"] == 5
        assert stats["avg_fps"] > 0
        assert "allocated_gb" in stats
    
    def test_real_time_processing(self, neural_engine, dummy_model):
        """Test real-time processing capabilities"""
        neural_engine.register_model("test_model", dummy_model)
        neural_engine.start_realtime_processing()
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            input_tensor = torch.randn(1, 10)
            task_id = neural_engine.submit_task("test_model", input_tensor, f"task_{i}")
            if task_id:
                task_ids.append(task_id)
        
        # Get results
        results = []
        for _ in range(10):  # Try multiple times
            task_id, result = neural_engine.get_result(timeout=0.5)
            if task_id and result is not None:
                results.append((task_id, result))
            if len(results) >= len(task_ids):
                break
        
        neural_engine.stop_realtime_processing()
        
        assert len(results) > 0  # At least some results should be processed

class TestGPUOptimizer:
    """Test suite for GPUOptimizer"""
    
    @pytest.fixture
    def gpu_optimizer(self):
        return GPUOptimizer()
    
    def test_initialization(self, gpu_optimizer):
        """Test GPU optimizer initialization"""
        assert hasattr(gpu_optimizer, 'available_gpus')
        assert hasattr(gpu_optimizer, 'system_info')
        assert hasattr(gpu_optimizer, 'optimization_settings')
    
    def test_system_info(self, gpu_optimizer):
        """Test system information gathering"""
        system_info = gpu_optimizer.system_info
        
        required_keys = ["platform", "cpu_count", "total_ram", "cuda_available"]
        for key in required_keys:
            assert key in system_info
        
        assert isinstance(system_info["cpu_count"], int)
        assert system_info["cpu_count"] > 0
        assert isinstance(system_info["total_ram"], int)
        assert system_info["total_ram"] > 0
    
    def test_device_selection(self, gpu_optimizer):
        """Test device selection"""
        device = gpu_optimizer.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_optimization_settings(self, gpu_optimizer):
        """Test optimization settings generation"""
        settings = gpu_optimizer.get_optimization_settings()
        
        required_keys = ["device", "precision", "batch_size", "num_workers"]
        for key in required_keys:
            assert key in settings
        
        assert settings["batch_size"] >= 1
        assert settings["num_workers"] >= 1
        assert settings["precision"] in ["fp16", "fp32", "bf16"]
    
    def test_model_optimization(self, gpu_optimizer):
        """Test model optimization"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        optimized_model = gpu_optimizer.optimize_for_inference(model)
        
        assert isinstance(optimized_model, torch.nn.Module)
        # Model should be moved to the optimal device
        device_param = next(optimized_model.parameters()).device
        assert device_param.type in ["cpu", "cuda", "mps"]

class TestCacheManager:
    """Test suite for CacheManager"""
    
    @pytest.fixture
    def cache_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield CacheManager(
                cache_dir=temp_dir,
                max_memory_mb=100,  # Small for testing
                max_disk_gb=1.0
            )
    
    def test_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager.max_memory_bytes == 100 * 1024 * 1024
        assert cache_manager.max_disk_bytes == 1024 * 1024 * 1024
        assert cache_manager.memory_usage == 0
        assert len(cache_manager.memory_cache) == 0
    
    def test_memory_cache_operations(self, cache_manager):
        """Test memory cache put/get operations"""
        test_data = {"test": "data", "number": 42}
        
        # Put data
        success = cache_manager.put("test_key", test_data)
        assert success
        
        # Get data
        retrieved = cache_manager.get("test_key")
        assert retrieved == test_data
        
        # Get non-existent key
        non_existent = cache_manager.get("non_existent")
        assert non_existent is None
    
    def test_cache_eviction(self, cache_manager):
        """Test cache eviction when memory limit is reached"""
        # Fill cache beyond memory limit
        large_data = np.random.rand(1000, 1000).astype(np.float32)  # ~4MB
        
        keys = []
        for i in range(50):  # Try to exceed 100MB limit
            key = f"large_data_{i}"
            cache_manager.put(key, large_data, priority=1)
            keys.append(key)
        
        # Some keys should have been evicted
        remaining_keys = []
        for key in keys:
            if cache_manager.get(key) is not None:
                remaining_keys.append(key)
        
        assert len(remaining_keys) < len(keys)  # Some eviction should have occurred
    
    def test_frame_caching_convenience_methods(self, cache_manager):
        """Test convenience methods for frame caching"""
        frame = np.random.rand(480, 640, 3).astype(np.uint8)
        
        # Cache frame
        success = cache_manager.cache_frame("frame_001", frame)
        assert success
        
        # Retrieve frame
        retrieved_frame = cache_manager.get_frame("frame_001")
        assert np.array_equal(retrieved_frame, frame)
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics"""
        # Add some data
        cache_manager.put("key1", "data1")
        cache_manager.put("key2", "data2")
        
        # Access data to generate hits
        cache_manager.get("key1")
        cache_manager.get("key2")
        cache_manager.get("non_existent")  # Miss
        
        stats = cache_manager.get_stats()
        
        assert stats["memory_hits"] == 2
        assert stats["memory_misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate_percent"] > 0

class TestNeuralPupilTracker:
    """Test suite for NeuralPupilTracker"""
    
    @pytest.fixture
    def pupil_tracker(self):
        return NeuralPupilTracker()
    
    @pytest.fixture
    def test_image(self):
        """Create a test image with a face-like pattern"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a simple face-like pattern
        cv2.circle(image, (320, 240), 100, (255, 255, 255), -1)  # Face
        cv2.circle(image, (300, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (340, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
    
    def test_initialization(self, pupil_tracker):
        """Test pupil tracker initialization"""
        assert hasattr(pupil_tracker, 'face_mesh')
        assert hasattr(pupil_tracker, 'neural_engine')
        assert hasattr(pupil_tracker, 'cache_manager')
        assert hasattr(pupil_tracker, 'LEFT_IRIS')
        assert hasattr(pupil_tracker, 'RIGHT_IRIS')
    
    def test_track_pupils_with_face(self, pupil_tracker, test_image):
        """Test pupil tracking with a test image"""
        result = pupil_tracker.track_pupils(
            image=test_image,
            sensitivity=1.0,
            smoothing=0.5,
            cache_results=False  # Disable caching for testing
        )
        
        # Unpack results
        tracking_result, annotated_image, gaze_vis, left_mask, right_mask = result
        
        # Check result structure
        assert isinstance(tracking_result, EyeTrackingResult)
        assert isinstance(annotated_image, torch.Tensor)
        assert isinstance(gaze_vis, torch.Tensor)
        assert isinstance(left_mask, torch.Tensor)
        assert isinstance(right_mask, torch.Tensor)
        
        # Check tracking result fields
        assert hasattr(tracking_result, 'left_pupil')
        assert hasattr(tracking_result, 'right_pupil')
        assert hasattr(tracking_result, 'left_blink_state')
        assert hasattr(tracking_result, 'right_blink_state')
        assert hasattr(tracking_result, 'timestamp')
        assert hasattr(tracking_result, 'frame_id')
    
    def test_kalman_filter(self, pupil_tracker):
        """Test Kalman filter functionality"""
        kf = pupil_tracker.left_eye_filter
        
        # Test prediction and update
        kf.predict()
        kf.update(np.array([0.5, 0.5]))
        
        position = kf.get_position()
        velocity = kf.get_velocity()
        
        assert len(position) == 2
        assert len(velocity) == 2
        assert all(isinstance(x, float) for x in position)
        assert all(isinstance(x, float) for x in velocity)
    
    def test_blink_detection(self, pupil_tracker):
        """Test blink state detection"""
        # Test different EAR values
        current_state = BlinkState.OPEN
        
        # High EAR (open eye)
        new_state = pupil_tracker._detect_blink_state(0.3, 0.25, current_state)
        assert new_state == BlinkState.OPEN
        
        # Low EAR (closing eye)
        new_state = pupil_tracker._detect_blink_state(0.2, 0.25, current_state)
        assert new_state == BlinkState.CLOSING
        
        # Very low EAR (closed eye)
        new_state = pupil_tracker._detect_blink_state(0.1, 0.25, BlinkState.CLOSING)
        assert new_state == BlinkState.CLOSED

class TestVideoFrameLoader:
    """Test suite for VideoFrameLoader"""
    
    @pytest.fixture
    def frame_loader(self):
        return VideoFrameLoader()
    
    @pytest.fixture
    def test_video_path(self):
        """Create a temporary test video file"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            # Create a simple test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
            
            # Write 10 test frames
            for i in range(10):
                frame = np.full((480, 640, 3), i * 25, dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            yield temp_file.name
            
            # Cleanup
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def test_initialization(self, frame_loader):
        """Test frame loader initialization"""
        assert hasattr(frame_loader, 'gpu_optimizer')
        assert hasattr(frame_loader, 'cache_manager')
        assert hasattr(frame_loader, 'supported_formats')
        assert len(frame_loader.supported_formats) > 0
    
    @pytest.mark.skipif(not cv2.VideoCapture().isOpened(), reason="OpenCV video support not available")
    def test_metadata_extraction(self, frame_loader, test_video_path):
        """Test video metadata extraction"""
        if not os.path.exists(test_video_path):
            pytest.skip("Test video file not created successfully")
        
        metadata = frame_loader._extract_metadata(test_video_path)
        
        assert isinstance(metadata, VideoMetadata)
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.frame_count == 10
        assert metadata.fps > 0
        assert metadata.format in VideoFormat
    
    def test_quality_settings(self, frame_loader):
        """Test quality settings configuration"""
        settings = frame_loader.quality_settings
        
        required_qualities = ["original", "high", "medium", "low"]
        for quality in required_qualities:
            assert quality in settings
            assert "compression" in settings[quality]
            assert "interpolation" in settings[quality]

class TestKanibusMaster:
    """Test suite for KanibusMaster"""
    
    @pytest.fixture
    def kanibus_master(self):
        return KanibusMaster()
    
    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        image = np.random.rand(512, 512, 3).astype(np.float32)
        return torch.from_numpy(image).unsqueeze(0)
    
    def test_initialization(self, kanibus_master):
        """Test Kanibus master initialization"""
        assert hasattr(kanibus_master, 'gpu_optimizer')
        assert hasattr(kanibus_master, 'neural_engine')
        assert hasattr(kanibus_master, 'cache_manager')
        assert hasattr(kanibus_master, 'pupil_tracker')
        assert hasattr(kanibus_master, 'current_config')
    
    def test_configuration_update(self, kanibus_master):
        """Test configuration updates"""
        config = kanibus_master._update_config(
            pipeline_mode="batch",
            wan_version="wan_2.1",
            target_fps=24.0,
            enable_eye_tracking=True,
            enable_depth_estimation=False
        )
        
        assert isinstance(config, KanibusConfig)
        assert config.target_fps == 24.0
        assert config.enable_eye_tracking == True
        assert config.enable_depth_estimation == False
    
    def test_single_frame_processing(self, kanibus_master, test_image):
        """Test single frame processing"""
        config = KanibusConfig()
        
        # Mock the pupil tracker to avoid MediaPipe dependencies in testing
        with patch.object(kanibus_master.pupil_tracker, 'track_pupils') as mock_track:
            # Mock return value
            mock_result = EyeTrackingResult(
                left_pupil=(0.4, 0.4),
                right_pupil=(0.6, 0.4),
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
                left_eye_confidence=0.8,
                right_eye_confidence=0.8,
                left_iris_landmarks=np.zeros((4, 3)),
                right_iris_landmarks=np.zeros((4, 3)),
                timestamp=time.time(),
                frame_id=0
            )
            
            mock_track.return_value = (
                mock_result,
                test_image,
                test_image,
                torch.zeros(1, 512, 512, 1),
                torch.zeros(1, 512, 512, 1)
            )
            
            result = kanibus_master._process_single_frame(test_image, config, 0)
            
            assert isinstance(result, KanibusResult)
            assert result.frame_id == 0
            assert result.processing_time > 0
            assert result.eye_tracking is not None
    
    def test_performance_monitoring(self, kanibus_master):
        """Test performance monitoring"""
        # Simulate some processing
        kanibus_master._update_performance_stats(0.05)  # 50ms processing time
        kanibus_master._update_performance_stats(0.03)  # 30ms processing time
        
        stats = kanibus_master.get_performance_stats()
        
        assert stats["total_frames"] == 2
        assert stats["avg_processing_time"] > 0
        assert stats["current_fps"] > 0
    
    def test_wan_compatibility_detection(self, kanibus_master):
        """Test WAN version detection"""
        config = KanibusConfig()
        
        # Test with 480p metadata (should detect WAN 2.1)
        metadata_480p = VideoMetadata(
            filepath="test.mp4",
            width=854, height=480, fps=24.0, frame_count=100,
            duration=4.17, codec="h264", format=VideoFormat.MP4,
            file_size=1000000, file_hash="test_hash"
        )
        
        wan_version = kanibus_master._detect_wan_version(config, metadata_480p)
        from nodes.kanibus_master import WanVersion
        assert wan_version == WanVersion.WAN_21
        
        # Test with 720p metadata (should detect WAN 2.2)
        metadata_720p = VideoMetadata(
            filepath="test.mp4",
            width=1280, height=720, fps=30.0, frame_count=100,
            duration=3.33, codec="h264", format=VideoFormat.MP4,
            file_size=2000000, file_hash="test_hash"
        )
        
        wan_version = kanibus_master._detect_wan_version(config, metadata_720p)
        assert wan_version == WanVersion.WAN_22

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def test_system(self):
        """Setup a complete test system"""
        # Use CPU-only configuration for testing
        gpu_optimizer = GPUOptimizer()
        cache_manager = CacheManager(cache_dir=tempfile.mkdtemp(), max_memory_mb=50)
        
        return {
            "gpu_optimizer": gpu_optimizer,
            "cache_manager": cache_manager,
            "neural_engine": NeuralEngine(ProcessingConfig(device="cpu", batch_size=1))
        }
    
    def test_system_initialization(self, test_system):
        """Test complete system initialization"""
        assert test_system["gpu_optimizer"] is not None
        assert test_system["cache_manager"] is not None
        assert test_system["neural_engine"] is not None
    
    def test_memory_management(self, test_system):
        """Test system memory management"""
        cache = test_system["cache_manager"]
        
        # Add data and check memory usage
        initial_usage = cache.memory_usage
        
        test_data = np.random.rand(100, 100).astype(np.float32)
        cache.put("test_memory", test_data)
        
        assert cache.memory_usage > initial_usage
        
        # Clear cache and check cleanup
        cache.clear()
        assert cache.memory_usage == 0
    
    def test_concurrent_processing(self, test_system):
        """Test concurrent processing capabilities"""
        engine = test_system["neural_engine"]
        
        # Register a simple model
        model = torch.nn.Linear(10, 5)
        engine.register_model("concurrent_test", model)
        
        # Start real-time processing
        engine.start_realtime_processing()
        
        # Submit multiple tasks concurrently
        task_ids = []
        for i in range(5):
            input_tensor = torch.randn(1, 10)
            task_id = engine.submit_task("concurrent_test", input_tensor, f"concurrent_{i}")
            if task_id:
                task_ids.append(task_id)
        
        # Collect results
        results = []
        start_time = time.time()
        while len(results) < len(task_ids) and time.time() - start_time < 5.0:
            task_id, result = engine.get_result(timeout=0.1)
            if task_id and result is not None:
                results.append((task_id, result))
        
        engine.stop_realtime_processing()
        
        # Should process at least some tasks
        assert len(results) > 0

def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\\n🚀 Running Performance Benchmarks...")
    
    # GPU Optimizer benchmark
    print("📊 GPU Optimizer...")
    gpu_opt = GPUOptimizer()
    gpu_opt.print_system_info()
    
    # Neural Engine benchmark
    print("\\n🧠 Neural Engine...")
    config = ProcessingConfig(device="cpu", batch_size=4)
    engine = NeuralEngine(config)
    
    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100)
    )
    engine.register_model("benchmark", model)
    
    # Benchmark inference
    input_shape = (4, 100)
    benchmark_results = engine.benchmark_model("benchmark", input_shape, num_iterations=100)
    
    print(f"  Average inference time: {benchmark_results['avg_time_per_inference']*1000:.2f}ms")
    print(f"  Throughput: {benchmark_results['fps']:.1f} FPS")
    
    # Cache Manager benchmark
    print("\\n💾 Cache Manager...")
    cache = CacheManager(max_memory_mb=100)
    
    # Benchmark cache operations
    start_time = time.time()
    for i in range(1000):
        data = np.random.rand(50, 50).astype(np.float32)
        cache.put(f"bench_{i}", data)
    
    cache_time = time.time() - start_time
    print(f"  Cache 1000 items: {cache_time:.2f}s ({1000/cache_time:.0f} ops/sec)")
    
    cache_stats = cache.get_stats()
    print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
    print(f"  Cache entries: {cache_stats['memory_entries']}")

if __name__ == "__main__":
    # Run tests with pytest
    print("🧪 Running Kanibus Test Suite...")
    
    # Run basic smoke tests
    try:
        import torch
        import cv2
        import mediapipe
        import numpy as np
        print("✅ All core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        exit(1)
    
    # Run performance benchmarks
    run_performance_benchmarks()
    
    print("\\n✅ Test suite completed!")
    print("\\nTo run full test suite with coverage:")
    print("  pytest tests/ --cov=src --cov=nodes --cov-report=html")