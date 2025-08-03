"""
Neural Processing Engine - Core GPU-optimized processing system
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum

class ProcessingMode(Enum):
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class ProcessingConfig:
    """Configuration for neural processing"""
    device: str = "cuda"
    precision: str = "fp16"  # fp16, fp32, bf16
    batch_size: int = 1
    max_workers: int = 4
    memory_limit: float = 0.8  # 80% of GPU memory
    enable_tensorrt: bool = True
    enable_onnx: bool = False
    mode: ProcessingMode = ProcessingMode.REALTIME

class NeuralEngine:
    """
    Core neural processing engine with GPU optimization, multi-threading,
    and automatic memory management for real-time eye-tracking.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.scaler = GradScaler() if self.config.precision == "fp16" else None
        
        # Performance monitoring
        self.performance_stats = {
            "frame_count": 0,
            "total_time": 0.0,
            "avg_fps": 0.0,
            "memory_usage": 0.0,
            "gpu_utilization": 0.0
        }
        
        # Threading for real-time processing
        self.processing_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self.workers = []
        self.running = False
        
        # Memory management
        self._setup_memory_management()
        
        # Initialize CUDA streams for parallel processing
        if self.device.type == "cuda":
            self.streams = [torch.cuda.Stream() for _ in range(self.config.max_workers)]
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"NeuralEngine initialized on {self.device}")
    
    def _setup_memory_management(self):
        """Setup automatic memory management"""
        if self.device.type == "cuda":
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.memory_limit)
            
            # Enable memory caching allocator
            torch.cuda.empty_cache()
            
            # Setup memory pool
            if hasattr(torch.cuda, 'memory_pool'):
                self.memory_pool = torch.cuda.memory_pool()
    
    def register_model(self, name: str, model: nn.Module, 
                      optimizer: torch.optim.Optimizer = None,
                      scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """Register a model with the engine"""
        model = model.to(self.device)
        
        # Enable mixed precision if configured
        if self.config.precision == "fp16":
            model = model.half()
        elif self.config.precision == "bf16" and hasattr(torch, "bfloat16"):
            model = model.to(torch.bfloat16)
        
        # Compile model for optimization (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info(f"Model {name} compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Failed to compile model {name}: {e}")
        
        self.models[name] = model
        if optimizer:
            self.optimizers[name] = optimizer
        if scheduler:
            self.schedulers[name] = scheduler
        
        self.logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> nn.Module:
        """Get a registered model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        return self.models[name]
    
    @autocast(enabled=True)
    def forward(self, model_name: str, inputs: torch.Tensor, 
               **kwargs) -> torch.Tensor:
        """Forward pass with automatic mixed precision"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        model = self.models[model_name]
        inputs = inputs.to(self.device, non_blocking=True)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(inputs, **kwargs)
        
        # Update performance stats
        self.performance_stats["frame_count"] += 1
        frame_time = time.time() - start_time
        self.performance_stats["total_time"] += frame_time
        self.performance_stats["avg_fps"] = self.performance_stats["frame_count"] / self.performance_stats["total_time"]
        
        return outputs
    
    def batch_forward(self, model_name: str, batch_inputs: List[torch.Tensor],
                     **kwargs) -> List[torch.Tensor]:
        """Batch processing with dynamic batching"""
        if not batch_inputs:
            return []
        
        # Stack inputs into batch
        batch_tensor = torch.stack(batch_inputs).to(self.device, non_blocking=True)
        
        # Process batch
        batch_outputs = self.forward(model_name, batch_tensor, **kwargs)
        
        # Split outputs back to list
        if isinstance(batch_outputs, torch.Tensor):
            return list(torch.unbind(batch_outputs, dim=0))
        else:
            # Handle multiple outputs
            return [list(torch.unbind(output, dim=0)) for output in batch_outputs]
    
    def start_realtime_processing(self):
        """Start real-time processing threads"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.config.max_workers} processing workers")
    
    def stop_realtime_processing(self):
        """Stop real-time processing threads"""
        self.running = False
        
        # Clear queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        self.workers.clear()
        self.logger.info("Stopped processing workers")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread loop for real-time processing"""
        stream = self.streams[worker_id] if self.device.type == "cuda" else None
        
        while self.running:
            try:
                # Get task from queue
                task = self.processing_queue.get(timeout=0.1)
                
                if task is None:  # Shutdown signal
                    break
                
                model_name, inputs, task_id, kwargs = task
                
                # Process with CUDA stream if available
                if stream:
                    with torch.cuda.stream(stream):
                        result = self.forward(model_name, inputs, **kwargs)
                        stream.synchronize()
                else:
                    result = self.forward(model_name, inputs, **kwargs)
                
                # Put result in result queue
                self.result_queue.put((task_id, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                # Put error result
                self.result_queue.put((task.get('task_id', 'unknown'), None))
    
    def submit_task(self, model_name: str, inputs: torch.Tensor, 
                   task_id: str = None, **kwargs) -> str:
        """Submit a task for real-time processing"""
        if not self.running:
            self.start_realtime_processing()
        
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        task = (model_name, inputs, task_id, kwargs)
        
        try:
            self.processing_queue.put(task, block=False)
            return task_id
        except queue.Full:
            self.logger.warning("Processing queue full, dropping frame")
            return None
    
    def get_result(self, timeout: float = 0.1) -> Tuple[str, torch.Tensor]:
        """Get processed result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "utilization": allocated / torch.cuda.get_device_properties(0).total_memory * 1024**3
            }
        else:
            return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0, "utilization": 0}
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats.update(self.get_memory_usage())
        return stats
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_memory(self):
        """Optimize memory usage"""
        self.clear_cache()
        
        # Defragment memory if available
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats()
            fragmentation = stats.get('segment.all.current', 0)
            if fragmentation > 0:
                self.logger.info("Defragmenting GPU memory")
                torch.cuda.empty_cache()
    
    def export_to_tensorrt(self, model_name: str, example_input: torch.Tensor,
                          output_path: str):
        """Export model to TensorRT for optimization"""
        if not self.config.enable_tensorrt:
            raise RuntimeError("TensorRT not enabled in config")
        
        try:
            import torch_tensorrt
            
            model = self.models[model_name]
            model.eval()
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16} if self.config.precision == "fp16" else {torch.float32}
            )
            
            # Save compiled model
            torch.jit.save(trt_model, output_path)
            self.logger.info(f"Model {model_name} exported to TensorRT: {output_path}")
            
        except ImportError:
            self.logger.error("TensorRT not available. Install torch-tensorrt")
            raise
    
    def export_to_onnx(self, model_name: str, example_input: torch.Tensor,
                      output_path: str):
        """Export model to ONNX format"""
        model = self.models[model_name]
        model.eval()
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Model {model_name} exported to ONNX: {output_path}")
    
    def benchmark_model(self, model_name: str, input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        model = self.models[model_name]
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.forward(model_name, dummy_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = self.forward(model_name, dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        return {
            "total_time": total_time,
            "avg_time_per_inference": avg_time,
            "fps": fps,
            "memory_usage": self.get_memory_usage()
        }
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_realtime_processing()
        self.clear_cache()