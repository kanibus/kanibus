"""
GPU Optimizer - Automatic GPU detection, optimization, and resource management
"""

import torch
import psutil
import platform

# GPUtil import with fallback
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("⚠️  GPUtil not available - GPU monitoring limited")
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

class GPUVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"

@dataclass
class GPUInfo:
    """GPU information structure"""
    id: int
    name: str
    vendor: GPUVendor
    total_memory: int  # MB
    free_memory: int   # MB
    utilization: float # Percentage
    temperature: float # Celsius
    power_usage: float # Watts
    compute_capability: Optional[Tuple[int, int]] = None
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_tensorrt: bool = False

class GPUOptimizer:
    """
    Automatic GPU detection, optimization, and resource management
    for maximum eye-tracking performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_gpus: List[GPUInfo] = []
        self.selected_gpu: Optional[GPUInfo] = None
        self.optimization_settings = {}
        
        # Detect system configuration
        self.system_info = self._get_system_info()
        self._detect_gpus()
        self._select_optimal_gpu()
        self._configure_optimization()
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "total_ram": psutil.virtual_memory().total // (1024**3),  # GB
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    
    def _detect_gpus(self):
        """Detect available GPUs"""
        self.available_gpus = []
        
        if torch.cuda.is_available():
            # NVIDIA GPUs via PyTorch CUDA
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get additional info via GPUtil if available
                if HAS_GPUTIL:
                    try:
                        gpus = GPUtil.getGPUs()
                        gpu_util = gpus[i] if i < len(gpus) else None
                        utilization = gpu_util.load * 100 if gpu_util else 0.0
                        temperature = gpu_util.temperature if gpu_util else 0.0
                        power_usage = getattr(gpu_util, 'powerDraw', 0.0) if gpu_util else 0.0
                    except:
                        utilization = temperature = power_usage = 0.0
                else:
                    utilization = temperature = power_usage = 0.0
                
                gpu_info = GPUInfo(
                    id=i,
                    name=props.name,
                    vendor=GPUVendor.NVIDIA,
                    total_memory=props.total_memory // (1024**2),  # MB
                    free_memory=torch.cuda.mem_get_info(i)[0] // (1024**2),  # MB
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=(props.major, props.minor),
                    supports_fp16=props.major >= 7,  # Volta+
                    supports_int8=props.major >= 6,  # Pascal+
                    supports_tensorrt=props.major >= 6
                )
                
                self.available_gpus.append(gpu_info)
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info = GPUInfo(
                id=0,
                name="Apple Silicon GPU",
                vendor=GPUVendor.APPLE,
                total_memory=self.system_info["total_ram"] * 1024,  # Unified memory
                free_memory=psutil.virtual_memory().available // (1024**2),
                utilization=0.0,
                temperature=0.0,
                power_usage=0.0,
                supports_fp16=True,
                supports_int8=True,
                supports_tensorrt=False
            )
            self.available_gpus.append(gpu_info)
        
        # AMD ROCm support (if available)
        try:
            if hasattr(torch, 'hip') and torch.hip.is_available():
                for i in range(torch.hip.device_count()):
                    gpu_info = GPUInfo(
                        id=i,
                        name=f"AMD GPU {i}",
                        vendor=GPUVendor.AMD,
                        total_memory=0,  # Would need ROCm specific calls
                        free_memory=0,
                        utilization=0.0,
                        temperature=0.0,
                        power_usage=0.0,
                        supports_fp16=True,
                        supports_int8=True,
                        supports_tensorrt=False
                    )
                    self.available_gpus.append(gpu_info)
        except:
            pass
        
        self.logger.info(f"Detected {len(self.available_gpus)} GPU(s)")
        for gpu in self.available_gpus:
            self.logger.info(f"  GPU {gpu.id}: {gpu.name} ({gpu.vendor.value}) - {gpu.total_memory}MB")
    
    def _select_optimal_gpu(self):
        """Select the optimal GPU for eye-tracking"""
        if not self.available_gpus:
            self.logger.warning("No GPUs available, using CPU")
            return
        
        # Scoring function for GPU selection
        def score_gpu(gpu: GPUInfo) -> float:
            score = 0.0
            
            # Memory score (40% weight)
            memory_score = min(gpu.free_memory / 8192, 1.0)  # 8GB baseline
            score += memory_score * 0.4
            
            # Compute capability score (30% weight)
            if gpu.compute_capability:
                cc_score = min((gpu.compute_capability[0] * 10 + gpu.compute_capability[1]) / 86, 1.0)  # 8.6 baseline
                score += cc_score * 0.3
            elif gpu.vendor == GPUVendor.APPLE:
                score += 0.25  # Apple Silicon is quite capable
            
            # Feature support score (20% weight)
            feature_score = 0.0
            if gpu.supports_fp16:
                feature_score += 0.5
            if gpu.supports_tensorrt:
                feature_score += 0.3
            if gpu.supports_int8:
                feature_score += 0.2
            score += feature_score * 0.2
            
            # Utilization penalty (10% weight)
            utilization_penalty = gpu.utilization / 100.0
            score += (1.0 - utilization_penalty) * 0.1
            
            return score
        
        # Select GPU with highest score
        best_gpu = max(self.available_gpus, key=score_gpu)
        self.selected_gpu = best_gpu
        
        # Set PyTorch device
        if best_gpu.vendor == GPUVendor.NVIDIA:
            torch.cuda.set_device(best_gpu.id)
        elif best_gpu.vendor == GPUVendor.APPLE:
            # MPS will be used automatically
            pass
        
        self.logger.info(f"Selected GPU: {best_gpu.name} (Score: {score_gpu(best_gpu):.2f})")
    
    def _configure_optimization(self):
        """Configure optimization settings based on selected GPU"""
        if not self.selected_gpu:
            self.optimization_settings = {
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 1,
                "num_workers": min(4, self.system_info["cpu_count"]),
                "memory_limit": 0.5,
                "enable_tensorrt": False,
                "enable_cudnn": False
            }
            return
        
        gpu = self.selected_gpu
        settings = {
            "device": self._get_device_string(),
            "batch_size": self._calculate_optimal_batch_size(),
            "num_workers": self._calculate_optimal_workers(),
            "memory_limit": self._calculate_memory_limit(),
            "enable_tensorrt": gpu.supports_tensorrt,
            "enable_cudnn": gpu.vendor == GPUVendor.NVIDIA
        }
        
        # Precision selection
        if gpu.supports_fp16 and gpu.total_memory >= 6144:  # 6GB+ for stable FP16
            settings["precision"] = "fp16"
        else:
            settings["precision"] = "fp32"
        
        # NVIDIA specific optimizations
        if gpu.vendor == GPUVendor.NVIDIA:
            settings.update({
                "enable_amp": gpu.supports_fp16,
                "enable_channels_last": gpu.compute_capability and gpu.compute_capability[0] >= 7,
                "enable_flash_attention": gpu.compute_capability and gpu.compute_capability[0] >= 8,
                "enable_nvfuser": True
            })
        
        # Apple Silicon optimizations
        elif gpu.vendor == GPUVendor.APPLE:
            settings.update({
                "enable_amp": True,
                "enable_metal_performance_shaders": True,
                "unified_memory": True
            })
        
        self.optimization_settings = settings
        self.logger.info(f"Optimization settings: {settings}")
    
    def _get_device_string(self) -> str:
        """Get PyTorch device string"""
        if not self.selected_gpu:
            return "cpu"
        
        if self.selected_gpu.vendor == GPUVendor.NVIDIA:
            return f"cuda:{self.selected_gpu.id}"
        elif self.selected_gpu.vendor == GPUVendor.APPLE:
            return "mps"
        elif self.selected_gpu.vendor == GPUVendor.AMD:
            return f"hip:{self.selected_gpu.id}"
        else:
            return "cpu"
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if not self.selected_gpu:
            return 1
        
        memory_mb = self.selected_gpu.free_memory
        
        # Conservative batch size calculation for eye-tracking
        # Assumes ~200MB per frame for full pipeline
        if memory_mb >= 16384:  # 16GB+
            return 8
        elif memory_mb >= 8192:  # 8GB+
            return 4
        elif memory_mb >= 4096:  # 4GB+
            return 2
        else:
            return 1
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads"""
        if not self.selected_gpu:
            return min(4, self.system_info["cpu_count"])
        
        # Balance between CPU cores and GPU capability
        gpu_workers = 4 if self.selected_gpu.total_memory >= 8192 else 2
        cpu_workers = min(self.system_info["cpu_count"] // 2, 8)
        
        return min(gpu_workers, cpu_workers)
    
    def _calculate_memory_limit(self) -> float:
        """Calculate safe memory limit"""
        if not self.selected_gpu:
            return 0.5
        
        # Leave some memory for system and other processes
        if self.selected_gpu.total_memory >= 16384:  # 16GB+
            return 0.9
        elif self.selected_gpu.total_memory >= 8192:  # 8GB+
            return 0.8
        else:
            return 0.7
    
    def apply_optimizations(self):
        """Apply all optimizations to PyTorch"""
        settings = self.optimization_settings
        
        # Enable cuDNN optimizations
        if settings.get("enable_cudnn", False):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable TensorFloat-32 (TF32) on Ampere GPUs
        if self.selected_gpu and self.selected_gpu.compute_capability and self.selected_gpu.compute_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Memory optimizations
        if settings.get("device", "").startswith("cuda"):
            # Enable memory pool
            torch.cuda.empty_cache()
            
            # Set memory fraction
            if "memory_limit" in settings:
                torch.cuda.set_per_process_memory_fraction(settings["memory_limit"])
        
        # Enable optimized attention (if available)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # This enables Flash Attention and other optimizations automatically
            pass
        
        self.logger.info("Applied GPU optimizations")
    
    def get_device(self) -> torch.device:
        """Get the optimal PyTorch device"""
        device_str = self.optimization_settings.get("device", "cpu")
        return torch.device(device_str)
    
    def get_optimization_settings(self) -> Dict:
        """Get current optimization settings"""
        return self.optimization_settings.copy()
    
    def benchmark_gpu(self, model: torch.nn.Module, input_shape: Tuple[int, ...],
                     num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark GPU performance with given model"""
        device = self.get_device()
        model = model.to(device)
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        return {
            "device": str(device),
            "total_time": total_time,
            "avg_time_per_inference": avg_time,
            "fps": fps,
            "memory_allocated": torch.cuda.memory_allocated() if device.type == "cuda" else 0,
            "memory_reserved": torch.cuda.memory_reserved() if device.type == "cuda" else 0
        }
    
    def monitor_gpu_usage(self) -> Dict[str, float]:
        """Monitor current GPU usage"""
        if not self.selected_gpu or self.selected_gpu.vendor != GPUVendor.NVIDIA or not HAS_GPUTIL:
            return {}
        
        try:
            gpus = GPUtil.getGPUs()
            if self.selected_gpu.id < len(gpus):
                gpu = gpus[self.selected_gpu.id]
                return {
                    "utilization": gpu.load * 100,
                    "memory_utilization": gpu.memoryUtil * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                    "power_draw": getattr(gpu, 'powerDraw', 0.0)
                }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU usage: {e}")
        
        return {}
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information"""
        if not self.selected_gpu:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        if self.selected_gpu.vendor == GPUVendor.NVIDIA:
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "total": torch.cuda.get_device_properties(0).total_memory
            }
        else:
            return {"allocated": 0, "reserved": 0, "total": self.selected_gpu.total_memory * 1024 * 1024}
    
    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        model.eval()
        
        # Move to optimal device
        model = model.to(self.get_device())
        
        # Apply precision optimization
        precision = self.optimization_settings.get("precision", "fp32")
        if precision == "fp16":
            model = model.half()
        elif precision == "bf16" and hasattr(torch, "bfloat16"):
            model = model.to(torch.bfloat16)
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")
        
        # Enable channels-last memory format for better performance
        if self.optimization_settings.get("enable_channels_last", False):
            try:
                model = model.to(memory_format=torch.channels_last)
                self.logger.info("Enabled channels-last memory format")
            except:
                pass
        
        return model
    
    def create_optimizer(self, model_parameters, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
        """Create optimized optimizer based on GPU capabilities"""
        if self.selected_gpu and self.selected_gpu.total_memory >= 8192:
            # Use AdamW for high-memory GPUs
            return torch.optim.AdamW(
                model_parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                amsgrad=True
            )
        else:
            # Use SGD for memory-constrained systems
            return torch.optim.SGD(
                model_parameters,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True
            )
    
    def print_system_info(self):
        """Print detailed system and GPU information"""
        print("🖥️  System Information:")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   CPU Cores: {self.system_info['cpu_count']}")
        print(f"   Total RAM: {self.system_info['total_ram']} GB")
        print(f"   Python: {self.system_info['python_version']}")
        print(f"   PyTorch: {self.system_info['pytorch_version']}")
        print(f"   CUDA Available: {self.system_info['cuda_available']}")
        if self.system_info['cuda_version']:
            print(f"   CUDA Version: {self.system_info['cuda_version']}")
        
        print(f"\\n🎮 Available GPUs ({len(self.available_gpus)}):")
        for gpu in self.available_gpus:
            selected = " ⭐" if gpu == self.selected_gpu else ""
            print(f"   GPU {gpu.id}: {gpu.name}{selected}")
            print(f"      Vendor: {gpu.vendor.value}")
            print(f"      Memory: {gpu.total_memory} MB total, {gpu.free_memory} MB free")
            if gpu.compute_capability:
                print(f"      Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            print(f"      Features: FP16={gpu.supports_fp16}, INT8={gpu.supports_int8}, TensorRT={gpu.supports_tensorrt}")
        
        print(f"\\n⚙️  Optimization Settings:")
        for key, value in self.optimization_settings.items():
            print(f"   {key}: {value}")