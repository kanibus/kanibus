# ⚙️ KANIBUS CONFIGURATION REFERENCE

## 📖 **COMPREHENSIVE CONFIGURATION GUIDE**

*Complete reference for all configuration parameters, settings, and optimization options*

---

## 🎛️ **GLOBAL SYSTEM CONFIGURATION**

### **Core System Settings**
Located in `config.json` in the root directory:

```json
{
  "system": {
    "version": "1.0.0",
    "debug_mode": false,
    "logging_level": "INFO",
    "max_concurrent_processes": 4,
    "default_timeout_seconds": 30
  },
  "performance": {
    "enable_gpu_optimization": true,
    "enable_mixed_precision": true,
    "enable_tensorrt": true,
    "memory_management": "automatic",
    "cache_strategy": "intelligent"
  },
  "hardware": {
    "gpu_memory_limit_gb": 8.0,
    "cpu_thread_limit": 8,
    "enable_multi_gpu": false,
    "preferred_device": "auto"
  }
}
```

### **Parameter Descriptions**

#### **System Parameters**
| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|---------|
| `version` | string | "1.0.0" | System version identifier | Metadata only |
| `debug_mode` | boolean | false | Enable debug logging and features | Performance: -20% |
| `logging_level` | enum | "INFO" | Log verbosity: DEBUG/INFO/WARN/ERROR | Disk I/O impact |
| `max_concurrent_processes` | int | 4 | Maximum parallel processing threads | CPU/Memory scaling |
| `default_timeout_seconds` | int | 30 | Default operation timeout | Reliability vs speed |

#### **Performance Parameters**
| Parameter | Type | Default | Description | Performance Impact |
|-----------|------|---------|-------------|-------------------|
| `enable_gpu_optimization` | boolean | true | GPU acceleration features | +300% speed |
| `enable_mixed_precision` | boolean | true | FP16/FP32 automatic selection | +50% speed, -50% VRAM |
| `enable_tensorrt` | boolean | true | TensorRT optimization (NVIDIA) | +100% speed |
| `memory_management` | enum | "automatic" | Memory allocation strategy | Stability vs performance |
| `cache_strategy` | enum | "intelligent" | Caching approach | Memory vs speed trade-off |

#### **Hardware Parameters**
| Parameter | Type | Default | Description | Resource Impact |
|-----------|------|---------|-------------|-----------------|
| `gpu_memory_limit_gb` | float | 8.0 | Maximum GPU memory usage | OOM prevention |
| `cpu_thread_limit` | int | 8 | Maximum CPU threads | CPU resource limit |
| `enable_multi_gpu` | boolean | false | Multi-GPU distribution | Scaling capability |
| `preferred_device` | enum | "auto" | Device selection: auto/cuda/mps/cpu | Performance tier |

---

## 🧠 **KANIBUS MASTER CONFIGURATION**

### **Complete Parameter Reference**

```json
{
  "KanibusMaster": {
    "processing": {
      "input_source": "video",
      "pipeline_mode": "streaming",
      "wan_version": "auto_detect",
      "target_fps": 30.0,
      "enable_gpu_optimization": true
    },
    "features": {
      "enable_eye_tracking": true,
      "enable_face_tracking": true,
      "enable_body_tracking": false,
      "enable_object_tracking": false,
      "enable_depth_estimation": true,
      "enable_emotion_analysis": false,
      "enable_hand_tracking": false
    },
    "quality": {
      "tracking_quality": "high",
      "depth_quality": "medium",
      "temporal_smoothing": 0.7
    },
    "controlnet": {
      "eye_mask_weight": 1.3,
      "depth_weight": 1.0,
      "normal_weight": 0.7,
      "landmarks_weight": 0.9,
      "pose_weight": 0.6,
      "hands_weight": 0.5
    },
    "performance": {
      "batch_size": 4,
      "max_workers": 4,
      "memory_limit": 0.8,
      "enable_caching": true,
      "cache_size_gb": 5.0
    }
  }
}
```

### **Parameter Impact Analysis**

#### **Processing Settings**
```yaml
input_source:
  Options: ["image", "video", "webcam"]
  Performance Impact:
    - image: Minimal latency, single frame
    - video: Batch optimization, high throughput
    - webcam: Real-time constraints, buffer management
  
pipeline_mode:
  Options: ["real_time", "batch", "streaming", "analysis"]
  Resource Usage:
    - real_time: High CPU, low latency (16ms target)
    - batch: High GPU, high throughput (>100 FPS)
    - streaming: Balanced, continuous processing
    - analysis: Maximum accuracy, slower processing
    
wan_version:
  Options: ["wan_2.1", "wan_2.2", "auto_detect"]
  Optimization Impact:
    - wan_2.1: 480p optimized, faster processing
    - wan_2.2: 720p optimized, higher quality
    - auto_detect: Dynamic optimization, slight overhead
```

#### **Feature Toggles Performance Matrix**
| Feature | CPU Impact | GPU Impact | Memory Impact | Accuracy Gain |
|---------|------------|------------|---------------|---------------|
| Eye Tracking | +5ms | +2ms | +500MB | Baseline |
| Face Tracking | +3ms | +5ms | +800MB | +15% |
| Body Tracking | +15ms | +10ms | +1.2GB | +25% |
| Object Tracking | +20ms | +15ms | +1.5GB | +30% |
| Depth Estimation | +8ms | +12ms | +1.0GB | +20% |
| Emotion Analysis | +5ms | +3ms | +600MB | +10% |
| Hand Tracking | +10ms | +8ms | +900MB | +15% |

---

## 👁️ **NEURAL PUPIL TRACKER CONFIGURATION**

### **Advanced Parameter Tuning**

```json
{
  "NeuralPupilTracker": {
    "detection": {
      "sensitivity": 1.0,
      "smoothing": 0.7,
      "blink_threshold": 0.25,
      "saccade_threshold": 300.0
    },
    "features": {
      "enable_3d_gaze": true,
      "enable_saccade_detection": true,
      "enable_pupil_dilation": true,
      "enable_micro_saccades": false
    },
    "filtering": {
      "kalman_q_process": 0.01,
      "kalman_r_measurement": 0.1,
      "temporal_window": 5,
      "outlier_rejection": true
    },
    "performance": {
      "cache_results": true,
      "parallel_processing": true,
      "precision_mode": "balanced"
    }
  }
}
```

### **Sensitivity Calibration Guide**

#### **Environmental Factors**
```yaml
Lighting Conditions:
  Bright Office (>500 lux):
    sensitivity: 1.2-1.5
    blink_threshold: 0.27-0.30
    
  Standard Office (200-500 lux):
    sensitivity: 1.0-1.2
    blink_threshold: 0.24-0.27
    
  Dim Environment (<200 lux):
    sensitivity: 0.6-0.9
    blink_threshold: 0.20-0.24
    
Camera Quality:
  High-end (1080p+, good optics):
    sensitivity: 1.0-1.3
    smoothing: 0.5-0.7
    
  Standard Webcam (720p):
    sensitivity: 0.8-1.1
    smoothing: 0.7-0.9
    
  Low-end Camera (<720p):
    sensitivity: 0.5-0.8
    smoothing: 0.8-1.0
```

#### **User Demographics Calibration**
```yaml
Age Groups:
  Children (5-12):
    blink_threshold: 0.22-0.26
    saccade_threshold: 250-350
    
  Adults (18-65):
    blink_threshold: 0.24-0.28
    saccade_threshold: 300-400
    
  Elderly (65+):
    blink_threshold: 0.26-0.32
    saccade_threshold: 200-300

Ethnicity Considerations:
  East Asian:
    blink_threshold: 0.20-0.24
    sensitivity: 0.9-1.1
    
  European:
    blink_threshold: 0.24-0.28
    sensitivity: 1.0-1.2
    
  African:
    blink_threshold: 0.26-0.30
    sensitivity: 1.1-1.3
```

---

## 🎬 **VIDEO FRAME LOADER CONFIGURATION**

### **Optimization Strategies**

```json
{
  "VideoFrameLoader": {
    "input": {
      "video_path": "/path/to/video.mp4",
      "start_frame": 0,
      "frame_count": -1,
      "step": 1
    },
    "processing": {
      "target_fps": -1.0,
      "resize_width": -1,
      "resize_height": -1,
      "quality": "high",
      "color_space": "RGB"
    },
    "performance": {
      "enable_caching": true,
      "batch_size": 8,
      "preload_frames": 32,
      "memory_limit_gb": 4.0,
      "disk_cache_gb": 20.0
    },
    "codec": {
      "prefer_hardware_decode": true,
      "thread_count": 4,
      "buffer_size": 16
    }
  }
}
```

### **Performance Tuning Matrix**

#### **Storage Type Optimization**
```yaml
NVMe SSD:
  batch_size: 16-32
  preload_frames: 64-128
  cache_strategy: "aggressive"
  
SATA SSD:
  batch_size: 8-16
  preload_frames: 32-64
  cache_strategy: "balanced"
  
HDD:
  batch_size: 4-8
  preload_frames: 16-32
  cache_strategy: "conservative"
  
Network Storage:
  batch_size: 2-4
  preload_frames: 8-16
  cache_strategy: "minimal"
  enable_compression: true
```

#### **Resolution-Based Optimization**
```yaml
4K Video (3840x2160):
  batch_size: 2-4
  memory_limit_gb: 8.0
  quality: "medium"
  
1080p Video (1920x1080):
  batch_size: 8-16
  memory_limit_gb: 4.0
  quality: "high"
  
720p Video (1280x720):
  batch_size: 16-32
  memory_limit_gb: 2.0
  quality: "high"
  
480p Video (854x480):
  batch_size: 32-64
  memory_limit_gb: 1.0
  quality: "original"
```

---

## 🎛️ **MULTI-CONTROLNET CONFIGURATION**

### **WAN Version Optimization**

```json
{
  "MultiControlNetApply": {
    "wan_2_1": {
      "eye_mask_weight": 1.2,
      "depth_weight": 0.9,
      "normal_weight": 0.6,
      "pose_weight": 0.8,
      "cfg_scale": 7.0,
      "start_percent": 0.0,
      "end_percent": 1.0
    },
    "wan_2_2": {
      "eye_mask_weight": 1.3,
      "depth_weight": 1.0,
      "normal_weight": 0.7,
      "pose_weight": 0.9,
      "cfg_scale": 7.5,
      "start_percent": 0.0,
      "end_percent": 1.0
    },
    "advanced": {
      "hand_weight": 0.5,
      "face_landmarks_weight": 0.8,
      "emotion_weight": 0.4,
      "temporal_consistency": 0.6
    }
  }
}
```

### **Weight Tuning Guidelines**

#### **Control Strength Matrix**
```yaml
Eye Mask Control:
  Subtle: 0.8-1.0 (natural look)
  Standard: 1.1-1.3 (balanced control)
  Strong: 1.4-1.6 (dominant control)
  Maximum: 1.7+ (override other controls)
  
Depth Control:
  Minimal: 0.5-0.7 (depth hints)
  Balanced: 0.8-1.0 (standard depth)
  Strong: 1.1-1.3 (depth emphasis)
  Dominant: 1.4+ (depth-driven generation)
  
Normal Map Control:
  Subtle: 0.4-0.6 (surface hints)
  Standard: 0.7-0.9 (balanced normals)
  Strong: 1.0-1.2 (surface emphasis)
  Maximum: 1.3+ (geometry-driven)
```

---

## 🔧 **ADVANCED OPTIMIZATION STRATEGIES**

### **Memory Optimization Profiles**

#### **Profile 1: Memory Constrained (<8GB)**
```json
{
  "memory_profile": "constrained",
  "settings": {
    "batch_size": 1,
    "quality": "medium",
    "cache_size_gb": 1.0,
    "enable_mixed_precision": true,
    "enable_gradient_checkpointing": true,
    "features": {
      "enable_body_tracking": false,
      "enable_object_tracking": false,
      "enable_emotion_analysis": false
    }
  }
}
```

#### **Profile 2: Balanced Performance (8-16GB)**
```json
{
  "memory_profile": "balanced",
  "settings": {
    "batch_size": 4,
    "quality": "high",
    "cache_size_gb": 3.0,
    "enable_mixed_precision": true,
    "features": {
      "enable_body_tracking": false,
      "enable_object_tracking": false,
      "enable_emotion_analysis": true
    }
  }
}
```

#### **Profile 3: High Performance (16GB+)**
```json
{
  "memory_profile": "high_performance",
  "settings": {
    "batch_size": 8,
    "quality": "ultra",
    "cache_size_gb": 5.0,
    "enable_mixed_precision": false,
    "features": {
      "enable_body_tracking": true,
      "enable_object_tracking": true,
      "enable_emotion_analysis": true,
      "enable_hand_tracking": true
    }
  }
}
```

### **CPU Optimization Profiles**

#### **Profile 1: Low-End CPU (<8 cores)**
```json
{
  "cpu_profile": "low_end",
  "settings": {
    "max_workers": 2,
    "thread_pool_size": 4,
    "enable_multiprocessing": false,
    "preprocessing_threads": 1,
    "postprocessing_threads": 1
  }
}
```

#### **Profile 2: Mid-Range CPU (8-16 cores)**
```json
{
  "cpu_profile": "mid_range",
  "settings": {
    "max_workers": 4,
    "thread_pool_size": 8,
    "enable_multiprocessing": true,
    "preprocessing_threads": 2,
    "postprocessing_threads": 2
  }
}
```

#### **Profile 3: High-End CPU (16+ cores)**
```json
{
  "cpu_profile": "high_end",
  "settings": {
    "max_workers": 8,
    "thread_pool_size": 16,
    "enable_multiprocessing": true,
    "preprocessing_threads": 4,
    "postprocessing_threads": 4
  }
}
```

---

## 📊 **MONITORING & METRICS CONFIGURATION**

### **Performance Metrics Collection**

```json
{
  "monitoring": {
    "enabled": true,
    "collection_interval_seconds": 5,
    "retention_days": 30,
    "metrics": {
      "system": {
        "cpu_usage": true,
        "memory_usage": true,
        "gpu_usage": true,
        "disk_io": true,
        "network_io": false
      },
      "application": {
        "processing_fps": true,
        "queue_depth": true,
        "error_rate": true,
        "latency_percentiles": [50, 90, 95, 99],
        "accuracy_scores": true
      },
      "business": {
        "throughput": true,
        "cost_per_frame": true,
        "user_satisfaction": false,
        "feature_usage": true
      }
    }
  }
}
```

### **Alerting Configuration**

```json
{
  "alerting": {
    "enabled": true,
    "channels": ["email", "slack", "webhook"],
    "rules": {
      "critical": {
        "gpu_memory_usage": {
          "threshold": 95,
          "duration": "2m",
          "action": "scale_down_batch"
        },
        "processing_failure_rate": {
          "threshold": 10,
          "duration": "5m",
          "action": "restart_service"
        }
      },
      "warning": {
        "fps_degradation": {
          "threshold": 30,
          "baseline": "rolling_average_1h",
          "duration": "10m"
        },
        "queue_buildup": {
          "threshold": 50,
          "duration": "5m",
          "action": "scale_out"
        }
      }
    }
  }
}
```

---

## 🔐 **SECURITY CONFIGURATION**

### **Authentication & Authorization**

```json
{
  "security": {
    "authentication": {
      "enabled": true,
      "method": "jwt",
      "token_expiry_hours": 24,
      "refresh_token_enabled": true
    },
    "authorization": {
      "rbac_enabled": true,
      "default_role": "viewer",
      "admin_users": ["admin@company.com"],
      "api_key_required": true
    },
    "encryption": {
      "data_at_rest": true,
      "data_in_transit": true,
      "algorithm": "AES-256-GCM",
      "key_rotation_days": 90
    },
    "privacy": {
      "anonymize_faces": false,
      "blur_faces": false,
      "data_retention_days": 30,
      "gdpr_compliant": true,
      "audit_logging": true
    }
  }
}
```

---

## 📋 **CONFIGURATION VALIDATION**

### **Validation Rules**

```json
{
  "validation": {
    "required_fields": [
      "system.version",
      "performance.enable_gpu_optimization",
      "hardware.gpu_memory_limit_gb"
    ],
    "ranges": {
      "target_fps": [1.0, 120.0],
      "batch_size": [1, 64],
      "memory_limit": [0.1, 1.0],
      "sensitivity": [0.1, 3.0]
    },
    "dependencies": {
      "enable_tensorrt": "nvidia_gpu_required",
      "enable_mixed_precision": "modern_gpu_required",
      "multi_gpu": "multiple_gpus_required"
    }
  }
}
```

### **Configuration Testing**

```bash
# Validate configuration
python -m kanibus.config.validate config.json

# Test configuration performance
python -m kanibus.config.benchmark config.json

# Generate optimal configuration
python -m kanibus.config.optimize --hardware-scan
```

---

*Configuration Reference for Kanibus v1.0.0*
*For configuration support: config-help@kanibus.ai*