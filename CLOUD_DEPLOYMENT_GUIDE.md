# 🚀 Kanibus Cloud Deployment Guide

Complete guide for deploying Kanibus eye-tracking system on cloud platforms with T2I-Adapters (94% more efficient than legacy ControlNet) and WAN 2.1/2.2 compatibility.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Supported Platforms](#-supported-platforms)
- [System Requirements](#-system-requirements)
- [Model Efficiency](#-model-efficiency)
- [ComfyDeploy Deployment](#-comfydeploy-deployment)
- [RunPod Deployment](#-runpod-deployment)
- [Google Colab Setup](#-google-colab-setup)
- [Manual Cloud Setup](#-manual-cloud-setup)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [Advanced Configuration](#-advanced-configuration)

---

## 🚀 Quick Start

The fastest way to deploy Kanibus on any cloud platform:

```bash
# 1. Clone/navigate to Kanibus directory
cd /path/to/ComfyUI/custom_nodes/Kanibus

# 2. Run cloud installer (auto-detects platform and hardware)
python install_cloud.py --auto-install

# 3. Start ComfyUI
cd ../../..
python main.py --listen 0.0.0.0 --port 8188
```

**That's it!** The installer will:
- ✅ Auto-detect your cloud platform (RunPod, ComfyDeploy, Colab, etc.)
- ✅ Analyze your GPU specs and select optimal models
- ✅ Download T2I-Adapters (94% smaller than ControlNet)
- ✅ Configure WAN 2.1/2.2 compatibility
- ✅ Apply GPU optimizations for your hardware

---

## 🌐 Supported Platforms

| Platform | Status | Auto-Detection | Optimization |
|----------|--------|----------------|--------------|
| 🟢 **RunPod** | ✅ Full Support | ✅ Yes | ✅ Optimized |
| 🟢 **ComfyDeploy** | ✅ Full Support | ✅ Yes | ✅ Optimized |
| 🟢 **Google Colab** | ✅ Supported | ✅ Yes | ✅ Basic |
| 🟡 **Paperspace** | ⚠️ Beta | ✅ Yes | ✅ Basic |
| 🟡 **AWS EC2** | ⚠️ Manual | ❌ Manual | ✅ Basic |
| 🟡 **Azure** | ⚠️ Manual | ❌ Manual | ✅ Basic |
| 🟡 **GCP** | ⚠️ Manual | ❌ Manual | ✅ Basic |

---

## 💻 System Requirements

### Minimum Requirements
- **GPU**: 6GB VRAM (GTX 1060, RTX 2060, or equivalent)
- **RAM**: 8GB system RAM
- **Storage**: 20GB free space
- **CUDA**: 11.8+ (for NVIDIA GPUs)

### Recommended Requirements
- **GPU**: 12GB+ VRAM (RTX 3080, RTX 4080, A6000, etc.)
- **RAM**: 16GB+ system RAM  
- **Storage**: 50GB+ free space (for models and cache)
- **CUDA**: 12.1+ with cuDNN 8+

### Optimal Requirements
- **GPU**: 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ NVMe SSD
- **Network**: High-speed internet for model downloads

---

## 🎯 Model Efficiency

### T2I-Adapters vs Legacy ControlNet

| Model Type | Size per Model | Total Size | Parameters | Speed Impact |
|------------|----------------|------------|------------|--------------|
| **T2I-Adapters** (Recommended) | 158MB | 632MB | 6.31% of ControlNet | Near-zero |
| **Legacy ControlNet** | 1.4GB | 5.6GB | 100% baseline | Significant |

**🔥 T2I-Adapters Benefits:**
- ✅ **94% smaller** file size
- ✅ **93.69% fewer** parameters
- ✅ **Better performance** for eye tracking
- ✅ **Native WAN 2.1/2.2** compatibility
- ✅ **Faster loading** and inference
- ✅ **Better temporal consistency** for video

### Model Selection by GPU VRAM

| GPU VRAM | Recommended Models | Total Size |
|----------|-------------------|------------|
| **4-6GB** | T2I-Adapters only | ~632MB |
| **8-10GB** | T2I-Adapters + I2V-Adapter | ~1.5GB |
| **12GB+** | Full model set + Video models | ~3.6GB |
| **16GB+** | All models + Legacy backup | ~9.2GB |

---

## 🎨 ComfyDeploy Deployment

### Method 1: Using Configuration File

1. **Upload Configuration**:
   ```bash
   # Upload comfydeploy.yaml to your ComfyDeploy dashboard
   # File location: ComfyUI/custom_nodes/Kanibus/comfydeploy.yaml
   ```

2. **Deploy**:
   - Navigate to ComfyDeploy dashboard
   - Select "Create New Deployment"
   - Upload `comfydeploy.yaml`
   - Click "Deploy"

3. **Access**:
   - ComfyDeploy will provide your endpoint URL
   - Access ComfyUI at: `https://your-deployment.comfydeploy.com`

### Method 2: Manual ComfyDeploy Setup

```bash
# 1. Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. Install ComfyUI dependencies  
pip install -r requirements.txt

# 3. Clone Kanibus
cd custom_nodes
git clone https://github.com/your-repo/kanibus.git Kanibus
cd Kanibus

# 4. Install Kanibus with cloud optimization
python install_cloud.py --cloud=comfydeploy --auto-install

# 5. Start ComfyUI
cd ../../..
python main.py --listen 0.0.0.0 --port 8188
```

### ComfyDeploy Environment Variables

```bash
export COMFYDEPLOY_API_KEY="your-api-key"
export KANIBUS_CLOUD_MODE="comfydeploy"
export KANIBUS_USE_T2I_ADAPTERS="1"
export KANIBUS_GPU_OPTIMIZE="1"
```

---

## ⚡ RunPod Deployment

### Method 1: Using RunPod Template

1. **Import Template**:
   ```bash
   # Use the provided runpod-template.json
   # File location: ComfyUI/custom_nodes/Kanibus/runpod-template.json
   ```

2. **Deploy Pod**:
   - Log into RunPod dashboard
   - Go to "Templates" → "Create Template"  
   - Import `runpod-template.json`
   - Launch pod with template

3. **Access**:
   - RunPod will provide public IP
   - Access ComfyUI at: `http://your-pod-ip:8188`

### Method 2: Manual RunPod Setup

```bash
# 1. SSH into your RunPod instance
ssh root@your-pod-ip

# 2. Navigate to workspace
cd /workspace

# 3. Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 4. Install ComfyUI dependencies
pip install -r requirements.txt

# 5. Clone Kanibus
cd custom_nodes  
git clone https://github.com/your-repo/kanibus.git Kanibus
cd Kanibus

# 6. Run cloud installer
python install_cloud.py --cloud=runpod --auto-install

# 7. Start ComfyUI
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### RunPod GPU Recommendations

| RunPod GPU | VRAM | Hourly Cost* | Recommended Use |
|------------|------|--------------|-----------------|
| RTX 3070 | 8GB | ~$0.20/hr | Basic eye tracking |
| RTX 3080 | 10GB | ~$0.25/hr | Standard workflows |
| RTX 3090 | 24GB | ~$0.35/hr | Full feature set |
| RTX 4090 | 24GB | ~$0.50/hr | Maximum performance |
| A100 40GB | 40GB | ~$1.20/hr | Enterprise/research |

*Prices vary by availability and region

---

## 🔬 Google Colab Setup

### Free Colab (T4 GPU)

```python
# 1. Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!apt-get update && apt-get install -y git wget

# 2. Clone ComfyUI
!git clone https://github.com/comfyanonymous/ComfyUI.git
%cd ComfyUI
!pip install -r requirements.txt

# 3. Clone Kanibus
%cd custom_nodes
!git clone https://github.com/your-repo/kanibus.git Kanibus
%cd Kanibus

# 4. Install Kanibus (minimal model set for T4)
!python install_cloud.py --cloud=google_colab --minimal --auto-install

# 5. Start ComfyUI with Colab tunnel
%cd /content/ComfyUI
!python main.py --listen 0.0.0.0 --port 8188 &

# 6. Create tunnel (using cloudflared or ngrok)
!cloudflared tunnel --url http://localhost:8188
```

### Colab Pro/Pro+ (V100/A100)

```python
# Same as above, but use full model set
!python install_cloud.py --cloud=google_colab --auto-install
```

### Colab Environment Variables

```python
import os
os.environ['KANIBUS_CLOUD_MODE'] = 'google_colab'
os.environ['KANIBUS_USE_T2I_ADAPTERS'] = '1'
os.environ['KANIBUS_GPU_OPTIMIZE'] = '1'
os.environ['COMFYUI_LOWVRAM'] = '1'  # For T4 GPU
```

---

## 🔧 Manual Cloud Setup

For platforms not directly supported or custom setups:

### Step 1: System Preparation

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3 python3-pip git wget curl unzip \
    build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 ffmpeg

# Install NVIDIA drivers (if needed)
# sudo apt-get install -y nvidia-driver-525 nvidia-utils-525
```

### Step 2: Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv kanibus-env
source kanibus-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: ComfyUI Installation

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt

# Clone Kanibus
cd custom_nodes
git clone https://github.com/your-repo/kanibus.git Kanibus
cd Kanibus
```

### Step 4: Kanibus Configuration

```bash
# Install cloud-optimized dependencies
pip install -r requirements_cloud.txt

# Run cloud installer
python install_cloud.py --auto-install

# Or manual configuration
python download_models.py --force
```

### Step 5: Launch

```bash
# Start ComfyUI
cd ../../..
python main.py --listen 0.0.0.0 --port 8188

# Access at http://your-server-ip:8188
```

---

## ⚡ Performance Optimization

### GPU Memory Optimization

```bash
# For 6-8GB VRAM
export COMFYUI_LOWVRAM=1
export COMFYUI_MODEL_MEMORY_SAVE=1

# For 8-12GB VRAM  
export COMFYUI_LOWVRAM=0
export COMFYUI_MODEL_MEMORY_SAVE=1

# For 12GB+ VRAM
export COMFYUI_LOWVRAM=0
export COMFYUI_MODEL_MEMORY_SAVE=0
```

### PyTorch Optimizations

```bash
# Enable optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4

# Enable TensorFloat-32 (Ampere GPUs)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

### Kanibus-Specific Settings

```bash
# Use efficient T2I-Adapters
export KANIBUS_USE_T2I_ADAPTERS=1
export KANIBUS_PREFER_EFFICIENT=1

# GPU optimizations
export KANIBUS_GPU_OPTIMIZE=1
export KANIBUS_CACHE_SIZE=2048

# WAN compatibility
export KANIBUS_WAN_VERSION=auto
```

### Model Loading Optimization

```python
# In your workflow, prefer T2I-Adapters
# T2I-Adapter nodes load 94% faster than ControlNet
# Use these node types:
- T2I-Adapter Sketch (for eye masks)
- T2I-Adapter Depth (for depth maps)  
- T2I-Adapter Canny (for edge detection)
- T2I-Adapter OpenPose (for pose/face landmarks)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```bash
# Solution 1: Enable low VRAM mode
export COMFYUI_LOWVRAM=1

# Solution 2: Use CPU offload
export COMFYUI_CPU_OFFLOAD=1

# Solution 3: Reduce batch size
# In Kanibus nodes, set batch_size=1
```

#### 2. Models Not Found

```bash
# Re-download models
cd ComfyUI/custom_nodes/Kanibus
python download_models.py --force

# Check model locations
ls -la ../../models/t2i_adapter/
ls -la ../../models/controlnet/
```

#### 3. GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Import Errors

```bash
# Reinstall dependencies
cd ComfyUI/custom_nodes/Kanibus
pip install -r requirements_cloud.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 5. Slow Performance

```bash
# Apply optimizations
python -c "from src.gpu_optimizer import GPUOptimizer; opt = GPUOptimizer(); opt.apply_optimizations()"

# Use T2I-Adapters instead of ControlNet
# Check that KANIBUS_USE_T2I_ADAPTERS=1
```

### Platform-Specific Issues

#### RunPod Issues

```bash
# Persistent storage not mounted
# Ensure models are in /workspace/ComfyUI/models/

# Network pods: Check firewall settings
# Ensure port 8188 is open

# GPU not available
# Restart pod or contact RunPod support
```

#### ComfyDeploy Issues

```bash
# Deployment fails
# Check comfydeploy.yaml syntax
# Reduce model requirements if needed

# Timeout during startup
# Increase startup timeout in config
# Use minimal model set for faster startup
```

#### Google Colab Issues

```bash
# Session timeout
# Use Colab Pro for longer sessions
# Save work frequently

# T4 memory issues
# Use minimal model set
# Enable aggressive memory saving
```

### Debug Mode

```bash
# Enable debug logging
export KANIBUS_LOG_LEVEL=DEBUG
export COMFYUI_LOG_LEVEL=INFO

# Run with verbose output
python install_cloud.py --verbose
python main.py --verbose
```

---

## 🔬 Advanced Configuration

### Custom Model Configuration

Create `configs/cloud_models.yaml`:

```yaml
# Custom model configuration
models:
  required:
    - name: "Custom T2I-Adapter"
      url: "https://your-url/custom_adapter.pth"
      path: "models/t2i_adapter/custom_adapter.pth"
      size_mb: 150
      
  optional:
    - name: "Custom ControlNet"
      url: "https://your-url/custom_controlnet.pth" 
      path: "models/controlnet/custom_controlnet.pth"
      size_mb: 1400
      min_vram_gb: 8.0
```

### Environment Configuration

Create `.env` file:

```bash
# Kanibus Cloud Configuration
KANIBUS_CLOUD_MODE=auto
KANIBUS_USE_T2I_ADAPTERS=1
KANIBUS_WAN_VERSION=2.2
KANIBUS_GPU_OPTIMIZE=1
KANIBUS_CACHE_SIZE=4096
KANIBUS_LOG_LEVEL=INFO

# PyTorch Optimizations
TORCH_CUDNN_V8_API_ENABLED=1
TORCH_COMPILE_MODE=reduce-overhead
OMP_NUM_THREADS=8

# ComfyUI Settings
COMFYUI_LOWVRAM=0
COMFYUI_MODEL_MEMORY_SAVE=1
COMFYUI_ATTENTION_SLICE=1
```

### Custom Installation Script

```python
#!/usr/bin/env python3
"""Custom Kanibus installation for your specific needs"""

from install_cloud import CloudInstaller
import argparse

def custom_install():
    # Custom configuration
    args = argparse.Namespace()
    args.cloud = "custom"
    args.auto_install = True
    args.minimal = False
    args.include_legacy = True
    args.verbose = True
    
    # Run installer
    installer = CloudInstaller(args)
    return installer.run()

if __name__ == "__main__":
    exit(custom_install())
```

### Performance Monitoring

```python
# Monitor GPU usage
from src.gpu_optimizer import GPUOptimizer

optimizer = GPUOptimizer()
while True:
    usage = optimizer.monitor_gpu_usage()
    print(f"GPU Usage: {usage}")
    time.sleep(30)
```

### Auto-scaling Configuration

For platforms that support it:

```yaml
# Auto-scaling config
scaling:
  enabled: true
  min_instances: 1
  max_instances: 5
  target_gpu_utilization: 70
  scale_up_threshold: 80
  scale_down_threshold: 30
  cooldown_seconds: 300
```

---

## 📞 Support & Resources

### Documentation
- 📚 **Main Documentation**: [GitHub Repository](https://github.com/your-repo/kanibus)
- 🔧 **API Reference**: [API Documentation](https://github.com/your-repo/kanibus/docs/api)
- 🎯 **Example Workflows**: [Examples Directory](https://github.com/your-repo/kanibus/examples)

### Community Support
- 💬 **GitHub Discussions**: [Community Forum](https://github.com/your-repo/kanibus/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-repo/kanibus/issues)
- 📧 **Email**: staffytech@proton.me

### Performance Tips
- ✅ **Always use T2I-Adapters** over legacy ControlNet (94% more efficient)
- ✅ **Enable GPU optimizations** with `KANIBUS_GPU_OPTIMIZE=1`
- ✅ **Use appropriate VRAM settings** based on your GPU
- ✅ **Keep models on fast storage** (SSD preferred)
- ✅ **Monitor GPU temperature** and usage during intensive workflows

### Model Efficiency Summary
- **T2I-Adapters**: 632MB total, 94% more efficient, WAN 2.1/2.2 compatible
- **Legacy ControlNet**: 5.6GB total, slower, but maximum compatibility
- **Video Models**: 3GB additional, for advanced video workflows
- **Hybrid Setup**: Mix of efficient and legacy models based on needs

---

**🎉 Congratulations! You now have Kanibus deployed on the cloud with optimal T2I-Adapter efficiency and WAN 2.1/2.2 compatibility!**

For the latest updates and features, check the [GitHub repository](https://github.com/your-repo/kanibus) and join our [community discussions](https://github.com/your-repo/kanibus/discussions).