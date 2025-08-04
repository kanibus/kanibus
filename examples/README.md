# 🎬 KANIBUS EXAMPLE WORKFLOWS

## 📋 **Available Workflows**

### **🚀 simple_eye_tracking.json**
**Basic eye tracking for testing installation**
- **Nodes**: VideoFrameLoader → NeuralPupilTracker → PreviewImage
- **Purpose**: Test if Kanibus is working properly
- **Performance**: Fast processing, minimal GPU usage
- **Use Case**: First-time testing, debugging

### **⚡ wan21_basic_tracking.json**  
**WAN 2.1 compatible workflow (480p)**
- **Resolution**: 854x480 (WAN 2.1 optimized)
- **FPS**: 24 FPS target
- **Features**: Eye tracking + ControlNet scribble
- **Use Case**: Standard video generation with eye control
- **Performance**: Balanced speed and quality

### **🎯 wan22_advanced_full.json**
**WAN 2.2 full pipeline (720p)**
- **Resolution**: 1280x720 (WAN 2.2 optimized)  
- **FPS**: 30 FPS target
- **Features**: Full multi-modal pipeline
  - Eye tracking
  - Facial landmarks (468 points)
  - Depth estimation
  - Emotion analysis
  - Multi-ControlNet application
- **Use Case**: Professional content creation
- **Performance**: High quality, requires powerful GPU

### **🌟 advanced_integrated_workflow.json** ⭐ **NEW**
**Advanced integrated pipeline with dual WAN compatibility**
- **Resolution**: Adaptive (720p default, 480p for WAN 2.1)
- **FPS**: 30 FPS target with temporal consistency
- **Features**: Complete integrated pipeline
  - Dual WAN 2.1/2.2 compatibility with auto-detection
  - High-precision eye tracking with 3D gaze estimation
  - 468-point facial landmarks
  - Real-time emotion analysis (7 basic + 15 micro-expressions)
  - Temporal frame consistency optimization
  - Multi-modal ControlNet integration
  - Performance monitoring and analytics
- **Use Case**: Enterprise-grade content creation and research
- **Performance**: Ultimate quality, requires 8GB+ GPU
- **Documentation**: See [ADVANCED_WORKFLOW_GUIDE.md](ADVANCED_WORKFLOW_GUIDE.md)

---

## 🛠️ **How to Use Workflows**

### **Method 1: Load in ComfyUI**
1. Open ComfyUI
2. Click "Load" button
3. Navigate to `ComfyUI/custom_nodes/kanibus/examples/`
4. Select desired workflow JSON file
5. Click "Load"

### **Method 2: Drag & Drop**
1. Open ComfyUI in browser
2. Drag the JSON file directly into the ComfyUI interface
3. Workflow will load automatically

---

## 📋 **Workflow Requirements**

### **✅ All Workflows Require:**
- **Kanibus nodes** installed and working
- **4 ControlNet models** downloaded (~5.6GB)
  - `control_v11p_sd15_scribble.pth`
  - `control_v11f1p_sd15_depth.pth`
  - `control_v11p_sd15_normalbae.pth` 
  - `control_v11p_sd15_openpose.pth`
- **Base model** (Stable Diffusion 1.5 or WAN compatible)
- **VAE model** (recommended: `vae-ft-mse-840000-ema-pruned.safetensors`)

### **🎯 Per Workflow:**

#### **simple_eye_tracking.json**
- **GPU Memory**: 4GB+ recommended
- **Models**: Only Kanibus (MediaPipe auto-downloads)
- **Input**: Video file or webcam

#### **wan21_basic_tracking.json**
- **GPU Memory**: 6GB+ recommended  
- **Models**: Base model + 2 ControlNet models minimum
- **Input**: Video file (will be resized to 480p)

#### **wan22_advanced_full.json**
- **GPU Memory**: 8GB+ recommended
- **Models**: Base model + all 4 ControlNet models
- **Input**: Video file (will be resized to 720p)
- **Processing**: Most demanding, highest quality

---

## 🔧 **Customization**

### **Common Parameters to Adjust:**

#### **VideoFrameLoader Node:**
```json
"widgets_values": [
  "your_video.mp4",  // Video file path
  0,                 // Start frame  
  -1,                // Frame count (-1 = all)
  1,                 // Step (1 = every frame)
  30.0,              // Target FPS
  1280,              // Width (-1 = original)
  720,               // Height (-1 = original)
  "high",            // Quality: low/medium/high/original
  true,              // Enable caching
  8,                 // Batch size
  32,                // Preload frames
  4.0,               // Memory limit GB
  "RGB"              // Color space
]
```

#### **NeuralPupilTracker Node:**
```json
"widgets_values": [
  1.0,     // Sensitivity (0.1-3.0)
  0.7,     // Smoothing (0.0-1.0)  
  0.25,    // Blink threshold (0.1-0.5)
  300.0,   // Saccade threshold (degrees/second)
  true,    // Enable 3D gaze
  true,    // Enable saccade detection
  true,    // Enable pupil dilation
  true     // Cache results
]
```

#### **KanibusMaster Node:**
```json
"widgets_values": [
  "video",        // Input source: image/video/webcam
  "streaming",    // Pipeline mode: real_time/batch/streaming/analysis
  "auto_detect",  // WAN version: wan_2.1/wan_2.2/auto_detect
  30.0,           // Target FPS
  true,           // Enable GPU optimization
  // ... feature enables and weights
]
```

---

## 🧪 **Testing Workflows**

### **Verification Steps:**
1. **Load workflow** in ComfyUI
2. **Check for red nodes** (missing models/nodes)
3. **Set input video** path in VideoFrameLoader
4. **Click "Queue Prompt"**
5. **Monitor progress** in ComfyUI console

### **Troubleshooting:**
- **Red nodes**: Missing models or nodes not installed
- **Slow processing**: Reduce batch size or resolution
- **Out of memory**: Lower quality settings or batch size
- **No output**: Check video file path and format

### **Performance Tips:**
- **Start with simple_eye_tracking.json** to test installation
- **Use wan21_basic_tracking.json** for standard use
- **Only use wan22_advanced_full.json** with powerful GPUs
- **Monitor GPU memory** usage during processing

---

## 📞 **Support**

If workflows don't work:
1. **Run diagnostic**: `python test_installation.py`
2. **Check models**: `python download_models.py --check-only`
3. **Verify installation**: Follow README.md installation guide
4. **Get help**: [GitHub Issues](https://github.com/kanibus/kanibus/issues)

---

*Example workflows for Kanibus v1.0.0 - Eye Tracking ControlNet System*