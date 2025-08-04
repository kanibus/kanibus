# 🧠 Kanibus Models Directory

## 📁 **Model Storage Strategy**

This directory is reserved for **custom and specialized models** that are not automatically downloaded by the standard libraries.

---

## 🎯 **Current Status: EMPTY (By Design)**

Kanibus v1.0.0 uses **external models** that are automatically downloaded:

### **🔄 Auto-Downloaded Models**
- **MediaPipe Face Mesh** - 468 landmarks + iris detection
- **Hugging Face Transformers** - DPT depth estimation models  
- **Ultralytics YOLO** - Object detection and pose estimation
- **Segment Anything (SAM)** - Advanced segmentation
- **PyTorch Vision Models** - Various computer vision models

These models are cached in standard locations:
- `~/.mediapipe/` - MediaPipe models
- `~/.cache/huggingface/` - Transformer models
- `~/.cache/torch/hub/` - PyTorch Hub models
- `~/.ultralytics/` - YOLO models

---

## 🔮 **Future Use Cases**

This directory is prepared for:

### **📁 Custom Models (v1.1+)**
```
models/
├── custom_eye_tracker_v2.pth      # Fine-tuned eye tracking
├── medical_gaze_analyzer.onnx     # Medical applications  
├── gaming_optimized_tracker.trt   # Gaming optimizations
├── emotion_micro_expressions.pth  # Advanced emotion AI
└── offline_models/                # Offline deployment models
    ├── face_mesh_offline.tflite
    ├── depth_estimation_lite.onnx
    └── pose_detection_mobile.trt
```

### **🎯 Specialized Models**
- **Domain-specific** - Medical, gaming, research variants
- **Performance-optimized** - TensorRT, ONNX, TensorFlow Lite
- **Offline variants** - For air-gapped environments
- **Fine-tuned models** - Trained on specific datasets
- **Quantized models** - For edge deployment

### **⚡ Performance Models**
- **TensorRT engines** - NVIDIA GPU optimization  
- **ONNX models** - Cross-platform inference
- **TensorFlow Lite** - Mobile/edge deployment
- **CoreML models** - Apple Silicon optimization

---

## 📋 **Model Requirements**

When adding custom models:

### **📝 Documentation**
- Add model description to this README
- Include performance benchmarks
- Document input/output specifications
- Provide usage examples

### **🔧 Code Integration**
- Add model loading in relevant node files
- Implement fallback to auto-downloaded models
- Add model validation and error handling
- Update configuration options

### **📊 Performance**
- Benchmark against baseline models
- Test memory usage and speed
- Validate accuracy metrics
- Check multi-GPU compatibility

---

## 🚀 **Adding Your Own Models**

### **1. Place Model Files**
```bash
# Copy your model files
cp your_model.pth models/
cp your_model.onnx models/
```

### **2. Update Node Code**
```python
# In your node file
model_path = os.path.join(os.path.dirname(__file__), "../models/your_model.pth")
if os.path.exists(model_path):
    custom_model = torch.load(model_path)
else:
    # Fallback to auto-downloaded model
    custom_model = load_default_model()
```

### **3. Update Documentation**
- Add to this README
- Update node documentation
- Include in CHANGELOG.md

---

## 📞 **Support**

For custom model integration:
- **Issues**: [GitHub Issues](https://github.com/kanibus/kanibus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kanibus/kanibus/discussions)
- **Email**: staffytech@proton.me

---

*This directory structure is prepared for future extensibility while keeping v1.0.0 simple and efficient.*