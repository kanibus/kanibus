# 🚀 ADVANCED INTEGRATED WORKFLOW GUIDE

## 📋 **Overview**

O **Advanced Integrated Workflow** é o workflow mais completo do Kanibus, combinando todas as capacidades dos sistemas WAN 2.1 e WAN 2.2 em um pipeline unificado e otimizado.

---

## ✨ **Features Integradas**

### **🎯 Processamento Dual-Compatible**
- **WAN 2.1 Support**: Otimização automática para 480p (854x480)
- **WAN 2.2 Support**: Processamento completo em 720p (1280x720)
- **Auto-Detection**: Detecção automática da versão WAN disponível
- **Adaptive Resolution**: Resolução adaptável baseada no modelo detectado

### **👁️ Advanced Eye Tracking**
- **Sub-pixel Accuracy**: Precisão de ±0.5 pixels
- **3D Gaze Estimation**: Vetores 3D de direção do olhar
- **Blink Detection**: Detecção avançada de piscadas
- **Saccade Analysis**: Análise de movimentos sacádicos
- **Pupil Dilation**: Monitoramento da dilatação pupilar

### **🧠 Comprehensive Analysis**
- **468-Point Landmarks**: Mapeamento facial completo
- **Emotion Recognition**: 7 emoções básicas + 15 micro-expressões
- **Real-time Processing**: Processamento em tempo real
- **Confidence Scoring**: Pontuação de confiança para todos os componentes

### **🎛️ Multi-Modal ControlNet**
- **Eye Mask Control**: Controle baseado em máscaras oculares
- **Depth Map Control**: Controle por mapas de profundidade
- **Normal Map Control**: Controle por mapas normais
- **Pose Control**: Controle opcional por pose corporal
- **Temporal Consistency**: Suavização temporal para vídeos

---

## 🏗️ **Architecture Overview**

### **📊 Pipeline Flow**
```
VideoFrameLoader (720p Adaptive)
    ↓
KanibusMaster (Full Pipeline) ──→ MultiControlNetApply (Dual WAN)
    ↓                                      ↓
[NeuralPupilTracker]              ControlNet Processing
    ↓                                      ↓
[LandmarkPro468] ──→ [EmotionAnalyzer]    Final Output
    ↓
TemporalSmoother ──→ Preview & Save
```

### **🔄 Processing Stages**

#### **Stage 1: Input Processing**
- **VideoFrameLoader**: Carregamento otimizado com cache inteligente
- **Adaptive Resolution**: 720p padrão com downscale automático para WAN 2.1
- **Batch Processing**: Processamento em lotes para eficiência

#### **Stage 2: Analysis Pipeline**
- **KanibusMaster**: Orchestração completa do pipeline
- **NeuralPupilTracker**: Rastreamento ocular de alta precisão
- **LandmarkPro468**: Detecção de 468 pontos faciais
- **EmotionAnalyzer**: Análise emocional em tempo real

#### **Stage 3: Temporal & ControlNet**
- **TemporalSmoother**: Consistência temporal para vídeos
- **MultiControlNetApply**: Aplicação integrada de múltiplos controles
- **WAN Optimization**: Otimização automática para versão detectada

#### **Stage 4: Output & Monitoring**
- **Preview Systems**: Visualização em tempo real
- **Save System**: Salvamento dos resultados finais
- **Performance Monitoring**: Métricas detalhadas de performance

---

## ⚙️ **Configuration Guide**

### **🎛️ Key Parameters**

#### **VideoFrameLoader Settings**
```json
{
  "target_fps": 30.0,        // FPS alvo para processamento
  "resize_width": 1280,      // Resolução padrão (adaptável)
  "resize_height": 720,      // Resolução padrão (adaptável)
  "quality": "high",         // Qualidade de processamento
  "batch_size": 8,           // Tamanho do lote
  "preload_frames": 64       // Frames pré-carregados
}
```

#### **KanibusMaster Settings**
```json
{
  "pipeline_mode": "streaming",    // Modo de pipeline
  "wan_version": "auto_detect",    // Detecção automática WAN
  "tracking_quality": "ultra",     // Qualidade máxima
  "temporal_smoothing": 0.8,       // Suavização temporal
  "enable_all_features": true      // Todos os recursos habilitados
}
```

#### **MultiControlNetApply Settings**
```json
{
  "eye_mask_weight": 1.3,     // Peso do controle ocular
  "depth_weight": 1.0,        // Peso do controle de profundidade
  "normal_weight": 0.7,       // Peso do controle normal
  "wan_version": "wan_2.2",   // Otimização WAN
  "cfg_scale": 7.5           // Escala CFG
}
```

---

## 📊 **Performance Requirements**

### **🖥️ Hardware Requirements**

#### **Minimum Configuration**
- **GPU**: 6GB VRAM (RTX 3060 ou similar)
- **RAM**: 16GB sistema
- **CPU**: 8 cores
- **Storage**: 10GB livres (SSD recomendado)

#### **Recommended Configuration**
- **GPU**: 8GB+ VRAM (RTX 3070+ ou similar)
- **RAM**: 32GB sistema
- **CPU**: 12+ cores
- **Storage**: 20GB livres em SSD

#### **Optimal Performance**
- **GPU**: 12GB+ VRAM (RTX 4070 Ti+ ou similar)
- **RAM**: 32GB+ sistema
- **CPU**: 16+ cores
- **Storage**: NVMe SSD

### **⚡ Performance Targets**

| Hardware Tier | Resolution | FPS | Latency | Features |
|---------------|------------|-----|---------|----------|
| **Minimum** | 480p | 15-20 FPS | <50ms | Core features only |
| **Recommended** | 720p | 24-30 FPS | <35ms | All features |
| **Optimal** | 720p | 30+ FPS | <25ms | Ultra quality |

---

## 🎯 **Usage Scenarios**

### **🎬 Video Production**
- **Input**: Arquivos MP4/AVI de alta qualidade
- **Output**: Controle preciso por eye tracking
- **Settings**: Quality="high", batch_size=8-16
- **Use Case**: Produção de conteúdo profissional

### **🎮 Real-time Applications**
- **Input**: Webcam ou stream ao vivo
- **Output**: Resposta em tempo real
- **Settings**: Quality="medium", temporal_smoothing=0.4
- **Use Case**: Aplicações interativas, jogos

### **🔬 Research & Analysis**
- **Input**: Datasets científicos
- **Output**: Dados detalhados de análise
- **Settings**: Quality="ultra", all_features=true
- **Use Case**: Pesquisa científica, análise comportamental

### **🏢 Enterprise Applications**  
- **Input**: Conteúdo corporativo
- **Output**: Controle consistente e profissional
- **Settings**: Quality="high", temporal_smoothing=0.8
- **Use Case**: Treinamento, apresentações, marketing

---

## 🔧 **Optimization Guide**

### **⚡ Performance Optimization**

#### **For WAN 2.1 (480p focus)**
```json
{
  "VideoFrameLoader": {
    "resize_width": 854,
    "resize_height": 480,
    "batch_size": 16
  },
  "KanibusMaster": {
    "wan_version": "wan_2.1",
    "tracking_quality": "medium"
  }
}
```

#### **For WAN 2.2 (720p focus)**
```json
{
  "VideoFrameLoader": {
    "resize_width": 1280,
    "resize_height": 720,
    "batch_size": 8
  },
  "KanibusMaster": {
    "wan_version": "wan_2.2",
    "tracking_quality": "high"
  }
}
```

### **💾 Memory Optimization**

#### **Low VRAM (6GB)**
- Reduce batch_size to 4
- Set quality to "medium"
- Disable non-essential features
- Use FP16 precision

#### **Standard VRAM (8GB)**
- Use default settings
- Enable all core features
- Monitor memory usage

#### **High VRAM (12GB+)**
- Increase batch_size to 16
- Set quality to "ultra"
- Enable all features
- Use maximum resolution

---

## 🧪 **Testing & Validation**

### **✅ Validation Checklist**

#### **Pre-execution**
- [ ] All 4 ControlNet models downloaded
- [ ] GPU memory >= 6GB available
- [ ] Input video file accessible
- [ ] ComfyUI base model loaded

#### **During execution**
- [ ] No red nodes (missing models/connections)
- [ ] GPU utilization 70-90%
- [ ] No memory overflow errors
- [ ] Smooth frame processing

#### **Post-execution**
- [ ] Output images generated successfully
- [ ] Eye tracking visualization accurate
- [ ] Emotion analysis showing results
- [ ] Temporal consistency maintained

### **🔍 Troubleshooting**

#### **Common Issues & Solutions**

**Issue**: Red nodes on load
```
Solution: Check ControlNet models installation
Command: python download_models.py --check-only
```

**Issue**: Out of memory errors
```
Solution: Reduce batch size and resolution
Settings: batch_size=4, quality="medium"
```

**Issue**: Slow processing
```
Solution: Check GPU utilization and optimization
Command: nvidia-smi (monitor usage)
```

**Issue**: Poor tracking quality
```
Solution: Increase sensitivity and quality settings
Settings: sensitivity=1.2, quality="high"
```

---

## 📈 **Monitoring & Analytics**

### **📊 Key Metrics**

#### **Performance Metrics**
- **Processing FPS**: Frames processados por segundo
- **GPU Utilization**: Utilização da GPU (70-90% ideal)
- **Memory Usage**: Uso de VRAM e RAM
- **Latency**: Tempo de processamento por frame

#### **Quality Metrics**
- **Eye Tracking Confidence**: Confiança do rastreamento ocular
- **Landmark Accuracy**: Precisão dos pontos faciais
- **Emotion Confidence**: Confiança da análise emocional
- **Temporal Consistency**: Consistência entre frames

### **📋 Output Data**

#### **Primary Outputs**
- **Processed Images**: Imagens finais processadas
- **Eye Tracking Data**: Dados completos de rastreamento
- **Emotion Scores**: Pontuações emocionais detalhadas
- **Performance Report**: Relatório de métricas

#### **Debug Outputs**
- **Confidence Scores**: Pontuações de confiança
- **Processing Times**: Tempos de processamento
- **Memory Usage**: Uso de recursos
- **Error Logs**: Logs de erro e avisos

---

## 🚀 **Getting Started**

### **Quick Start**
1. **Load Workflow**: Abrir `advanced_integrated_workflow.json` no ComfyUI
2. **Set Input**: Configurar caminho do vídeo no VideoFrameLoader
3. **Check Models**: Verificar se todos os modelos ControlNet estão carregados
4. **Execute**: Clicar "Queue Prompt" para iniciar processamento
5. **Monitor**: Acompanhar progresso e métricas de performance

### **First Time Setup**
1. **Install Models**: `python download_models.py`
2. **Test System**: `python test_installation.py`
3. **Load Workflow**: Carregar workflow no ComfyUI
4. **Test Run**: Executar com vídeo de teste pequeno
5. **Optimize**: Ajustar parâmetros baseado na performance

---

## 📞 **Support**

### **Documentation**
- [Main README](../README.md) - Instalação e configuração geral
- [Node Documentation](../docs/ENTERPRISE_NODE_DOCUMENTATION.md) - Referência completa dos nodes
- [Configuration Reference](../docs/CONFIGURATION_REFERENCE.md) - Parâmetros de configuração

### **Troubleshooting**
- [Installation Test](../test_installation.py) - Teste completo do sistema
- [Model Downloader](../download_models.py) - Download automático de modelos
- [GitHub Issues](https://github.com/kanibus/kanibus/issues) - Suporte da comunidade

---

**🎊 O Advanced Integrated Workflow representa o estado da arte em processamento de eye tracking para ComfyUI, combinando todas as capacidades do sistema Kanibus em um pipeline unificado e otimizado.**